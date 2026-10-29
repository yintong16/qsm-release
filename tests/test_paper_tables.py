"""Paper commands count fresh gates, not stored counts or scaling formulas."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.mul import accumulate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/paper_tables.py"
spec = importlib.util.spec_from_file_location("paper_tables", SCRIPT)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def toy_square(extra=False):
    c = Circuit()
    x, acc = c.input("x", 5), c.input("acc", 5)
    result = accumulate.modsqrsub_inplace(c, x, acc, 31, carry_window=3,
        cmp_bits=3, window=3, ripple=2, diag_add_fn=accumulate._flat_diag)
    c.rebind("acc", result)
    if extra:
        c.ccx(x[0], x[1], result[0])
    return SimpleNamespace(circuit=c)


def test_square_observation_preserves_all_gates_and_counts_exclusive_spans():
    before = toy_square()
    observed, totals, calls, spans, parameters = paper.traced_square(toy_square, accumulate)
    assert observed.circuit.ops == before.circuit.ops
    assert count(observed.circuit) == count(before.circuit)
    assert sum(totals.values()) == count(before.circuit).toffoli
    assert calls["modhalve_inplace"] == calls["moddouble"] == 5
    assert calls["_signed_row"]+calls["_narrow_row"] == 4
    assert calls["modsub_direct"] == 1 and calls["modadd_lazy"] == 2
    assert parameters["ripple"] == 2 and parameters["ctrl"] is None
    ordered = sorted(spans, key=lambda s: s["start"])
    assert all(a["stop"] <= b["start"] for a, b in zip(ordered, ordered[1:]))


def test_unclassified_gates_are_not_hidden_in_the_diagonal_row():
    with pytest.raises(ValueError, match="unclassified CCX"):
        paper.traced_square(lambda: toy_square(extra=True), accumulate)


def test_trace_patches_are_restored_even_when_construction_fails():
    original = accumulate.moddouble
    def broken():
        raise RuntimeError("deliberate construction failure")
    with pytest.raises(RuntimeError, match="deliberate"):
        paper.traced_square(broken, accumulate)
    assert accumulate.moddouble is original


def test_formatting_derives_values_and_savings_from_measurements():
    values = {"25519": {k:i+2 for i,(k,_) in enumerate(paper.ROWS["table2"])},
              "secp256k1": {k:3*i+5 for i,(k,_) in enumerate(paper.ROWS["table2"])}}
    rows = paper.table_rows("table2", values)
    assert rows[0] == ["Modular addition", 2, 5, 3]
    assert rows[-1] == ["Square subtraction", 5, 14, 9]
    tex = paper.latex("table2", rows)
    assert r"\begin{tabularx}{\linewidth}{Xrrr}" in tex
    assert r"Multiplication by $F$" in tex
    assert "Square subtraction & 5 & 14 & 9" in tex
    assert "69,134" not in tex
    assert "| Square subtraction | 5 | 14 | 9 |" in paper.markdown("table2", rows)


def test_table1_caption_distinguishes_ripple_from_reduction_windows():
    values = {field: {key: i+1 for i, (key, _) in enumerate(paper.ROWS["table1"])}
              for field in paper.FIELDS}
    tex = paper.latex("table1", paper.table_rows("table1", values))
    assert "common 40-bit narrow-row ripple margin" in tex
    assert "45 bits for Curve25519 and 73 bits for secp256k1" in tex
    assert "legacy" not in tex
    assert dict(paper.PRIMITIVES)["square"] == paper.SQUARE_VERSION


def test_table3_formats_resource_pairs_and_computes_ccx_percentages():
    values = {
        "25519": {"default": {"ccx": 100, "peak_qubits": 1000},
                  "matched": {"ccx": 100, "peak_qubits": 1000}},
        "secp256k1": {"default": {"ccx": 150, "peak_qubits": 2000},
                      "matched": {"ccx": 125, "peak_qubits": 999}},
    }
    rows = paper.table_rows("table3", values)
    assert rows[0][3] == pytest.approx(100/3)
    assert rows[1][3] == 20
    tex = paper.latex("table3", rows)
    assert r"\label{tab:point-fields}" in tex
    assert r"261 / 264 & 100 / 1,000 & 150 / 2,000 & 33.33\%" in tex
    assert r"261 / 261 & 100 / 1,000 & 125 / 999 & 20.00\%" in tex
    assert "CCX savings relative to" in tex
    assert "| 261 / 261 | 100 / 1,000 | 125 / 999 | 20.00% |" in paper.markdown("table3", rows)
    assert "1,580,778" not in tex


@pytest.mark.parametrize("changes", [
    {"peak_qubits": None}, {"peak_qubits": True}, {"peak_qubits": 0},
    {"iterations": 264}, {"ccx": 0}, {"ccx": True}, {"circuit_key": "missing"},
])
def test_table3_rejects_invalid_resource_or_iteration_events(changes):
    event = dict(row="default", ccx=100, peak_qubits=10, iterations=261, circuit_key="iters_261")
    assert paper.measured_value("table3", "25519", event)["peak_qubits"] == 10
    with pytest.raises(ValueError):
        paper.measured_value("table3", "25519", {**event, **changes})


def test_point_observation_preserves_gates_and_records_actual_configuration(monkeypatch):
    from qsm import params
    from qsm.inversion.config import InversionConfig
    from qsm.point import affine

    def point(circ, x, y, xp, yp, ctrl=None, p=None, const_kw=None, sqr_kw=None, inv_kw=None):
        circ.ccx(x[0], y[0], x[1])
        return x, y

    monkeypatch.setattr(affine, "add_const_affine_inplace", point)
    config = InversionConfig.tuned(params.U, params.PRIME, iters=261)
    build = lambda: affine._affine_case(False, config)
    before = build()
    observed, actual = paper.traced_point(build, affine)
    assert observed.circuit.ops == before.circuit.ops
    assert actual["inversion_config"] == paper.asdict(config)
    assert actual["square"]["ripple"] == 40
    assert actual["ctrl"] is None
    assert actual["constant_point"] == {"x": affine.curve.GX, "y": affine.curve.GY}
    assert affine.add_const_affine_inplace is point
    with pytest.raises(ValueError, match="exactly one"):
        paper.traced_point(lambda: None, affine)
    assert affine.add_const_affine_inplace is point
    def broken():
        raise RuntimeError("deliberate point build failure")
    with pytest.raises(RuntimeError, match="deliberate"):
        paper.traced_point(broken, affine)
    assert affine.add_const_affine_inplace is point


def test_point_worker_reuses_only_identical_fresh_configuration(monkeypatch):
    from qsm import params
    from qsm.bench import registry
    from qsm.point import affine

    def get_builder(version):
        assert version == paper.POINT_VERSION
        return affine._build_add_const_affine_inplace_v2
    monkeypatch.setattr(registry, "get_builder", get_builder)
    seen = []
    def point(circ, x, y, xp, yp, ctrl=None, p=None, const_kw=None, sqr_kw=None, inv_kw=None):
        seen.append(inv_kw["config"].iters)
        circ.ccx(x[0], y[0], x[1])
        return x, y

    monkeypatch.setattr(affine, "add_const_affine_inplace", point)
    events = []
    def send(kind, **kwargs):
        events.append(dict(kind=kind, **kwargs))
    records, values = paper.measure_points(params.FIELD, params, send)
    expected = [261] if params.FIELD == "25519" else [264, 261]
    assert seen == expected
    assert len(records) == len(expected)
    assert [e["iterations"] for e in events if e["kind"] == "building"] == expected
    assert [e["row"] for e in events if e["kind"] == "value"] == ["default", "matched"]
    for row, value in values.items():
        assert value["iterations"] == paper.POINT_ITERATIONS[params.FIELD][row]
        assert value["ccx"] == 1
        assert records[value["circuit_key"]]["actual_parameters"]["inversion_config"]["iters"] == value["iterations"]
    if params.FIELD == "25519":
        assert values["default"] == values["matched"]
    else:
        assert records["iters_261"]["overrides"] == {"iters": 261}


def test_help_and_list_do_not_build_or_create_outputs(tmp_path):
    for args in ([], ["--help"], ["--list"]):
        result = subprocess.run([sys.executable, "-I", str(SCRIPT), *args], cwd=tmp_path,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert not list(tmp_path.iterdir())
        assert "table1" in result.stdout
        if args == ["--list"]:
            assert "legacy" not in result.stdout
            assert result.stdout.count("ripple40/40") == 2


def test_existing_output_path_is_not_overwritten(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("user data")
    result = subprocess.run([sys.executable, "-I", str(SCRIPT), "table1", "--out", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert sentinel.read_text() == "user data"
    assert not paper.benchmark_files.paths(tmp_path, "table1")["tex"].exists()


def test_wrapper_fails_clearly_without_a_project_interpreter(tmp_path):
    wrapper = tmp_path / "paper"
    wrapper.write_bytes((ROOT / "paper").read_bytes())
    wrapper.chmod(0o755)
    result = subprocess.run([str(wrapper), "table1"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "./setup.sh" in result.stderr


@pytest.mark.parametrize("table,expected", [
    ("table1", [[11475,18688,7213], [11475,18688,7213], [44911,46325,1414],
                [1273,1616,343], [69134,85317,16183]]),
    ("table2", [[339,368,29], [45,73,28], [2299,4495,2196], [69134,85317,16183]]),
])
def test_reviewer_wrapper_freshly_builds_both_fields_from_any_directory(tmp_path, table, expected):
    output = tmp_path / table
    result = subprocess.run([str(ROOT / "paper"), table, "--out", str(output)], cwd=tmp_path,
        env=dict(os.environ, QSM_FIELD="deliberately-invalid-parent-value"),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    data = json.loads(paper.benchmark_files.paths(output, table)["json"].read_text())
    assert data["complete"] and data["measurement_mode"].startswith("fresh")
    assert [r[1:] for r in data["rows"]] == expected
    assert set(data["fields"]) == {"25519", "secp256k1"}
    assert "fresh circuit builds started" in result.stdout
    assert "building and counting" in result.stderr
    assert paper.benchmark_files.paths(output, table)["tex"].read_text() == paper.latex(table, data["rows"])
    assert paper.benchmark_files.paths(output, table)["md"].read_text() == paper.markdown(table, data["rows"])
    for field, result in data["fields"].items():
        assert result["source_digest"] == data["source_digest"]
        assert result["runner_sha256"] == data["runner_sha256"]
        for circuit in result["circuits"].values():
            assert circuit["operation_nodes"] > 0 and len(circuit["ir_digest"]) == 64
        square = result["circuits"]["square"]
        assert square["version"] == paper.SQUARE_VERSION
        assert square["actual_parameters"]["ripple"] == 40
        assert square["actual_parameters"]["carry_window"] == (45 if field == "25519" else 73)
        assert sum(square["components"].values()) == square["resources"]["toffoli"]


@pytest.mark.slow
def test_table3_reviewer_wrapper_rebuilds_three_distinct_point_circuits(tmp_path):
    table = "table3"
    output = tmp_path / table
    result = subprocess.run([str(ROOT / "paper"), "table3", "--out", str(output)], cwd=tmp_path,
        env=dict(os.environ, QSM_FIELD="deliberately-invalid-parent-value"),
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout+result.stderr
    data = json.loads(paper.benchmark_files.paths(output, table)["json"].read_text())
    assert data["complete"] and data["measurement_mode"].startswith("fresh")
    expected = [(1580778, 1173, 1658011, 1206), (1580778, 1173, 1647581, 1199)]
    for (_, a, b, saving), (ac, aq, bc, bq) in zip(data["rows"], expected, strict=True):
        assert (a["ccx"], a["peak_qubits"], b["ccx"], b["peak_qubits"]) == (ac, aq, bc, bq)
        assert saving == pytest.approx(100*(bc-ac)/bc)
    assert "1,580,778 / 1,173" in result.stdout
    assert "4.66%" in result.stdout and "4.05%" in result.stdout
    assert result.stderr.count("building and counting") == 3
    assert paper.benchmark_files.paths(output, table)["tex"].read_text() == paper.latex("table3", data["rows"])
    assert paper.benchmark_files.paths(output, table)["md"].read_text() == paper.markdown("table3", data["rows"])
    for field, measured in data["fields"].items():
        assert measured["source_digest"] == data["source_digest"]
        assert measured["runner_sha256"] == data["runner_sha256"]
        expected_iters = {261} if field == "25519" else {264, 261}
        assert set(measured["circuits"]) == {f"iters_{i}" for i in expected_iters}
        for key, record in measured["circuits"].items():
            actual = record["actual_parameters"]
            assert key == f"iters_{actual['inversion_config']['iters']}"
            assert actual["square"]["ripple"] == 40
            assert actual["ctrl"] is None
            assert record["version"] == paper.POINT_VERSION
            assert record["operation_nodes"] > 0 and len(record["ir_digest"]) == 64
        default, matched = (measured["values"][key] for key in ("default", "matched"))
        if field == "25519":
            assert default == matched
        else:
            assert measured["circuits"]["iters_264"]["overrides"] == {}
            assert measured["circuits"]["iters_261"]["overrides"] == {"iters": 261}


def test_child_failure_never_falls_back_to_published_numbers(tmp_path):
    table = "table2"
    project = tmp_path / "broken-project"
    script_dir = project / "scripts"
    script_dir.mkdir(parents=True)
    script = script_dir / SCRIPT.name
    script.write_bytes(SCRIPT.read_bytes())
    naming = Path(paper.benchmark_files.__file__)
    (script_dir / naming.name).write_bytes(naming.read_bytes())
    # The isolated child sees this broken local package rather than the installed checkout.
    package = project / "qsm"
    package.mkdir()
    (package / "__init__.py").write_text('raise RuntimeError("deliberately broken live source")\n')
    output = tmp_path / "failed-run"
    result = subprocess.run([sys.executable, "-I", str(script), "table2", "--out", str(output)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    data = json.loads(paper.benchmark_files.paths(output, table)["json"].read_text())
    assert data["complete"] is False and data["error"]
    assert not paper.benchmark_files.paths(output, table)["tex"].exists()
    assert not paper.benchmark_files.paths(output, table)["md"].exists()


@pytest.mark.parametrize("failure", [None, "worker", "source"])
def test_table_prints_once_after_both_workers_finish_and_validation_passes(monkeypatch, tmp_path, failure):
    import io

    table, output = "table2", tmp_path / "completed"
    files = paper.benchmark_files.paths(output, table)
    launched, finished, printed = [], set(), []

    class Process:
        def __init__(self, command, **kwargs):
            self.field = command[command.index("--worker-field")+1]
            self.code = int(failure == "worker" and self.field == "secp256k1")
            launched.append(self.field)
            values = {key: i+1 for i, (key, _) in enumerate(paper.ROWS[table])}
            source = "mismatched-source" if failure == "source" and self.field == "secp256k1" else "test-source"
            events = [dict(kind="building", field=self.field, version="test.primitive")]
            events += [dict(kind="value", field=self.field, row=key, ccx=value)
                       for key, value in values.items()]
            events.append(dict(kind="complete", field=self.field, values=values, circuits={},
                               source_digest=source, runner_sha256=paper.file_sha(Path(paper.__file__))))
            self.stdout = io.StringIO("".join(json.dumps(event)+"\n" for event in events))
        def wait(self, timeout=None):
            assert set(launched) == set(paper.FIELDS)
            finished.add(self.field)
            return self.code
        def poll(self):
            return self.code

    def capture(*args, **kwargs):
        text = " ".join(map(str, args))
        printed.append(text)
        if "Modular addition" in text or text.lstrip().startswith("Operation"):
            assert finished == set(paper.FIELDS)
            assert json.loads(files["json"].read_text())["complete"]

    monkeypatch.setattr(paper.subprocess, "Popen", Process)
    monkeypatch.setattr(paper, "print", capture, raising=False)
    if failure is None:
        assert paper.run(table, output) == output
        blocks = [text for text in printed if text.lstrip().startswith("Operation")]
        assert len(blocks) == 1
        assert all(label in blocks[0] for _, label in paper.ROWS[table])
        assert any("building and counting" in text for text in printed)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            paper.run(table, output)
        assert not any("Modular addition" in text or text.lstrip().startswith("Operation") for text in printed)
        assert not json.loads(files["json"].read_text())["complete"]
        assert not files["tex"].exists() and not files["md"].exists()
