"""Table 6 uses one fresh odd-only worker plus explicitly quoted reference rows."""
import io
import json
import math
from pathlib import Path

import pytest

from qsm.bench import calibration
from qsm.bench.calibration import source_digest
from test_signed_paper_bench import bench, data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_real_workers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Table 6 tests must never launch a real scalar benchmark")
    monkeypatch.setattr(bench.subprocess, "Popen", forbidden)


def sample(source, *, addition=37, peak=65):
    result = data("25519", source)
    row, = result["rows"]
    row["variant"] = "odd_torsion_add"
    row.pop("zero_safe")
    for stage in row["stages"][1:]:
        stage["toffoli"] = addition
        stage["global_peak_so_far"] = peak
    row["resources"].update(toffoli=5+28*addition, peak_qubits=peak)
    return result


def summary(**kwargs):
    return bench.verify(sample("fresh", **kwargs), "25519", "fresh", variant="odd_torsion_add")


def mock_worker(monkeypatch, *, mutate=None, code=0, emit=True, on_wait=None):
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs["cwd"] == ROOT
            result = sample(source_digest())
            if mutate is not None:
                mutate(result["rows"][0])
            output = Path(command[command.index("--json")+1])
            if emit:
                output.write_text(json.dumps(result))
            self.stdout = io.StringIO("synthetic odd-only progress\n")
            self.on_wait = on_wait
            launched.append(command)

        def wait(self, timeout=None):
            assert len(launched) == 1
            if self.on_wait is not None:
                self.on_wait()
                self.on_wait = None
            return code

        def poll(self):
            return code

    monkeypatch.setattr(bench.subprocess, "Popen", Process)
    return launched


def failed_manifest(output):
    paths = bench.benchmark_files.paths(output, "table6")
    manifest = json.loads(paths["json"].read_text())
    assert manifest["complete"] is False and manifest["error"]
    assert not paths["tex"].exists() and not paths["md"].exists()
    assert len(manifest["reference_rows"]) == 2
    return manifest


def test_table6_job_is_exactly_the_fresh_wei_odd_schedule(tmp_path):
    job, = bench.comparison_jobs(tmp_path)
    assert job == bench.scalar_job(tmp_path, "25519", "odd_torsion_add", "odd_torsion_add")
    command = job["command"]
    assert command[1:6] == ["-I", "-B", "-u", "-m", "qsm"]
    assert "full-scalar" in command
    for option, value in (("--field", "25519"), ("--variants", "odd_torsion_add"),
                          ("--u-bits", "253"), ("--v-bits", "208"),
                          ("--window-bits", "16"), ("--samples", "4"),
                          ("--seed", str(bench.SEED))):
        assert command[command.index(option)+1] == value
    assert "secp256k1" not in command and "signed_zero_safe" not in command
    assert summary()["diagnostics"] == 6


@pytest.mark.parametrize("addition,peak", [(37, 65), (1000, 77)])
def test_table6_formatter_uses_dynamic_wei_values_and_literal_quoted_rows(addition, peak):
    result = summary(addition=addition, peak=peak)
    tex, md = bench.comparison_table([result])
    assert r"\caption{Overall resource comparison with published \texttt{secp256k1} estimates.}" in tex
    assert r"\label{tab:full-comparison}" in tex
    assert r"\begin{tabularx}{\linewidth}{Xrr}" in tex
    assert r"Construction & Toffoli count & Logical qubits\\" in tex
    for label, exponent, qubits in (("space-optimized", "26.11", 1208),
                                    ("gate-optimized", "25.78", 1462)):
        quoted = rf"\texttt{{secp256k1}}, {label}~\cite{{Schrottenloher26}}"
        assert quoted+rf" & $2^{{{exponent}}}$ & {qubits:,}\\" in tex
        assert rf"| secp256k1, {label} (Schrottenloher26) | $2^{{{exponent}}}$ | {qubits:,} |" in md
    total = 5+28*addition
    count = rf"${total:,}\approx 2^{{{math.log2(total):.2f}}}$"
    assert f"Wei25519, this work & {count} & {peak:,}"+r"\\" in tex
    assert f"| Wei25519, this work | {count} | {peak:,} |" in md
    assert tex.count(r"\cite{Schrottenloher26}") == 2
    assert tex.index("space-optimized") < tex.index("gate-optimized") < tex.index("Wei25519")
    assert "47,199,530" not in tex and "1189" not in tex


@pytest.mark.parametrize("change", [
    {"field": "secp256k1"}, {"variant": "signed_zero_safe"}, {"allocation": "255/208"},
    {"strict_clean": 5}, {"diagnostics": 7}, {"total_ccx": 0}, {"total_ccx": True},
    {"total_ccx": 1.5}, {"peak_qubits": None}, {"peak_qubits": -1}, {"peak_qubits": False},
])
def test_table6_formatter_rejects_wrong_or_failed_measurement(change):
    with pytest.raises(ValueError):
        bench.comparison_table([{**summary(), **change}])


def test_table6_formatter_rejects_missing_or_extra_fresh_rows():
    for results in ([], [summary(), summary()]):
        with pytest.raises(ValueError, match="exactly one"):
            bench.comparison_table(results)


def test_table6_formats_reference_resource_values():
    result = dict(field="25519", variant="odd_torsion_add", allocation="253/208",
                  total_ccx=47199530, peak_qubits=1189, strict_clean=6, diagnostics=6)
    tex, md = bench.comparison_table([result])
    assert r"Wei25519, this work & $47,199,530\approx 2^{25.49}$ & 1,189\\" in tex
    assert r"| Wei25519, this work | $47,199,530\approx 2^{25.49}$ | 1,189 |" in md


def test_table6_dispatch_writes_one_measurement_and_separate_quoted_references(monkeypatch, tmp_path, capsys):
    launched = mock_worker(monkeypatch)
    output = tmp_path/"fresh"
    assert bench.run(output, table6=True) == 0
    assert launched == [bench.comparison_jobs(output)[0]["command"]]
    paths = bench.benchmark_files.paths(output, "table6")
    assert paths["json"].name == "resource-comparison.json"
    manifest = json.loads(paths["json"].read_text())
    assert manifest["complete"] and manifest["table"] == "table6"
    assert "Wei25519 circuit only" in manifest["measurement_mode"]
    assert "fixed user-supplied literature quotations" in manifest["measurement_mode"]
    assert manifest["results"] == [summary()]
    assert len(manifest["jobs"]) == 1
    job, = manifest["jobs"]
    assert job["variant"] == "odd_torsion_add" and job["field"] == "25519"
    assert job["output_sha256"] and job["exit_code"] == 0
    assert manifest["source_digest"] == source_digest()
    assert manifest["runner_sha256"] and manifest["output_naming_sha256"]
    assert [(r["toffoli_log2"], r["logical_qubits"]) for r in manifest["reference_rows"]] == [(26.11, 1208), (25.78, 1462)]
    for reference in manifest["reference_rows"]:
        assert reference["citation_key"] == "Schrottenloher26"
        assert reference["provenance"] == "quoted user-supplied literature reference values"
        assert reference["fresh_measurement"] is False and reference["independently_verified"] is False
    tex, md = bench.comparison_table(manifest["results"])
    assert paths["tex"].read_text() == tex and paths["md"].read_text() == md
    assert sorted(p.name for p in output.iterdir()) == [
        "odd_torsion_add.json", "odd_torsion_add.log", "resource-comparison.json",
        "resource-comparison.md", "resource-comparison.tex",
    ]
    printed = capsys.readouterr().out
    assert "only the Wei25519 odd-only circuit is built" in printed
    assert "External secp256k1 rows are fixed quoted user-supplied" in printed
    assert "not fresh measurements or independently verified" in printed
    assert "Measured Wei25519, this work: 1,041 CCX total; 65 logical qubits" in printed
    assert "Completed fresh table6 measurements" in printed


@pytest.mark.parametrize("key,value,error", [
    ("source_digest", "stale", "identity mismatch"),
    ("variant", "signed_zero_safe", "identity mismatch"),
    ("full_width_test_settings", True, "identity mismatch"),
    ("restoration_exact_reduction", False, "exact restoration"),
    ("coordinate_restoration", "subtract", "restoration policy"),
    ("restoration_unlookup", "unpaired", "restoration policy"),
    ("point_additions", 27, "incomplete"),
    ("diagnostics", [], "incomplete"),
])
def test_table6_rejects_identity_and_policy_failures(monkeypatch, tmp_path, key, value, error):
    mock_worker(monkeypatch, mutate=lambda row: row.update({key: value}))
    output = tmp_path/"failed"
    with pytest.raises(ValueError, match=error):
        bench.run(output, table6=True)
    failed_manifest(output)


@pytest.mark.parametrize("part,key,value,error", [
    ("resources", "toffoli", 1, "resource mismatch"),
    ("resources", "clifford", 1, "resource mismatch"),
    ("resources", "measurements", 1, "resource mismatch"),
    ("resources", "peak_qubits", 64, "peak mismatch"),
    ("initial", "bits", 16, "initializer"),
    ("addition", "kind", "initial_lookup", "addition costs"),
])
def test_table6_reuses_stage_resource_and_peak_validation(monkeypatch, tmp_path, part, key, value, error):
    def mutate(row):
        target = row["resources"] if part == "resources" else row["stages"][0 if part == "initial" else 1]
        target[key] = value
    mock_worker(monkeypatch, mutate=mutate)
    output = tmp_path/"failed"
    with pytest.raises(ValueError, match=error):
        bench.run(output, table6=True)
    failed_manifest(output)


@pytest.mark.parametrize("key,value", [
    ("observed_pass", False), ("active", False), ("phase", 1), ("output", [9, 9]),
    ("phase_stages", [2]), ("noncanonical_stages", [2]), ("value_failure", "failure"),
    ("control_failure", "failure"), ("domain_failure", "failure"), ("dirty_failure", "failure"),
])
def test_table6_strict_failures_gate_completion_even_when_worker_exits_zero(monkeypatch, tmp_path, key, value):
    mock_worker(monkeypatch, mutate=lambda row: row["diagnostics"][0].update({key: value}))
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="Table 6 diagnostics failed"):
        bench.run(output, table6=True)
    manifest = failed_manifest(output)
    assert manifest["results"][0]["strict_clean"] == 5
    assert Path(manifest["jobs"][0]["output"]).exists()


@pytest.mark.parametrize("code,emit,dirty,error", [
    (2, True, False, "build failed"), (1, False, False, "build failed"),
    (1, True, False, "worker failed despite"), (1, True, True, "diagnostics failed"),
])
def test_table6_worker_failures_never_emit_a_completed_table(monkeypatch, tmp_path, code, emit, dirty, error):
    def mutate(row):
        if dirty:
            row["diagnostics"][0]["phase_stages"] = [2]
    mock_worker(monkeypatch, code=code, emit=emit, mutate=mutate)
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match=error):
        bench.run(output, table6=True)
    failed_manifest(output)


@pytest.mark.parametrize("changed", ["source", "runner", "naming"])
def test_table6_detects_provenance_changes_during_measurement(monkeypatch, tmp_path, changed):
    on_wait = None
    if changed == "source":
        digests = iter([source_digest(), "changed"])
        monkeypatch.setattr(calibration, "source_digest", lambda: next(digests))
    elif changed == "naming":
        digests = iter(["original", "changed"])
        monkeypatch.setattr(bench.benchmark_files, "digest", lambda: next(digests))
    else:
        runner = tmp_path/"runner.py"
        runner.write_text("original runner")
        monkeypatch.setattr(bench, "__file__", str(runner))
        on_wait = lambda: runner.write_text("changed runner")
    mock_worker(monkeypatch, on_wait=on_wait)
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="changed while measuring"):
        bench.run(output, table6=True)
    failed_manifest(output)


@pytest.mark.parametrize("kwargs", [
    {"table4": True, "table6": True}, {"table6": True, "field": "25519"},
    {"table6": True, "field": "secp256k1"},
])
def test_table6_invalid_modes_rejected_before_creating_output(tmp_path, kwargs):
    output = tmp_path/"never-created"
    with pytest.raises(ValueError):
        bench.run(output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("args", [
    ["--table4", "--table6"], ["--table6", "--field", "25519"],
    ["--table6", "--field", "secp256k1"],
])
def test_table6_cli_rejects_conflicting_options_before_dispatch(monkeypatch, args):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid Table 6 CLI options reached run")
    monkeypatch.setattr(bench, "run", forbidden)
    with pytest.raises(SystemExit) as error:
        bench.main(args)
    assert error.value.code == 2


def test_table6_cli_forwards_mode_and_output(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bench, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or 7)
    assert bench.main(["--table6", "--out", str(tmp_path)]) == 7
    assert calls == [((tmp_path, None), {"table4": False, "table6": True})]


def test_paper_interface_dispatches_table6_without_a_field_override(monkeypatch, tmp_path):
    from test_paper_tables import paper
    commands = []
    monkeypatch.setattr(paper.subprocess, "call", lambda command: commands.append(command) or 7)
    assert paper.main(["table6", "--out", str(tmp_path/"new")]) == 7
    command, = commands
    assert str(ROOT/"scripts/signed_paper_bench.py") in command
    assert "--table6" in command and "--field" not in command
    with pytest.raises(SystemExit):
        paper.main(["table6", "--field", "secp256k1"])
    assert len(commands) == 1


def test_table6_never_overwrites_existing_output(tmp_path):
    sentinel = tmp_path/"keep"
    sentinel.write_text("previous measurements")
    with pytest.raises(FileExistsError):
        bench.run(tmp_path, table6=True)
    assert sentinel.read_text() == "previous measurements"
    assert list(tmp_path.iterdir()) == [sentinel]
