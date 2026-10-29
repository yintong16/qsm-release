"""Exclusive odd-encoding savings, provenance and reviewer dispatch."""
import importlib.util
import io
import json
from pathlib import Path

import pytest

from qsm.bench.calibration import source_digest
from qsm.core.resources import count
from qsm.scalarmul import signed_window as signed, zero_point as odd
from qsm.mul import accumulate
from test_paper_tables import paper
from test_signed_zero_safe import _case
from test_zero_point import _kwargs, _table

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("odd_savings_bench", ROOT/"scripts/odd_savings_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("kind", ["signed", "odd"])
def test_tracing_preserves_real_toy_circuit_and_exact_component_spans(kind):
    build = (lambda: _case(31, 2)) if kind == "signed" else (
        lambda: odd.build_zero_point_case(3, restoration="add", phase_correction=False,
                                         **_kwargs(_table(31, 3))))
    original = build()
    saved = signed._signed_enabled, signed._endpoint_xor, accumulate._and, accumulate._unand, odd._restore_positive
    case, trace = bench.traced_addition(build)
    assert case.circuit.ops == original.circuit.ops
    assert count(case.circuit) == count(original.circuit)
    assert saved == (signed._signed_enabled, signed._endpoint_xor, accumulate._and, accumulate._unand, odd._restore_positive)
    expected = dict(enable=2, endpoint=2, copy_compute=5, copy_erase=5) if kind == "signed" else {}
    assert trace["calls"] == expected
    assert trace["components"]["square_copy"] == (5 if kind == "signed" else 0)
    assert trace["signatures"]["division"] == trace["signatures"]["multiplication"]
    assert trace["signatures"]["restoration"]["unlookup"] == "paired"
    spans = sorted(trace["component_spans"], key=lambda s: s["start"])
    assert all(a["stop"] <= b["start"] for a, b in zip(spans, spans[1:]))
    if kind == "signed":
        assert all(s["toffoli"] == 0 for s in spans if s["operation"] == "copy_erase")


def test_trace_patches_restore_after_builder_failure():
    saved = signed._signed_enabled, accumulate._and, odd._restore_positive
    def broken():
        raise RuntimeError("deliberate builder failure")
    with pytest.raises(RuntimeError, match="deliberate"):
        bench.traced_addition(broken)
    assert saved == (signed._signed_enabled, accumulate._and, odd._restore_positive)


def sample(kind):
    meta = dict(coordinate_restoration="add", restoration_exact_reduction=True,
                phase_correction=False, restoration_unlookup="paired", zero_safe=True)
    components = dict(recoding=2, enable=3, endpoint=5, square_copy=7) if kind == "signed" else dict.fromkeys(dict(bench.LABELS), 0)
    return dict(kind=kind, field="25519", word_bits=16, table_entries=32768,
        input_widths=dict(word=16, x=255, y=255), curve=dict(p=(1 << 255)-19, a=1, b=2), public_base=[3, 4],
        source_digest=source_digest(), tools=bench.tool_hashes(), case_metadata=meta,
        components=components, shared_regions=dict(arithmetic=100),
        resources=dict(toffoli=100+sum(components.values())),
        signatures=dict(square=dict(controlled=kind == "signed", width=255, ripple=40),
                        division=dict(iters=261), multiplication=dict(iters=261), restoration=dict(unlookup="paired")),
        calls=dict(enable=2, endpoint=2, copy_compute=255, copy_erase=255) if kind == "signed" else {})


def test_savings_and_latex_are_derived_not_quoted():
    a, b = sample("signed"), sample("odd")
    rows = bench.savings(a, b)
    assert [v for _, v in rows] == [2, 3, 5, 7, 17]
    tex, md = bench.tables(rows)
    assert r"\label{tab:odd}" in tex and r"\begin{tabularx}{\linewidth}{Xr}" in tex
    assert "Controlled square-source copy & 7" in tex
    assert "Total & 17" in tex and "543" not in tex
    assert "| Zero-digit enable logic | 3 |" in md


@pytest.mark.parametrize("change", ["total", "shared", "support", "source", "copy", "unsafe"])
def test_unexplained_differences_or_unmatched_support_are_rejected(change):
    a, b = sample("signed"), sample("odd")
    if change == "total": a["resources"]["toffoli"] += 1
    elif change == "shared": a["shared_regions"]["arithmetic"] += 1
    elif change == "support": a["signatures"]["restoration"]["unlookup"] = "split"
    elif change == "source": b["source_digest"] = "old"
    elif change == "copy": a["calls"]["copy_compute"] = 254
    else: a["case_metadata"]["zero_safe"] = False
    with pytest.raises(ValueError):
        bench.savings(a, b)


def mock_workers(monkeypatch, broken=False):
    launched = []
    class Process:
        def __init__(self, command, **kwargs):
            kind = command[command.index("--worker")+1]
            assert kwargs["env"]["QSM_FIELD"] == "25519"
            target = Path(command[command.index("--json")+1])
            self.code = 1 if broken and kind == "odd" else 0
            if self.code == 0:
                target.write_text(json.dumps(sample(kind)))
            launched.append(kind)
            self.stdout = io.StringIO(f"synthetic {kind} progress\n")
        def wait(self, timeout=None):
            assert launched == ["signed", "odd"]
            return self.code
        def poll(self): return self.code
    monkeypatch.setattr(bench.subprocess, "Popen", Process)
    return launched


def test_parent_runs_matched_builds_in_parallel_and_saves_named_outputs(monkeypatch, tmp_path):
    mock_workers(monkeypatch)
    output = tmp_path/"fresh"
    assert bench.run(output) == output
    files = bench.benchmark_files.paths(output, "table5")
    record = json.loads(files["json"].read_text())
    assert record["complete"] and record["rows"][-1] == ["Total", 17]
    tex, md = bench.tables(record["rows"])
    assert files["tex"].read_text() == tex and files["md"].read_text() == md
    assert files["tex"].name == "odd-encoding-savings.tex"
    assert all(job["sha256"] for job in record["jobs"])


def test_failed_worker_never_emits_a_completed_table(monkeypatch, tmp_path):
    mock_workers(monkeypatch, broken=True)
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="odd build failed"):
        bench.run(output)
    files = bench.benchmark_files.paths(output, "table5")
    assert not json.loads(files["json"].read_text())["complete"]
    assert not files["tex"].exists() and not files["md"].exists()


def test_existing_output_is_preserved_and_wrapper_dispatch_is_table5_only(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("existing output reached worker launch")
    monkeypatch.setattr(bench.subprocess, "Popen", forbidden)
    with pytest.raises(FileExistsError):
        bench.run(tmp_path)
    commands = []
    monkeypatch.setattr(paper.subprocess, "call", lambda command: commands.append(command) or 0)
    assert paper.main(["table5", "--out", str(tmp_path/"new")]) == 0
    command, = commands
    assert str(ROOT/"scripts/odd_savings_bench.py") in command
    with pytest.raises(SystemExit):
        paper.main(["table5", "--field", "25519"])
    assert len(commands) == 1
