"""Signed-only reviewer dispatch and provenance checks without unrelated rebuilds."""
import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

from test_paper_tables import paper

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("signed_paper_bench", ROOT/"scripts/signed_paper_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def data(field, source):
    bits = 253 if field == "25519" else 256
    stages = [dict(kind="initial_lookup", bits=bits-240, toffoli=5, clifford=2,
                   measurements=1, global_peak_so_far=10)]
    stages += [dict(kind="point_addition", bits=16, toffoli=100, clifford=20,
                    measurements=10, global_peak_so_far=30) for _ in range(28)]
    good = dict(observed_pass=True, active=True, phase=0, output=[1, 2], expected=[1, 2],
                phase_stages=[], noncanonical_stages=[], value_failure=None,
                control_failure=None, domain_failure=None, dirty_failure=None)
    return dict(field=field, rows=[dict(variant="signed_zero_safe", u_bits=bits, v_bits=208,
        window_bits=16, source_digest=source, seed=bench.SEED, recycled=True,
        full_width_test_settings=False, zero_safe=True, restoration_exact_reduction=True,
        coordinate_restoration="add", restoration_unlookup="paired", point_additions=28,
        stages=stages, resources=dict(toffoli=2805, clifford=562, measurements=281, peak_qubits=30),
        diagnostics=[copy.deepcopy(good) for _ in range(6)])])


def test_dispatch_builds_only_selected_signed_schedules(tmp_path):
    jobs = bench.jobs(tmp_path)
    assert [j["field"] for j in jobs] == ["25519", "secp256k1"]
    assert [j["u_bits"] for j in jobs] == [253, 256]
    for job in jobs:
        args = job["command"]
        assert args[args.index("--variants")+1] == "signed_zero_safe"
        assert "odd_torsion_add" not in args and "unsigned" not in args
        assert args[args.index("--samples")+1] == "4"
    assert len(bench.jobs(tmp_path, ("25519",))) == 1


@pytest.mark.parametrize("field", bench.FIELDS)
def test_verification_counts_stages_and_preserves_failed_diagnostics(field):
    original = data(field, "current")
    result = bench.verify(original, field, "current")
    assert result["addition_ccx"] == 100 and result["total_ccx"] == 2805
    assert result["strict_clean"] == 6
    changed = copy.deepcopy(original)
    changed["rows"][0]["diagnostics"][0]["phase_stages"] = [2]
    assert bench.verify(changed, field, "current")["strict_clean"] == 5
    for key, value in (("source_digest", "old"), ("zero_safe", False),
                       ("restoration_exact_reduction", False), ("recycled", False)):
        changed = copy.deepcopy(original)
        changed["rows"][0][key] = value
        with pytest.raises(ValueError):
            bench.verify(changed, field, "current")
    changed = copy.deepcopy(original)
    changed["rows"][0]["stages"][-1]["toffoli"] += 1
    with pytest.raises(ValueError, match="resource mismatch"):
        bench.verify(changed, field, "current")


def test_parallel_parent_starts_both_fields_before_waiting(monkeypatch, tmp_path):
    from qsm.bench.calibration import source_digest
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            field = command[command.index("--field")+1]
            output = Path(command[command.index("--json")+1])
            output.write_text(json.dumps(data(field, source_digest())))
            launched.append(field)
            self.stdout = io.StringIO("synthetic test progress\n")
        def wait(self, timeout=None):
            assert len(launched) == 2
            return 0
        def poll(self):
            return 0

    monkeypatch.setattr(bench.subprocess, "Popen", Process)
    output = tmp_path/"fresh"
    assert bench.run(output) == 0
    result = json.loads(bench.benchmark_files.paths(output, "signed")["json"].read_text())
    assert result["complete"] and len(result["results"]) == 2
    assert "2,805" in bench.benchmark_files.paths(output, "signed")["md"].read_text()
    assert not list(output.glob("*odd*"))


def test_existing_directory_is_refused_without_launch(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("existing output directory reached worker launch")
    monkeypatch.setattr(bench.subprocess, "Popen", forbidden)
    with pytest.raises(FileExistsError):
        bench.run(tmp_path)
    assert not list(tmp_path.iterdir())


def test_child_failure_produces_no_completed_table(monkeypatch, tmp_path):
    class Failed:
        stdout = io.StringIO("synthetic build failure\n")
        def wait(self, timeout=None): return 1
        def poll(self): return 1
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k: Failed())
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="build failed"):
        bench.run(output, "25519")
    assert not json.loads(bench.benchmark_files.paths(output, "signed")["json"].read_text())["complete"]
    assert not bench.benchmark_files.paths(output, "signed")["md"].exists()


def test_paper_wrapper_forwards_only_signed_request(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(paper.subprocess, "call", lambda command: commands.append(command) or 7)
    assert paper.main(["signed", "--field", "25519", "--out", str(tmp_path/"new")]) == 7
    command, = commands
    assert str(ROOT/"scripts/signed_paper_bench.py") in command
    assert command[-2:] == ["--field", "25519"]
    with pytest.raises(SystemExit):
        paper.main(["table1", "--field", "25519"])


def test_point_horizontal_fixtures_are_selected_algebraically():
    from test_signed_zero_safe import _case
    spec = importlib.util.spec_from_file_location("signed_point_bench", ROOT/"scripts/bench_zero_safe_signed.py")
    point = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(point)
    case = _case(31, 2)
    table = case.analysis_context["window_table"]
    fixtures, selection = point.horizontal_fixtures(table, 3)
    assert fixtures and selection["selection"].endswith("no circuit-outcome filtering")
    for item in fixtures:
        raw, x, y = (item["input"][k] for k in ("word", "x", "y"))
        digit = raw-4
        j, s = abs(digit) % 4, int(digit < 0)
        assert y == table.operand(j, s)[1]
        assert (y*y-x*x*x-table.a*x-table.b) % table.p == 0
        expected = case.reference(dict(j=j, s=s, x=x, y=y))
        assert expected["y"] == (y if digit == 0 else -y % table.p)
