"""Table 4 dispatch/formatting tests; no expensive full-width builds are run."""
import copy
import io
import json
from pathlib import Path
import subprocess

import pytest

from qsm.bench.calibration import source_digest
from test_paper_tables import paper
from test_signed_paper_bench import bench, data

ROOT = Path(__file__).resolve().parents[1]


def sample(variant, source):
    result = data("25519", source)
    row, = result["rows"]
    row["variant"] = variant
    if variant != "signed_zero_safe":
        row.pop("zero_safe")
    if variant == "unsigned":
        for key in ("coordinate_restoration", "restoration_unlookup", "restoration_exact_reduction"):
            row.pop(key)
    cost = {"unsigned": 19, "signed_zero_safe": 17, "odd_torsion_add": 13}[variant]
    for stage in row["stages"][1:]:
        stage["toffoli"] = cost
    row["resources"]["toffoli"] = 5+28*cost
    return result


def test_table4_jobs_are_exactly_three_fresh_wei_schedules(tmp_path):
    jobs = bench.schedule_jobs(tmp_path)
    assert [j["variant"] for j in jobs] == ["unsigned", "signed_zero_safe", "odd_torsion_add"]
    assert len({j["output"] for j in jobs}) == len({j["name"] for j in jobs}) == 3
    for job in jobs:
        command = job["command"]
        for option, expected in (("--field", "25519"), ("--variants", job["variant"]),
                                 ("--u-bits", "253"), ("--v-bits", "208"), ("--window-bits", "16")):
            assert command[command.index(option)+1] == expected
        assert "secp256k1" not in command and job["field"] == "25519"


def test_table4_renderer_uses_measurements_for_rows_and_caption():
    results = [bench.verify(sample(v, "fresh"), "25519", "fresh", variant=v) for v, _ in bench.SCHEDULES]
    tex, md = bench.schedule_table(list(reversed(results)))
    assert r"\label{tab:schedules}" in tex
    assert r"\begin{tabularx}{\linewidth}{Xrr}" in tex
    assert "one 5-CCX initialization lookup and 28 additions" in tex
    assert "30 logical qubits" in tex
    assert "Signed half-tables & 17 & 481" in tex
    assert "| Complete odd-only construction | 13 | 369 |" in md
    assert tex.index("Unsigned tables") < tex.index("Signed half-tables") < tex.index("Complete odd-only construction")
    assert "47,214,734" not in tex and "1,189" not in tex
    for key, value in (("peak_qubits", 31), ("initializer_ccx", 6), ("allocation", "255/208")):
        changed = copy.deepcopy(results)
        changed[0][key] = value
        with pytest.raises(ValueError):
            bench.schedule_table(changed)
    with pytest.raises(ValueError):
        bench.schedule_table([results[0], results[0], results[2]])


def test_table4_formats_reference_resource_values():
    values = [("unsigned", 1780046, 49849478),
              ("signed_zero_safe", 1685948, 47214734),
              ("odd_torsion_add", 1685405, 47199530)]
    results = [dict(variant=variant, field="25519", allocation="253/208",
                    initializer_ccx=8190, point_additions=28, peak_qubits=1189,
                    addition_ccx=addition, total_ccx=total)
               for variant, addition, total in values]
    tex, md = bench.schedule_table(results)
    for line in ("Unsigned tables & 1,780,046 & 49,849,478",
                 "Signed half-tables & 1,685,948 & 47,214,734",
                 "Complete odd-only construction & 1,685,405 & 47,199,530"):
        assert line in tex
    assert "8,190-CCX" in tex and "1,189 logical qubits" in tex
    assert "47,214,734" in md


def mock_workers(monkeypatch, fault=None):
    launched = []
    class Process:
        def __init__(self, command, **kwargs):
            variant = command[command.index("--variants")+1]
            path = Path(command[command.index("--json")+1])
            result = sample(variant, source_digest())
            self.code = 0
            if variant == "signed_zero_safe":
                if fault == "diagnostics":
                    result["rows"][0]["diagnostics"][0]["phase_stages"] = [3]
                    self.code = 1
                elif fault == "source":
                    result["rows"][0]["source_digest"] = "stale"
                elif fault == "build":
                    self.code = 2
            if self.code != 2:
                path.write_text(json.dumps(result))
            launched.append(variant)
            self.stdout = io.StringIO(f"synthetic {variant} progress\n")
        def wait(self, timeout=None):
            assert len(launched) == 3
            return self.code
        def poll(self):
            return self.code
    monkeypatch.setattr(bench.subprocess, "Popen", Process)
    return launched


def test_table4_launches_all_three_before_waiting_and_writes_all_formats(monkeypatch, tmp_path, capsys):
    launched = mock_workers(monkeypatch)
    output = tmp_path/"fresh-table4"
    assert bench.run(output, table4=True) == 0
    assert launched == [v for v, _ in bench.SCHEDULES]
    record = json.loads(bench.benchmark_files.paths(output, "table4")["json"].read_text())
    assert record["complete"] and record["table"] == "table4"
    assert record["measurement_mode"].startswith("fresh full emitted")
    tex, md = bench.schedule_table(record["results"])
    assert bench.benchmark_files.paths(output, "table4")["tex"].read_text() == tex
    assert bench.benchmark_files.paths(output, "table4")["md"].read_text() == md
    assert len(list(output.glob("*.log"))) == 3
    assert all(j["output_sha256"] and j["exit_code"] == 0 for j in record["jobs"])
    printed = capsys.readouterr().out
    assert "Measured Signed half-tables: 17 CCX/addition; 481 CCX total" in printed
    assert "Completed fresh table4 measurements" in printed


@pytest.mark.parametrize("fault,error", [("build", "build failed"), ("source", "identity mismatch"),
                                         ("diagnostics", "diagnostics failed")])
def test_table4_failures_never_emit_a_completed_table(monkeypatch, tmp_path, fault, error):
    mock_workers(monkeypatch, fault)
    output = tmp_path/fault
    with pytest.raises((ValueError, RuntimeError), match=error):
        bench.run(output, table4=True)
    result = json.loads(bench.benchmark_files.paths(output, "table4")["json"].read_text())
    assert not result["complete"] and result["error"]
    assert not bench.benchmark_files.paths(output, "table4")["tex"].exists() and not bench.benchmark_files.paths(output, "table4")["md"].exists()


def test_table4_wrapper_dispatch_and_field_rejection(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(paper.subprocess, "call", lambda command: commands.append(command) or 0)
    assert paper.main(["table4", "--out", str(tmp_path/"new")]) == 0
    command, = commands
    assert str(ROOT/"scripts/signed_paper_bench.py") in command and "--table4" in command
    assert command[command.index("--out")+1] == str(tmp_path/"new")
    with pytest.raises(SystemExit):
        paper.main(["table4", "--field", "25519"])
    with pytest.raises(ValueError):
        bench.run(tmp_path/"never-created", "25519", table4=True)
    assert not (tmp_path/"never-created").exists() and len(commands) == 1


def test_real_wrapper_lists_table4_and_refuses_existing_directory(tmp_path):
    wrapper = str(ROOT/"paper")
    listed = subprocess.run([wrapper, "--list"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert listed.returncode == 0 and "table4" in listed.stdout
    sentinel = tmp_path/"keep"
    sentinel.write_text("previous measurements")
    refused = subprocess.run([wrapper, "table4", "--out", str(tmp_path)], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert refused.returncode == 1
    assert sentinel.read_text() == "previous measurements"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep"]
