import json

import pytest

from qsm.audit import signed_window as audit
from qsm.bench.registry import Case
from qsm.core.circuit import Circuit
from qsm.core.resources import count


def _case(bits, strategy):
    c = Circuit()
    reg = c.input("value", 3)
    start = len(c.ops)
    c.ccx(reg[0], reg[1], reg[2])
    c.ccx(reg[0], reg[1], reg[2])
    return Case(c, lambda v: dict(v), lambda rng: {"value": rng.randrange(8)}, meta={
        "empirical_only": True, "fail_analytic": None,
        "resource_regions": [("test", start, len(c.ops))]})


def test_region_counts_are_exclusive_and_include_live_inputs():
    case = _case(2, "unsigned")
    rows = audit.region_resources(case.circuit, case.meta["resource_regions"])
    total = count(case.circuit)
    assert sum(r["toffoli"] for r in rows.values()) == total.toffoli == 2
    assert rows["test"]["peak_live_qubits"] == 3
    assert max(r["peak_live_qubits"] for r in rows.values()) == total.peak_qubits
    with pytest.raises(ValueError, match="overlap"):
        audit.region_resources(case.circuit, [("one", 0, 2), ("two", 1, 3)])


def test_runner_counts_all_strategies_without_persistence(monkeypatch):
    calls = []
    def build(bits, strategy):
        calls.append((bits, strategy))
        return _case(bits, strategy)
    monkeypatch.setattr(audit, "_build", build)
    result = audit.run(address_bits=[2, 2], strategies=audit.STRATEGIES, trials=2)
    assert calls == [(2, s) for s in audit.STRATEGIES]
    assert [r["radix_bits"] for r in result["rows"]] == [2, 3, 3]
    assert all(r["result"]["observed_pass"] for r in result["rows"])
    assert all(r["input_qubits"] == 3 for r in result["rows"])
    assert result["production_changed"] is False
    assert "NOT included" in result["scope"]
    assert "unpaired" in result["scope"]


def test_runner_detects_source_changes(monkeypatch):
    monkeypatch.setattr(audit, "_build", _case)
    digests = iter(("before", "after"))
    monkeypatch.setattr(audit, "source_digest", lambda: next(digests))
    with pytest.raises(RuntimeError, match="source changed"):
        audit.run(address_bits=[2], strategies=["unsigned"], trials=1)


@pytest.mark.parametrize("kwargs", [
    {"address_bits": []}, {"address_bits": [True]}, {"address_bits": [18]},
    {"address_bits": [0]}, {"strategies": []}, {"strategies": ["unknown"]},
    {"trials": 0}, {"trials": True}, {"meas_reps": 0},
])
def test_invalid_requests_fail_before_building(kwargs):
    with pytest.raises(ValueError):
        audit.run(**kwargs)


def test_cli_writes_new_archive_and_refuses_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_build", _case)
    path = tmp_path / "results.json"
    args = ["--address-bits", "2", "--strategies", "unsigned", "conjugate", "--trials", "2",
            "--json", str(path)]
    assert audit.main(args) == 0
    result = json.loads(path.read_text())
    assert len(result["rows"]) == 2
    assert result["source_digest"]
    with pytest.raises(SystemExit):
        audit.main(args)


def test_cli_surfaces_smoke_failure_without_hiding_archive(monkeypatch, tmp_path):
    def bad_case(bits, strategy):
        case = _case(bits, strategy)
        case.reference = lambda v: {"value": v["value"] ^ 1}
        return case
    monkeypatch.setattr(audit, "_build", bad_case)
    path = tmp_path / "failed.json"
    assert audit.main(["--address-bits", "2", "--strategies", "unsigned", "--trials", "1",
                       "--json", str(path)]) == 1
    result = json.loads(path.read_text())
    assert result["rows"][0]["result"]["output_seen"] == 1
