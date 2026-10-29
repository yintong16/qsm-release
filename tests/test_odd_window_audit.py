import json

import pytest

from qsm.audit import odd_window as audit
from qsm.bench.registry import Case
from qsm.core.circuit import Circuit


@pytest.fixture
def fake_cases(monkeypatch):
    calls = []

    def build(bits, strategy):
        calls.append((bits, strategy))
        c = Circuit()
        c.input("word", bits)
        x, y = c.input("x", 2), c.input("y", 2)
        start = len(c.ops)
        c.x(x[0])
        c.x(x[0])
        return Case(c, lambda v: dict(v), lambda rng: {}, meta={
            "resource_regions": [("recoding", start, len(c.ops))]})

    monkeypatch.setattr(audit, "_build", build)
    monkeypatch.setattr(audit.curve, "mul", lambda *args, **kwargs: (1, 2))
    monkeypatch.setattr(audit, "group_add", lambda left, *args: left)
    monkeypatch.setattr(audit, "source_digest", lambda: "fixed")
    return calls, build


def test_same_raw_inputs_edges_and_separate_measurement_streams(fake_cases):
    calls, _ = fake_cases
    result = audit.run(word_bits=3, random_cases=2)
    assert calls == [(3, "centered"), (3, "odd")]
    a, b = result["rows"]
    assert [d["input"]["word"] for d in a["diagnostics"][:4]] == [0, 3, 4, 7]
    assert [d["input"] for d in a["diagnostics"]] == [d["input"] for d in b["diagnostics"]]
    assert all(x["measurement_seed"] != y["measurement_seed"] for x, y in zip(a["diagnostics"], b["diagnostics"]))
    assert all(row["observed_pass"] for row in result["rows"])
    assert result["ideal_shift_checks"] == 6
    assert result["saved_toffoli"] == 0
    assert all(row["table_entries"] == 4 for row in result["rows"])
    assert "no initialization/full scalar schedule change" in result["scope"]


def test_domain_rejection_is_retained_without_resampling(fake_cases, monkeypatch):
    _, base = fake_cases
    def build(bits, strategy):
        case = base(bits, strategy)
        if strategy == "odd":
            def reference(v):
                if v["word"] == 0:
                    raise AssertionError("diagnostic domain rejection")
                return dict(v)
            case.reference = reference
        return case
    monkeypatch.setattr(audit, "_build", build)
    result = audit.run(word_bits=3, random_cases=0)
    assert result["rows"][0]["observed_pass"]
    assert not result["rows"][1]["observed_pass"]
    assert len(result["rows"][1]["diagnostics"]) == 4
    assert result["rows"][1]["diagnostics"][0]["domain_failure"]
    assert result["ideal_shift_checks"] == 3


def test_phase_failures_are_not_hidden_by_correct_outputs(fake_cases, monkeypatch):
    _, base = fake_cases
    def build(bits, strategy):
        case = base(bits, strategy)
        start = len(case.circuit.ops)
        case.circuit.z(case.circuit.inputs["x"][0])
        case.meta["resource_regions"].append(("phase", start, len(case.circuit.ops)))
        return case
    monkeypatch.setattr(audit, "_build", build)
    result = audit.run(word_bits=3, random_cases=0)
    assert all(not row["observed_pass"] for row in result["rows"])
    assert all(d["phase"] == 1 and d["value_failure"] is False for row in result["rows"] for d in row["diagnostics"])


def test_source_change_is_rejected(fake_cases, monkeypatch):
    values = iter(("before", "after"))
    monkeypatch.setattr(audit, "source_digest", lambda: next(values))
    with pytest.raises(RuntimeError, match="sources changed"):
        audit.run(word_bits=3, random_cases=0)


@pytest.mark.parametrize("kwargs", [{"word_bits": 1}, {"word_bits": 18}, {"word_bits": True},
                                    {"random_cases": -1}, {"random_cases": True}])
def test_bad_requests_fail_before_building(kwargs):
    with pytest.raises(ValueError):
        audit.run(**kwargs)


def test_cli_creates_new_result_and_refuses_overwrite(fake_cases, tmp_path):
    path = tmp_path / "study.json"
    args = ["--word-bits", "3", "--random-cases", "0", "--json", str(path)]
    assert audit.main(args) == 0
    assert json.loads(path.read_text())["source_digest"] == "fixed"
    with pytest.raises(SystemExit):
        audit.main(args)
