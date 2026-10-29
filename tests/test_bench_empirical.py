"""Empirical diagnostics never invent an analytic model or suppress phase failures."""

import json
import math
from dataclasses import asdict

import pytest

from qsm.bench import harness, registry, report
from qsm.bench.statistics import clopper_pearson_upper
from qsm.core.circuit import Circuit


def _case(*, wrong=False, phase=False, dirty=False, measured=True, meta=None):
    c = Circuit()
    x = c.input("x", 1)
    if measured:
        c.measure(c.alloc(1)[0])
    if phase:
        c.z(x[0])
    if dirty:
        a = c.alloc(1)
        c.x(a[0])
        c.free(a)
    return registry.Case(
        c, (lambda v: {"x": 1 - v["x"]}) if wrong else dict,
        lambda rng: {"x": 1},
        {"empirical_only": True, "fail_analytic": None, **(meta or {})},
    )


@pytest.mark.parametrize("wrong,phase,dirty", [
    (False, False, False), (True, False, False), (False, True, False),
    (True, True, False), (False, False, True), (True, True, True),
])
@pytest.mark.parametrize("tolerates_phase", [False, True])
def test_empirical_collects_all_diagnostics(wrong, phase, dirty, tolerates_phase, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("empirical mode must not call analytic regression inference")
    monkeypatch.setattr(harness, "binomial_tail", forbidden)
    case = _case(wrong=wrong, phase=phase, dirty=dirty,
                 meta={"tolerates_phase": tolerates_phase})
    r = harness.benchmark_case("test.empirical.v1", case, n_trials=5, meas_reps=3,
                               persist=False)
    assert r.empirical_only and r.correct is None
    assert r.p_correct is r.fail_analytic is r.log2_fail is None
    assert r.batch_fail_analytic is r.regression_pvalue is None
    assert r.inputs_executed == r.batches_completed == r.n_trials == 5
    assert r.meas_reps == 3 and r.trials == 15 and not r.aborted
    failed = wrong or phase or dirty
    assert r.observed_pass is (not failed)
    assert r.fail_seen == 15 * failed  # Combined failures count once.
    assert r.dirty_seen == 15 * dirty
    assert r.output_seen == 15 * (wrong and not dirty)
    assert r.phase_seen == 15 * (phase and not dirty)
    assert r.batches_failed == 5 * failed
    assert r.fail_bound95 == pytest.approx(clopper_pearson_upper(5 * failed, 5))
    assert "sampled batch diagnostics, not quantum norm" in r.note


def test_empirical_no_measurements_execute_one_shot_per_input():
    r = harness.benchmark_case("test.empirical.v1", _case(measured=False),
                               n_trials=10, meas_reps=8, persist=False)
    assert r.meas_reps == 1 and r.trials == r.batches_completed == 10
    assert r.fail_bound95 == pytest.approx(-math.expm1(math.log(.05) / 10))


def test_mixed_empirical_batches_have_batch_not_shot_cp_limit():
    case = _case()
    inputs = iter([1, 0, 0, 0, 0])
    case.gen_input = lambda rng: {"x": next(inputs)}
    case.reference = lambda v: {"x": 0}
    r = harness.benchmark_case("test.empirical.v1", case, n_trials=5, meas_reps=4,
                               persist=False)
    assert r.output_seen == r.fail_seen == 4
    assert r.batches_failed == 1 and r.batches_completed == 5 and r.trials == 20
    assert r.fail_bound95 == pytest.approx(clopper_pearson_upper(1, 5))
    assert r.fail_bound95 != pytest.approx(clopper_pearson_upper(4, 20))


@pytest.mark.parametrize("meta", [
    {"fail_analytic": 0.0}, {"fail_analytic": .01},
    {"empirical_only": "yes"}, {"empirical_only": 1},
    {"empirical_only": False},
])
def test_empirical_declaration_is_explicit_and_unambiguous(meta):
    with pytest.raises(ValueError):
        harness.benchmark_case("test.empirical.v1", _case(meta=meta), persist=False)


def test_empirical_requires_explicit_unknown_failure_declaration():
    case = _case()
    del case.meta["fail_analytic"]
    with pytest.raises(ValueError, match="fail_analytic"):
        harness.benchmark_case("test.empirical.v1", case, persist=False)


def test_empirical_metadata_and_seed_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_PATH", tmp_path / "results.json")
    meta = {"window_bits": 3, "lookup_backend": "unary",
            "table_fingerprint": "abc123", "dummy_policy": "nonidentity",
            "tuning": {"chunk": 4, "cmp_bits": 2, "windows": (3, 4)},
            "circuit_config": {"custom": "kept", "window_bits": 99}}
    case = _case(meta=meta)
    r = harness.benchmark_case("test.empirical.v1", case, n_trials=2, seed=31)
    row = report.load_history()[0]
    assert row == asdict(r)
    assert row["seed"] == 31 and row["empirical_only"] is True
    assert row["correct"] is row["fail_analytic"] is row["p_correct"] is None
    config = row["circuit_config"]
    assert config["window_bits"] == 3 and config["custom"] == "kept"
    for key in ("lookup_backend", "table_fingerprint", "dummy_policy"):
        assert config[key] == meta[key]
    assert config["tuning"] == {"chunk": 4, "cmp_bits": 2, "windows": [3, 4]}
    case.meta["tuning"]["chunk"] = 99
    assert r.circuit_config["tuning"]["chunk"] == 4
    json.dumps(asdict(r), allow_nan=False)


@pytest.mark.parametrize("bad", [object(), float("nan"), {"nested": float("inf")}])
def test_config_must_be_json_safe_even_without_persist(bad):
    with pytest.raises((TypeError, ValueError)):
        harness.benchmark_case("test.empirical.v1", _case(meta={"tuning": bad}),
                               persist=False)


def test_registered_wrapper_and_built_case_agree(monkeypatch):
    case = _case(phase=True, wrong=True)
    monkeypatch.setitem(registry._REGISTRY, "test.empirical.v1", lambda: case)
    args = dict(n_trials=2, seed=71, meas_reps=3, persist=False)
    assert harness.benchmark("test.empirical.v1", **args) == harness.benchmark_case(
        "test.empirical.v1", case, **args)


def test_cli_and_history_label_empirical_without_fabricated_exact(tmp_path, monkeypatch, capsys):
    from qsm.bench import __main__ as cli

    monkeypatch.setattr(report, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setitem(registry._REGISTRY, "test.empirical.v1", lambda: _case())
    monkeypatch.setattr("sys.argv", ["qsm.bench", "test.empirical.v1", "--trials", "2",
                                     "--meas-reps", "3"])
    cli.main()
    output = capsys.readouterr().out
    assert "[EMPIRICAL]" in output and "unknown" in output
    assert "[OK" not in output and "[FAIL]" not in output and "exact" not in output
    assert "observed_pass=True" in output and "failed batches=0/2" in output
    assert "failed calls=0/6" in output and "output=0, phase=0, dirty=0" in output
    assert "CP95 upper=" in output and "not a quantum-norm bound" in output
    before = report.RESULTS_PATH.read_bytes()
    monkeypatch.setattr("sys.argv", ["qsm.bench", "--history", "test.empirical.v1"])
    cli.main()
    output = capsys.readouterr().out
    assert "[EMPIRICAL]" in output and "unknown" in output and "exact" not in output
    assert report.RESULTS_PATH.read_bytes() == before


def test_exact_history_stays_exact_and_legacy_rows_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_PATH", tmp_path / "results.json")
    old = {"version": "old", "log2_fail": None, "correct": True}
    report.append_result(old)
    r = harness.benchmark_case("test.exact.v1", _case(meta={"empirical_only": False,
                                                          "fail_analytic": 0.0}),
                               n_trials=1, persist=False)
    report.append_result(asdict(r))
    with pytest.warns(report.UnknownProvenanceWarning):
        table = report.compare(include_legacy=True)
    assert "exact" in table and "[EMPIRICAL]" not in table
    assert report.load_history()[0] == old


def _residue_case(expected, output_moduli=None):
    c = Circuit()
    c.input("x", 3)
    c.input("y", 3)
    c.input("j", 3)
    c.input("ctrl", 1)
    meta = {"empirical_only": True, "fail_analytic": None}
    if output_moduli is not None:
        meta["output_moduli"] = output_moduli
    return registry.Case(c, lambda v: expected,
                         lambda rng: {"x": 6, "y": 7, "j": 6, "ctrl": 1}, meta)


@pytest.mark.parametrize("changed", [None, "x", "y", "j", "ctrl", "missing", "extra"])
def test_explicit_residue_comparison_does_not_normalize_controls_or_keys(changed):
    expected = {"x": 1, "y": 2, "j": 6, "ctrl": 1}
    if changed in ("x", "y"):
        expected[changed] += 1
    elif changed == "j":
        expected["j"] = 1  # Congruent modulo 5, but j must compare exactly.
    elif changed == "ctrl":
        expected["ctrl"] = 0
    elif changed == "missing":
        del expected["y"]
    elif changed == "extra":
        expected["extra"] = 0
    case = _residue_case(expected, {"x": 5, "y": 5})
    r = harness.benchmark_case("test.residue.v1", case, n_trials=2, persist=False)
    assert r.observed_pass is (changed is None)
    assert r.output_seen == 2 * (changed is not None)
    assert r.circuit_config["output_moduli"] == {"x": 5, "y": 5}


def test_without_explicit_moduli_comparison_remains_raw():
    case = _residue_case({"x": 1, "y": 2, "j": 6, "ctrl": 1})
    r = harness.benchmark_case("test.residue.v1", case, n_trials=2, persist=False)
    assert r.observed_pass is False and r.output_seen == 2


@pytest.mark.parametrize("mapping", [
    {"unknown": 5}, {"x": 0}, {"x": -1}, {"x": True}, {"x": 5.0},
    {"x": "5"}, {"x": None}, [("x", 5)], "x", None,
])
def test_output_moduli_are_validated_before_sampling(mapping):
    case = _residue_case({})
    case.meta["output_moduli"] = mapping
    with pytest.raises(ValueError, match="output_moduli"):
        harness.benchmark_case("test.residue.v1", case, persist=False)
