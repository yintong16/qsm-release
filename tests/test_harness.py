import math

import pytest

from qsm import params
from qsm.bench import harness, registry
from qsm.bench.registry import Case, circuit_version
from qsm.core.circuit import Circuit


def test_duplicate_version_raises(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})

    @circuit_version("test.dup.v1")
    def _a() -> Case:
        c = Circuit()
        c.input("x", 1)
        return Case(c, dict, lambda rng: {"x": rng.getrandbits(1)})

    with pytest.raises(ValueError):
        circuit_version("test.dup.v1")(_a)


def test_benchmark_demo_and():
    import qsm.bench.demos  # noqa: F401 — registers demo circuits
    r = harness.benchmark("demo.and.v1", n_trials=32, persist=False)
    assert r.correct
    assert r.toffoli == 8
    assert r.peak_qubits == 24


def _case(monkeypatch, *, fail=0.0, wrong=False, dirty=False, measured=False,
          meta=None, gen=None):
    c = Circuit()
    c.input("x", 1)
    if measured:
        c.measure(c.alloc(1)[0])
    if dirty:
        a = c.alloc(1)
        c.x(a[0])
        c.free(a)
    case = Case(c, (lambda v: {"x": 1 - v["x"]}) if wrong else dict,
                gen or (lambda rng: {"x": 0}),
                {"fail_analytic": fail, **(meta or {})})
    monkeypatch.setitem(registry._REGISTRY, "test.harness.v1", lambda: case)
    return "test.harness.v1"


def test_benchmark_catches_wrong_reference(monkeypatch):
    r = harness.benchmark(_case(monkeypatch, wrong=True), n_trials=32, persist=False)
    assert not r.correct
    assert "wrong output" in r.note


def test_always_wrong_one_shot_approximate_rejected(monkeypatch):
    r = harness.benchmark(_case(monkeypatch, wrong=True, fail=1e-12),
                          n_trials=1, persist=False)
    assert not r.correct
    assert r.fail_seen == r.trials == 1


@pytest.mark.parametrize("dirty", [False, True])
def test_early_abort_counts_only_executed_calls(monkeypatch, dirty):
    r = harness.benchmark(_case(monkeypatch, wrong=not dirty, dirty=dirty,
                               measured=True), n_trials=32, meas_reps=4, persist=False)
    assert not r.correct
    assert r.n_trials == 32
    assert r.trials == 1
    assert r.fail_seen == 1
    assert r.dirty_seen == int(dirty)
    assert r.inputs_executed == 1
    assert r.batches_completed == r.batches_failed == 0
    assert r.aborted and r.regression_pvalue is None
    assert r.fail_bound95 == 1.0


def test_repetitions_are_not_independent_input_evidence(monkeypatch):
    r = harness.benchmark(_case(monkeypatch, measured=True),
                          n_trials=10, meas_reps=8, persist=False)
    assert r.trials == 80
    assert r.fail_bound95 == pytest.approx(-math.expm1(math.log(0.05) / 10))


@pytest.mark.parametrize("fail, accepted", [(0.009, False), (0.01, True), (0.02, True)])
def test_named_regression_threshold(monkeypatch, fail, accepted):
    r = harness.benchmark(_case(monkeypatch, wrong=True, fail=fail),
                          n_trials=1, persist=False)
    assert r.correct is accepted
    assert r.regression_alpha == 0.01
    assert r.regression_pvalue == pytest.approx(fail)
    assert r.batches_failed == r.batches_completed == 1


def test_correlated_shots_test_batches_not_shots(monkeypatch):
    from qsm.bench.statistics import binomial_tail, clopper_pearson_upper

    inputs = iter([1] + [0] * 9)
    name = _case(monkeypatch, fail=0.01, measured=True,
                 gen=lambda rng: {"x": next(inputs)})
    registry.get_builder(name)().reference = lambda v: {"x": 0}
    r = harness.benchmark(name, n_trials=10, meas_reps=8, persist=False)
    assert r.correct  # Eight correlated failures are one failed input, not eight.
    assert r.n_trials == r.inputs_executed == r.batches_completed == 10
    assert r.fail_seen == 8 and r.trials == 80 and r.batches_failed == 1
    assert r.batch_fail_analytic == 0.08
    assert r.regression_pvalue == pytest.approx(binomial_tail(1, 10, 0.08))
    assert r.fail_bound95 == pytest.approx(clopper_pearson_upper(1, 10))


def test_dirty_approximate_calls_complete_batches(monkeypatch):
    r = harness.benchmark(_case(monkeypatch, fail=1e-12, dirty=True, measured=True),
                          n_trials=3, meas_reps=4, persist=False)
    assert not r.correct and not r.aborted
    assert r.trials == r.dirty_seen == r.fail_seen == 12
    assert r.batches_failed == r.batches_completed == r.inputs_executed == 3
    assert "dirty-ancilla" in r.note


def test_batch_union_bound_is_capped(monkeypatch):
    r = harness.benchmark(_case(monkeypatch, fail=0.9, wrong=True, measured=True),
                          n_trials=3, meas_reps=4, persist=False)
    assert r.batch_fail_analytic == r.regression_pvalue == 1.0
    assert r.correct and r.fail_bound95 == 1.0


def test_no_measurements_execute_one_shot_per_input(monkeypatch):
    r = harness.benchmark(_case(monkeypatch), n_trials=3, meas_reps=9, persist=False)
    assert r.correct
    assert r.n_trials == r.inputs_executed == r.batches_completed == r.trials == 3
    assert r.meas_reps == 1 and r.batches_failed == 0


@pytest.mark.parametrize("argument", ["n_trials", "meas_reps"])
@pytest.mark.parametrize("bad", [0, -1, 1.5, True, float("nan"), float("inf"), "2", None])
def test_invalid_trial_inputs_rejected_before_build(monkeypatch, argument, bad):
    def unexpected(version):
        pytest.fail("invalid trial arguments must not build a circuit")
    monkeypatch.setattr(registry, "get_builder", unexpected)
    with pytest.raises(ValueError, match=argument):
        harness.benchmark("not-built", persist=False, **{argument: bad})


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True, "0.1", None])
def test_invalid_analytic_probability(monkeypatch, bad):
    name = _case(monkeypatch, fail=bad)
    with pytest.raises(ValueError, match="fail_analytic"):
        harness.benchmark(name, persist=False)


def test_result_provenance_and_metadata(monkeypatch):
    meta = {"input_distribution": "uniform canonical nonzero",
            "arithmetic_contract": {"domain": "canonical", "exact": False}}
    r = harness.benchmark(_case(monkeypatch, meta=meta), n_trials=1, persist=False)
    assert (r.field, r.field_u, r.field_prime) == (params.FIELD, params.U, params.PRIME)
    assert r.input_distribution == meta["input_distribution"]
    assert r.arithmetic_contract == meta["arithmetic_contract"]


def test_persist_false_does_not_touch_history(monkeypatch):
    def unexpected(row):
        pytest.fail("persist=False must not touch local history")
    monkeypatch.setattr(harness.report, "append_result", unexpected)
    r = harness.benchmark(_case(monkeypatch), n_trials=1, persist=False)
    assert r.input_distribution is None and r.arithmetic_contract is None


def test_registry_lists_demos():
    import qsm.bench.demos  # noqa: F401
    versions = registry.all_versions()
    assert "demo.xor.v1" in versions
    assert "demo.and.v1" in versions
