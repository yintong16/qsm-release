"""Fixed-input calibration contracts on an enumerated seven-point toy group.

No production-width arithmetic, source-tree hashing, or benchmark history is used.
The evaluator is mocked except where a tiny raw circuit exercises the full path.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import random
from types import SimpleNamespace

import pytest

from qsm import params
from qsm.bench import calibration, harness
from qsm.core import symbolic_batch
from qsm.bench.harness import Result
from qsm.bench.registry import Case
from qsm.core.circuit import Circuit
from qsm.core.symbolic_batch import BatchHazards
from qsm.point.group import scalar_mul
from qsm.scalarmul.window import WindowTable, add_window_affine_inplace_ref


SOURCE = "a" * 64
DOMAIN = calibration.WindowDomain("toy13", 4, 13, 0, 7, 7, 1)
POINTS = tuple((x, y) for x in range(13) for y in range(13)
               if (y*y - x*x*x - 7) % 13 == 0)
CATEGORIES = ("value", "noncanonical", "dirty", "phase", "constant_phase",
              "summary_mismatch")


def unexpected(*args, **kwargs):
    pytest.fail("unexpected production build, case sampler/reference, or history access")


def tiny_case():
    table = WindowTable(13, 0, 7, (None, (7, 5)))
    circ = Circuit()
    circ.input("j", 1)
    circ.input("x", 4)
    circ.input("y", 4)
    return Case(circ, unexpected, unexpected, {
        "empirical_only": True, "fail_analytic": None,
        "window_bits": 1, "lookup_backend": "test.v1",
        "table_fingerprint": table.fingerprint, "dummy_policy": "T[1]",
        "tuning": {"inversion": {"iters": 5, "widths": [4, 3, 2]}},
        "output_moduli": {"x": 13, "y": 13},
        "circuit_config": {"operand_strategy": "paired"},
    }, analysis_context={"window_table": table})


def hazards(size, **masks):
    values = {name: masks.get(name, 0) for name in CATEGORIES}
    union = 0
    for value in values.values():
        union |= value
    return BatchHazards(samples=size, bad_mask=union, outputs=[],
                        **{name + "_mask": value for name, value in values.items()})


@pytest.fixture(autouse=True)
def fixed_source_and_no_history(monkeypatch):
    from qsm.bench import report
    monkeypatch.setattr(calibration, "source_digest", lambda: SOURCE)
    monkeypatch.setattr(report, "load_history", unexpected)
    monkeypatch.setattr(report, "append_result", unexpected)


@pytest.fixture
def evaluator(monkeypatch):
    calls = SimpleNamespace(compiled=[], batches=[], sampler_args=[], plan=None)

    def compile_(circ):
        calls.compiled.append(circ)
        calls.plan = symbolic_batch.compile_symbolic(circ)
        return calls.plan

    class UniformToySampler:
        def __init__(self, *args):
            calls.sampler_args.append(args)
            self.stats = {"fixture_group_order": 7}

        def sample(self, rng):
            return rng.choice(POINTS)

    def simulate(plan_, inputs, expected, *, output_moduli):
        assert plan_ is calls.plan
        calls.batches.append((deepcopy(inputs), deepcopy(expected), dict(output_moduli)))
        return hazards(len(inputs))

    monkeypatch.setattr(calibration, "compile_symbolic", compile_)
    monkeypatch.setattr(calibration, "simulate_batch", simulate)
    monkeypatch.setattr(calibration, "SubgroupSampler", UniformToySampler)
    return calls


def run(case=None, **kwargs):
    return calibration.calibrate_window(tiny_case() if case is None else case,
                                        domain=DOMAIN, **kwargs)


def result_for(bound):
    return Result(
        "test.window.v1", None, 4, 0, 0, 1, 9, 1, 1,
        empirical_only=True, observed_pass=False,
        p_correct=None, fail_analytic=None, batch_fail_analytic=None,
        field=DOMAIN.field, field_u=DOMAIN.n, field_prime=DOMAIN.p,
        analysis_ir_digest=bound.analysis_ir_digest,
        trials=12, fail_seen=3, phase_seen=2, dirty_seen=1,
        inputs_executed=4, batches_completed=4, batches_failed=2,
        meas_reps=3, fail_bound95=0.91,
        circuit_config={"table_fingerprint": bound.table_fingerprint,
                        **deepcopy(bound.circuit_config)},
    )


def test_fixture_is_nonsingular_prime_order_group():
    assert len(POINTS) + 1 == DOMAIN.order == 7
    assert (4 * DOMAIN.a**3 + 27 * DOMAIN.b**2) % DOMAIN.p != 0
    assert set(POINTS) == {scalar_mul(k, 7, 5, p=13, a=0) for k in range(1, 7)}
    assert scalar_mul(7, 7, 5, p=13, a=0) is None


@pytest.mark.parametrize("changes", [
    {"field": ""}, {"field": None}, {"n": 1}, {"n": True}, {"n": 4.0},
    {"p": 3}, {"p": 12}, {"p": 17}, {"p": True},
    {"order": 5}, {"order": 8}, {"order": True}, {"order": 7.0},
    {"cofactor": 0}, {"cofactor": 7}, {"cofactor": True},
    {"a": 0.0}, {"b": False},
])
def test_domain_rejects_invalid_basic_assumptions(changes):
    with pytest.raises(ValueError):
        replace(DOMAIN, **changes)


def test_domain_normalizes_coefficients():
    assert replace(DOMAIN, a=26, b=-6) == DOMAIN


@pytest.mark.parametrize("kwargs", [
    {"samples": 0}, {"samples": -1}, {"samples": True}, {"samples": 1.5},
    {"batch_size": 0}, {"batch_size": -1}, {"batch_size": True},
    {"batch_size": 1.5}, {"confidence": 0}, {"confidence": 1},
    {"confidence": float("nan")}, {"confidence": float("inf")},
    {"confidence": -0.1}, {"confidence": True},
    {"seed": True}, {"seed": 1.5},
])
def test_invalid_options_rejected_before_compilation(monkeypatch, kwargs):
    monkeypatch.setattr(calibration, "compile_symbolic", unexpected)
    with pytest.raises(ValueError):
        run(**({"samples": 2} | kwargs))


@pytest.mark.parametrize("mutation", [
    lambda case: case.analysis_context.clear(),
    lambda case: case.analysis_context.update(window_table=object()),
    lambda case: case.analysis_context.update(
        window_table=WindowTable(13, 0, 6, (None, (2, 1)))),
    lambda case: case.meta.update(table_fingerprint="wrong"),
    lambda case: case.meta.update(output_moduli={"x": 17, "y": 17}),
    lambda case: case.meta.pop("output_moduli"),
    lambda case: case.meta.update(empirical_only=False),
    lambda case: case.meta.update(fail_analytic=0.0),
    lambda case: case.circuit.input("extra", 1),
    lambda case: case.circuit.inputs.pop("j"),
])
def test_invalid_case_provenance_rejected_before_compilation(monkeypatch, mutation):
    case = tiny_case()
    mutation(case)
    monkeypatch.setattr(calibration, "compile_symbolic", unexpected)
    with pytest.raises(ValueError):
        run(case, samples=2)


def test_fixed_samples_partial_last_batch_and_joint_rejection_law(evaluator):
    case = tiny_case()
    stream = io.StringIO()
    result = run(case, samples=7, batch_size=3, seed=29, stream=stream)
    assert evaluator.compiled == [case.circuit]
    assert evaluator.sampler_args == [(13, 0, 7, 7, 1)]
    assert [len(inputs) for inputs, _, _ in evaluator.batches] == [3, 3, 1]
    # Replay independent joint proposals; rejecting only a digit or only a point
    # would change this exact accepted sequence and the number of proposals.
    rng = random.Random(29)
    expected_inputs, expected_outputs, attempts = [], [], 0
    table = case.analysis_context["window_table"]
    while len(expected_inputs) < 7:
        x, y = rng.choice(POINTS)
        j = rng.randrange(2)
        attempts += 1
        try:
            xr, yr = add_window_affine_inplace_ref(x, y, j, table)
        except AssertionError:
            continue
        expected_inputs.append({"j": j, "x": x, "y": y})
        expected_outputs.append({"j": j, "x": xr, "y": yr})
    assert [v for inp, _, _ in evaluator.batches for v in inp] == expected_inputs
    assert [v for _, exp, _ in evaluator.batches for v in exp] == expected_outputs
    assert all(moduli == {"x": 13, "y": 13} for _, _, moduli in evaluator.batches)
    assert result.samples == 7
    assert result.attempted_inputs == attempts > 7
    assert result.rejected_inputs == attempts - 7
    assert "7/7" in stream.getvalue()
    assert result.source_digest == SOURCE
    assert result.analysis_ir_digest == evaluator.plan.ir_digest
    assert result.evaluator_stats == dict(evaluator.plan.stats)
    assert result.sampler_stats == {"fixture_group_order": 7}
    assert result.domain == asdict(DOMAIN)
    assert result.table_fingerprint == table.fingerprint
    assert "every measurement outcome" in result.success_event
    assert "conditioned" in result.input_distribution
    assert "shor" not in json.dumps(asdict(result)).lower()


@pytest.mark.parametrize("samples,confidence", [(1, .99), (10, .95), (257, .995)])
def test_zero_bad_inputs_has_exact_one_sided_cp_bound(evaluator, samples, confidence):
    result = run(samples=samples, confidence=confidence, batch_size=17)
    expected_lower = (1 - confidence) ** (1 / samples)
    assert result.bad_inputs == 0 and result.clean_fraction == 1
    assert result.success_lower == pytest.approx(expected_lower, abs=1e-15)
    assert result.failure_upper == pytest.approx(1 - expected_lower, abs=1e-15)
    assert 0 < result.success_lower < 1
    assert result.confidence == confidence


def test_overlapping_hazards_are_one_bernoulli_input_not_fractional_shots(
        evaluator, monkeypatch):
    def simulate(plan, inputs, expected, **kwargs):
        assert len(inputs) == 4
        # Lane 0 has both output and phase faults, lane 2 constant phase,
        # lane 3 dirty ancillas/failed summary. Counts must not be summed.
        return hazards(4, value=0b0001, phase=0b0101, constant_phase=0b0100,
                       dirty=0b1000, summary_mismatch=0b1000, noncanonical=0b0001)

    monkeypatch.setattr(calibration, "simulate_batch", simulate)
    result = run(samples=4)
    assert result.bad_inputs == 3 and result.clean_fraction == .25
    assert result.counts == {"value": 1, "phase": 2, "constant_phase": 1,
                             "dirty": 1, "summary_mismatch": 1, "noncanonical": 1}
    # P[Binomial(4, upper) <= 3] = 1-upper**4 = 1-confidence.
    assert result.failure_upper == pytest.approx(.99 ** .25)


def test_all_bad_inputs_do_not_stop_sampling_early(evaluator, monkeypatch):
    sizes = []

    def simulate(plan, inputs, expected, **kwargs):
        sizes.append(len(inputs))
        return hazards(len(inputs), phase=(1 << len(inputs)) - 1)

    monkeypatch.setattr(calibration, "simulate_batch", simulate)
    result = run(samples=11, batch_size=4)
    assert sizes == [4, 4, 3]
    assert result.samples == result.bad_inputs == result.counts["phase"] == 11
    assert result.clean_fraction == result.success_lower == 0
    assert result.failure_upper == 1


def test_seed_reproducible_and_does_not_use_global_rng(evaluator):
    state = random.getstate()
    first = run(samples=10, seed=17, batch_size=3)
    batches = deepcopy(evaluator.batches)
    evaluator.batches.clear()
    second = run(samples=10, seed=17, batch_size=3)
    assert evaluator.batches == batches
    assert first.attempted_inputs == second.attempted_inputs
    assert random.getstate() == state


def test_source_change_during_calibration_refuses_certificate(evaluator, monkeypatch):
    digests = iter((SOURCE, "c" * 64))
    monkeypatch.setattr(calibration, "source_digest", lambda: next(digests))
    with pytest.raises(ValueError, match="sources changed"):
        run(samples=2)


def test_wrong_evaluator_lane_count_rejected(evaluator, monkeypatch):
    monkeypatch.setattr(calibration, "simulate_batch", lambda *a, **kw: hazards(1))
    with pytest.raises(ValueError, match="lane count"):
        run(samples=2)


def test_calibration_metadata_is_detached_and_json_safe(evaluator):
    case = tiny_case()
    result = run(case, samples=3)
    before = asdict(result)
    case.meta["tuning"]["inversion"]["widths"].append(999)
    assert asdict(result) == before
    assert json.loads(json.dumps(before, allow_nan=False)) == before
    assert "window_table" not in json.dumps(before)


def test_attach_keeps_pilot_shots_and_analytic_unknowns_separate(evaluator):
    bound = run(samples=7, confidence=.995)
    result = result_for(bound)
    before = asdict(result)
    calibration.attach_calibration(result, bound)
    after = asdict(result)
    changed = {"success_lower", "success_confidence", "success_samples", "bad_inputs",
               "clean_input_fraction", "circuit_config"}
    assert {k: v for k, v in after.items() if k not in changed} == {
        k: v for k, v in before.items() if k not in changed}
    assert result.success_samples == 7 != result.trials
    assert result.success_lower == pytest.approx(.005 ** (1 / 7))
    assert result.success_confidence == .995
    assert result.bad_inputs == 0 and result.clean_input_fraction == 1
    assert result.correct is result.p_correct is result.fail_analytic is None
    assert result.circuit_config["success_calibration"] == asdict(bound)
    assert result.circuit_config["success_calibration"]["samples"] == 7
    bound.counts["phase"] = 99
    assert result.circuit_config["success_calibration"]["counts"]["phase"] == 0


@pytest.mark.parametrize("field,value", [
    ("empirical_only", False), ("fail_analytic", 0.0),
    ("field", "other"), ("field_u", 5), ("field_prime", 17),
])
def test_attach_rejects_empirical_or_field_mismatch_without_partial_write(
        evaluator, field, value):
    bound = run(samples=2)
    result = result_for(bound)
    setattr(result, field, value)
    before = asdict(result)
    with pytest.raises(ValueError):
        calibration.attach_calibration(result, bound)
    assert asdict(result) == before


@pytest.mark.parametrize("key,value", [
    ("table_fingerprint", "c" * 64), ("lookup_backend", "other.v1"),
    ("dummy_policy", "T[2]"), ("window_bits", 2),
    ("tuning", {"inversion": {"iters": 6}}),
])
def test_attach_rejects_config_mismatch_without_partial_write(evaluator, key, value):
    bound = run(samples=2)
    result = result_for(bound)
    result.circuit_config[key] = value
    before = asdict(result)
    with pytest.raises(ValueError):
        calibration.attach_calibration(result, bound)
    assert asdict(result) == before


@pytest.mark.parametrize("fault", ["identity", "phase", "constant_phase", "dirty", "noncanonical", "index"])
def test_real_small_pipeline_uses_raw_outputs_and_all_measurement_outcomes(monkeypatch, fault):
    # An identity window is intentionally wrong for nonzero digits. Add tiny
    # orthogonal faults; the independent group reference must still catch them.
    domain = replace(DOMAIN, n=5)
    case = tiny_case()
    circ = Circuit()
    j, x, _y = circ.input("j", 1), circ.input("x", 5), circ.input("y", 5)
    case.circuit = circ
    if fault in ("phase", "dirty"):
        ancilla = circ.alloc(1)[0]
        circ.cx(x[0], ancilla)
        if fault == "phase":
            circ.measure(ancilla)  # x_0*m survives; half the shots may look clean.
    elif fault == "constant_phase":
        circ.z(x[0])
    elif fault == "noncanonical":
        # Accepted toy inputs have x in {8, 11}. This reversible affine map
        # sends them to {21, 24}: exactly x+p, not merely a wrong residue.
        for bit in (1, 2, 3):
            circ.cx(x[0], x[bit])
        for bit in (0, 2, 3, 4):
            circ.x(x[bit])
    elif fault == "index":
        circ.x(j[0])
    observed = []
    simulate = calibration.simulate_batch

    def spy(plan, inputs, expected, *, output_moduli):
        got = simulate(plan, inputs, expected, output_moduli=output_moduli)
        observed.extend(zip(inputs, expected, got.outputs, strict=True))
        return got

    monkeypatch.setattr(calibration, "simulate_batch", spy)
    result = calibration.calibrate_window(case, samples=23, batch_size=8,
                                           seed=41, domain=domain)
    expected_counts = dict.fromkeys(CATEGORIES, 0)
    expected_bad = 0
    for inputs, expected, actual in observed:
        raw = dict(inputs)
        if fault == "noncanonical":
            raw["x"] += 13
        elif fault == "index":
            raw["j"] ^= 1
        assert actual == raw
        # Residue mismatch and noncanonical encoding are separate diagnostics;
        # either one makes the strict raw-output success event fail.
        reduced = {**raw, "x": raw["x"] % 13, "y": raw["y"] % 13}
        value = reduced != expected
        noncanonical = raw["x"] >= 13 or raw["y"] >= 13
        phase = fault in ("phase", "constant_phase") and bool(inputs["x"] & 1)
        dirty = fault == "dirty" and bool(inputs["x"] & 1)
        expected_counts["value"] += value
        expected_counts["noncanonical"] += noncanonical
        expected_counts["phase"] += phase
        expected_counts["constant_phase"] += phase and fault == "constant_phase"
        expected_counts["dirty"] += dirty
        expected_bad += value or noncanonical or phase or dirty
    assert len(observed) == result.samples == 23
    assert result.counts == expected_counts
    assert result.bad_inputs == expected_bad
    assert result.clean_fraction == (23 - expected_bad) / 23
    if fault in ("phase", "constant_phase", "dirty", "noncanonical"):
        assert result.counts["phase" if fault == "constant_phase" else fault] > 0
    if fault == "noncanonical":
        assert any(inputs["j"] == 0 for inputs, _, _ in observed)
        for inputs, expected, actual in observed:
            if inputs["j"] == 0:
                assert actual["x"] % 13 == expected["x"]
                assert actual != expected  # Modular equality cannot certify raw output.
        assert result.counts["value"] < result.counts["noncanonical"] == result.bad_inputs == 23
    assert result.analysis_ir_digest
    assert result.sampler_stats


@pytest.mark.parametrize("digest", [None, "", "d" * 64])
def test_attach_requires_nonempty_matching_benchmark_ir(evaluator, digest):
    bound = run(samples=3)
    result = result_for(bound)
    result.analysis_ir_digest = digest
    before = asdict(result)
    # Every other identity matches. Equal labels/config/source cannot replace
    # evidence that the benchmark and calibration analyzed the same circuit.
    assert result.circuit_config["table_fingerprint"] == bound.table_fingerprint
    assert result.field == bound.domain["field"]
    with pytest.raises(ValueError, match="(?i)(ir|digest|circuit|analysis)"):
        calibration.attach_calibration(result, bound)
    assert asdict(result) == before


def test_real_distinct_ir_cannot_share_calibration_despite_identical_metadata(monkeypatch):
    monkeypatch.setattr(params, "FIELD", DOMAIN.field)
    monkeypatch.setattr(params, "U", DOMAIN.n)
    monkeypatch.setattr(params, "PRIME", DOMAIN.p)
    case_a, case_b = tiny_case(), tiny_case()
    case_b.circuit.x(case_b.circuit.inputs["j"][0])
    assert case_a.meta == case_b.meta
    for case in (case_a, case_b):
        case.gen_input = lambda rng: {"j": 0, "x": 8, "y": 8}
        case.reference = dict
    plan_a = symbolic_batch.compile_symbolic(case_a.circuit)
    plan_b = symbolic_batch.compile_symbolic(case_b.circuit)
    assert plan_a.ir_digest != plan_b.ir_digest
    result_a = harness.benchmark_case("same-label.v1", case_a, n_trials=1,
                                      persist=False, analysis_plan=plan_a)
    result_b = harness.benchmark_case("same-label.v1", case_b, n_trials=1,
                                      persist=False, analysis_plan=plan_b)
    assert result_a.circuit_config == result_b.circuit_config
    assert result_a.analysis_ir_digest == plan_a.ir_digest
    assert result_b.analysis_ir_digest == plan_b.ir_digest
    bound = run(case_a, samples=7, plan=plan_a)
    before_b = asdict(result_b)
    with pytest.raises(ValueError, match="(?i)(ir|digest|circuit|analysis)"):
        calibration.attach_calibration(result_b, bound)
    assert asdict(result_b) == before_b
    calibration.attach_calibration(result_a, bound)
    assert result_a.success_samples == 7
    assert result_a.circuit_config["success_calibration"]["analysis_ir_digest"] == plan_a.ir_digest


def test_precompiled_plan_is_reused_without_recompiling(monkeypatch):
    case = tiny_case()
    plan = symbolic_batch.compile_symbolic(case.circuit)
    monkeypatch.setattr(calibration, "compile_symbolic", unexpected)
    bound = run(case, samples=3, plan=plan)
    assert bound.analysis_ir_digest == plan.ir_digest


@pytest.mark.parametrize("timing", ["before", "during"])
def test_calibration_rejects_circuit_changed_since_plan(monkeypatch, timing):
    case = tiny_case()
    plan = symbolic_batch.compile_symbolic(case.circuit)
    if timing == "before":
        case.circuit.x(case.circuit.inputs["j"][0])
        monkeypatch.setattr(calibration, "simulate_batch", unexpected)
    else:
        simulate = calibration.simulate_batch

        def mutate(*args, **kwargs):
            got = simulate(*args, **kwargs)
            case.circuit.x(case.circuit.inputs["j"][0])
            return got

        monkeypatch.setattr(calibration, "simulate_batch", mutate)
    with pytest.raises(ValueError, match="(?i)(plan|snapshot|circuit|changed|binding)"):
        run(case, samples=3, plan=plan)


@pytest.mark.parametrize("timing", ["before", "during"])
def test_benchmark_rejects_circuit_changed_since_analysis_plan(monkeypatch, timing):
    case = tiny_case()
    case.gen_input = lambda rng: {"j": 0, "x": 8, "y": 8}
    case.reference = dict
    plan = symbolic_batch.compile_symbolic(case.circuit)
    if timing == "before":
        case.circuit.x(case.circuit.inputs["j"][0])
        case.gen_input = unexpected
    else:
        simulate = harness.simulate

        def mutate(*args, **kwargs):
            got = simulate(*args, **kwargs)
            case.circuit.x(case.circuit.inputs["j"][0])
            return got

        monkeypatch.setattr(harness, "simulate", mutate)
    with pytest.raises(ValueError, match="(?i)(plan|snapshot|circuit|changed|binding)"):
        harness.benchmark_case("mutable.v1", case, n_trials=1,
                               persist=False, analysis_plan=plan)


def test_attach_rejects_stale_source_without_partial_write(evaluator, monkeypatch):
    bound = run(samples=2)
    result = result_for(bound)
    before = asdict(result)
    monkeypatch.setattr(calibration, "source_digest", lambda: "c" * 64)
    with pytest.raises(ValueError, match="stale"):
        calibration.attach_calibration(result, bound)
    assert asdict(result) == before
