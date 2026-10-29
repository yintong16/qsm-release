"""Build, simulate against a classical reference, count resources and record a result.

Regression acceptance is not a correctness proof: analytics and inference both depend
on the declared input distribution. Measurement shots sharing an input form one batch.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field as dataclass_field

from .. import params
from ..core.circuit import DirtyAncillaError
from ..core.resources import count
from ..core.simulator import simulate
from . import registry, report
from .statistics import (
    REGRESSION_ALPHA, binomial_tail, clopper_pearson_upper, probability, trial_count,
)


@dataclass
class Result:
    version: str
    correct: bool | None  # None = empirical diagnostics, no model acceptance.
    n_trials: int  # Requested independent inputs; retained for API compatibility.
    toffoli: int
    clifford: int
    measurements: int
    peak_qubits: int
    depth: int
    n_gates: int
    toffoli_depth: int = 0
    volume: int = 0
    toffoli_volume: int = 0
    circuit_volume: int = 0
    # Analytic estimates, NOT evidence of observed regression success.
    p_correct: float | None = 1.0
    log2_fail: float | None = None
    fail_analytic: float | None = 0.0
    # Per simulator call: empirical failures are a union, with independent phase/output
    # counters. Numeric-model mode retains its historical phase precedence.
    fail_seen: int = 0
    dirty_seen: int = 0
    phase_seen: int = 0
    trials: int = 0  # Actual simulator calls, including DirtyAncillaError.
    fail_bound95: float = 1.0
    note: str = ""
    inputs_executed: int = 0
    batches_completed: int = 0
    batches_failed: int = 0
    meas_reps: int = 1  # Effective repetitions; one for circuits without measurement.
    aborted: bool = False
    batch_fail_analytic: float | None = 0.0
    regression_pvalue: float | None = None
    regression_alpha: float = REGRESSION_ALPHA
    # Missing provenance remains unknown when old Result dictionaries are loaded.
    field: str | None = None
    field_u: int | None = None
    field_prime: int | None = None
    input_distribution: object = None
    arithmetic_contract: object = None
    empirical_only: bool = False
    observed_pass: bool | None = None  # Unknown for historical rows.
    output_seen: int = 0
    seed: int | None = None
    circuit_config: dict = dataclass_field(default_factory=dict)
    # Fixed-sample all-outcome input calibration, separate from smoke-test shots.
    success_lower: float | None = None
    success_confidence: float | None = None
    success_samples: int | None = None
    bad_inputs: int | None = None
    clean_input_fraction: float | None = None
    analysis_ir_digest: str | None = None


def _outputs_match(actual: dict, expected: dict, moduli: dict) -> bool:
    """Normalize only explicitly declared residues; key sets and controls stay exact."""
    if not moduli:
        return actual == expected
    if actual.keys() != expected.keys():
        return False
    return all((value % moduli[key] == expected[key] % moduli[key]
                if key in moduli else value == expected[key])
               for key, value in actual.items())


def benchmark(
    version: str,
    n_trials: int = 64,
    seed: int = 0,
    meas_reps: int = 4,
    persist: bool = True,
) -> Result:
    """Build a registered case, then run the same diagnostics as benchmark_case."""
    trial_count(n_trials, "n_trials", minimum=1)
    trial_count(meas_reps, "meas_reps", minimum=1)
    return benchmark_case(version, registry.get_builder(version)(), n_trials, seed,
                          meas_reps, persist)


def benchmark_case(
    version: str,
    case: registry.Case,
    n_trials: int = 64,
    seed: int = 0,
    meas_reps: int = 4,
    persist: bool = True,
    *, analysis_plan=None,
) -> Result:
    """Benchmark an already-built case without registering a global version.

    Empirical mode requires explicit empirical_only=True and fail_analytic=None.
    Its CP upper limit describes sampled input-batch diagnostics, assuming IID input
    batches; it is not a quantum-norm guarantee or a transferable analytic model.
    Dirty-ancilla exceptions expose neither final output nor phase.
    """
    trial_count(n_trials, "n_trials", minimum=1)
    trial_count(meas_reps, "meas_reps", minimum=1)
    if analysis_plan is not None:
        from ..core.symbolic_batch import validate_plan_binding
        validate_plan_binding(analysis_plan, case.circuit)
    rng = random.Random(seed)
    empirical = case.meta.get("empirical_only", False)
    if type(empirical) is not bool:
        raise ValueError("empirical_only must be a bool")
    declared_fail = case.meta.get("fail_analytic", 0.0)
    if empirical:
        if declared_fail is not None:
            raise ValueError("empirical_only requires fail_analytic=None")
        fail_analytic = None
    else:
        fail_analytic = probability(declared_fail, "fail_analytic")
    config = dict(case.meta.get("circuit_config", {}))
    for key in ("window_bits", "lookup_backend", "table_fingerprint", "dummy_policy", "tuning"):
        if key in case.meta:
            config[key] = case.meta[key]
    output_moduli = case.meta.get("output_moduli", {})
    if (not isinstance(output_moduli, dict) or
            any(key not in case.circuit.inputs or type(modulus) is not int or modulus <= 0
                for key, modulus in output_moduli.items())):
        raise ValueError("output_moduli must map known output names to positive integer moduli")
    if "output_moduli" in case.meta:
        config["output_moduli"] = output_moduli
    # Validate and detach metadata before sampling; results are JSON-safe even without
    # persistence, and mutable builder metadata cannot change a completed result.
    circuit_config = json.loads(json.dumps(config, allow_nan=False))
    approximate = fail_analytic is not None and fail_analytic > 0.0
    # Approximation alone never licenses phase; exact circuits reject every failure.
    tolerates_phase = approximate and bool(case.meta.get("tolerates_phase", False))

    correct = None if empirical else True
    note = ""
    fail_seen = dirty_seen = phase_seen = output_seen = trials = 0
    inputs_executed = batches_completed = batches_failed = 0
    aborted = False
    has_meas = any(op.kind == "MEASURE" for op in case.circuit.ops)
    reps = meas_reps if has_meas else 1
    batch_fail_analytic = None if empirical else min(1.0, reps * fail_analytic)

    for _ in range(n_trials):
        inp = case.gen_input(rng)
        expected = case.reference(inp)
        inputs_executed += 1
        batch_failed = False
        calls = 0
        for _r in range(reps):
            trials += 1
            calls += 1
            try:
                res = simulate(case.circuit, inp, rng=rng)
            except DirtyAncillaError:
                dirty_seen += 1
                fail_seen += 1
                batch_failed = True
                if not approximate and not empirical:
                    correct, note = False, f"dirty ancilla on {inp}"
                    break
                continue
            if empirical:
                # Never hide output errors behind a phase error, nor license phase
                # via tolerates_phase. A shot contributes at most one union failure.
                phase_failed = res.phase != 0
                output_failed = not _outputs_match(res.outputs, expected, output_moduli)
                phase_seen += int(phase_failed)
                output_seen += int(output_failed)
                failed = phase_failed or output_failed
                fail_seen += int(failed)
                batch_failed |= failed
                continue
            if res.phase != 0:
                phase_seen += 1
                fail_seen += 1
                batch_failed = True
                if not tolerates_phase:
                    correct, note = False, f"dirty phase (kickback) on {inp}"
                    break
                continue
            if not _outputs_match(res.outputs, expected, output_moduli):
                output_seen += 1
                fail_seen += 1
                batch_failed = True
                if not approximate:
                    correct = False
                    note = f"wrong output on {inp}: {res.outputs} != {expected}"
                    break
        if calls == reps:
            batches_completed += 1
            batches_failed += int(batch_failed)
        if correct is False:
            # Even a fatal failure on the last shot follows the fail-fast policy.
            aborted = True
            break

    regression_pvalue = None
    if empirical:
        note = (f"empirical only: {batches_failed}/{batches_completed} failed input batches; "
                f"{output_seen} output, {phase_seen} phase, {dirty_seen} dirty-ancilla; "
                "no analytic model or regression acceptance; CP upper limit concerns "
                "sampled batch diagnostics, not quantum norm")
    elif not aborted:
        # Test only once, after all planned batches; no optional-stopping p-values.
        regression_pvalue = binomial_tail(
            batches_failed, batches_completed, batch_fail_analytic)
        if approximate:
            correct = regression_pvalue >= REGRESSION_ALPHA
            note = (f"approximate: designed P(fail)={fail_analytic:.3e}; "
                    f"{batches_failed}/{batches_completed} failed input batches; "
                    f"binomial tail={regression_pvalue:.3e} "
                    f"{'<' if not correct else '>='} alpha={REGRESSION_ALPHA:g}")
            if dirty_seen:
                note += f"; {dirty_seen} dirty-ancilla"
            if phase_seen:
                note += f"; {phase_seen} phase (vented)"

    fail_bound95 = (1.0 if aborted else
                    clopper_pearson_upper(batches_failed, batches_completed))
    r = count(case.circuit)
    if analysis_plan is not None:
        validate_plan_binding(analysis_plan, case.circuit)
    result = Result(
        version=version,
        correct=correct,
        n_trials=n_trials,
        toffoli=r.toffoli,
        clifford=r.clifford,
        measurements=r.measurements,
        peak_qubits=r.peak_qubits,
        depth=r.depth,
        n_gates=r.n_gates,
        toffoli_depth=r.toffoli_depth,
        volume=r.volume,
        toffoli_volume=r.toffoli_volume,
        circuit_volume=r.circuit_volume,
        p_correct=None if empirical else 1.0 - fail_analytic,
        log2_fail=math.log2(fail_analytic) if approximate else None,
        fail_analytic=fail_analytic,
        fail_seen=fail_seen,
        dirty_seen=dirty_seen,
        phase_seen=phase_seen,
        trials=trials,
        fail_bound95=fail_bound95,
        note=note,
        inputs_executed=inputs_executed,
        batches_completed=batches_completed,
        batches_failed=batches_failed,
        meas_reps=reps,
        aborted=aborted,
        batch_fail_analytic=batch_fail_analytic,
        regression_pvalue=regression_pvalue,
        field=params.FIELD,
        field_u=params.U,
        field_prime=params.PRIME,
        input_distribution=case.meta.get("input_distribution"),
        arithmetic_contract=case.meta.get("arithmetic_contract"),
        empirical_only=empirical,
        observed_pass=fail_seen == 0,
        output_seen=output_seen,
        seed=seed,
        circuit_config=circuit_config,
        analysis_ir_digest=None if analysis_plan is None else analysis_plan.ir_digest,
    )
    if persist:
        report.append_result(asdict(result))
    return result
