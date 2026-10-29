"""Fixed-sample bounds on all-measurement-clean window inputs.

The probability is over the named, rejection-conditioned input law. It is not
an analytic arithmetic rate or a discrete-log recovery probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .. import params
from ..core.symbolic_batch import compile_symbolic, simulate_batch, validate_plan_binding
from ..point import active as curve
from ..point.sampling import SubgroupSampler
from ..scalarmul.window import WindowTable, add_window_affine_inplace_ref
from .registry import Case
from .statistics import clopper_pearson_upper, probability, trial_count


MODEL_KIND = "fixed-sample-all-outcome-window-bound.v1"
SUCCESS_EVENT = ("canonical group output; unchanged index; clean released and residual ancillas; "
                 "zero residual phase for every measurement outcome")
INPUT_LAW = ("uniform nonidentity prime-subgroup point and uniform window digit, "
             "jointly conditioned only on the declared algebraic preconditions")


@dataclass(frozen=True)
class WindowDomain:
    field: str
    n: int
    p: int
    a: int
    b: int
    order: int
    cofactor: int

    def __post_init__(self):
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("domain field name must be nonempty")
        if type(self.n) is not int or self.n < 2:
            raise ValueError("domain width must be an integer at least two")
        if type(self.p) is not int or not 3 < self.p < 1 << self.n or self.p % 2 == 0:
            raise ValueError("domain requires an odd prime fitting its register")
        if type(self.order) is not int or self.order <= 6 or self.order % 2 == 0:
            raise ValueError("calibration requires an odd prime subgroup order greater than six")
        if (type(self.cofactor) is not int or self.cofactor < 1 or
                math.gcd(self.order, self.cofactor) != 1):
            raise ValueError("cofactor must be positive and coprime to subgroup order")
        if type(self.a) is not int or type(self.b) is not int:
            raise ValueError("curve coefficients must be integers")
        object.__setattr__(self, "a", self.a % self.p)
        object.__setattr__(self, "b", self.b % self.p)

    @classmethod
    def active(cls):
        return cls(params.FIELD, params.U, params.PRIME, curve.CURVE.A, curve.CURVE.B,
                   curve.ORDER, curve.CURVE.COFACTOR)


def source_digest() -> str:
    """Identify all local QSM Python sources, including uncommitted modules."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationResult:
    model_kind: str
    samples: int
    bad_inputs: int
    clean_fraction: float
    failure_upper: float
    success_lower: float
    confidence: float
    counts: dict[str, int]
    attempted_inputs: int
    rejected_inputs: int
    seed: int
    batch_size: int
    input_distribution: str
    success_event: str
    domain: dict
    table_fingerprint: str
    circuit_config: dict
    source_digest: str
    analysis_ir_digest: str
    evaluator_stats: dict
    sampler_stats: dict
    elapsed_seconds: float


def _validate_case(case: Case, domain: WindowDomain) -> WindowTable:
    table = case.analysis_context.get("window_table")
    if not isinstance(table, WindowTable):
        raise ValueError("calibration requires the window builder's explicit table context")
    if (table.p, table.a, table.b) != (domain.p, domain.a, domain.b):
        raise ValueError("calibration domain does not match the window table")
    if case.meta.get("table_fingerprint") != table.fingerprint:
        raise ValueError("calibration table fingerprint does not match the case")
    outputs = case.circuit.inputs
    if (set(outputs) != {"j", "x", "y"} or len(outputs["j"]) != table.bits or
            len(outputs["x"]) != domain.n or len(outputs["y"]) != domain.n):
        raise ValueError("calibration requires one window's j/x/y output interface")
    if case.meta.get("output_moduli") != {"x": domain.p, "y": domain.p}:
        raise ValueError("calibration field does not match the declared output residues")
    if case.meta.get("empirical_only") is not True or case.meta.get("fail_analytic") is not None:
        raise ValueError("window calibration must not overwrite a declared analytic model")
    return table


def calibrate_window(
    case: Case, *, samples: int, confidence: float = 0.99, seed: int = 1,
    batch_size: int = 1024, domain: WindowDomain | None = None, stream=None, plan=None,
) -> CalibrationResult:
    """Calibrate a fixed number of IID accepted inputs, with no optional stopping.

    Group order and cofactor are domain assumptions, not inferred from samples.
    An input with a possible phase error counts once as bad even when only half
    of its measurement outcomes would fail. No repeated measurement histories
    enter this binomial sample count.
    """
    trial_count(samples, "success samples", minimum=1)
    trial_count(batch_size, "success batch size", minimum=1)
    confidence = probability(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if type(seed) is not int:
        raise ValueError("calibration seed must be an integer")
    domain = WindowDomain.active() if domain is None else domain
    table = _validate_case(case, domain)
    before_source = source_digest()
    start_time = time.monotonic()
    if stream is not None:
        action = "Compiling" if plan is None else "Using bound"
        print(f"{action} all-outcome checker for {samples} fixed input samples...",
              file=stream, flush=True)
    plan = compile_symbolic(case.circuit) if plan is None else plan
    validate_plan_binding(plan, case.circuit)
    sampler = SubgroupSampler(domain.p, domain.a, domain.b, domain.order, domain.cofactor)
    rng = random.Random(seed)
    counts = dict.fromkeys(("value", "noncanonical", "dirty", "phase", "constant_phase",
                            "summary_mismatch"), 0)
    bad = accepted = attempted = rejected = 0
    while accepted < samples:
        size = min(batch_size, samples - accepted)
        inputs, expected = [], []
        while len(inputs) < size:
            x, y = sampler.sample(rng)
            digit = rng.randrange(1 << table.bits)
            attempted += 1
            try:
                xr, yr = add_window_affine_inplace_ref(x, y, digit, table)
            except AssertionError:
                rejected += 1
                continue
            inputs.append({"j": digit, "x": x, "y": y})
            expected.append({"j": digit, "x": xr, "y": yr})
        hazards = simulate_batch(plan, inputs, expected,
                                 output_moduli={"x": domain.p, "y": domain.p})
        if hazards.samples != size:
            raise ValueError("symbolic evaluator returned an incorrect lane count")
        bad += hazards.bad_count
        for name in counts:
            counts[name] += getattr(hazards, f"{name}_count")
        accepted += size
        if stream is not None:
            print(f"Calibration inputs {accepted}/{samples}: {bad} hazardous inputs",
                  file=stream, flush=True)
    validate_plan_binding(plan, case.circuit)
    if source_digest() != before_source:
        raise ValueError("QSM sources changed during calibration; no bound is certified")
    upper = clopper_pearson_upper(bad, samples, confidence=confidence)
    config = {"lookup_backend": case.meta.get("lookup_backend"),
              "dummy_policy": case.meta.get("dummy_policy"),
              "tuning": case.meta.get("tuning"),
              "window_bits": table.bits}
    config = json.loads(json.dumps(config, allow_nan=False))
    stats = getattr(sampler, "stats", {})
    stats = stats() if callable(stats) else stats
    return CalibrationResult(
        model_kind=MODEL_KIND, samples=samples, bad_inputs=bad,
        clean_fraction=(samples-bad)/samples, failure_upper=upper,
        success_lower=1.0-upper, confidence=confidence, counts=counts,
        attempted_inputs=attempted, rejected_inputs=rejected, seed=seed,
        batch_size=batch_size, input_distribution=INPUT_LAW, success_event=SUCCESS_EVENT,
        domain=asdict(domain), table_fingerprint=table.fingerprint,
        circuit_config=config, source_digest=before_source,
        analysis_ir_digest=plan.ir_digest, evaluator_stats=dict(plan.stats),
        sampler_stats=json.loads(json.dumps(stats, allow_nan=False)),
        elapsed_seconds=time.monotonic()-start_time,
    )


def attach_calibration(result, calibration: CalibrationResult) -> None:
    """Keep simulation-shot counters separate from input-hazard calibration."""
    if not result.empirical_only or result.fail_analytic is not None:
        raise ValueError("statistical calibration cannot replace an analytic declaration")
    if (not calibration.analysis_ir_digest or
            result.analysis_ir_digest != calibration.analysis_ir_digest):
        raise ValueError("calibration IR digest does not match the benchmarked circuit")
    domain = calibration.domain
    if ((result.field, result.field_u, result.field_prime) !=
            (domain["field"], domain["n"], domain["p"])):
        raise ValueError("calibration field does not match the benchmark result")
    if result.circuit_config.get("table_fingerprint") != calibration.table_fingerprint:
        raise ValueError("calibration table does not match the benchmark result")
    for key, value in calibration.circuit_config.items():
        if result.circuit_config.get(key) != value:
            raise ValueError(f"calibration {key} does not match the benchmark result")
    if source_digest() != calibration.source_digest:
        raise ValueError("calibration source digest is stale")
    result.success_lower = calibration.success_lower
    result.success_confidence = calibration.confidence
    result.success_samples = calibration.samples
    result.bad_inputs = calibration.bad_inputs
    result.clean_input_fraction = calibration.clean_fraction
    result.circuit_config["success_calibration"] = asdict(calibration)
