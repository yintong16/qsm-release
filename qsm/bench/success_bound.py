"""Conditional probability algebra, NOT a success certificate for current Shor code.

No circuit is inspected or changed, no offset is installed, no schedule length
(such as 28) is inferred, and no ideal success probability p0 is supplied here.
In particular FRESH_RANDOM_NONZERO_GROUP_OFFSET is a prerequisite for a proposed
composition argument, not something the existing scalar-multiplication circuit
is claimed to implement. Supplying textual evidence below records assumptions;
it does not verify them. Even a complete result is labelled CONDITIONAL.

For a nonzero addend T in a prime subgroup G of odd order r, the exceptional
set for the incomplete affine window is contained in:
  infinity; points with x=x(T); points with y=y(T); points with x(P+T)=x(T).
The first contributes one, x=x(T) contributes +/-T, the cubic y=y(T) adds at
most two more (T is already counted), and the final test adds at most -2T
(the other preimage is infinity). Hence at most six points of G. For a disabled
zero digit with a finite nonzero dummy T, the final output-x test is absent,
giving at most five. These are algebraic exclusions, not arithmetic success
claims. For an accepted-domain all-outcome failure upper bound U, a safe bound
under uniform G is U + 6/r (we deliberately do not need the acceptance rate).

A fresh uniform R in G minus {0}, independent of uniform binary input digits, makes
each *ideal* prefix P=R+C uniform on G minus {C}, for each fixed digit string. This
translate kernel has density at most r/(r-1) relative to uniform G. Thus each
covered window has epsilon <= min(1, r/(r-1)*(U + 6/r)). No independence of
window failures is needed for a union bound over the complete ideal schedule.

The final 2*sqrt(sum epsilon_i) bound REQUIRES the separate common-good-input
projection argument: the implementation and ideal oracle agree coherently on
the good input subspace, including canonical logical encoding, residual phase,
clean ancillas, and every measurement outcome's common input-independent
environment (including outcome amplitudes, not only classical corrections). Classical
value agreement alone cannot establish this. On a common good subspace of
weight >=1-delta, the triangle inequality bounds the state distance by
2*sqrt(delta), hence bounds total variation under any final measurement. The
ideal p0 must concern the SAME final success event/offset-aware postprocessing.

Profile confidence uses a union bound, not independence. Reuse of an identical
profile in many occurrences is charged once for confidence and once PER
occurrence for failure mass. Distinct profiles, including those from repeated
RNG seeds, are never silently pooled; duplicate identity tuples are refused.
The identities and complete schedule are explicit caller inputs, not guessed
from hashes, counts, source paths, or circuit registry names.

Arithmetic stays rational, including 6/r. Finite float inputs are interpreted
as their round-trip decimal strings; use Fraction or Decimal for explicit exact
probabilities. Square roots are rounded UP to a 2**-256 grid, not to nearest;
success lower bounds consequently round down. This can loosen extremely tiny
bounds but never drop a nonzero term by floating-point underflow.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from fractions import Fraction
from math import isfinite, isqrt
from typing import Iterable


Probability = int | float | Decimal | Fraction
Identity = tuple[str, str, str]


def _rational(value: Probability, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, Fraction)):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction(str(value))
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return Fraction(value)


def _probability(value: Probability, name: str) -> Fraction:
    result = _rational(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _order(order: int) -> int:
    if type(order) is not int or order <= 2 or order % 2 == 0:
        raise ValueError("order must be an assumed odd prime greater than two")
    return order  # Primality is a documented caller assumption, not a test here.


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty explicit string")


def domain_exception_bound(order: int, *, dummy_zero: bool = False) -> Fraction:
    """Uniform-full-G algebraic domain upper bound; assume nonzero subgroup T.

    This is NOT a bound under the nonidentity law (whose denominator is r-1),
    nor a bound for arbitrary finite T outside G. Tiny-order bounds are capped
    at one. The dummy-zero refinement applies only to that disabled branch.
    """
    _order(order)
    if type(dummy_zero) is not bool:
        raise ValueError("dummy_zero must be boolean")
    return min(Fraction(1), Fraction(5 if dummy_zero else 6, order))


def extended_failure_bound(failure_upper: Probability, order: int) -> Fraction:
    """Extend a domain-conditioned profile to a fresh nonzero-offset prefix law.

    Assumes the profile's joint point/digit law and success event match those
    stated above. The result alone does not establish coherent composition.
    """
    _order(order)
    upper = _probability(failure_upper, "failure_upper")
    return min(Fraction(1), Fraction(order, order - 1) *
               (upper + domain_exception_bound(order)))


def total_variation_bound(failure_sum: Probability) -> Fraction:
    """Conservative min(1, 2*sqrt(failure_sum)); requires the projection argument."""
    value = _rational(failure_sum, "failure_sum")
    if value < 0:
        raise ValueError("failure_sum must be nonnegative")
    if value >= Fraction(1, 4):
        return Fraction(1)
    if value == 0:
        return Fraction(0)
    scale = 1 << 256
    squared = 4 * value
    numerator = squared.numerator * scale * scale
    root = isqrt(numerator // squared.denominator)
    if root * root * squared.denominator < numerator:
        root += 1
    return Fraction(root, scale)


@dataclass(frozen=True)
class FailureProfile:
    """One supplied fixed-sample upper bound; not a binomial pooling interface.

    Each identity denotes one particular table, arithmetic config, and source
    version. Identical seeds do not create independent evidence; independence
    between profiles is neither asserted nor required by the confidence union.
    The provenance establishing validity of each profile belongs in the
    profile_input_law assumption. This record does not certify an estimator.
    """

    table_id: str
    config_id: str
    source_digest: str
    failure_upper: Probability
    confidence: Probability

    def __post_init__(self):
        for name in ("table_id", "config_id", "source_digest"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "failure_upper", _probability(self.failure_upper, "failure_upper"))
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))

    @property
    def identity(self) -> Identity:
        return self.table_id, self.config_id, self.source_digest


@dataclass(frozen=True)
class ScheduleRequirement:
    """ONE ordered occurrence; repeat this record for repeated window use."""

    table_id: str
    config_id: str
    source_digest: str

    def __post_init__(self):
        for name in ("table_id", "config_id", "source_digest"):
            _text(getattr(self, name), name)

    @property
    def identity(self) -> Identity:
        return self.table_id, self.config_id, self.source_digest


@dataclass(frozen=True)
class CompositionAssumptions:
    """Unverified, caller-supplied evidence references, NOT 'verified' booleans.

    Each non-None field must explain the claimed prerequisite or identify its
    proof/configuration. None leaves the result unavailable. Strings do not
    become code certificates: every value is echoed in the returned result.
    """

    group_structure: str | None = None
    # Odd prime r, group order h*r, cofactor, and every finite table/dummy in G minus {0}.
    uniform_binary_digits: str | None = None
    # Independent uniform binary digits, with the actual schedule's window widths.
    fresh_random_nonzero_group_offset: str | None = None
    # FRESH_RANDOM_NONZERO_GROUP_OFFSET, independent and fresh per run; NOT implemented here.
    canonical_logical_encoding: str | None = None
    # No noncanonical coordinate encodings at composed logical boundaries.
    all_outcome_common_environment: str | None = None
    # Coherent all-outcome value/phase/ancilla agreement with a common
    # input-independent measurement environment, including outcome amplitudes.
    ideal_prefix_window_coverage: str | None = None
    # The supplied ordered list is the COMPLETE ideal-prefix window schedule.
    profile_input_law: str | None = None
    # Each fixed-sample profile's ANY-outcome-success event must include canonical
    # boundary encoding, cleanup, phase and common input-independent measurement
    # environment, under the specified joint point/digit rejection-conditioned law.
    composition_projection_argument: str | None = None
    # Common-good-input subspace argument, not just classical output correctness.
    ideal_success_event: str | None = None
    # p0 and actual output postprocessing have the same offset-aware success event.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                _text(value, field.name)


@dataclass(frozen=True)
class ConditionalSuccessBound:
    status: str
    label: str
    missing_conditions: tuple[str, ...]
    assumptions: tuple[tuple[str, str | None], ...]
    used_profile_identities: tuple[Identity, ...]
    global_alpha: Fraction
    confidence_spent: Fraction
    ideal_success_probability: Fraction | None
    occurrence_epsilons: tuple[Fraction, ...] = ()
    failure_union_upper: Fraction | None = None
    total_variation_upper: Fraction | None = None
    success_lower: Fraction | None = None


def conditional_success_bound(
    schedule: Iterable[ScheduleRequirement], profiles: Iterable[FailureProfile], *,
    order: int, global_alpha: Probability,
    assumptions: CompositionAssumptions | None = None,
    ideal_success_probability: Probability | None = None,
) -> ConditionalSuccessBound:
    """Evaluate only when every stated prerequisite and exact profile match is supplied.

    Malformed numeric/typed inputs raise ValueError. Missing evidence, absent or
    mismatched profile identities, duplicate profiles, incomplete (empty)
    coverage, missing p0, or excess confidence spending return 'unavailable'
    with no probability result. No fresh-offset adoption is inferred. Complete
    input gives only a conditional mathematical bound, never certified code.
    Unused profiles spend no confidence. Correlated profiles are allowed because
    a union bound does not require statistical independence; no samples merge.
    """
    _order(order)
    alpha = _probability(global_alpha, "global_alpha")
    p0 = (None if ideal_success_probability is None else
          _probability(ideal_success_probability, "ideal_success_probability"))
    if assumptions is None:
        assumptions = CompositionAssumptions()
    if not isinstance(assumptions, CompositionAssumptions):
        raise ValueError("assumptions must be CompositionAssumptions, not a verified flag")
    stated = tuple((field.name, getattr(assumptions, field.name)) for field in fields(assumptions))
    missing = [name for name, value in stated if value is None]
    if p0 is None:
        missing.append("ideal_success_probability")
    schedule = tuple(schedule)
    if not schedule:
        missing.append("explicit_nonempty_schedule")
    by_identity: dict[Identity, FailureProfile] = {}
    for profile in profiles:
        if not isinstance(profile, FailureProfile):
            raise ValueError("profiles must contain FailureProfile records")
        if profile.identity in by_identity:
            missing.append(f"ambiguous_duplicate_profile:{profile.identity!r}")
        else:
            by_identity[profile.identity] = profile
    used: dict[Identity, FailureProfile] = {}
    for i, requirement in enumerate(schedule):
        if not isinstance(requirement, ScheduleRequirement):
            raise ValueError("schedule must contain ScheduleRequirement records")
        if requirement.identity not in by_identity:
            missing.append(f"profile_coverage[{i}]:{requirement.identity!r}")
        else:
            used[requirement.identity] = by_identity[requirement.identity]
    spent = sum((1 - profile.confidence for profile in used.values()), Fraction(0))
    if spent > alpha:
        missing.append("global_confidence_allocation")
    common = dict(
        assumptions=stated, used_profile_identities=tuple(used),
        global_alpha=alpha, confidence_spent=spent, ideal_success_probability=p0,
    )
    if missing:
        return ConditionalSuccessBound(
            status="unavailable", label="Unavailable: composition prerequisites not established",
            missing_conditions=tuple(missing), **common,
        )
    epsilons = tuple(extended_failure_bound(by_identity[item.identity].failure_upper, order)
                     for item in schedule)
    union = min(Fraction(1), sum(epsilons, Fraction(0)))
    tv = total_variation_bound(union)
    return ConditionalSuccessBound(
        status="conditional", label="Conditional mathematical bound; NOT certified current-code success",
        missing_conditions=(), occurrence_epsilons=epsilons,
        failure_union_upper=union, total_variation_upper=tv,
        success_lower=max(Fraction(0), p0 - tv), **common,
    )
