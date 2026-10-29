"""Conditional algebra tests, not current-code Shor probability certification."""
from collections import Counter
from dataclasses import fields, replace
from decimal import Decimal
from fractions import Fraction
import math

import numpy as np
import pytest

from qsm.bench.success_bound import (
    CompositionAssumptions, FailureProfile, ScheduleRequirement,
    conditional_success_bound, domain_exception_bound, extended_failure_bound,
    total_variation_bound,
)


ORDER = 1009  # Prime, deliberately small enough to expose the 6/r correction.


def evidence():
    # These are test hypotheses, NOT evidence about any production circuit.
    return CompositionAssumptions(**{
        field.name: f"toy hypothetical proof of {field.name}"
        for field in fields(CompositionAssumptions)
    })


def profile(table="table-A", config="config-A", source="source-A", upper=0, confidence=0.99):
    return FailureProfile(table, config, source, upper, confidence)


def requirement(item):
    return ScheduleRequirement(*item.identity)


def compute(schedule=None, profiles=None, **kwargs):
    item = profile()
    parameters = dict(order=ORDER, global_alpha=Decimal("0.01"),
                      assumptions=evidence(), ideal_success_probability=Fraction(3, 4))
    parameters.update(kwargs)
    return conditional_success_bound(
        [requirement(item)] if schedule is None else schedule,
        [item] if profiles is None else profiles, **parameters,
    )


def test_exception_and_extension_arithmetic_exact_even_for_tiny_probabilities():
    assert domain_exception_bound(13) == Fraction(6, 13)
    assert domain_exception_bound(13, dummy_zero=True) == Fraction(5, 13)
    assert domain_exception_bound(3) == 1
    assert extended_failure_bound(0, ORDER) == Fraction(6, ORDER - 1)
    assert extended_failure_bound(Fraction(1, 100), ORDER) == \
        Fraction(ORDER, ORDER - 1) * (Fraction(1, 100) + Fraction(6, ORDER))
    assert extended_failure_bound(1, ORDER) == 1
    huge_order = 2**252 + 27742317777372353535851937790883648493
    assert domain_exception_bound(huge_order) == Fraction(6, huge_order)
    assert extended_failure_bound(0, huge_order) == Fraction(6, huge_order - 1)
    # No underflow is allowed even well below the minimum positive binary float.
    tiny = Fraction(1, 10**1000)
    assert 0 < total_variation_bound(tiny) <= Fraction(1, 2**255)
    assert total_variation_bound(tiny)**2 >= 4 * tiny


@pytest.mark.parametrize("mass", [0, Fraction(1, 100), Fraction(1, 16), Fraction(1, 4), 1, 20])
def test_projection_bound_endpoints_and_upward_rounding(mass):
    upper = total_variation_bound(mass)
    assert 0 <= upper <= 1
    if mass >= Fraction(1, 4):
        assert upper == 1
    else:
        assert upper * upper >= 4 * mass
        if mass:
            assert (upper - Fraction(1, 2**256))**2 < 4 * mass
    if mass == Fraction(1, 16):
        assert upper == Fraction(1, 2)


def test_complete_result_is_always_conditional_and_echoes_all_assumptions():
    result = compute()
    epsilon = Fraction(6, ORDER - 1)
    assert result.status == "conditional"
    assert "NOT certified" in result.label
    assert result.missing_conditions == ()
    assert result.occurrence_epsilons == (epsilon,)
    assert result.confidence_spent == result.global_alpha == Fraction(1, 100)
    assert result.failure_union_upper == epsilon
    assert result.total_variation_upper == total_variation_bound(epsilon)
    assert result.success_lower == Fraction(3, 4) - result.total_variation_upper
    assert dict(result.assumptions) == {field.name: getattr(evidence(), field.name)
                                       for field in fields(CompositionAssumptions)}


def test_defaults_do_not_infer_offset_schedule_or_ideal_success_probability():
    item = profile()
    result = conditional_success_bound([requirement(item)], [item], order=ORDER,
                                       global_alpha=0.01)
    assert result.status == "unavailable"
    assert "fresh_random_nonzero_group_offset" in result.missing_conditions
    assert "ideal_success_probability" in result.missing_conditions
    assert "canonical_logical_encoding" in result.missing_conditions
    assert result.success_lower is result.total_variation_upper is None
    assert result.occurrence_epsilons == ()
    empty = compute(schedule=[], profiles=[])
    assert "explicit_nonempty_schedule" in empty.missing_conditions


@pytest.mark.parametrize("name", [field.name for field in fields(CompositionAssumptions)])
def test_each_prerequisite_is_individually_required(name):
    result = compute(assumptions=replace(evidence(), **{name: None}))
    assert result.status == "unavailable"
    assert result.missing_conditions == (name,)
    assert result.success_lower is None


@pytest.mark.parametrize("name", ["table_id", "config_id", "source_digest"])
def test_profile_coverage_requires_all_three_exact_identities(name):
    item = profile()
    wrong = replace(requirement(item), **{name: "different"})
    result = compute(schedule=[wrong])
    assert result.status == "unavailable"
    assert any(missing.startswith("profile_coverage[0]") for missing in result.missing_conditions)
    assert result.used_profile_identities == ()
    assert result.confidence_spent == 0


def test_one_missing_occurrence_refuses_entire_composition():
    item = profile()
    result = compute(schedule=[requirement(item), ScheduleRequirement("missing", "c", "s")])
    assert result.status == "unavailable"
    assert any("profile_coverage[1]" in entry for entry in result.missing_conditions)
    assert result.success_lower is None


def test_repeated_profile_charged_once_for_confidence_each_time_for_failure():
    item = profile(upper=Fraction(1, 1000))
    occurrence = requirement(item)
    result = compute(schedule=[occurrence] * 5, profiles=[item])
    epsilon = extended_failure_bound(item.failure_upper, ORDER)
    assert result.status == "conditional"
    assert result.used_profile_identities == (item.identity,)
    assert result.confidence_spent == Fraction(1, 100)
    assert result.occurrence_epsilons == (epsilon,) * 5
    assert result.failure_union_upper == 5 * epsilon
    # Enough explicit occurrences cap the error at one and success at zero.
    result = compute(schedule=[occurrence] * 500, profiles=[item])
    assert len(result.occurrence_epsilons) == 500
    assert result.failure_union_upper == result.total_variation_upper == 1
    assert result.success_lower == 0


def test_confidence_union_allocation_is_not_an_independence_product():
    first, second = profile(), profile(table="table-B")
    schedule = [requirement(first), requirement(second)]
    result = compute(schedule=schedule, profiles=[first, second])
    assert result.status == "unavailable"
    assert result.confidence_spent == Fraction(1, 50)
    assert result.missing_conditions == ("global_confidence_allocation",)
    exact = compute(schedule=schedule, profiles=[first, second], global_alpha=0.02)
    assert exact.status == "conditional"
    # Unused evidence is not charged, and cannot improve a used profile's bound.
    unused = compute(profiles=[first, profile(table="unused", confidence=0, upper=1)])
    assert unused.confidence_spent == Fraction(1, 100)
    assert unused.occurrence_epsilons == compute().occurrence_epsilons


@pytest.mark.parametrize("copy", [profile(), profile(upper=Fraction(1, 10)), profile(confidence=1)])
def test_duplicate_or_repeated_seed_profile_records_never_silently_pool(copy):
    # Re-running the SAME table/config/source (including with the same seed)
    # cannot turn two input records into stronger confidence or doubled trials.
    result = compute(profiles=[profile(), copy])
    assert result.status == "unavailable"
    assert any("ambiguous_duplicate_profile" in entry for entry in result.missing_conditions)
    assert result.success_lower is None


def test_correlated_different_table_profiles_need_no_independence_claim():
    # Even if both calibration experiments used the same seed, Boole's union
    # bound is valid. The API does not combine samples or multiply confidences.
    items = [profile(table="A"), profile(table="B")]
    result = compute(schedule=list(map(requirement, items)), profiles=items, global_alpha=0.02)
    assert result.status == "conditional"
    assert result.confidence_spent == Fraction(2, 100)
    assert result.occurrence_epsilons == (Fraction(6, ORDER - 1),) * 2


def test_two_995_profiles_spend_exactly_one_percent_without_pooling():
    items = [profile(table="table-A", confidence=0.995),
             profile(table="table-B", confidence=0.995)]
    result = compute(schedule=list(map(requirement, items)), profiles=items, global_alpha=0.01)
    assert result.status == "conditional"
    assert result.confidence_spent == result.global_alpha == Fraction(1, 100)
    assert len(result.used_profile_identities) == 2


def test_success_probability_endpoints_are_not_ideal_probability_defaults():
    assert compute(ideal_success_probability=0).success_lower == 0
    best = compute(ideal_success_probability=1)
    assert best.success_lower == 1 - best.total_variation_upper
    assert compute(profiles=[profile(upper=1)]).success_lower == 0
    assert compute(profiles=[profile(confidence=1)], global_alpha=0).status == "conditional"


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -float("inf"),
                                 Decimal("NaN"), Decimal("Infinity"), -0.001, 1.001, "0.5", None])
def test_strict_probability_validation(bad):
    with pytest.raises(ValueError):
        profile(upper=bad)
    with pytest.raises(ValueError):
        profile(confidence=bad)
    with pytest.raises(ValueError):
        compute(global_alpha=bad)
    with pytest.raises(ValueError):
        extended_failure_bound(bad, ORDER)
    if bad is not None:  # None for p0 means unavailable, never a numerical default.
        with pytest.raises(ValueError):
            compute(ideal_success_probability=bad)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1, "0.2"])
def test_projection_rejects_invalid_mass(bad):
    with pytest.raises(ValueError):
        total_variation_bound(bad)


@pytest.mark.parametrize("bad", [True, 0, 2, 4, 13.0, "13"])
def test_subgroup_order_checks_are_not_implicit_coercions(bad):
    with pytest.raises(ValueError):
        domain_exception_bound(bad)
    with pytest.raises(ValueError):
        compute(order=bad)


@pytest.mark.parametrize("bad", [True, False, "", " ", 1])
def test_assumptions_are_not_blind_verified_booleans(bad):
    with pytest.raises(ValueError):
        CompositionAssumptions(canonical_logical_encoding=bad)
    with pytest.raises(ValueError):
        compute(assumptions=bad)


def test_identity_strings_and_record_types_are_explicit():
    for bad in (True, "", " ", 123):
        with pytest.raises(ValueError):
            profile(table=bad)
        with pytest.raises(ValueError):
            ScheduleRequirement("table", bad, "source")
    with pytest.raises(ValueError):
        compute(schedule=[True])
    with pytest.raises(ValueError):
        compute(profiles=[True])
    with pytest.raises(ValueError):
        domain_exception_bound(13, dummy_zero=1)


# Separate small-group oracle: no production scalar multiplication or affine refs.
def group_add(p, a, left, right):
    if left is None:
        return right
    if right is None:
        return left
    x, y = left
    u, v = right
    if x == u and (y + v) % p == 0:
        return None
    m = ((3 * x * x + a) * pow(2 * y, p - 2, p) if left == right else
         (v - y) * pow((u - x) % p, p - 2, p)) % p
    z = (m * m - x - u) % p
    return z, (m * (x - z) - y) % p


def cyclic_group(p, a, b, r):
    points = [(x, y) for x in range(p) for y in range(p)
              if (y * y - x**3 - a * x - b) % p == 0]
    for base in points:
        group, current = [None], base
        while current is not None and len(group) <= len(points):
            group.append(current)
            current = group_add(p, a, current, base)
        if len(group) == r:
            assert len(set(group)) == r and current is None
            return group
    raise AssertionError("fixture contains no order-r subgroup")


def excluded(point, addend, p, a, *, dummy=False):
    if point is None:
        return True
    if point[0] == addend[0] or point[1] == addend[1]:
        return True
    if dummy:
        return False
    output = group_add(p, a, point, addend)
    return output is None or output[0] == addend[0]


@pytest.mark.parametrize("p,a,b,r", [(7, 0, 1, 3), (13, 0, 2, 19),
                                     (43, 0, 2, 13), (43, 0, 7, 31)])
def test_exact_small_group_domain_exceptions_and_translate_kernel(p, a, b, r):
    group = cyclic_group(p, a, b, r)
    for addend in group[1:]:
        full = sum(excluded(point, addend, p, a) for point in group)
        dummy = sum(excluded(point, addend, p, a, dummy=True) for point in group)
        assert full <= 6 and dummy <= 5
        assert Fraction(full, r) <= domain_exception_bound(r)
        assert Fraction(dummy, r) <= domain_exception_bound(r, dummy_zero=True)
    for prefix in group:
        translated = Counter(group_add(p, a, offset, prefix) for offset in group[1:])
        assert translated == Counter({point: 1 for point in group if point != prefix})
        assert all(Fraction(translated[point], r - 1) <= Fraction(r, r - 1) / r
                   for point in group)
        assert translated[prefix] == 0
    # Without the fresh offset the point mass at a fixed prefix is NOT bounded
    # by r/(r-1) times uniform G. A correlated walk cannot supply this lemma.
    assert 1 > Fraction(r, r - 1) / r


def test_exact_conditioning_and_offset_extension_on_joint_point_digit_law():
    p, a, b, r = 43, 0, 7, 31
    group = cyclic_group(p, a, b, r)
    # Two binary digits, with a nonzero dummy for j=0; no quantum circuits built.
    addends = [group[1], group[1], group[2], group[3]]
    pairs = [(point, j) for point in group for j in range(4)]
    domain_bad = {(point, j) for point, j in pairs
                  if excluded(point, addends[j], p, a, dummy=j == 0)}
    accepted = [pair for pair in pairs if pair not in domain_bad]
    assert all(point is not None for point, _ in accepted)
    # Several deterministic artificial arithmetic bad sets on the valid domain.
    for stride in (1, 2, 5, 17, len(accepted) + 1):
        bad = set(accepted[::stride])
        conditional_upper = Fraction(len(bad), len(accepted))
        extended = extended_failure_bound(conditional_upper, r)
        for prefix in group:
            failures = sum((group_add(p, a, offset, prefix), j) in domain_bad | bad
                           for offset in group[1:] for j in range(4))
            assert Fraction(failures, 4 * (r - 1)) <= extended


def test_residue_equality_cannot_replace_canonical_logical_encoding():
    # Two basis encodings of residue zero in a width-three register over F_7.
    # A modulo comparison accepts both, but the logical states are orthogonal.
    canonical = np.zeros(8)
    noncanonical = np.zeros(8)
    canonical[0] = noncanonical[7] = 1
    assert 0 % 7 == 7 % 7
    assert np.vdot(canonical, noncanonical) == 0
    result = compute(assumptions=replace(evidence(), canonical_logical_encoding=None))
    assert result.status == "unavailable"
    assert result.missing_conditions == ("canonical_logical_encoding",)


def test_graph_state_phase_fault_projection_and_value_only_counterexample():
    # |graph(f)> for three input bits and one parity-output bit. A phase error on
    # one input leaves EVERY classical value and computational probability intact.
    n = 8
    ideal = np.zeros(2 * n, dtype=complex)
    actual = np.zeros_like(ideal)
    for x in range(n):
        index = 2 * x + x.bit_count() % 2
        ideal[index] = 1 / math.sqrt(n)
        actual[index] = (-1 if x == 0 else 1) / math.sqrt(n)
    assert np.allclose(abs(actual)**2, abs(ideal)**2)
    assert np.linalg.norm(actual - ideal) == pytest.approx(2 * math.sqrt(1 / n))
    trace_distance = math.sqrt(1 - abs(np.vdot(ideal, actual))**2)
    assert trace_distance <= float(total_variation_bound(Fraction(1, n)))
    # A final Hadamard-basis measurement sees an error despite zero value errors.
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    transform = h
    for _ in range(3):
        transform = np.kron(transform, h)
    tv = 0.5 * np.sum(abs(abs(transform @ actual)**2 - abs(transform @ ideal)**2))
    assert tv > 0
    assert tv <= trace_distance <= float(total_variation_bound(Fraction(1, n)))
    assert total_variation_bound(0) == 0 < tv  # Value-only profiles are unsound here.
    # The API therefore cannot certify composition without the phase/environment
    # prerequisite, even when all supplied numerical profiles report zero errors.
    result = compute(assumptions=replace(evidence(), all_outcome_common_environment=None))
    assert result.status == "unavailable" and result.success_lower is None
