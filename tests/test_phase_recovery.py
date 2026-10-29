"""Exact classical recovery; dense tests use the actual +/+ inverse-QFT oracle.

No full-size BSGS or quantum circuit is run. Large examples are integer-only
resource plans (or reject before invoking any backend callback).
"""
from dataclasses import replace
from fractions import Fraction
import json
import numpy as np
import pytest

from qsm.audit import phase_recovery as recovery
from qsm.audit.phase_recovery import (
    GroupOps, RecoveryPlan, WEI25519_ORDER, recover_sample, round_ratio,
    signed_interval_bsgs, tail_bound,
)


def cyclic(order):
    return GroupOps(order, 0, lambda x, y: (x + y) % order,
                    lambda x: -x % order,
                    lambda x: type(x) is int and 0 <= x < order)


def _dense_distribution(plan, logarithm, translation=0, *, bias_address_bits=None):
    """Fourier transform actual computational group-output amplitudes, not Eq23."""
    n, m, r = 1 << plan.p_control_bits, 1 << plan.q_control_bits, plan.order
    u, v = np.arange(n), np.arange(m)
    if bias_address_bits is not None:
        w = bias_address_bits + 1
        def signed_sum(values, bits):
            # Compute digit sums directly, including zero-padded partial top.
            result = np.zeros_like(values)
            for offset in range(0, bits, w):
                digit = ((values >> offset) & ((1 << w) - 1)) - (1 << bias_address_bits)
                result += digit << offset
            return result
        u, v = signed_sum(u, plan.p_control_bits), signed_sum(v, plan.q_control_bits)
    p_factor = pow(2, plan.p_exponent_offset, r)
    q_factor = pow(2, plan.q_exponent_offset, r)
    output = (p_factor * u[:, None]
              + plan.oracle_sign * q_factor * logarithm * v[None, :] + translation) % r
    probabilities = np.zeros((n, m))
    for point in range(r):
        amplitudes = (output == point).astype(complex) / np.sqrt(n * m)
        for axis, sign in ((0, plan.p_qft_sign), (1, plan.q_qft_sign)):
            transform = np.fft.ifft if sign == 1 else np.fft.fft
            amplitudes = transform(amplitudes, axis=axis, norm="ortho")
        probabilities += abs(amplitudes) ** 2
    return probabilities


def _dirichlet_probabilities(size, phase):
    indices = np.arange(size)
    return abs(np.exp(2j * np.pi * indices[:, None]
                      * (indices[None, :] / size - phase)).mean(axis=0)) ** 2


def _mixture(plan, logarithm):
    n, m, r = 1 << plan.p_control_bits, 1 << plan.q_control_bits, plan.order
    d_prime = logarithm * pow(2, plan.q_exponent_offset, r) * pow(
        pow(2, plan.p_exponent_offset, r), -1, r) % r
    canonical = sum(
        _dirichlet_probabilities(n, c / r)[:, None]
        * _dirichlet_probabilities(m, -(c * d_prime % r) / r)[None, :] / r
        for c in range(r)
    )
    j_indices = plan.p_qft_sign * np.arange(n) % n
    k_indices = -plan.oracle_sign * plan.q_qft_sign * np.arange(m) % m
    return canonical[np.ix_(j_indices, k_indices)]


def _cyclic_distance_bins(sample, size, numerator, denominator):
    delta = (Fraction(sample, size) - Fraction(numerator, denominator)) % 1
    return size * min(delta, 1 - delta)


@pytest.mark.parametrize("numerator,denominator,expected", [
    (1, 2, 1), (-1, 2, 0), (-3, 2, -1), (3, 2, 2), (8, 3, 3),
    ((1 << 300) + 1, 2, (1 << 299) + 1),
])
def test_exact_rounding(numerator, denominator, expected):
    assert round_ratio(numerator, denominator) == expected


def test_tail_and_probability_metadata_are_exact_not_physical_or_heuristic():
    plan = RecoveryPlan()
    assert (plan.p_control_bits, plan.q_control_bits) == (255, 255)
    assert tail_bound(101) == Fraction(102, 20402)
    expected = Fraction(WEI25519_ORDER - 1, WEI25519_ORDER) * (1 - tail_bound(101)) ** 2
    assert plan.ideal_success_lower_bound == expected
    assert expected > Fraction(9900259753, 10_000_000_000)
    meta = plan.metadata()
    assert meta["order_bits"] == 253 and meta["first_register_padding"] == 2
    assert meta["probability_is_heuristic"] is False
    assert meta["physical_success_lower_bound"] is None
    assert meta["production_omission_enabled"] is False
    assert meta["bound_requires_budget_acceptance"] is True
    assert meta["search_fits_budget"] is True
    assert Fraction(**meta["ideal_success_lower_bound"]) == expected
    json.dumps(meta)  # Fractions are intentionally serialized as exact numerator/denominator.
    with pytest.raises(ValueError, match="no heuristic"):
        RecoveryPlan(probability_model="ekera_heuristic")


@pytest.mark.parametrize("changes", [
    {"order": 2}, {"order": 15}, {"order": 341550071728321},
    {"order": (1 << 127) - 1},  # Unsupported large prime is not probable-prime accepted.
    {"order": True}, {"p_control_bits": 0}, {"q_control_bits": 4097},
    {"p_exponent_offset": -1}, {"q_exponent_offset": 4097},
    {"oracle_sign": 0}, {"p_qft_sign": True}, {"q_qft_sign": 2},
    {"first_bin_cutoff": 0}, {"second_bin_cutoff": 0},
    {"max_baby_entries": 0}, {"max_group_operations": False},
])
def test_invalid_plan_rejected(changes):
    with pytest.raises(ValueError):
        RecoveryPlan(**changes)


def test_supported_64_bit_prime_and_invalid_rounding_inputs():
    assert RecoveryPlan(order=(1 << 61) - 1).order == (1 << 61) - 1
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            tail_bound(value)
    for args in ((1, 0), (1, True), (0.5, 1)):
        with pytest.raises(ValueError):
            round_ratio(*args)


@pytest.mark.parametrize("order", [5, 7, 11, 13, 15])
def test_signed_interval_bsgs_exhaustive_including_duplicate_points(order):
    group = cyclic(order)
    for radius in range(order + 2):
        for target in range(order):
            result = signed_interval_bsgs(group, 1, target, radius)
            candidates = [t for t in range(-radius, radius + 1) if t % order == target]
            assert result.group_operations <= result.resources.group_operations_upper_bound
            if candidates:
                assert result.status == "recovered"
                assert result.value in candidates
                assert result.baby_entries == min(order, result.resources.baby_entries)
            else:
                assert result.status == "search_exhausted"
                assert result.value is None


def test_interval_overshoot_identity_and_failed_candidate_verification():
    group = cyclic(31)
    # Radius 3 => width 7; the s=3 rectangle also represents offsets 7 and 8.
    assert signed_interval_bsgs(group, 1, 4, 3).status == "search_exhausted"
    assert signed_interval_bsgs(group, 1, 5, 3).status == "search_exhausted"
    assert signed_interval_bsgs(group, 1, 0, 0).value == 0
    # A faulty negation can manufacture a table hit; the candidate must be checked.
    broken = replace(group, neg=lambda x: x)
    result = signed_interval_bsgs(broken, 1, 27, 3)
    assert result.status == "verification_failed" and result.value is None


@pytest.mark.parametrize("order", [5, 7, 11])
def test_dense_actual_plus_inverse_oracle_and_probability_weighted_recovery(order):
    plan = RecoveryPlan(order=order, p_control_bits=order.bit_length() + 1,
                        q_control_bits=order.bit_length() - 1,
                        first_bin_cutoff=2, second_bin_cutoff=2)
    group = cyclic(order)
    n, m = 1 << plan.p_control_bits, 1 << plan.q_control_bits
    for d in range(order):
        dense = _dense_distribution(plan, d)
        assert dense.sum() == pytest.approx(1, abs=2e-14)
        np.testing.assert_allclose(dense, _mixture(plan, d), rtol=0, atol=2e-14)
        np.testing.assert_allclose(dense, _dense_distribution(plan, d, 3), rtol=0, atol=2e-15)
        np.testing.assert_allclose(dense, _dense_distribution(plan, d, bias_address_bits=1),
                                   rtol=0, atol=2e-15)
        success_mass = 0.0
        for j in range(n):
            for k in range(m):
                result = recover_sample(plan, j, k, 1, d, group)
                assert result.normalized_measurements == (-j % n, k)
                assert result.group_operations <= result.resources.group_operations_upper_bound
                assert result.status in ("recovered", "search_exhausted")
                if result.status == "recovered":
                    assert result.logarithm == d  # No accepted wrong candidate, including d=0.
                    assert abs(result.t) <= plan.interval_radius
                    assert abs(result.eta) <= plan.eta_radius
                    success_mass += dense[j, k]
                # Directly check all guaranteed-good latent eigenphase labels.
                for c in range(1, order):
                    canonical_j, canonical_k = result.normalized_measurements
                    good = (_cyclic_distance_bins(canonical_j, n, c, order) <= 2
                            and _cyclic_distance_bins(canonical_k, m, -c*d, order) <= 2)
                    if good:
                        assert result.status == "recovered"
        assert success_mass + 2e-13 >= float(plan.ideal_success_lower_bound)


@pytest.mark.parametrize("oracle_sign,p_sign,q_sign", [
    (1, -1, -1), (1, 1, 1), (-1, 1, 1), (-1, -1, -1), (1, -1, 1),
])
def test_all_sign_conventions_against_dense_oracle(oracle_sign, p_sign, q_sign):
    plan = RecoveryPlan(order=7, p_control_bits=5, q_control_bits=4,
                        oracle_sign=oracle_sign, p_qft_sign=p_sign, q_qft_sign=q_sign,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    np.testing.assert_allclose(_dense_distribution(plan, 3), _mixture(plan, 3),
                               rtol=0, atol=2e-14)


def test_default_inverse_qft_sign_regression_has_a_wrong_sign_witness():
    plan = RecoveryPlan(order=31, p_control_bits=7, q_control_bits=7,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    dense = _dense_distribution(plan, 7)
    assert dense[41, 33] > 0.01  # A real high-probability sample of the actual oracle.
    result = recover_sample(plan, 41, 33, 1, 7, cyclic(31))
    assert result.status == "recovered" and result.logarithm == 7
    assert result.normalized_measurements == (87, 33)
    wrong = recover_sample(replace(plan, p_qft_sign=1), 41, 33, 1, 7, cyclic(31))
    assert wrong.status == "search_exhausted"


@pytest.mark.parametrize("p_offset,q_offset", [(0, 3), (2, 0), (3, 4)])
def test_shifted_contiguous_exponents_recover_original_logarithm(p_offset, q_offset):
    plan = RecoveryPlan(order=31, p_control_bits=7, q_control_bits=6,
                        p_exponent_offset=p_offset, q_exponent_offset=q_offset,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    d = 9
    dense = _dense_distribution(plan, d)
    np.testing.assert_allclose(dense, _mixture(plan, d), rtol=0, atol=3e-14)
    d_prime = d * pow(2, q_offset, 31) * pow(pow(2, p_offset, 31), -1, 31) % 31
    for c in range(1, 31):
        j = round_ratio((1 << 7) * c, 31) % (1 << 7)
        k = round_ratio((1 << 6) * (c * d_prime % 31), 31) % (1 << 6)
        assert dense[j, k] > 0
        got = recover_sample(plan, j, k, 1, d, cyclic(31))
        assert got.status == "recovered" and got.logarithm == d


def test_invalid_samples_points_and_backend_order():
    plan = RecoveryPlan(order=7, p_control_bits=4, q_control_bits=3,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    for j, k in ((-1, 0), (16, 0), (0, 8), (True, 0), (0, 0.5)):
        with pytest.raises(ValueError, match="Fourier"):
            recover_sample(plan, j, k, 1, 2, cyclic(7))
    with pytest.raises(ValueError, match="order differs"):
        recover_sample(plan, 0, 0, 1, 2, cyclic(11))
    with pytest.raises(ValueError, match="identity"):
        recover_sample(plan, 0, 0, 0, 2, cyclic(7))
    with pytest.raises(ValueError, match="invalid group point"):
        recover_sample(plan, 0, 0, 1, 7, cyclic(7))
    lying_order = replace(cyclic(11), order=7)
    with pytest.raises(ValueError, match="torsion"):
        recover_sample(plan, 0, 0, 1, 2, lying_order)
    unhashable = GroupOps(7, [], lambda x,y: [], lambda x: [], lambda x: True)
    with pytest.raises(ValueError, match="hashable"):
        signed_interval_bsgs(unhashable, [1], [2], 1)


def test_candidate_verification_not_merely_table_lookup(monkeypatch):
    plan = RecoveryPlan(order=31, p_control_bits=7, q_control_bits=7,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    monkeypatch.setattr(recovery, "_interval_match", lambda *args: 0)
    got = recover_sample(plan, 41, 33, 1, 7, cyclic(31))
    assert got.status == "verification_failed"
    assert got.logarithm is None


def test_zero_eigenlabel_does_not_divide_by_zero():
    plan = RecoveryPlan(order=31, p_control_bits=7, q_control_bits=7,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    result = recover_sample(plan, 0, 0, 1, 7, cyclic(31))
    assert result.status == "search_exhausted"  # eta=0 is skipped, no inverse(0).


@pytest.mark.parametrize("a,b,h,t,entries,dominant", [
    (253, 208, 52, 1776810790486017, 59612261, 6318899456),
    (255, 208, 14, 1776810790486017, 59612261, 1788367772),
    (253, 204, 52, 28428972647776257, 238449042, 25275598242),
    (255, 204, 14, 28428972647776257, 238449042, 7153471202),
])
def test_full_size_is_parameter_only_and_rejects_before_any_backend_call(a,b,h,t,entries,dominant):
    plan = RecoveryPlan(p_control_bits=a, q_control_bits=b)
    assert plan.eta_radius == h and plan.interval_radius == t
    assert plan.resources().baby_entries == entries
    assert plan.resources().dominant_additions == dominant
    assert not plan.metadata()["search_fits_budget"]
    def forbidden(*args):
        pytest.fail("Full-size rejected plans must not call the group backend")
    backend = GroupOps(WEI25519_ORDER, None, forbidden, forbidden, forbidden)
    result = recover_sample(plan, 0, 0, None, None, backend)
    assert result.status == "budget_rejected"
    assert result.group_operations == result.baby_entries == 0
    interval = signed_interval_bsgs(backend, None, None, t)
    assert interval.status == "budget_rejected" and interval.group_operations == 0


def test_both_budget_limits_and_worst_case_admission():
    plan = RecoveryPlan(order=31, p_control_bits=7, q_control_bits=7,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    resources = plan.resources()
    for changes in ({"max_baby_entries": resources.baby_entries - 1},
                    {"max_group_operations": resources.group_operations_upper_bound - 1}):
        denied = recover_sample(replace(plan, **changes), 41, 33, 1, 7, cyclic(31))
        assert denied.status == "budget_rejected" and denied.group_operations == 0
    admitted = replace(plan, max_baby_entries=resources.baby_entries,
                       max_group_operations=resources.group_operations_upper_bound)
    assert recover_sample(admitted, 41, 33, 1, 7, cyclic(31)).logarithm == 7
    interval = signed_interval_bsgs(cyclic(31), 1, 2, 3, max_group_operations=1)
    assert interval.status == "budget_rejected" and interval.group_operations == 0


def test_complete_elliptic_curve_backend_including_identity_and_negative_points():
    # E/F17: y^2=x^3+2x+2 has prime order 19, generated by (5,1).
    # Reuse the independent complete classical toy law, not an incomplete adder.
    from qsm.audit.signed_chain import group_add, group_mul

    def contains(point):
        if point is None:
            return True
        return (isinstance(point, tuple) and len(point) == 2
                and all(type(x) is int and 0 <= x < 17 for x in point)
                and (point[1]**2-point[0]**3-2*point[0]-2) % 17 == 0)

    backend = GroupOps(19, None, lambda p,q: group_add(p,q,17,2),
                       lambda p: None if p is None else (p[0], -p[1] % 17), contains)
    base = (5, 1)
    plan = RecoveryPlan(order=19, p_control_bits=7, q_control_bits=5,
                        first_bin_cutoff=1, second_bin_cutoff=1)
    assert group_mul(19, base, 17, 2) is None
    for d in range(19):
        target = group_mul(d, base, 17, 2)
        for c in range(1, 19):
            j = round_ratio(128*c, 19) % 128
            k = round_ratio(32*(c*d % 19), 19) % 32
            result = recover_sample(plan, j, k, base, target, backend)
            assert result.status == "recovered" and result.logarithm == d
            assert result.group_operations <= result.resources.group_operations_upper_bound
    for t in range(-4, 5):
        result = signed_interval_bsgs(backend, base, group_mul(t, base, 17, 2), 4)
        assert result.status == "recovered" and result.value == t


def test_elementary_tail_bound_against_direct_phase_probabilities():
    for size in (8, 16, 32, 64):
        for phase in np.linspace(0, 1, 37, endpoint=False):
            probabilities = _dirichlet_probabilities(size, phase)
            distance = abs((np.arange(size) - size * phase + size/2) % size - size/2)
            for cutoff in (1, 2, 3, 5, 101):
                assert probabilities[distance > cutoff].sum() <= float(tail_bound(cutoff)) + 1e-13
