"""Bounded classical phase recovery; no quantum schedule or omission is enabled.

The default experiment is the parent's +uP +vQ oracle, inverse QFT on BOTH
controls, A=B=255. Frequencies are normalized to the positive-QFT, +uP-vQ
convention of Ekerå's equation (23): j'=-j mod 2**A, k'=k. In that convention
z=nearest(r*j'/2**A), v=nearest(r*k'/2**B), and d=(t-v)/(z+eta) mod r.

The exact rational ideal-success bound is the independently derived Dirichlet
bound in decision 28, NOT Ekerå's heuristic probability estimate and NOT a
physical-circuit success claim. Group operations and Fourier transforms in its
hypotheses are exact. The classical backend must supply canonical, hashable
points with exact equality, including the identity, and exact complete addition.
No external recovery implementation is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Callable, Hashable


WEI25519_ORDER = (1 << 252) + 27742317777372353535851937790883648493
DEFAULT_MAX_BABY_ENTRIES = 10_000
DEFAULT_MAX_GROUP_OPERATIONS = 1_000_000
_MAX_PARAMETER_BITS = 4096


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _ceil_ratio(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def round_ratio(numerator: int, denominator: int) -> int:
    """Exact nearest integer, ties toward +infinity, including negative inputs."""
    _integer("denominator", denominator, 1)
    if type(numerator) is not int:
        raise ValueError("numerator must be an integer")
    return (2 * numerator + denominator) // (2 * denominator)


def tail_bound(cutoff: int) -> Fraction:
    """Exact upper bound outside cutoff cyclic phase bins; no floating point."""
    _integer("cutoff", cutoff, 1)
    return min(Fraction(1), Fraction(cutoff + 1, 2 * cutoff * cutoff))


def _known_prime(order: int) -> bool:
    """Known Wei25519 prime, or deterministic Miller-Rabin below 2**64.

    Unknown large orders are rejected, never accepted on a probable-prime test.
    The seven bases below are sufficient over the unsigned 64-bit domain.
    """
    if order == WEI25519_ORDER:
        return True
    if not 2 <= order < 1 << 64:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if order % prime == 0:
            return order == prime
    odd, shifts = order - 1, 0
    while odd % 2 == 0:
        odd //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        base %= order
        if base == 0:
            continue
        value = pow(base, odd, order)
        if value in (1, order - 1):
            continue
        for _ in range(shifts - 1):
            value = value * value % order
            if value == order - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class GroupOps:
    """Exact classical backend; points must be canonical and hashable.

    ``contains`` validates encodings and curve/group membership. Recovery also
    checks [r]P=[r]Q=O and P!=O. Q in <P> is a theorem precondition: the r-torsion
    checks alone do not prove this in a group whose r-torsion is not cyclic.
    Never use an incomplete affine circuit reference as this backend's add.
    """

    order: int
    identity: Hashable
    add: Callable[[Hashable, Hashable], Hashable]
    neg: Callable[[Hashable], Hashable]
    contains: Callable[[Hashable], bool]


@dataclass(frozen=True)
class SearchResources:
    interval_width: int
    baby_entries: int
    giant_steps_per_target: int
    eta_targets: int
    dominant_additions: int
    group_operations_upper_bound: int


def _resources(radius: int, targets: int, order_bits: int, *, recovery: bool) -> SearchResources:
    width = 2 * radius + 1
    babies = isqrt(width - 1) + 1
    giants = _ceil_ratio(width, babies)
    dominant = babies + targets * (giants - 1)
    # Count calls to add AND neg. Binary multiplication needs at most 2m-1
    # additions. Recovery: 2 order checks, 2 generator scalings, 2 initial
    # target scalings, 1 candidate verification. Standalone: 2 order checks,
    # 1 radius shift, 1 verification. Other costs: stride negation, target
    # addition and (for recovery) one Q addition between adjacent eta values.
    scalar_mults = 7 if recovery else 4
    extra = (targets - 1 if recovery else 0) + 2
    return SearchResources(width, babies, giants, targets, dominant,
                           dominant + scalar_mults * (2 * order_bits - 1) + extra)


@dataclass(frozen=True)
class RecoveryPlan:
    order: int = WEI25519_ORDER
    p_control_bits: int = 255
    q_control_bits: int = 255
    p_exponent_offset: int = 0
    q_exponent_offset: int = 0
    oracle_sign: int = 1
    p_qft_sign: int = -1
    q_qft_sign: int = -1
    first_bin_cutoff: int = 101
    second_bin_cutoff: int = 101
    max_baby_entries: int = DEFAULT_MAX_BABY_ENTRIES
    max_group_operations: int = DEFAULT_MAX_GROUP_OPERATIONS
    probability_model: str = "exact_dirichlet_tail"

    def __post_init__(self) -> None:
        _integer("order", self.order, 3)
        if self.order.bit_length() > _MAX_PARAMETER_BITS or not _known_prime(self.order):
            raise ValueError("order must be a supported proven odd prime (Wei25519 or <2**64)")
        for name in ("p_control_bits", "q_control_bits"):
            value = getattr(self, name)
            _integer(name, value, 1)
            if value > _MAX_PARAMETER_BITS:
                raise ValueError(f"{name} exceeds parameter size limit")
        for name in ("p_exponent_offset", "q_exponent_offset"):
            value = getattr(self, name)
            _integer(name, value)
            if value > _MAX_PARAMETER_BITS:
                raise ValueError(f"{name} exceeds parameter size limit")
        for name in ("oracle_sign", "p_qft_sign", "q_qft_sign"):
            if type(getattr(self, name)) is not int or getattr(self, name) not in (-1, 1):
                raise ValueError(f"{name} must be -1 or +1")
        for name in ("first_bin_cutoff", "second_bin_cutoff", "max_baby_entries",
                     "max_group_operations"):
            _integer(name, getattr(self, name), 1)
        if self.probability_model != "exact_dirichlet_tail":
            raise ValueError("only exact_dirichlet_tail is implemented; no heuristic bound is substituted")

    @property
    def eta_radius(self) -> int:
        n = 1 << self.p_control_bits
        return _ceil_ratio(2 * self.order * self.first_bin_cutoff + n, 2 * n)

    @property
    def interval_radius(self) -> int:
        m = 1 << self.q_control_bits
        return _ceil_ratio(2 * self.order * self.second_bin_cutoff + m, 2 * m)

    @property
    def ideal_success_lower_bound(self) -> Fraction:
        return (Fraction(self.order - 1, self.order)
                * (1 - tail_bound(self.first_bin_cutoff))
                * (1 - tail_bound(self.second_bin_cutoff)))

    def normalize_measurements(self, j: int, k: int) -> tuple[int, int]:
        """Normalize actual measured frequencies; default +/+ inverse -> (-j,k)."""
        n, m = 1 << self.p_control_bits, 1 << self.q_control_bits
        for name, value, bound in (("j", j, n), ("k", k, m)):
            if type(value) is not int or not 0 <= value < bound:
                raise ValueError(f"{name} outside declared Fourier register")
        return (self.p_qft_sign * j % n,
                -self.oracle_sign * self.q_qft_sign * k % m)

    def resources(self) -> SearchResources:
        return _resources(self.interval_radius, 2 * self.eta_radius + 1,
                          self.order.bit_length(), recovery=True)

    def metadata(self) -> dict:
        bound = self.ideal_success_lower_bound
        resources = self.resources()
        return {
            "scope": "isolated classical reference; no quantum omission enabled",
            "order": self.order,
            "order_bits": self.order.bit_length(),
            "p_control_bits": self.p_control_bits,
            "q_control_bits": self.q_control_bits,
            "first_register_padding": self.p_control_bits - self.order.bit_length(),
            "p_exponent_offset": self.p_exponent_offset,
            "q_exponent_offset": self.q_exponent_offset,
            "oracle_sign": self.oracle_sign,
            "p_qft_sign": self.p_qft_sign,
            "q_qft_sign": self.q_qft_sign,
            "normalization": "j'=p_qft_sign*j; k'=-oracle_sign*q_qft_sign*k (mod register sizes)",
            "probability_model": self.probability_model,
            "probability_is_heuristic": False,
            "probability_scope": "exact oracle and QFT; complete deterministic classical search",
            "ideal_success_lower_bound": {"numerator": bound.numerator, "denominator": bound.denominator},
            "physical_success_lower_bound": None,
            "assumptions": ["P has declared prime order", "Q belongs to <P>",
                            "canonical exact group backend", "clean exact linear oracle up to fixed translation",
                            "uniform contiguous scalar domains and exact Fourier transforms"],
            "first_bin_cutoff": self.first_bin_cutoff,
            "second_bin_cutoff": self.second_bin_cutoff,
            "eta_radius": self.eta_radius,
            "interval_radius": self.interval_radius,
            "resources": vars(resources).copy(),
            "max_baby_entries": self.max_baby_entries,
            "max_group_operations": self.max_group_operations,
            "search_fits_budget": _fits(resources, self.max_baby_entries, self.max_group_operations),
            "bound_requires_budget_acceptance": True,
            "production_omission_enabled": False,
        }


@dataclass(frozen=True)
class IntervalResult:
    status: str
    value: int | None
    group_operations: int
    baby_entries: int
    resources: SearchResources


@dataclass(frozen=True)
class RecoveryResult:
    status: str
    logarithm: int | None
    eta: int | None
    t: int | None
    normalized_measurements: tuple[int, int]
    group_operations: int
    baby_entries: int
    resources: SearchResources


def _fits(resources: SearchResources, entries: int, operations: int) -> bool:
    return (resources.baby_entries <= entries
            and resources.group_operations_upper_bound <= operations)


class _Meter:
    def __init__(self, group: GroupOps, limit: int):
        self.group, self.limit, self.used = group, limit, 0

    def _charge(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError("internal group-operation budget exceeded")
        self.used += 1

    def add(self, left: Hashable, right: Hashable) -> Hashable:
        self._charge()
        return self.group.add(left, right)

    def neg(self, point: Hashable) -> Hashable:
        self._charge()
        return self.group.neg(point)

    def mul(self, scalar: int, point: Hashable) -> Hashable:
        # Do NOT reduce the scalar here: [r]P validation must actually run.
        if scalar < 0:
            return self.neg(self.mul(-scalar, point))
        result = self.group.identity
        while scalar:
            if scalar & 1:
                result = self.add(result, point)
            scalar >>= 1
            if scalar:
                point = self.add(point, point)
        return result


def _validate_points(meter: _Meter, base: Hashable, target: Hashable) -> None:
    group = meter.group
    for point in (group.identity, base, target):
        if not group.contains(point):
            raise ValueError("invalid group point")
        try:
            hash(point)
        except TypeError as error:
            raise ValueError("group points must be canonical and hashable") from error
    if base == group.identity:
        raise ValueError("base must not be the identity")
    if (meter.mul(group.order, base) != group.identity
            or meter.mul(group.order, target) != group.identity):
        raise ValueError("base/target is not in declared order torsion")


def _babies(meter: _Meter, base: Hashable, count: int) -> tuple[dict, Hashable]:
    table = {}
    point = meter.group.identity
    for index in range(count):
        # Keep the smallest exponent when a toy interval exceeds the order.
        table.setdefault(point, index)
        point = meter.add(point, base)
    return table, meter.neg(point)  # -[count]P, including the last build addition


def _interval_match(meter: _Meter, shifted: Hashable, table: dict,
                    stride: Hashable, resources: SearchResources) -> int | None:
    point = shifted
    for step in range(resources.giant_steps_per_target):
        index = table.get(point)
        if index is not None:
            value = step * resources.baby_entries + index
            if value < resources.interval_width:  # final rectangle may overshoot
                return value
        if step + 1 < resources.giant_steps_per_target:
            point = meter.add(point, stride)
    return None


def signed_interval_bsgs(
    group: GroupOps, base: Hashable, target: Hashable, radius: int, *,
    max_baby_entries: int = DEFAULT_MAX_BABY_ENTRIES,
    max_group_operations: int = DEFAULT_MAX_GROUP_OPERATIONS,
) -> IntervalResult:
    """Find and verify t in [-radius,radius] with [t]base=target, deterministically.

    Reject the worst-case budget before point validation, group work or allocation.
    The backend order may be composite here; unlike RecoveryPlan, this routine
    makes no phase-success/invertibility claim.
    """
    _integer("order", group.order, 2)
    _integer("radius", radius)
    _integer("max_baby_entries", max_baby_entries, 1)
    _integer("max_group_operations", max_group_operations, 1)
    resources = _resources(radius, 1, group.order.bit_length(), recovery=False)
    if not _fits(resources, max_baby_entries, max_group_operations):
        return IntervalResult("budget_rejected", None, 0, 0, resources)
    meter = _Meter(group, max_group_operations)
    _validate_points(meter, base, target)
    table, stride = _babies(meter, base, resources.baby_entries)
    shifted = meter.add(target, meter.mul(radius % group.order, base))
    match = _interval_match(meter, shifted, table, stride, resources)
    if match is None:
        return IntervalResult("search_exhausted", None, meter.used, len(table), resources)
    value = match - radius
    verified = meter.mul(value % group.order, base) == target
    return IntervalResult("recovered" if verified else "verification_failed",
                          value if verified else None, meter.used, len(table), resources)


def recover_sample(
    plan: RecoveryPlan, j: int, k: int, P: Hashable, Q: Hashable, group: GroupOps,
) -> RecoveryResult:
    """Recover a verified logarithm from one actual measured frequency pair.

    P,Q are the ORIGINAL public points. Exponent-offset generator scaling and
    final logarithm rescaling are performed here. A budget rejection contributes
    no success guarantee, even though plan.metadata describes the ideal experiment.
    Every point addition/negation, including validation and candidate verification,
    is metered. Primality/field arithmetic, hashing and memory overhead are not
    included in the group-operation ledger.
    """
    if group.order != plan.order:
        raise ValueError("backend order differs from recovery plan")
    normalized = plan.normalize_measurements(j, k)
    resources = plan.resources()
    if not _fits(resources, plan.max_baby_entries, plan.max_group_operations):
        return RecoveryResult("budget_rejected", None, None, None, normalized, 0, 0, resources)
    meter = _Meter(group, plan.max_group_operations)
    _validate_points(meter, P, Q)
    r = plan.order
    p_factor = pow(2, plan.p_exponent_offset, r)
    q_factor = pow(2, plan.q_exponent_offset, r)
    base, target = meter.mul(p_factor, P), meter.mul(q_factor, Q)
    z = round_ratio(r * normalized[0], 1 << plan.p_control_bits)
    v = round_ratio(r * normalized[1], 1 << plan.q_control_bits)
    h, radius = plan.eta_radius, plan.interval_radius
    table, stride = _babies(meter, base, resources.baby_entries)
    shifted = meter.add(meter.mul((v + radius) % r, base),
                        meter.mul((z - h) % r, target))
    rescale = p_factor * pow(q_factor, -1, r) % r
    for eta in range(-h, h + 1):
        coefficient = (z + eta) % r
        if gcd(coefficient, r) == 1:
            match = _interval_match(meter, shifted, table, stride, resources)
            if match is not None:
                t = match - radius
                candidate = (t - v) * pow(coefficient, -1, r) * rescale % r
                verified = meter.mul(candidate, P) == Q
                return RecoveryResult("recovered" if verified else "verification_failed",
                                      candidate if verified else None, eta, t, normalized,
                                      meter.used, len(table), resources)
        if eta < h:
            shifted = meter.add(shifted, target)
    return RecoveryResult("search_exhausted", None, None, None, normalized,
                          meter.used, len(table), resources)
