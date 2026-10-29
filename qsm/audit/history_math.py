"""Exact branch-prefix populations for the classical jump-2 trajectory.

No width schedule or approximate comparison changes this trace. A prefix is the
arithmetic progression of *initial* canonical inputs sharing all decisions so far;
it is captured after this iteration's parity decisions, before its ordering bit.
The sampled input must be coprime to p. Populations range over all 1 <= x < p
(which are all coprime when p is prime), not a separately filtered totient domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterator


def affine_floor_sum(count: int, modulus: int, slope: int, intercept: int) -> int:
    """Sum floor((intercept + slope*t)/modulus), 0 <= t < count, exactly.

    Signed coefficients are reduced first. The remaining lattice-point count is
    transposed repeatedly, giving a Euclidean algorithm rather than enumeration.
    """
    if count < 0 or modulus <= 0:
        raise ValueError("count must be nonnegative and modulus positive")
    total = 0
    while True:
        q, slope = divmod(slope, modulus)
        total += q * count * (count - 1) // 2
        q, intercept = divmod(intercept, modulus)
        total += q * count
        top = slope * count + intercept
        if top < modulus:
            return total
        count, intercept = divmod(top, modulus)
        modulus, slope = slope, modulus


def _interval(count: int, c: int, d: int, low: int, high: int) -> tuple[int, int]:
    """Half-open t interval satisfying low <= c+d*t <= high."""
    if not d:
        return (0, count) if low <= c <= high else (0, 0)
    if d > 0:
        start = max(0, (low - c + d - 1) // d)
        stop = min(count, (high - c) // d + 1)
    else:
        start = max(0, (c - high - d - 1) // -d)
        stop = min(count, (c - low) // -d + 1)
    return (start, stop) if start < stop else (0, 0)


def _bucket(k: int, w: int) -> int:
    if k < 0 or w < 1:
        raise ValueError("k must be nonnegative and w positive")
    return 1 << max(0, w - k)


@dataclass(frozen=True, slots=True)
class Prefix:
    """x=x0+step*t, u=u0+du*t, v=v0+dv*t for 0 <= t < count.

    The inverse first row satisfies p=a*u+b*v; S counts all halvings.
    Singleton slopes are canonicalized to zero. Empty restrictions are allowed
    internally, but records always carry a nonempty prefix.
    """

    x0: int
    step: int
    count: int
    u0: int
    du: int
    v0: int
    dv: int
    a: int = 1
    b: int = 0
    S: int = 0

    @property
    def m(self) -> int:
        return self.a + self.b

    @property
    def p(self) -> int:
        return self.a * self.u0 + self.b * self.v0

    def _rebase(self, start: int, stride: int, count: int) -> Prefix:
        return Prefix(
            self.x0 + self.step * start, self.step * stride, count,
            self.u0 + self.du * start, self.du * stride if count > 1 else 0,
            self.v0 + self.dv * start, self.dv * stride if count > 1 else 0,
            self.a, self.b, self.S,
        )

    def _cut(self, c: int, d: int, low: int, high: int) -> Prefix:
        start, stop = _interval(self.count, c, d, low, high)
        if start == 0 and stop == self.count:
            return self
        return self._rebase(start, 1, stop - start)

    def _parity(self, parity: int) -> Prefix:
        if not self.dv & 1:
            if self.v0 & 1 != parity:
                raise ValueError("parity branch has no members")
            return self
        start = (parity - self.v0) & 1
        return self._rebase(start, 2, (self.count + 1 - start) // 2)

    def _halve(self) -> Prefix:
        return Prefix(
            self.x0, self.step, self.count, self.u0, self.du,
            self.v0 // 2, self.dv // 2, self.a, 2 * self.b, self.S + 1,
        )

    def _ordered_subtract(self, swap: bool) -> Prefix:
        d0, dd = self.u0 - self.v0, self.du - self.dv
        q = self._cut(d0, dd, 1 if swap else -self.p, self.p if swap else 0)
        if swap:
            return Prefix(
                q.x0, q.step, q.count, q.v0, q.dv,
                q.u0 - q.v0, q.du - q.dv, q.a + q.b, q.a, q.S,
            )
        return Prefix(
            q.x0, q.step, q.count, q.u0, q.du,
            q.v0 - q.u0, q.dv - q.du, q.a + q.b, q.b, q.S,
        )

    def _require_odd(self) -> None:
        if (
            self.count < 1 or not self.u0 & 1 or not self.v0 & 1
            or (self.count > 1 and (self.du & 1 or self.dv & 1))
            or min(self.u0, self.u0 + self.du * (self.count - 1),
                   self.v0, self.v0 + self.dv * (self.count - 1)) < 1
        ):
            raise ValueError("requires a positive odd precomparison prefix")

    def bad_count(self, k: int, w: int, *, fit_only: bool = False) -> int:
        """Exact count of u>v with equal high k-bit prefixes at width w.

        By default *every* member must fit w bits or ValueError is raised.
        fit_only=True instead counts only members with both values in [0,2**w).
        Neither choice conditions on earlier width clips. All members must have
        positive odd operands, as at an enabled steady precomparison.
        """
        B = _bucket(k, w)
        self._require_odd()
        limit = (1 << w) - 1
        q = self
        if fit_only:
            q = q._cut(q.u0, q.du, 0, limit)
            q = q._cut(q.v0, q.dv, 0, limit)
        elif max(self.u0, self.u0 + self.du * (self.count - 1),
                 self.v0, self.v0 + self.dv * (self.count - 1)) > limit:
            raise ValueError("prefix operands do not all fit w bits; use fit_only=True")
        # Odd operands have even difference, so dropping at most one bit is exact.
        if B <= 2 or not q.count:
            return 0
        q = q._cut(q.u0 - q.v0, q.du - q.dv, 1, B - 1)
        return (q.count - affine_floor_sum(q.count, B, q.du, q.u0)
                + affine_floor_sum(q.count, B, q.dv, q.v0))

    @property
    def residue_first_positive(self) -> int:
        """First positive even d in its residue class modulo 2m (1 <= r <= 2m).

        p=m*v+a*d with v odd implies a*d == p-m (mod 2m).
        The inverse row of a live odd prefix has gcd(a,2m)=1.
        """
        self._require_odd()
        modulus = 2 * self.m
        return ((self.p - self.m) * pow(self.a, -1, modulus)) % modulus or modulus

    def residue_candidates(self, k: int, w: int) -> int:
        """Number of positive even residue candidates d < B; not a probability."""
        B = _bucket(k, w)
        r = self.residue_first_positive
        return max(0, (B - 1 - r) // (2 * self.m) + 1)

    def residue_bound(self, k: int, w: int) -> int:
        """Rigorous bad-count upper bound, capped by the prefix population.

        Each d fixes v=(p-a*d)/m and u=v+d, hence at most one initial x.
        This intentionally does not tighten for width fit or history inequalities.
        """
        return min(self.count, self.residue_candidates(k, w))

    def continuous_model(self, k: int, w: int) -> float:
        """Continuous candidate-count MODEL B/(2m), emphatically not a bound."""
        self._require_odd()
        return _bucket(k, w) / (2 * self.m)


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    """One live iteration, measured after shifts and before the order decision.

    active excludes boundary i=0 and even-v iterations. tape contains the actual
    (shifts, oddness, order) for cross-checks, but its *current* order bit is NOT
    part of prefix or history_key. entry_bits measures operands before shifts.
    """

    i: int
    active: bool
    prefix: Prefix
    u: int
    v: int
    rank: int
    history_key: int
    entry_bits: int
    tape: tuple[int, int, int]

    @property
    def S(self) -> int:
        return self.prefix.S

    @property
    def a(self) -> int:
        return self.prefix.a

    @property
    def b(self) -> int:
        return self.prefix.b

    @property
    def m(self) -> int:
        return self.prefix.m


def trace_history(p: int, x: int) -> Iterator[HistoryRecord]:
    """Yield all live jump-2 iterations, including boundary, stopping at v=0.

    i is zero-based, and the number of records is the total live iteration count.
    No terminal padding is emitted. Keys are collision-free integers encoding a
    leading sentinel, all previous tape entries, and current shifts/oddness.
    Keys identify histories for a fixed p, not just coincident arithmetic states.
    """
    if p <= 2 or not p & 1:
        raise ValueError("p must be odd and greater than two")
    if not 1 <= x < p or gcd(p, x) != 1:
        raise ValueError("x must be canonical, nonzero, and coprime to p")
    prefix = Prefix(1, 1, p - 1, p, 0, 1, 1)
    u, v, i, past = p, x, 0, 1
    while v:
        entry_bits = max(u, v).bit_length()
        # Other inputs can have terminated on the previous equality branch.
        prefix = prefix._cut(prefix.v0, prefix.dv, 1, p)
        shifts = 0
        for j in range(2):
            if i > 0 and j == 0:
                prefix = prefix._halve()
                v //= 2
                shifts += 1
            else:
                parity = v & 1
                prefix = prefix._parity(parity)
                if parity:
                    break
                prefix = prefix._halve()
                v //= 2
                shifts += 1
        odd = v & 1
        prefix = prefix._parity(odd)
        swap = int(bool(odd and u > v))
        rank = (x - prefix.x0) // prefix.step
        yield HistoryRecord(
            i, bool(i and odd), prefix, u, v, rank,
            (past << 4) | (shifts << 1) | odd,
            entry_bits, (shifts, odd, swap),
        )
        past = (past << 4) | (shifts << 2) | (odd << 1) | swap
        if odd:
            prefix = prefix._ordered_subtract(bool(swap))
            if swap:
                u, v = v, u
            v -= u
        i += 1
