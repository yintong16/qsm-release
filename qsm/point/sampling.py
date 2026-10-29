"""Independent classical draws from a nonidentity prime-order subgroup.

Assumptions, NOT certifications made by this module: p is prime, the nonsingular
curve has order h*r, r is an odd prime, and gcd(h, r) = 1. Published parameters
supply these facts for Wei25519 and secp256k1. The constructor checks necessary
arithmetic conditions (including Hasse's interval), not primality or #E(F_p).

Each (x, sign) proposal has probability 1/(2*p). A nonzero square RHS gives its
two affine points, once each. A zero RHS is accepted for just one sign, so its
single point has the SAME proposal mass. Thus accepted affine points are
uniform. Multiplication by h has kernel size h and image the order-r subgroup;
each nonzero image has h finite preimages. Removing the identity therefore
leaves the uniform law on the r-1 nonidentity points. No correlated group walk
or discrete-log sampling is used. IID claims require independent uniform draws
from the caller's RNG; seeded random.Random is for reproducible experiments.
"""
from __future__ import annotations

from math import gcd, isqrt
from typing import Protocol

from .group import scalar_mul


class RandomSource(Protocol):
    def randrange(self, stop: int, /) -> int: ...


def _integer(value: int, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, not a boolean")
    return value


class _SquareRoot:
    """Cache the exponent and sqrt(-1); no primality assertion is inferred."""

    def __init__(self, p: int):
        _integer(p, "p")
        if p <= 3 or p % 2 == 0:
            raise ValueError("p must be an assumed odd prime greater than three")
        self.p = p
        if p % 4 == 3:
            self.exponent = (p + 1) // 4
            self.sqrt_minus_one = None
        elif p % 8 == 5:
            self.exponent = (p + 3) // 8
            self.sqrt_minus_one = pow(2, (p - 1) // 4, p)
            if self.sqrt_minus_one**2 % p != p - 1:
                raise ValueError("p does not satisfy the required sqrt(-1) identity")
        else:
            raise ValueError("supported fields require p % 4 == 3 or p % 8 == 5")

    def __call__(self, value: int) -> int | None:
        value %= self.p
        root = pow(value, self.exponent, self.p)
        if root * root % self.p != value and self.sqrt_minus_one is not None:
            root = root * self.sqrt_minus_one % self.p
        # This check is mandatory even in the p % 4 == 3 branch.
        return root if root * root % self.p == value else None


def sqrt_mod(value: int, p: int) -> int | None:
    """Return a canonical square root, or None for a nonsquare in a supported field.

    Requires an assumed prime p > 3 of residue 3 mod 4 or 5 mod 8. Unsupported
    fields raise ValueError; this helper does not certify primality, nor promise
    a complete square-root algorithm for composite moduli. Negative values are
    reduced modulo p. The root returned is always checked by squaring it.
    """
    _integer(value, "value")
    return _SquareRoot(p)(value)


class SubgroupSampler:
    """Reusable sampler with cached field constants and cumulative diagnostics.

    Parameters are explicit: there is no default active field or generator.
    Each call starts with fresh independent x/sign proposals. Cofactor one
    bypasses scalar_mul (and its unnecessary final doubling). Membership is not
    rechecked with a full order multiplication on every draw: it follows only
    under the group-order assumptions in the module docstring.
    """

    def __init__(self, p: int, a: int, b: int, order: int, cofactor: int):
        self._root = _SquareRoot(p)
        for name, value in (("a", a), ("b", b), ("order", order), ("cofactor", cofactor)):
            _integer(value, name)
        if order <= 2 or order % 2 == 0:
            raise ValueError("order must be an assumed odd prime greater than two")
        if cofactor < 1 or gcd(cofactor, order) != 1:
            raise ValueError("cofactor must be positive and coprime to order")
        if abs(cofactor * order - (p + 1)) > isqrt(4 * p):
            raise ValueError("cofactor * order is outside the Hasse interval")
        self.p, self.a, self.b = p, a % p, b % p
        self.order, self.cofactor = order, cofactor
        if (4 * self.a**3 + 27 * self.b**2) % p == 0:
            raise ValueError("curve must be nonsingular")
        self.attempted_x = 0
        self.nonsquare_reject = 0
        self.zero_root_reject = 0
        self.identity_reject = 0
        self.accepted = 0

    @property
    def proposal_reject(self) -> int:
        return self.nonsquare_reject + self.zero_root_reject

    @property
    def stats(self) -> dict[str, int]:
        """A fresh JSON-safe snapshot; mutating it does not mutate the sampler."""
        return {"attempted_x": self.attempted_x,
                "proposal_reject": self.proposal_reject,
                "nonsquare_reject": self.nonsquare_reject,
                "zero_root_reject": self.zero_root_reject,
                "identity_reject": self.identity_reject,
                "accepted": self.accepted}

    def sample(self, rng: RandomSource) -> tuple[int, int]:
        """Draw one canonical, finite, nonidentity subgroup point."""
        p = self.p
        while True:
            self.attempted_x += 1
            x = rng.randrange(p)
            y = self._root((x * x % p * x + self.a * x + self.b) % p)
            if y is None:
                self.nonsquare_reject += 1
                continue
            sign = rng.randrange(2)
            if y == 0 and sign:
                self.zero_root_reject += 1
                continue
            if sign:
                y = (-y) % p
            point = ((x, y) if self.cofactor == 1 else
                     scalar_mul(self.cofactor, x, y, p=p, a=self.a))
            if point is None:
                self.identity_reject += 1
                continue
            self.accepted += 1
            return point


def sample_subgroup_point(rng: RandomSource, *, p: int, a: int, b: int,
                          order: int, cofactor: int) -> tuple[int, int]:
    """One-shot wrapper; reuse SubgroupSampler for large calibration batches."""
    return SubgroupSampler(p, a, b, order, cofactor).sample(rng)
