"""Wei25519 — the short Weierstrass model of Curve25519, and its group law in Python.

No circuits here. This is the classical side: the curve constants, the birational map that
licenses using them, and a plain affine addition for the circuit references to be checked
against.

WHY A WEIERSTRASS CURVE IN AN ED25519 PROJECT. Curve25519, Ed25519 and Wei25519 are three
models of ONE group, related by birational maps defined over F_q. Shor's algorithm measures
a *scalar*, and a scalar is invariant under a group isomorphism: if `A = [k]P` on Ed25519
and `phi` is the map, then `phi(A) = [k] phi(P)` on Wei25519 with the same `k`. So a
Wei25519 circuit solves the Ed25519 ECDLP, unchanged. The paper states the same transport
for the Curve25519 -> Ed25519 direction in its section 2.1; the argument does not care
which of the three models sits at either end.

The map from Montgomery `B v^2 = u^3 + A u^2 + u` (A = 486662, B = 1) to
`y^2 = x^3 + a x + b` is

    x = u/B + A/(3B)        y = v/B

so on Curve25519 the x-coordinate is a SHIFT of the Montgomery u by the constant
`A/3 mod q`, and nothing else. `u = 9` maps to `G` below.

The exceptional points (`u = 0` and the two-torsion) are the same set in both models, and
the incomplete addition law here fails on exactly the same inputs it always did -- see
`qsm.point.affine` for how the point layer discharges that.

Values are DERIVED below, not transcribed, and `test_point_affine.py` checks them against
the constants in draft-ietf-lwig-curve-representations. The derived `Gy` is the even
representative; the draft publishes `q - Gy`. Both generate the same group.
"""

from __future__ import annotations

from ..params import PRIME
from .group import scalar_mul

#: Montgomery Curve25519: B v^2 = u^3 + A u^2 + u.
MONT_A = 486662
MONT_B = 1

_INV3 = pow(3, -1, PRIME)

#: `x = u + SHIFT` carries a Montgomery u-coordinate to its Wei25519 x-coordinate.
SHIFT = (MONT_A * _INV3) % PRIME

#: Wei25519: y^2 = x^3 + A x + B over F_q, q = 2^255 - 19.
A = (_INV3 * (3 - MONT_A * MONT_A) * pow(MONT_B * MONT_B, -1, PRIME)) % PRIME
B = ((2 * MONT_A**3 - 9 * MONT_A) * pow(27 * MONT_B**3, -1, PRIME)) % PRIME

#: The base point, the image of Montgomery u = 9.
GX = (9 + SHIFT) % PRIME
GY = 43114425171068552920764898935933967039370386198203806730763910166200978582548

#: Group order of the prime-order subgroup, and the cofactor.
ORDER = (1 << 252) + 27742317777372353535851937790883648493
COFACTOR = 8


def is_on_curve(x: int, y: int, p: int = PRIME, a: int = A, b: int = B) -> bool:
    return (y * y - (x * x * x + a * x + b)) % p == 0


def add(x1: int, y1: int, x2: int, y2: int,
        p: int = PRIME) -> tuple[int, int]:
    """Incomplete affine addition, `x1 != x2` REQUIRED. The function the circuit computes.

    No doubling branch and no identity branch, deliberately: the circuit has neither
    either, and a reference that quietly handled the exceptional cases would hide the one
    precondition the point layer actually has to discharge.
    """
    assert x1 % p != x2 % p, "incomplete addition: x1 == x2 is doubling or the identity"
    lam = ((y1 - y2) * pow(x1 - x2, -1, p)) % p
    x3 = (lam * lam - x2 - x1) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3


def neg(x: int, y: int, p: int = PRIME) -> tuple[int, int]:
    return x % p, (-y) % p


def mul(k: int, x: int, y: int, p: int = PRIME) -> tuple[int, int] | None:
    """`[k](x, y)` by double-and-add, returning None for the identity.

    Only for tests -- it branches on the exceptional cases the circuit refuses, using the
    full doubling formula, so it is a group-law oracle rather than a model of any circuit.
    """
    return scalar_mul(k, x, y, p=p, a=A)
