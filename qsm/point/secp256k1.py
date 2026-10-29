"""secp256k1 — the control curve, in the same shape as `qsm.point.curve`.

WHY THIS FILE EXISTS. Every published ECDLP circuit we measure against — Babbush et al.,
Schrottenloher 2026, ecdsa.fail — is over secp256k1. Comparing our Wei25519 numbers to
theirs therefore confounds two different things: whether our *circuit* is better, and
whether our *prime* is better. Those want separating, and the separation is worth stating
in the paper rather than hand-waving, because the second effect is known to be large:
Schrottenloher's own special-prime and any-prime rows differ by 2^21.78 / 2^21.19 = 1.51x.

Both curves here are short Weierstrass and the circuit uses the same incomplete addition
formula on both, so there is no *curve-model* difference to confound the experiment with
either. Running the identical construction over `QSM_FIELD=secp256k1` changes exactly one
thing — `(U, F)`, and the carry window that F implies at a fixed accuracy — so whatever
Toffoli difference appears is attributable to the field and to nothing else.

    Wei25519   p = 2^255 - 19            F = 19          (5 bits)   -> L = 45
    secp256k1  p = 2^256 - 2^32 - 977    F = 2^32 + 977  (33 bits)  -> L = 73

Parameters are from SEC 2 v2 section 2.4.1 (a = 0, b = 7, h = 1). They are transcribed
rather than derived — unlike Wei25519's, they are not the image of anything — so
`test_point_secp256k1.py` checks G against the curve equation and checks that [n]G is the
identity, which between them pin every constant below.
"""

from __future__ import annotations

from .group import scalar_mul

#: p = 2^256 - 2^32 - 977. Written in the pseudo-Mersenne form the circuits actually use.
P = (1 << 256) - (1 << 32) - 977

#: y^2 = x^3 + 7 over F_p. The curve has j-invariant 0 and no CM structure we use.
A = 0
B = 7

#: The SEC 2 base point.
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

#: Prime group order, and the cofactor (1 — unlike 25519's 8).
ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
COFACTOR = 1


def is_on_curve(x: int, y: int, p: int = P, a: int = A, b: int = B) -> bool:
    return (y * y - (x * x * x + a * x + b)) % p == 0


def add(x1: int, y1: int, x2: int, y2: int, p: int = P) -> tuple[int, int]:
    """Incomplete affine addition, `x1 != x2` REQUIRED. Same function the circuit computes.

    Deliberately identical to `curve.add` — no doubling branch, no identity branch. The
    whole point of this module is that only the field changes.
    """
    assert x1 % p != x2 % p, "incomplete addition: x1 == x2 is doubling or the identity"
    lam = ((y1 - y2) * pow(x1 - x2, -1, p)) % p
    x3 = (lam * lam - x2 - x1) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3


def neg(x: int, y: int, p: int = P) -> tuple[int, int]:
    return x % p, (-y) % p


def mul(k: int, x: int, y: int, p: int = P) -> tuple[int, int] | None:
    """`[k](x, y)` by double-and-add, returning None for the identity. Tests only."""
    return scalar_mul(k, x, y, p=p, a=A)
