"""Small classical group-law helper; not a quantum circuit or an order assumption."""

from __future__ import annotations


Point = tuple[int, int] | None


def scalar_mul(k: int, x: int, y: int, *, p: int, a: int) -> Point:
    """Multiply a point on a short Weierstrass curve over a prime field p > 3.

    The caller supplies a valid point and the curve coefficient a. The constant b
    is not needed by the group law. No subgroup order is assumed: in particular,
    order checks and small-torsion inputs must perform the actual arithmetic.
    None represents infinity, including intermediate doublings; negative scalars
    multiply the inverse point.
    """
    def full_add(pt: Point, qt: Point) -> Point:
        if pt is None:
            return qt
        if qt is None:
            return pt
        px, py = pt
        qx, qy = qt
        if px == qx:
            if (py + qy) % p == 0:
                return None
            lam = (3 * px * px + a) * pow(2 * py, -1, p) % p
        else:
            lam = (qy - py) * pow(qx - px, -1, p) % p
        rx = (lam * lam - px - qx) % p
        return rx, (lam * (px - rx) - py) % p

    base: Point = (x % p, (-y if k < 0 else y) % p)
    acc: Point = None
    k = abs(k)
    while k:
        if k & 1:
            acc = full_add(acc, base)
        base = full_add(base, base)
        k >>= 1
    return acc
