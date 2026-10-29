"""Canonical affine addition with deterministic arithmetic on its incomplete domain.

The two slope operations require nonzero x differences. Unlike the experimental lazy
variant, a horizontal chord (zero slope) is supported. Infinity and exceptional point
addition are not encoded. Every field register is canonical throughout the schedule;
there is no conversion that discards the distinction between zero and a lazy modulus.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.canonical import (
    add_inplace,
    double_inplace,
    halve_inplace,
    neg_inplace,
    sub_inplace,
)
from ..inversion.inplace import moddiv_conservative_inplace, modmul_conservative_inplace
from ..params import PRIME, U
from . import active as curve


def _constant_inplace(circ: Circuit, b: Register, value: int, p: int,
                      *, subtract: bool = False, ctrl: int | None = None) -> Register:
    r = circ.alloc(len(b))
    bits = [j for j in range(len(b)) if (value % p >> j) & 1]
    for j in bits:
        circ.x(r[j])
    operation = sub_inplace if subtract else add_inplace
    b = operation(circ, r, b, p, ctrl=ctrl)
    for j in bits:
        circ.x(r[j])
    circ.free(r)
    return b


def square_sub_conservative_inplace(
    circ: Circuit, source: Register, acc: Register, p: int = PRIME,
    ctrl: int | None = None,
) -> Register:
    """Preserve source and subtract [ctrl] source² from a canonical accumulator."""
    n = len(source)
    assert len(acc) == n
    t = circ.alloc(n)
    for a, b in zip(source, t):
        circ.cx(a, b)
    gate = circ.alloc(1) if ctrl is not None else None
    for j in range(n):
        bit = source[j]
        if gate is not None:
            circ.ccx(ctrl, bit, gate[0])
            bit = gate[0]
        acc = sub_inplace(circ, t, acc, p, ctrl=bit)
        if gate is not None:
            circ.ccx(ctrl, source[j], gate[0])
        if j + 1 < n:
            t = double_inplace(circ, t, p)
    for _ in range(n - 1):
        t = halve_inplace(circ, t, p)
    if gate is not None:
        circ.free(gate)
    for a, b in zip(source, t):
        circ.cx(a, b)
    circ.free(t)
    return acc


def add_const_affine_conservative_inplace(
    circ: Circuit, x: Register, y: Register, xp: int, yp: int,
    ctrl: int | None = None, p: int = PRIME, config=None,
) -> tuple[Register, Register]:
    """Add a classical point, returning canonical, possibly relabelled coordinates.

    Inputs must be canonical with x != xp. In the enabled branch the result must also
    have x != xp. These are domain assumptions, not data-dependent runtime checks.
    """
    assert len(x) == len(y)
    xp, yp = xp % p, yp % p
    x = _constant_inplace(circ, x, xp, p, subtract=True)
    y = _constant_inplace(circ, y, yp, p, subtract=True)
    x, y = moddiv_conservative_inplace(circ, x, y, p, config=config)
    x = square_sub_conservative_inplace(circ, y, x, p, ctrl)
    x = neg_inplace(circ, x, p, ctrl=ctrl)
    x = _constant_inplace(circ, x, 3 * xp, p, subtract=True, ctrl=ctrl)
    x, y = modmul_conservative_inplace(circ, x, y, p, config=config)
    x = _constant_inplace(circ, x, xp, p)
    y = _constant_inplace(circ, y, yp, p)
    y = neg_inplace(circ, y, p, ctrl=ctrl)
    return x, y


def add_const_affine_conservative_inplace_ref(
    x: int, y: int, xp: int, yp: int, ctrl: int = 1, p: int = PRIME,
) -> tuple[int, int]:
    """Canonical rational-map reference; independent point tests use the group law."""
    assert 0 <= x < p and 0 <= y < p
    assert ctrl in (0, 1)
    xp, yp = xp % p, yp % p
    u = (x - xp) % p
    assert u, "exceptional: x == x_P"
    if not ctrl:
        return x, y
    slope = (y - yp) * pow(u, -1, p) % p
    xr = (slope * slope - x - xp) % p
    assert xr != xp, "exceptional: x_3 == x_P"
    return xr, (slope * (xp - xr) - yp) % p


def _case(controlled: bool) -> Case:
    circ = Circuit()
    c = circ.input("c", 1) if controlled else None
    x, y = circ.input("x", U), circ.input("y", U)
    x, y = add_const_affine_conservative_inplace(
        circ, x, y, curve.GX, curve.GY, None if c is None else c[0], PRIME)
    circ.rebind("x", x)
    circ.rebind("y", y)

    def ref(v):
        xr, yr = add_const_affine_conservative_inplace_ref(
            v["x"], v["y"], curve.GX, curve.GY, v.get("c", 1), PRIME)
        out = {"x": xr, "y": yr}
        if controlled:
            out["c"] = v["c"]
        return out

    def gen(rng: random.Random):
        while True:
            point = curve.mul(rng.randrange(1, curve.ORDER), curve.GX, curve.GY)
            if point is None:
                continue
            v = {"x": point[0], "y": point[1]}
            if controlled:
                v["c"] = rng.randrange(2)
            try:
                ref(v)
            except AssertionError:
                continue
            return v

    return Case(circ, ref, gen, meta={
        "fail_analytic": 0.0,
        "input_distribution": "nonexceptional prime-subgroup points",
        "arithmetic_contract": "canonical incomplete affine; x != xp and enabled output x != xp",
        "automatic_resource_winner": False,
    })


@circuit_version("point.add_const_affine_conservative_inplace.v1")
def _build_conservative() -> Case:
    return _case(False)


@circuit_version("point.add_const_affine_conservative_ctrl_inplace.v1")
def _build_conservative_ctrl() -> Case:
    return _case(True)
