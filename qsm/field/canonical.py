"""Matched exact arithmetic on canonical field elements, including zero.

All operands are in [0,p), p is odd and fits the register width. These are
permutations on that domain, not many-to-one canonicalizers of lazy inputs.
The exact adders/comparators may erase temporary ANDs by measurement with their
exact phase correction; no measurement-bearing IR is reversed here.
"""
from __future__ import annotations

from ..core.circuit import Circuit, Register
from .compare import cmp_ge_const, cmp_lt
from .constadd import addsub_p
from .modarith import modadd_cmp
from .shift import controlled_lshift1, controlled_rshift1
from .sub import sub
from .swap import cswap

__all__ = ["add_inplace", "sub_inplace", "double_inplace", "halve_inplace", "neg_inplace"]


def _check(b: Register, p: int) -> int:
    n = len(b)
    if not 1 < p < (1 << n) or not p & 1:
        raise ValueError("canonical arithmetic requires an odd modulus fitting the register")
    return n


def _operand(c: Circuit, a: Register, ctrl: int | None) -> Register:
    if ctrl is None:
        return a
    scratch = c.alloc(len(a))
    for aq, sq in zip(a, scratch):
        c.ccx(ctrl, aq, sq)
    return scratch


def _clear_operand(c: Circuit, a: Register, scratch: Register, ctrl: int | None) -> None:
    if ctrl is not None:
        for aq, sq in zip(a, scratch):
            c.ccx(ctrl, aq, sq)
        c.free(scratch)


def add_inplace(c: Circuit, a: Register, b: Register, p: int,
                ctrl: int | None = None) -> Register:
    """b <- b+a mod p (if ctrl), preserving a and the inactive branch."""
    _check(b, p)
    if len(a) != len(b) or set(a.ids) & set(b.ids):
        raise ValueError("add requires disjoint equal-width operands")
    operand = _operand(c, a, ctrl)
    modadd_cmp(c, operand, b, p, vent=False)
    _clear_operand(c, a, operand, ctrl)
    return b


def sub_inplace(c: Circuit, a: Register, b: Register, p: int,
                ctrl: int | None = None) -> Register:
    """b <- b-a mod p (if ctrl), the explicit arithmetic inverse of add.

    Recreate reduction = [b<a]; add p on that flag in n+1 bits; erase the
    flag with the FULL [B>=p] comparison; subtract a. This is not the lazy
    overflow surrogate: the n+1-bit intermediate represents b+p exactly.
    """
    n = _check(b, p)
    if len(a) != n or set(a.ids) & set(b.ids):
        raise ValueError("subtract requires disjoint equal-width operands")
    operand = _operand(c, a, ctrl)
    hi, flag = c.alloc(1), c.alloc(1)
    wide = Register(list(b.ids) + [hi[0]])
    cmp_lt(c, operand, wide, flag[0])
    addsub_p(c, wide, n, (1 << n) - p, adding=True, ctrl=flag[0])
    cmp_ge_const(c, wide, p, flag[0])
    sub(c, operand, wide)
    c.free(flag)
    c.free(hi)
    _clear_operand(c, a, operand, ctrl)
    return b


def double_inplace(c: Circuit, b: Register, p: int,
                   ctrl: int | None = None) -> Register:
    """b <- 2b mod p. Return the result handle (uncontrolled shift relabels).

    Shift into n+1 bits; compare against p, subtract p, erase the reduction
    flag with output parity. The high bit is zero because the result is <p.
    """
    n = _check(b, p)
    fresh, flag = c.alloc(1), c.alloc(1)
    if ctrl is None:
        wide = Register([fresh[0]] + list(b.ids))
    else:
        wide = Register(list(b.ids) + [fresh[0]])
        controlled_lshift1(c, ctrl, wide)
    cmp_ge_const(c, wide, p, flag[0])
    addsub_p(c, wide, n, (1 << n) - p, adding=False, ctrl=flag[0])
    if ctrl is None:
        c.cx(wide[0], flag[0])
    else:
        c.ccx(ctrl, wide[0], flag[0])
    c.free(flag)
    c.free(Register([wide[n]]))
    return Register(wide.ids[:n], b.name)


def halve_inplace(c: Circuit, b: Register, p: int,
                  ctrl: int | None = None) -> Register:
    """b <- b/2 mod p, exact even for odd b below the prime deficit.

    Recreate parity, add p in n+1 bits if odd, erase the flag by [B>=p],
    then shift the now-even integer right. Explicit inverse of double.
    """
    n = _check(b, p)
    hi, flag = c.alloc(1), c.alloc(1)
    wide = Register(list(b.ids) + [hi[0]])
    if ctrl is None:
        c.cx(b[0], flag[0])
    else:
        c.ccx(ctrl, b[0], flag[0])
    addsub_p(c, wide, n, (1 << n) - p, adding=True, ctrl=flag[0])
    cmp_ge_const(c, wide, p, flag[0])
    c.free(flag)
    if ctrl is None:
        c.free(Register([wide[0]]))
        return Register(wide.ids[1:], b.name)
    controlled_rshift1(c, ctrl, wide)
    c.free(hi)
    return b


def neg_inplace(c: Circuit, b: Register, p: int,
                ctrl: int | None = None) -> Register:
    """b <- -b mod p (if ctrl), including 0 -> 0, with clean scratch."""
    _check(b, p)
    scratch = c.alloc(len(b))
    sub_inplace(c, b, scratch, p, ctrl=ctrl)
    if ctrl is None:
        for bq, sq in zip(b, scratch):
            c.cx(bq, sq)
            c.cx(sq, bq)
            c.cx(bq, sq)
    else:
        cswap(c, ctrl, b, scratch)
    add_inplace(c, b, scratch, p, ctrl=ctrl)
    c.free(scratch)
    return b
