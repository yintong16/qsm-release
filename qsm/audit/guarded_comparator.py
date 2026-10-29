"""Isolated high-prefix comparator experiment; no production schedule changes.

The circuit computes exactly the top min(width, k+b)-bit ordering, not an
exact full-word ordering unless that prefix covers the word. The high b-bit
ordering and the next k-bit ordering are mutually exclusive contributions.
No data-dependent gate skipping or resource-optimality claim is implied.
"""
from __future__ import annotations

import random

from ..bench.registry import Case
from ..core.circuit import Circuit, Register
from ..field.add import _unand
from ..field.compare import cmp_lt
from ..scalarmul.lookup import xor_nonzero


def _validate_window(k: int, b: int) -> None:
    for name, value in (("k", k), ("b", b)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if b > k:
        raise ValueError("b must not exceed k for one-step sliding equivalence")


def _xor_equal(circ: Circuit, a: Register, B: Register, target: int) -> None:
    """XOR exact equality, borrowing B for differences and restoring it.

    xor_nonzero uses 2*b-3 Toffolis and b-2 transient ancillas for b>=2
    (zero Toffolis/scratch for b=1). Its scratch is gone on return.
    """
    for aq, bq in zip(a, B):
        circ.cx(aq, bq)
    circ.x(target)
    xor_nonzero(circ, B, target)
    for aq, bq in zip(a, B):
        circ.cx(aq, bq)


def cmp_lt_guarded(
    circ: Circuit,
    a: Register,
    B: Register,
    flag: int,
    *,
    k: int = 100,
    b: int = 8,
    ctrl: int | None = None,
) -> None:
    """flag ^= ctrl & (B_prefix < a_prefix), with clean temporary scratch.

    Both operands have width w. The prefix has min(w,k+b) bits; omitted ctrl
    means 1. k,b must be positive integers with b<=k. For w<=k, emit the
    existing comparator directly. Otherwise use effective b=min(b,w-k).

    Compare the high b bits first, then compare the NEXT k bits controlled
    by exact high-bit equality (and ctrl). Equality's ladder is uncomputed
    before the k-bit carry ladder starts; only its one-bit result and an
    optional conjunction survive across that comparison. All are erased
    before returning, so the primitive adds no persistent tape state.

    This also equals the one-step sliding rule: use the ordinary top-k
    comparison if the high b bits differ, otherwise drop b high bits and
    compare the next k. The restriction b<=k is needed for that rule, not
    for lexicographic comparison itself. Ties in the retained prefix are
    false, regardless of discarded low bits or operand parity.
    """
    _validate_window(k, b)
    if len(a) != len(B):
        raise ValueError("guarded comparison requires equal-width operands")
    w = len(B)
    effective_b = min(b, max(0, w - k))
    if effective_b == 0:
        cmp_lt(circ, a, B, flag, k=k, ctrl=ctrl)
        return

    split = w - effective_b
    high_a, high_B = a[split:w], B[split:w]
    cmp_lt(circ, high_a, high_B, flag, ctrl=ctrl)

    equal = circ.alloc(1)
    _xor_equal(circ, high_a, high_B, equal[0])
    lower_ctrl = equal[0]
    if ctrl is not None:
        conjunction = circ.alloc(1)
        lower_ctrl = conjunction[0]
        circ.ccx(ctrl, equal[0], lower_ctrl)

    cmp_lt(circ, a[split-k:split], B[split-k:split], flag, ctrl=lower_ctrl)

    if ctrl is not None:
        _unand(circ, ctrl, equal[0], lower_ctrl)
    _xor_equal(circ, high_a, high_B, equal[0])
    circ.free(equal)


def build_compare_case(
    width: int, k: int, b: int, strategy: str, controlled: bool = True,
) -> Case:
    """Build an unregistered baseline/wider/guarded benchmark Case.

    The reference is the strategy's *prefix* function, deliberately not full
    comparison. Compare separately against full ordering to study errors.
    Inputs are named a,b,flag and (only if controlled) ctrl.
    """
    _validate_window(k, b)
    if not isinstance(width, int) or isinstance(width, bool):
        raise TypeError("width must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")
    if strategy not in ("baseline", "wider", "guarded"):
        raise ValueError("strategy must be baseline, wider, or guarded")
    if not isinstance(controlled, bool):
        raise TypeError("controlled must be a bool")

    circ = Circuit()
    a = circ.input("a", width)
    operand_b = circ.input("b", width)
    flag = circ.input("flag", 1)
    ctrl = circ.input("ctrl", 1)[0] if controlled else None
    prefix_bits = min(width, k if strategy == "baseline" else k + b)
    if strategy == "guarded":
        cmp_lt_guarded(circ, a, operand_b, flag[0], k=k, b=b, ctrl=ctrl)
    else:
        cmp_lt(circ, a, operand_b, flag[0], k=prefix_bits, ctrl=ctrl)

    def reference(values: dict[str, int]) -> dict[str, int]:
        shift = width - prefix_bits
        predicate = int((values["b"] >> shift) < (values["a"] >> shift))
        return {**values, "flag": values["flag"] ^ (values.get("ctrl", 1) & predicate)}

    def gen_input(rng: random.Random) -> dict[str, int]:
        values = {"a": rng.getrandbits(width), "b": rng.getrandbits(width),
                  "flag": rng.getrandbits(1)}
        if controlled:
            values["ctrl"] = rng.getrandbits(1)
        return values

    return Case(circ, reference, gen_input, meta={
        "strategy": strategy, "width": width, "k": k, "b": b,
        "effective_b": min(b, max(0, width-k)), "prefix_bits": prefix_bits,
        "controlled": controlled,
    })
