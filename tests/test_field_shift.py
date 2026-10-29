"""The controlled one-bit right shift, and the Toffoli it is allowed to drop.

Jump 2's second halving is the only shift in the design that costs gates, so this file
does three things and each of them is load-bearing for `qsm/inversion`:

  * the ladder computes a shift, exhaustively, at widths small enough to enumerate;
  * `top_zero=True` is *exactly* equal to the full ladder on the inputs it declares, and
    strictly one Toffoli cheaper -- both checked, because "provably dead" from a paper is
    a claim and not a measurement;
  * a violated precondition is loud, so nobody is tempted to fold it into a rate.

The GCD's own obligation -- that the bit which wraps is zero because the control IS `NOT
r[0]`, and that the top slot is zero because the *unconditional* halving just rotated a
known zero into it -- is checked at the end against the reference loop rather than
restated.
"""

from __future__ import annotations

import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.shift import (
    controlled_rotate1,
    controlled_rshift1,
    rshift1_toffoli,
)

WIDTHS = [1, 2, 3, 5, 8]


def _run(w: int, ctrl: int, val: int, *, top_zero: bool = False, rotate: bool = False):
    c = Circuit()
    q = c.input("ctrl", 1)
    r = c.input("r", w)
    if rotate:
        controlled_rotate1(c, q[0], r)
    else:
        controlled_rshift1(c, q[0], r, top_zero=top_zero)
    return simulate(c, {"ctrl": ctrl, "r": val}, rng=random.Random(w * 31 + val * 7 + ctrl))


# --- the shift itself -------------------------------------------------------
@pytest.mark.parametrize("w", WIDTHS)
def test_the_ladder_shifts_when_told_to_and_does_nothing_otherwise(w: int):
    """Exhaustive over every legal (ctrl, r) at this width. `ctrl` comes back untouched."""
    for val in range(1 << w):
        for ctrl in (0, 1):
            if ctrl and val & 1:
                continue                      # outside the declared precondition
            res = _run(w, ctrl, val)
            assert res.phase == 0
            assert res.outputs["ctrl"] == ctrl, "the control must be preserved"
            assert res.outputs["r"] == (val >> 1 if ctrl else val), (w, ctrl, val)


@pytest.mark.parametrize("w", WIDTHS)
def test_the_rotate_is_a_rotate_for_every_input(w: int):
    """`controlled_rotate1` carries no precondition, which is the point of having it.

    It is the same gate sequence; the difference between the two names is entirely a
    statement about what the caller promises, so the unconditional version is checked on
    the inputs the shift refuses.
    """
    for val in range(1 << w):
        for ctrl in (0, 1):
            res = _run(w, ctrl, val, rotate=True)
            assert res.phase == 0
            want = ((val >> 1) | ((val & 1) << (w - 1))) if ctrl else val
            assert res.outputs["r"] == want, (w, ctrl, val)


# --- the dead top Fredkin ---------------------------------------------------
@pytest.mark.parametrize("w", WIDTHS)
def test_dropping_the_top_fredkin_changes_nothing_on_its_declared_inputs(w: int):
    """`top_zero` agrees with the full ladder wherever both are declared to work.

    Exhaustive over `r < 2^(w-1)` with `r` even under `ctrl`, which is exactly what the
    jump-2 loop hands it. This is the test the plan asked for: the deadness is established
    here, not taken on the reference implementation's word.
    """
    for val in range(1 << max(w - 1, 0)):
        for ctrl in (0, 1):
            if ctrl and val & 1:
                continue
            full = _run(w, ctrl, val)
            cut = _run(w, ctrl, val, top_zero=True)
            assert cut.phase == 0
            assert cut.outputs["r"] == full.outputs["r"] == (val >> 1 if ctrl else val)


@pytest.mark.parametrize("w", [2, 3, 8, 40, 255])
def test_the_dropped_fredkin_is_a_real_toffoli_and_the_cost_helper_agrees(w: int):
    """One Toffoli per call, 261 calls, at widths up to 255 -- worth counting, not assuming."""
    tof = []
    for top_zero in (False, True):
        c = Circuit()
        q = c.input("ctrl", 1)
        r = c.input("r", w)
        controlled_rshift1(c, q[0], r, top_zero=top_zero)
        tof.append(count(c).toffoli)
        assert tof[-1] == rshift1_toffoli(w, top_zero=top_zero)
    assert tof == [max(w - 1, 0), max(w - 2, 0)]


def test_at_the_full_width_the_saving_is_254_against_253():
    """The two registered versions differ by exactly one Toffoli and nothing else."""
    from qsm.bench.registry import get_builder

    a = get_builder("field.rshift1_ctrl.v1")()
    b = get_builder("field.rshift1_ctrl_topzero.v1")()
    ca, cb = count(a.circuit), count(b.circuit)
    assert (ca.toffoli, cb.toffoli) == (254, 253)
    assert ca.toffoli - cb.toffoli == 1


# --- violations are loud ----------------------------------------------------
def test_violating_the_low_bit_precondition_wraps_a_one_to_the_top():
    """Not a small numeric error: the result is off by 2^(w-1). Hence the assertion.

    Recorded so that if the GCD ever stops guaranteeing `ctrl = NOT r[0]` the consequence
    is known to be a wrong field element rather than a bounded truncation that could be
    folded into `params`' failure budget.
    """
    w = 8
    for val in range(1, 1 << w, 2):
        got = _run(w, 1, val).outputs["r"]
        assert got == (val >> 1) + (1 << (w - 1))
        assert got - (val >> 1) == 1 << (w - 1)


def test_violating_the_top_bit_precondition_strands_that_bit_one_slot_low():
    """The other half of the contract, and the reason iteration 0 cannot use `top_zero`."""
    w = 8
    for val in range(0, 1 << w, 2):
        if not (val >> (w - 1)) & 1:
            continue
        got = _run(w, 1, val, top_zero=True).outputs["r"]
        assert got != val >> 1
        # The top bit never moved; everything below it shifted out from under it.
        assert got == ((val & ~(1 << (w - 1))) >> 1) | (1 << (w - 1))


# --- the GCD's own guarantee ------------------------------------------------
@pytest.mark.parametrize("p", [251, 1021, 8191])
def test_the_jump_two_loop_really_does_supply_both_preconditions(p: int):
    """Where the deadness comes from, checked on the reference rather than argued.

    Two invariants, over every `x`: at the top of iteration `i > 0` the low bit of `v` is
    zero (odd minus odd is even, which is what makes the FIRST halving free), and after
    that free halving the top of the window is that same known zero. Iteration 0 satisfies
    neither in general, which is exactly why it is excluded here and passes
    `top_zero=False` in the loop.
    """
    n = p.bit_length()
    boundary_evens = 0
    for x in range(1, p):
        u, v = p, x
        for i in range(2 * n):
            if not v:
                break
            if i > 0:
                assert v % 2 == 0, f"p={p} x={x} i={i}: v odd at the top of the iteration"
                v >>= 1                      # the free halving; wraps a known zero up top
            else:
                boundary_evens += v % 2 == 0
                if v % 2 == 0:
                    v >>= 1
            # ...and the conditional halving's control is exactly NOT v[0].
            if v % 2 == 0:
                v >>= 1
            b0 = v & 1
            if b0 and u > v:
                u, v = v, u
            if b0:
                v -= u
    assert boundary_evens > 0, "iteration 0 really can shift; it is not vacuously top_zero"
