"""The jump-J dialog reference: does it compute the multiply, and what does it cost?

Two separate things are pinned here, and only the first is about correctness.

The first is that moving the loop boundary and batching halvings does not change the
function. `to_dialog_jump` ends each iteration where `to_dialog` begins it, so the two
tapes are not comparable entry-for-entry -- but the composite must still be the multiply,
with the same direction convention and no stray factor of two from the moved shift.

The second is the cost model, which is the reason jump-2 is being built at all. The claims
it rests on are arithmetic, so they are asserted rather than described:

  * the alphabet is 2J+1 states, not 4J, because an early-stopped halving chain forces
    `b0 = 1`. This is what makes the tape shrink, and the tape is the binding constraint.
  * iterations fall to ~2/3 at J=2, from E[ceil(tz/2)]/E[tz] with tz ~ Geom(1/2).
  * the w-proportional gate work is INVARIANT in J, so the entire gate win is comparator
    count. A test that showed jump-2 saving adder work would mean the model is wrong.

The measured numbers quoted below are at n = 255 over 20,000 random x: J=1 359.93
iterations, J=2 240.37 (ratio 0.668), J=3 206.25 (0.573).
"""

from __future__ import annotations

import math
import random

import pytest

from qsm.inversion.reference import (
    JUMP_BOUNDARY_STATE,
    apply_dialog_jump,
    apply_dialog_jump_circuit,
    apply_dialog_jump_lazy,
    dialog_widths_jump,
    jump_boundary_states,
    jump_states,
    modmul_inplace_ref,
    to_dialog_jump,
    un_dialog_jump,
)

PRIMES = [29, 61, 127, 251, 1021]
JUMPS = [1, 2, 3, 4]


def _iters(p: int, jump: int) -> int:
    """Generous fixed budget: the loop is allowed to converge and then idle."""
    return 4 * p.bit_length()


# --- correctness ------------------------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("jump", JUMPS)
def test_the_jump_tape_replays_to_the_multiply_exhaustively(p, jump):
    """`apply_dialog_jump(tape(x), y, 0) == (0, x*y)` for every x, every y.

    Same direction convention as jump 1 -- which is not automatic. The jump loop's shift
    sits at the top of the iteration rather than the bottom, so the tape carries one
    halving more or less at its ends than the jump-1 tape does, and a mismatch there would
    show up as the whole result being off by a factor of two.
    """
    it = _iters(p, jump)
    for x in range(1, p):
        tape, u, v = to_dialog_jump(p, x, it, jump)
        assert (u, v) == (1, 0), f"p={p} J={jump} x={x}: did not converge in {it}"
        for y in range(p):
            a, b = apply_dialog_jump(tape, y, 0, p)
            assert (a, b) == (0, (x * y) % p), f"p={p} J={jump} x={x} y={y}"


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("jump", JUMPS)
def test_the_jump_tape_agrees_with_the_jump_one_construction(p, jump):
    """The batched loop computes what the unbatched reference computes.

    `modmul_inplace_ref` goes through `to_dialog` / `apply_dialog`, so this is the
    end-to-end statement that jump-J is the same circuit specification, not merely an
    internally consistent one.
    """
    it = _iters(p, jump)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump(p, x, it, jump)
        for y in (1, 2, p - 1, p // 2):
            _, b = apply_dialog_jump(tape, y, 0, p)
            assert b == modmul_inplace_ref(p, x, y), f"p={p} J={jump} x={x} y={y}"


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("jump", JUMPS)
def test_the_jump_tape_run_backwards_reconstructs_p_and_x(p, jump):
    """`un_dialog_jump` is `apply_dialog_jump` with the modulus deleted, and returns (p, x).

    This is the Bezout identity that fixes the direction, and it is also the pass the
    composite needs in order to hand `x` back after the replay has spent it.
    """
    it = _iters(p, jump)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump(p, x, it, jump)
        assert un_dialog_jump(tape) == (p, x), f"p={p} J={jump} x={x}"


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("jump", [1, 2, 3])
def test_the_lazy_and_wrapping_references_agree_where_the_domain_closes(p, jump):
    """`_lazy` and `_circuit` agree exactly on every run in which no value leaves [0, 2^u).

    The two cannot be compared modulo 2^u after a non-closure: `_lazy` keeps the oversized
    value and keeps computing with it, so the very next branch may take a different path
    and the two references diverge in TRAJECTORY, not just in representation. So the run
    is instrumented, and equality is asserted only where the domain actually closed.

    p = 127 is the f = 1 case, where `2p < 2^u` cannot be reached and no run ever wraps --
    which is the reason for keeping it in the list.
    """
    u = p.bit_length()
    m = 1 << u
    it = _iters(p, jump)

    def lazy_wraps(tape, a, b):
        """`apply_dialog_jump_lazy`, plus whether any reduction failed to close."""
        wrapped = False
        for s, b0, b1 in reversed(tape):
            if b0:
                t = a + b
                b = t - p if t >= m else t
                wrapped |= b >= m
            if b1:
                a, b = b, a
            for _ in range(s):
                b = 2 * b - p if 2 * b >= m else 2 * b
                wrapped |= b >= m
        return a, b, wrapped

    closed = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_jump(p, x, it, jump)
        for y in (1, 3, p - 1):
            la, lb, wrapped = lazy_wraps(tape, y, 0)
            if wrapped:
                continue
            closed += 1
            assert (la, lb) == apply_dialog_jump_lazy(tape, y, 0, p, u)
            assert (la, lb) == apply_dialog_jump_circuit(tape, y, 0, p, u), (
                f"p={p} J={jump} x={x} y={y}: references differ with no wrap"
            )
    assert closed, f"p={p} J={jump}: every run wrapped, the test asserted nothing"
    if p == 127:
        assert closed == 3 * (p - 1), "f = 1 leaves no room to wrap"


# --- the cost model ---------------------------------------------------------
@pytest.mark.parametrize("jump", JUMPS)
def test_the_alphabet_is_two_jump_plus_one_states(jump):
    """2J+1, not 4J: a halving chain that stopped early stopped because `v` was odd.

    This is the whole reason the tape shrinks. (shifts, b0, b1) is nominally
    J x 2 x 2 = 4J combinations; `shifts < J` forces `b0 = 1`, which removes 2(J-1) of
    them, and `b0 = 0` forces `b1 = 0`, which removes one more.
    """
    states = jump_states(jump)
    assert len(states) == 2 * jump + 1
    assert len(set(states)) == len(states)
    for s, b0, b1 in states:
        assert 1 <= s <= jump
        if s < jump:
            assert b0 == 1, f"an early-stopped chain stopped on an odd v: {(s, b0, b1)}"
        if b0 == 0:
            assert b1 == 0, f"b1 is only defined under b0: {(s, b0, b1)}"


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("jump", JUMPS)
def test_no_tape_entry_falls_outside_the_declared_alphabet(p, jump):
    """Exhaustive: every logged state is one the packing will have room for.

    Entry 0 is allowed one extra state because its first halving is conditional; every
    other entry, INCLUDING the ones logged after convergence, must be in `jump_states`.
    Post-convergence entries are the ones that catch a stray `v == 0` guard: without the
    halving chain running on through zero, they log `(1, 0, 0)` and widen the alphabet.
    """
    it = _iters(p, jump)
    body, boundary = set(jump_states(jump)), set(jump_boundary_states(jump))
    for x in range(1, p):
        tape, _, _ = to_dialog_jump(p, x, it, jump)
        assert tape[0] in boundary, f"p={p} J={jump} x={x}: entry 0 is {tape[0]}"
        for i, st in enumerate(tape[1:], start=1):
            assert st in body, f"p={p} J={jump} x={x}: entry {i} is {st}, outside {body}"


@pytest.mark.parametrize("p", [251, 1021])
@pytest.mark.parametrize("jump", JUMPS)
def test_every_declared_state_is_actually_reachable(p, jump):
    """The alphabet is tight as well as sufficient -- no state is paid for and never used.

    A slack alphabet would inflate the base and so the packed tape, which is the one
    resource this whole exercise is spending gates to buy.

    Only the larger primes: reaching `shifts == J` needs J consecutive trailing zeros,
    which is a 2^-J event per iteration, and 28 or 60 inputs is not enough trials at J = 4.
    That is a shortage of samples, not a slack alphabet -- p = 251 and p = 1021 hit all
    2J+1 states for every J tested.
    """
    it = _iters(p, jump)
    seen = set()
    for x in range(1, p):
        tape, _, _ = to_dialog_jump(p, x, it, jump)
        seen.update(tape[1:])
    assert seen == set(jump_states(jump)), f"p={p} J={jump}: unreachable {set(jump_states(jump)) - seen}"


def test_the_boundary_state_belongs_to_iteration_zero_and_nowhere_else():
    """`(0, 1, 1)` is logged exactly when `x` is odd, and only by entry 0.

    Iteration 0 is the only one whose first halving is conditional. Its `b1` is 1 whenever
    `b0` is, because `u = p` still exceeds any `v = x < p` -- so it needs one extra state,
    not four, and that state can be encoded once instead of widening every group.
    """
    for p in PRIMES:
        it = _iters(p, 2)
        for x in range(1, p):
            tape, _, _ = to_dialog_jump(p, x, it, 2)
            assert (tape[0] == JUMP_BOUNDARY_STATE) == bool(x & 1), f"p={p} x={x}"
            assert JUMP_BOUNDARY_STATE not in tape[1:], f"p={p} x={x}"


def test_jump_two_needs_two_thirds_the_iterations():
    """Measured against the derivation, not against a remembered number.

    Trailing zeros after a subtraction are Geom(1/2) on {1,2,...}: E[tz] = 2 while
    E[ceil(tz/2)] = 4/3, so batching two halvings per iteration should cost 2/3 as many
    iterations. Anything materially off that would mean the halvings are correlated with
    the branch in a way the cost model does not know about.
    """
    p = (1 << 255) - 19
    rng = random.Random(20260906)
    xs = [rng.randrange(1, p) for _ in range(400)]

    def converge(x, jump):
        u, v, i = p, x, 0
        while v:
            for j in range(jump):
                if j == 0 and i > 0:
                    v >>= 1
                elif v % 2 == 0 and v:
                    v >>= 1
                else:
                    break
            if v & 1:
                if u > v:
                    u, v = v, u
                v -= u
            i += 1
        return i

    m1 = sum(converge(x, 1) for x in xs) / len(xs)
    m2 = sum(converge(x, 2) for x in xs) / len(xs)
    assert m2 / m1 == pytest.approx(2 / 3, abs=0.02), f"J=1 {m1:.1f}, J=2 {m2:.1f}"


def test_batching_saves_comparators_and_not_adder_work():
    """The w-proportional work is invariant in J -- so the tape, not the gates, decides J.

    Summing the live window over the whole loop is a proxy for every w-proportional
    region: the cswap, the controlled subtraction, and (for J > 1) the controlled
    rotations. Fewer iterations at more rotations each is the same integral, which is why
    the derivation predicts a gate win of only `(1 - 2/3) * comparator`.
    """
    p = (1 << 255) - 19
    rng = random.Random(4)
    xs = [rng.randrange(1, p) for _ in range(60)]

    def work(x, jump):
        """(sum of window widths over w-proportional ops, number of comparators).

        Gates are counted as EMITTED, not as effected. A circuit lays down the controlled
        rotation, the cswap and the controlled adder on every iteration; the control
        decides what they do, not whether they are there. Counting only the taken branches
        is what makes jump-2 look like it saves adder work -- it does not.
        """
        w_ops = cmps = 0
        u, v, i = p, x, 0
        while v:
            for j in range(jump):
                if j == 0 and i > 0:
                    v >>= 1               # unconditional: a relabel, no gates
                elif v % 2 == 0 and v:
                    v >>= 1
                else:
                    break
            w = max(max(u, v).bit_length(), 1)
            # (jump-1) controlled rotations + cswap + controlled subtract, all emitted.
            w_ops += (jump - 1) * w + 2 * w
            cmps += 1
            if v & 1:
                if u > v:
                    u, v = v, u
                v -= u
            i += 1
        return w_ops, cmps

    w1 = sum(work(x, 1)[0] for x in xs) / len(xs)
    w2 = sum(work(x, 2)[0] for x in xs) / len(xs)
    c1 = sum(work(x, 1)[1] for x in xs) / len(xs)
    c2 = sum(work(x, 2)[1] for x in xs) / len(xs)
    assert w2 / w1 == pytest.approx(1.0, abs=0.05), f"adder work moved: {w1:.0f} -> {w2:.0f}"
    assert c2 / c1 < 0.75, f"comparator count did not fall: {c1:.0f} -> {c2:.0f}"


def _best_packing(base: int, max_group: int = 15) -> tuple[float, int]:
    """(bits per iteration, group size) for the cheapest base-`base` packing available."""
    return min(
        (math.ceil(math.log2(base ** g)) / g, g) for g in range(1, max_group + 1)
    )


def test_jump_two_packs_to_a_smaller_tape_than_jump_one():
    """The tape is the only resource jump-2 actually buys, so price it at BOTH packings.

    Against the packing that exists today (3 states, 5 bits per 3 iterations) jump-2 looks
    better than it is, because 5-in-3 is not base 3's optimum -- 8 bits per 5 iterations
    is, and that is an independent jump-1 improvement already on the deferred list. So the
    comparison is made twice, and the honest incremental figure is the second one.

    Base 5 is the reason J=2 and not J=3: 5^3 = 125 <= 2^7 lands within 0.5% of the
    entropy bound at a group of 3, while base 7 needs a group of 11 to get close and sits
    at a flat 3.0 bits/iteration for anything practical. At 0.573x the iterations that
    still leaves J=3's tape LONGER than J=1's.
    """
    (b1_opt, g1), (b2_opt, g2), (b3_opt, _) = (
        _best_packing(3), _best_packing(5), _best_packing(7)
    )
    b1_now = math.ceil(math.log2(3 ** 3)) / 3          # what gcd.py packs today
    assert (b1_now, g1, b2_opt, g2) == (5 / 3, 5, 7 / 3, 3)
    assert b1_opt == 1.6 < b1_now, "8 bits per 5 iterations beats 5 per 3 at jump 1"

    # Relative tape length = bits per iteration x relative iteration count.
    r2, r3 = 2 / 3, 0.573
    assert b2_opt * r2 < b1_now, "jump 2 must beat the packing that exists"
    assert b2_opt * r2 < b1_opt, "jump 2 must beat the best jump 1, not just the current one"
    assert b3_opt * r3 > b1_opt, "jump 3's tape is worse than jump 1's; base 7 packs badly"

    # And in qubits, at the budgets each J actually needs.
    assert math.ceil(399 / 3) * 5 == 665                       # today
    assert math.ceil(399 / 5) * 8 == 640                       # denser jump-1 packing
    assert math.ceil(261 / 3) * 7 == 609                       # jump 2
    assert 609 + 255 + 255 < 1175, "the point of all this: slack under PEAK_BUDGET"


@pytest.mark.parametrize("p", [251, 1021])
def test_the_window_shrinks_faster_per_iteration_under_jump_two(p):
    """Per-iteration width decay scales with halvings consumed, so it steepens with J.

    The schedule cannot be `dialog_widths` resampled -- an iteration eats a variable
    number of halvings -- which is why `dialog_widths_jump` exists and why the jump-2
    width rate has to be measured rather than scaled from `GCD_WIDTH_RATE`.
    """
    it = _iters(p, 2)
    n = p.bit_length()

    def rate(jump):
        tot = 0.0
        for x in range(1, p):
            w = dialog_widths_jump(p, x, it, jump)
            live = [b for b in w if b > 1]
            tot += (n - live[-1]) / max(1, len(live))
        return tot / (p - 1)

    r1, r2 = rate(1), rate(2)
    assert r2 > r1, f"p={p}: jump-1 {r1:.3f} bits/iter, jump-2 {r2:.3f}"
    assert 1.2 < r2 / r1 < 1.8, f"p={p}: decay ratio {r2/r1:.3f} is not ~1.5"
