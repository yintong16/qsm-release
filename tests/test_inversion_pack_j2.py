"""The 7-per-3 jump-2 dialog packing, classically and as a circuit.

Same obligations as the base-3 packing: the two halves must be exactly inverse, and the
accumulator must be exactly base 5, or `unfold_digit_j2` cannot read `t = R div 5^j` back
out and the replay has no tape. Both are small enough to check exhaustively, and this file
does, at every digit position and every reachable accumulator value.

Two things here that the base-3 file does not have to worry about:

  * there is no free digit 0. The three signal qubits are not the binary form of any legal
    digit assignment, so digit 0 folds arithmetically like the others.
  * the fold can overflow its own accumulator if the terms are applied in the wrong order.
    `t = 3c + b1 + (b0 - 1)` has a negative term, and taking it last -- conditioned on NOT
    b0 -- is what keeps R inside 5^(j+1). There is a test for exactly that, because the
    natural ordering is the broken one and it only breaks at j = 2.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.inversion import pack

#: The five reachable (c, b0, b1) patterns. c = 0 forces b0 = 1 (the halving chain only
#: stops early on an odd v); b0 = 0 forces b1 = 0 (b1 is only defined under b0).
PATTERNS = [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]

#: The same five as `(shifts, b0, b1)`, which is what `to_dialog_jump` actually emits.
TAPE_PATTERNS = [(1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0), (2, 1, 1)]


def _digit(c: int, b0: int, b1: int) -> int:
    return 3 * c + b1 + (b0 - 1)


# --- the classical map -----------------------------------------------------
def test_the_digit_formula_is_a_bijection_onto_zero_through_four():
    """t = 3c + b1 + (b0 - 1). Affine in the raw signals, so the fold needs no lookup."""
    got = {p: _digit(*p) for p in PATTERNS}
    assert sorted(got.values()) == [0, 1, 2, 3, 4]
    assert got == pack._DIGIT_J2


@pytest.mark.parametrize("length", range(1, 8))
def test_pack_round_trips_exhaustively(length: int):
    """Every tape of this length survives, including the partial trailing group."""
    for tape in itertools.product(TAPE_PATTERNS, repeat=length):
        tape = list(tape)
        d = pack.pack_dialog_j2(tape)
        assert pack.unpack_dialog_j2(d, length) == tape
        assert d < 1 << pack.packed_bits_j2(length), "packed value overflows its own width"


def test_pack_is_injective_on_a_full_group():
    """125 states into 7 qubits, with no collisions -- the whole claim, on one group."""
    seen = {pack.pack_dialog_j2(list(t)) for t in itertools.product(TAPE_PATTERNS, repeat=3)}
    assert len(seen) == 125
    assert max(seen) < 1 << pack.GROUP_BITS_J2


def test_the_packed_width_is_609_and_sits_within_half_a_percent_of_the_floor():
    """609 at 261 iterations, against 665 for jump-1 at 399 and a 607-qubit floor.

    2.3333 bits per iteration against an entropy bound of 2.3219 -- 0.5% over, where the
    base-3 packing in use today spends 5.2%. That tightness is why the answer is base 5:
    base 7 (jump 3) needs a group of 11 to get close and sits at a flat 3.0 for any group
    small enough to decode, which leaves ITS tape longer than jump-1's.
    """
    assert pack.packed_bits_j2(261) == 609
    floor = math.ceil(261 * math.log2(5))
    assert floor == 607
    assert pack.packed_bits_j2(261) / floor < 1.005
    assert pack.packed_bits_j2(261) + 2 * 255 == 1119     # 56 under PEAK_BUDGET


def test_the_tape_the_reference_emits_packs_without_translation():
    """`to_dialog_jump`'s alphabet and `pack_dialog_j2`'s are the same five states.

    Pinned across the module boundary rather than assumed: the reference speaks
    `(shifts, b0, b1)` and the packing speaks `(c, b0, b1)`, and `c = [shifts == 2]` is the
    only bridge between them.
    """
    from qsm.inversion.reference import jump_states

    assert sorted(jump_states(2)) == sorted(TAPE_PATTERNS)
    for s, b0, b1 in TAPE_PATTERNS:
        assert pack._DIGIT_J2[(1 if s == 2 else 0, b0, b1)] == _digit(1 if s == 2 else 0, b0, b1)


# --- the fold circuit ------------------------------------------------------
def _run(j: int, r_in: int, sig_in: int = 0, invert: bool = False):
    """Run fold_digit_j2 (or its inverse) on one concrete state and read it all back.

    Unlike base 3 there is no digit-0 special case: the signals always live in their own
    3-qubit register and the accumulator is always separate.
    """
    c = Circuit()
    r = c.input("r", pack.DIGIT_WIDTH_J2[j])
    sig = c.input("s", 3)
    fn = pack.unfold_digit_j2 if invert else pack.fold_digit_j2
    fn(c, r, j, sig[0], sig[1], sig[2])
    return simulate(c, {"r": r_in, "s": sig_in},
                    rng=random.Random(j * 97 + r_in * 7 + sig_in))


def _sig(c: int, b0: int, b1: int) -> int:
    return c | (b0 << 1) | (b1 << 2)


@pytest.mark.parametrize("j", range(pack.GROUP_J2))
def test_fold_digit_is_base_5_and_clears_the_signals(j: int):
    """R += 5^j * t, and the three qubits the digit arrived on come back |0>.

    Exhaustive over every accumulator value reachable at this digit position, which is
    what licenses reading `t = R div 5^j` back out -- the property `unfold_digit_j2` and
    therefore the whole replay depends on.
    """
    for prev in range(5 ** j):
        for c, b0, b1 in PATTERNS:
            res = _run(j, prev, _sig(c, b0, b1))
            assert res.phase == 0
            expected = prev + (5 ** j) * _digit(c, b0, b1)
            assert res.outputs["r"] == expected, (j, prev, c, b0, b1, res.outputs["r"])
            assert res.outputs["s"] == 0, f"digit {j} left {res.outputs['s']} behind"


@pytest.mark.parametrize("j", range(pack.GROUP_J2))
def test_unfold_digit_inverts_fold_digit(j: int):
    """The replay reads the tape with this, so 'inverse' has to mean bit-for-bit."""
    for prev in range(5 ** j):
        for c, b0, b1 in PATTERNS:
            res = _run(j, prev + (5 ** j) * _digit(c, b0, b1), 0, invert=True)
            assert res.phase == 0
            assert res.outputs["r"] == prev, (j, prev, c, b0, b1, res.outputs["r"])
            assert res.outputs["s"] == _sig(c, b0, b1), (j, prev, c, b0, b1)


@pytest.mark.parametrize("j", range(pack.GROUP_J2))
def test_the_accumulator_never_leaves_its_declared_width_mid_fold(j: int):
    """The ordering constraint, checked rather than trusted.

    `t` has a negative term, so the fold is not a sum of non-negative pieces and its
    transient maximum depends on the order. Taking `-5^j` last bounds the accumulator by
    `5*5^j`; adding all three positive terms first and subtracting afterwards reaches
    `5^(j+1) + 5^j - 1`, which is 149 at j = 2 and does not fit in 7 qubits. The
    simulation would not catch it -- the register is sized from DIGIT_WIDTH_J2 -- so the
    bound is asserted directly.
    """
    k = 5 ** j
    for prev in range(5 ** j):
        for c, b0, b1 in PATTERNS:
            running = prev
            for step in (3 * k * c, k * b1, -k * (1 - b0)):
                running += step
                assert 0 <= running < (1 << pack.DIGIT_WIDTH_J2[j]), (
                    f"j={j} prev={prev} state={(c, b0, b1)}: transient {running} "
                    f"escapes {pack.DIGIT_WIDTH_J2[j]} qubits"
                )
            assert running == prev + k * _digit(c, b0, b1)
    # ...and the broken ordering really does break, so the test above is not vacuous.
    if j == 2:
        worst = (5 ** j - 1) + 3 * k + k + k
        assert worst >= 1 << pack.DIGIT_WIDTH_J2[j]


def test_a_group_costs_little_against_the_loop_it_rides():
    """No free digit 0, so this is dearer than base 3's -- the question is whether it matters.

    Three controlled constant adds and six constant compares per digit, on registers 3 to
    7 qubits wide, against the ~1,600 Toffoli the group's own divsteps cost. Recorded as a
    number rather than a claim, because the base-3 scheme got its digit 0 for free and the
    honest comparison has to show what dropping that costs.
    """
    tof = []
    for j in range(pack.GROUP_J2):
        c = Circuit()
        r = c.input("r", pack.DIGIT_WIDTH_J2[j])
        sig = c.input("s", 3)
        pack.fold_digit_j2(c, r, j, sig[0], sig[1], sig[2])
        tof.append(count(c).toffoli)
    assert tof[0] > 0, "base 5 has no free digit 0; if this ever passes, say why"
    assert sum(tof) <= 140, f"a jump-2 group costs {sum(tof)} Toffoli"


# --- the boundary iteration -------------------------------------------------
#: `(s0, s1, b0)` for iteration 0, and the `(shifts, b0, b1)` tape entry each produces.
BOUNDARY = [((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
            ((1, 1, 0), (2, 0, 0)), ((1, 1, 1), (2, 1, 1))]


def test_the_boundary_iteration_has_four_states_and_still_fits_a_base_five_digit():
    """The reason jump-2's tape is 609 and not 610: iteration 0 does not need a wider group.

    It loses one constraint -- the first halving is conditional, so `shifts` takes three
    values and `c = [shifts == 2]` cannot separate them -- and gains another at the same
    time: `u = p` and `v <= x < p`, so `u > v` outright and `b1 = b0` is not a wire. Four
    states, not six. `reference.jump_boundary_states` over-approximates to six on purpose;
    the two it adds need `u < v`, which iteration 0 cannot produce.
    """
    from qsm.inversion.reference import jump_boundary_states, to_dialog_jump

    assert set(pack._DIGIT_J2_BOUNDARY) == {t for _, t in BOUNDARY}
    assert sorted(pack._DIGIT_J2_BOUNDARY.values()) == [0, 1, 2, 3]
    assert len(pack._DIGIT_J2_BOUNDARY) < pack.RADIX_J2, "must fit one base-5 digit"
    assert set(pack._DIGIT_J2_BOUNDARY) < set(jump_boundary_states(2)), "a strict subset"

    # ...and the two extra states really are unreachable, over every x at a small prime.
    p = 251
    seen = {to_dialog_jump(p, x, 2 * p.bit_length())[0][0] for x in range(1, p)}
    assert seen == set(pack._DIGIT_J2_BOUNDARY), f"iteration 0 produced {seen}"


def test_the_boundary_digit_formula_is_s0_plus_two_s1_plus_b0_minus_one():
    for (s0, s1, b0), entry in BOUNDARY:
        assert pack._DIGIT_J2_BOUNDARY[entry] == s0 + 2 * s1 + b0 - 1


def test_a_real_tape_round_trips_through_the_boundary_aware_pack():
    """`boundary=True` is what a `to_dialog_jump` tape needs; the default must stay off."""
    from qsm.inversion.reference import to_dialog_jump

    p, iters = 1021, 30
    for x in range(1, p, 7):
        tape, _, _ = to_dialog_jump(p, x, iters)
        d = pack.pack_dialog_j2(tape, boundary=True)
        assert pack.unpack_dialog_j2(d, iters, boundary=True) == tape
        assert d < 1 << pack.packed_bits_j2(iters)


def _run_boundary(r_in: int, sig_in: int = 0, invert: bool = False):
    c = Circuit()
    r = c.input("r", pack.DIGIT_WIDTH_J2[0])
    sig = c.input("s", 3)
    fn = pack.unfold_digit_j2_boundary if invert else pack.fold_digit_j2_boundary
    fn(c, r, sig[0], sig[1], sig[2])
    return simulate(c, {"r": r_in, "s": sig_in}, rng=random.Random(r_in * 11 + sig_in))


@pytest.mark.parametrize("sig,entry", BOUNDARY)
def test_the_boundary_fold_is_base_five_and_clears_its_three_signals(sig, entry):
    s0, s1, b0 = sig
    res = _run_boundary(0, s0 | (s1 << 1) | (b0 << 2))
    assert res.phase == 0
    assert res.outputs["r"] == pack._DIGIT_J2_BOUNDARY[entry]
    assert res.outputs["s"] == 0, "the boundary signals did not come back |0>"


@pytest.mark.parametrize("sig,entry", BOUNDARY)
def test_the_boundary_unfold_inverts_the_fold(sig, entry):
    s0, s1, b0 = sig
    res = _run_boundary(pack._DIGIT_J2_BOUNDARY[entry], 0, invert=True)
    assert res.phase == 0
    assert res.outputs["r"] == 0
    assert res.outputs["s"] == s0 | (s1 << 1) | (b0 << 2)


def test_the_boundary_fold_never_leaves_three_qubits_mid_way():
    """`r` starts at 0 and the digit has a `-1`, so the order is the whole correctness argument.

    `s0 = 0` implies `b0 = 1`, so the two positive units cover the subtraction and every
    transient stays in [0, 3]. Putting the `-1` first underflows on `(1, 1, 0)`, which is
    asserted below so the ordering is not silently reshuffled later.
    """
    for (s0, s1, b0), entry in BOUNDARY:
        running = 0
        for step in (s0, b0, -1, 2 * s1):
            running += step
            assert 0 <= running < (1 << pack.DIGIT_WIDTH_J2[0]), (s0, s1, b0, running)
        assert running == pack._DIGIT_J2_BOUNDARY[entry]
    assert any(s0 == 0 for (s0, _, _), _ in BOUNDARY), "s0 = 0 must be reachable"
    assert all(s0 or b0 for (s0, _, b0), _ in BOUNDARY), "s0 = 0 forces b0 = 1"
    assert any(b0 == 0 for (_, _, b0), _ in BOUNDARY), "-1 first would underflow there"
