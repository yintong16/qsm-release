"""Phase 1 with two halvings per iteration, against the reference and against jump 1.

The jump-2 loop is not the jump-1 loop with a bigger stride. Three things change shape and
each is checked here rather than argued:

  * the free halving moves to the TOP of the iteration, because that is the only place
    `v` is known even, and the conditional second halving becomes real gates;
  * iteration 0 is different in kind: `x` may be even, so both of its halvings are
    conditional and it logs `(s0, s1, b0)` instead of `(c, b0, b1)`. It saves a comparator
    in exchange, because `u = p > v` makes `b1 = b0`;
  * the win is the *tape*, 609 qubits against 665, and on gates it is one third of one
    comparator per iteration. The last test pins that, because the 261/399 ratio invites a
    1.53x claim the arithmetic does not support.

Correctness is exhaustive over every `x` at small primes, in every tape layout, because a
tape that is one entry wrong replays to a wrong field element with no other symptom.
"""

from __future__ import annotations

import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.inversion import gcd, pack, schedule
from qsm.inversion.reference import to_dialog_jump

LAYOUTS = ["flat", "pooled", "packed"]


def _build(p: int, iters: int, layout: str, narrow: bool):
    n = p.bit_length()
    circ = Circuit()
    x = circ.input("x", n)
    if layout == "flat":
        tape = gcd.FlatTapeJ2(circ.input("d", 3 * iters))
    else:
        circ.input("d", 0)
        tape = (gcd.PooledTapeJ2 if layout == "pooled" else gcd.PackedTapeJ2)(circ)
    widths = schedule.widths_j2(n, iters) if narrow else None
    d = gcd.to_dialog_j2(circ, x, tape, p, iters=iters, widths=widths)
    if layout != "flat":
        circ.rebind("d", d)
    return circ


def _expect(p: int, x: int, iters: int, layout: str) -> int:
    tape, _, _ = to_dialog_jump(p, x, iters)
    if layout == "packed":
        return pack.pack_dialog_j2(tape, boundary=True)
    return gcd.encode_dialog_j2(tape)


# --- correctness ------------------------------------------------------------
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("p", [29, 61])
def test_the_loop_logs_the_reference_tape_for_every_x(p: int, layout: str):
    """Exhaustive, wide window: the gates alone, with no schedule to blame."""
    iters = 2 * p.bit_length()
    for x in range(1, p):
        circ = _build(p, iters, layout, narrow=False)
        res = simulate(circ, {"x": x, "d": 0}, rng=random.Random(p * 1000 + x))
        assert res.phase == 0, f"p={p} x={x}: phase leak"
        assert res.outputs["d"] == _expect(p, x, iters, layout), f"p={p} x={x} {layout}"
        assert res.outputs["x"] == 0, "x must be consumed into the tape"


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("p", [61, 251])
def test_the_narrowing_window_does_not_change_the_tape(p: int, layout: str):
    """Same answers with the schedule on -- so the window is shedding only zeros.

    A clipped window does not truncate a value here, it scribbles on the tape, so this is
    the test that says the schedule is sound at these widths rather than merely cheaper.
    """
    iters = 2 * p.bit_length()
    for x in range(1, p):
        circ = _build(p, iters, layout, narrow=True)
        res = simulate(circ, {"x": x, "d": 0}, rng=random.Random(p * 7717 + x))
        assert res.phase == 0
        assert res.outputs["d"] == _expect(p, x, iters, layout), f"p={p} x={x} {layout}"


@pytest.mark.parametrize("p", [29, 61, 251])
def test_a_run_that_did_not_converge_frees_a_dirty_qubit(p: int):
    """The iteration budget's failure mode has to be loud, exactly as at jump 1.

    With too few iterations `u != 1` at the end and `to_dialog_j2` frees a qubit that is
    not |0>, which the simulator raises on. If this ever silently returns a wrong tape the
    budget stops being a declared failure rate and becomes a silent one.
    """
    from qsm.core.simulator import DirtyAncillaError

    short = 2                              # nowhere near enough
    x = next(x for x in range(1, p) if to_dialog_jump(p, x, short)[1] != 1)
    circ = _build(p, short, "flat", narrow=False)
    with pytest.raises(DirtyAncillaError):
        simulate(circ, {"x": x, "d": 0}, rng=random.Random(1))


# --- the boundary iteration -------------------------------------------------
@pytest.mark.parametrize("p", [29, 61, 251, 1021])
def test_iteration_zero_needs_no_comparator_because_u_is_the_prime(p: int):
    """`b1 = b0 & (u > v)` and `u = p > v` for every `x < p`, so `b1` IS `b0`.

    One comparator saved out of 261, and -- more usefully -- one state removed from the
    boundary alphabet, which is what lets iteration 0 share the uniform base-5 digit
    instead of widening the first group. Checked over every `x`, at four primes, on the
    reference; the circuit's own use of it is checked by the exhaustive tests above.
    """
    iters = 2 * p.bit_length()
    for x in range(1, p):
        s, b0, b1 = to_dialog_jump(p, x, iters)[0][0]
        assert b1 == b0, f"p={p} x={x}: iteration 0 gave b1={b1}, b0={b0}"
        assert s <= 2


def test_the_boundary_costs_one_extra_shift_ladder_and_saves_one_comparator():
    """Iteration 0 pays a full-width ladder that no other iteration pays.

    Its first halving is conditional and cannot drop the top Fredkin (`x`'s top bit is
    live), so it is `w-1` Toffoli against `w-2`, on top of a second ladder that every
    iteration has. Set against the comparator it does not emit, it is a net saving at any
    comparator width above ~2 bits -- which is every width this design uses.
    """
    n, k = 32, schedule.compare_bits(32)
    one = count(_iter_circuit(n, iters=1, cmp_bits=k)).toffoli
    two = count(_iter_circuit(n, iters=2, cmp_bits=k)).toffoli
    three = count(_iter_circuit(n, iters=3, cmp_bits=k)).toffoli
    steady = three - two
    assert two - one == steady, "iterations past the boundary must all cost the same"
    assert one < steady, (
        f"boundary iteration {one} should be cheaper than a steady one {steady}: "
        "it trades a comparator for a shift ladder"
    )


def _iter_circuit(n: int, iters: int, cmp_bits: int | None = None) -> Circuit:
    """`iters` jump-2 iterations at full width `n`, tape in a flat register."""
    circ = Circuit()
    u = circ.input("u", n)
    v = circ.input("v", n)
    d = circ.input("d", 3 * iters)
    gcd.gcd_loop_j2(circ, u, v, gcd.FlatTapeJ2(d), iters=iters, cmp_bits=cmp_bits)
    return circ


def _iter_circuit_j1(n: int, iters: int, cmp_bits: int | None = None) -> Circuit:
    circ = Circuit()
    u = circ.input("u", n)
    v = circ.input("v", n)
    d = circ.input("d", 2 * iters)
    gcd.gcd_loop(circ, u, v, gcd.FlatTape(d), iters=iters, cmp_bits=cmp_bits)
    return circ


# --- what jump 2 actually buys ----------------------------------------------
def _steady(fn, n, k):
    """Toffoli for ONE mid-loop iteration: the difference of two runs, boundary removed."""
    return count(fn(n, 31, k)).toffoli - count(fn(n, 30, k)).toffoli


@pytest.mark.parametrize("n", [32, 64, 128, 255])
@pytest.mark.parametrize("k", [5, 20, None])
def test_the_per_iteration_cost_is_exactly_3w_plus_k_at_jump_one_and_one_ladder_more_at_two(
    n: int, k: int | None
):
    """Both cost lines, fitted with no residual -- which is what makes the ratio quotable.

    `3w` is one cswap plus a controlled subtract, and the subtract is `2w` because a
    controlled Gidney add is two Toffoli per bit. Getting that wrong is what made an
    earlier draft conclude the `w` terms cancelled between the jumps; they do not, and the
    consequence was a 4-8% estimate of a win that is really 11-17%. Pinning the closed
    form here means the next person who reasons about jump 3 starts from a checked line.
    """
    w, kk = n, n if k is None else min(k, n)
    assert _steady(_iter_circuit_j1, n, k) == 3 * w + kk + 1
    assert _steady(_iter_circuit, n, k) == 4 * w + kk - 1


@pytest.mark.parametrize("n", [32, 64, 128, 255])
@pytest.mark.parametrize("k", [5, 20, None])
def test_jump_two_wins_on_gates_for_every_width_and_comparator_not_just_wide_ones(
    n: int, k: int | None
):
    """The win does not depend on the comparator, and is nothing like the 261/399 ratio.

    Dropping one iteration in three saves a whole `3w + k + 1`; the ladder that buys it
    costs `w - 2`. The inequality reduces to `0 < w + k + 5`, so there is no width or
    comparator width at which jump 2 loses -- including `k = 5`, where the comparator is
    almost free and an "the saving IS the comparator" story would predict a loss.
    """
    s1, s2 = _steady(_iter_circuit_j1, n, k), _steady(_iter_circuit, n, k)
    ratio = (s2 * (2 / 3)) / s1                 # jump 2 needs 2/3 as many iterations
    assert ratio < 1.0, f"n={n} k={k}: jump-2 costs {ratio:.4f} of jump-1 per unit progress"
    assert 0.82 < ratio < 0.89, f"n={n} k={k}: ratio {ratio:.4f} outside the measured band"
    assert 261 / 399 < ratio, "the iteration ratio is NOT the gate ratio; do not quote it"

    w, kk = n, n if k is None else min(k, n)
    assert abs(ratio - (8 * w + 2 * kk - 2) / (9 * w + 3 * kk + 3)) < 1e-12


def test_the_tape_is_609_qubits_against_665_which_is_the_bigger_prize():
    """...but the 56 is a claim about the REPLAY, and Phase 1 is not evidence for it.

    Phase 1's peak sits early in the loop, at full window width and an almost empty tape,
    so shrinking the tape moves it by 5 qubits (1112 packed against 1107) rather than 56.
    The replay is where the whole tape is live at once beside both operands, and that is
    the only place `d + a + b` is the peak. Left unverified here on purpose.
    """
    assert schedule.iterations_j2(255) == 261
    assert pack.packed_bits_j2(261) == schedule.dialog_bits_j2(iters=261) == 609
    assert schedule.dialog_bits(iters=schedule.iterations(255)) == 665
    assert 609 + 2 * 255 == 1119, "56 under PEAK_BUDGET, where jump 1 sits at exactly 1175"
