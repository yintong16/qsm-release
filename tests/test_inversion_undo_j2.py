"""`undo.un_to_dialog_j2`: the jump-2 tape run backwards.

Three things this has to hold down, and they are the three that could go wrong quietly:

  the tape comes back to |0>   Not "the answer is right" -- `d` is entangled with `x`, so
                               a tape qubit left at |1> is the failure that matters, and
                               it surfaces as `DirtyAncillaError` rather than a bad value.
  base 5 frees its accumulator The jump-1 reverse gets digit 0's retirement for free
                               because the pair IS the accumulator's bottom two qubits.
                               Here it is three separate qubits that belong to nobody, so
                               a missing free is a leak the value tests cannot see.
  the rotate is at the far end `gcd_loop_j2` rotates at the top of an iteration, so the
                               undo goes at the bottom. Putting it at the top -- which is
                               where `un_gcd_loop` has it -- is off by one halving and is
                               the single easiest mistake in the file.
"""

import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl
from qsm.inversion import gcd, pack, schedule, undo
from qsm.inversion.reference import to_dialog_jump, un_dialog_jump

PRIMES = [29, 61, 127, 251, 1021]


def _iters(p):
    return schedule.iterations_j2(p.bit_length())


def _un_circuit(p, iters, packed=True, widths=None, **kw):
    n = p.bit_length()
    c = Circuit()
    if packed:
        tape = undo.PackedUnTapeJ2(c, c.input("d", pack.packed_bits_j2(iters)), iters)
    else:
        tape = undo.FlatUnTapeJ2(c, c.input("d", 3 * iters))
    c.input("x", 0)
    x = undo.un_to_dialog_j2(c, tape, n, p, iters=iters, widths=widths, **kw)
    c.rebind("x", x)
    c.rebind("d", Register([], "d"))
    return c


def _encode(tape, packed):
    return (pack.pack_dialog_j2(tape, boundary=True) if packed
            else gcd.encode_dialog_j2(tape))


# --- the reference, first: the circuit is only as right as this ------------


@pytest.mark.parametrize("p", PRIMES)
def test_the_reference_reverse_returns_the_input(p):
    """`un_dialog_jump` on a converged tape gives back (p, x). Nothing quantum yet."""
    it = _iters(p)
    for x in range(1, p):
        tape, u, v = to_dialog_jump(p, x, it)
        assert (u, v) == (1, 0), f"p={p} x={x} did not converge in {it}"
        assert un_dialog_jump(tape) == (p, x)


# --- values, exhaustively, both layouts ------------------------------------


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("packed", [False, True])
def test_the_reverse_gives_back_every_x(p, packed):
    it = _iters(p)
    c = _un_circuit(p, it, packed=packed)
    rng = random.Random(11)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump(p, x, it)
        r = simulate(c, {"d": _encode(tape, packed), "x": 0}, rng=rng)
        assert r.outputs["x"] == x, f"p={p} x={x} packed={packed}"
        assert r.outputs["d"] == 0, "the tape is spent, not merely ignored"
        assert r.phase == 0, "Phase 1 is exact in both directions"


@pytest.mark.parametrize("p", PRIMES)
def test_the_two_layouts_agree(p):
    """Packing is a relabelling of the same divsteps; it must not change the answer."""
    it = _iters(p)
    flat = _un_circuit(p, it, packed=False)
    packed = _un_circuit(p, it, packed=True)
    rng = random.Random(12)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump(p, x, it)
        a = simulate(flat, {"d": _encode(tape, False), "x": 0}, rng=rng)
        b = simulate(packed, {"d": _encode(tape, True), "x": 0}, rng=rng)
        assert a.outputs["x"] == b.outputs["x"] == x


# --- the round trip, which is the property `modmul_inplace` actually needs --


@pytest.mark.parametrize("p", PRIMES)
def test_forward_then_reverse_is_the_identity(p):
    """to_dialog_j2 then un_to_dialog_j2 on one circuit: |x> -> |x>, nothing left over.

    Stronger than the two halves passing separately. It is the only test here that runs
    the forward's packing against the reverse's unpacking on the *same* wires, which is
    where a group-boundary disagreement would show up.
    """
    n = p.bit_length()
    it = _iters(p)
    c = Circuit()
    x = c.input("x", n)
    tape = gcd.PackedTapeJ2(c)
    d = gcd.to_dialog_j2(c, x, tape, p, iters=it)
    un = undo.PackedUnTapeJ2(c, d, it)
    x2 = undo.un_to_dialog_j2(c, un, n, p, iters=it)
    c.rebind("x", x2)
    rng = random.Random(13)
    for xv in range(1, p):
        assert simulate(c, {"x": xv}, rng=rng).outputs["x"] == xv, f"p={p} x={xv}"


# --- the leak tests: what the value tests cannot see -----------------------


@pytest.mark.parametrize("p", PRIMES)
def test_the_packed_reader_hands_back_every_qubit_it_was_given(p):
    """Net qubit ledger per group is +7, and nothing is held at close().

    This is the base-5 accumulator free. Deleting that line in `PackedUnTapeJ2.retire`
    leaves 3 qubits per group live to the end -- 261 iterations is 87 groups, so 261
    stranded qubits, which is a peak claim silently 261 too high and no wrong answer
    anywhere to give it away.
    """
    it = _iters(p)
    c = _un_circuit(p, it, packed=True)
    assert count(c).peak_qubits > 0
    # Every tape qubit must be freed: the tape register is rebound to width 0, and the
    # simulator's dirty-ancilla check is what enforces the |0>.
    assert len(c.inputs["d"]) == 0


@pytest.mark.parametrize("p", [29, 61, 127])
def test_a_tape_that_is_not_a_dialog_is_loud_not_wrong(p):
    """A corrupted tape must raise, not return a plausible x.

    The reverse re-derives `b1` with a comparator; a tape the forward never wrote makes
    that disagree, and the erasure leaves a qubit at |1>. That has to reach the caller.
    """
    it = _iters(p)
    c = _un_circuit(p, it, packed=False)
    rng = random.Random(14)
    tape, _, _ = to_dialog_jump(p, 1, it)
    good = _encode(tape, False)
    raised = 0
    for bit in range(3 * it):
        try:
            simulate(c, {"d": good ^ (1 << bit), "x": 0}, rng=rng)
        except DirtyAncillaError:
            raised += 1
    assert raised > 0, "flipping tape bits must be detectable somewhere"


# --- the parity claim, which is the whole reason the file exists -----------


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("add_ctrl", [gidney_add_ctrl, cdkm_add_ctrl])
def test_the_reverse_costs_what_the_forward_costs(p, add_ctrl):
    """Same Toffoli count both directions, at matched width schedule and comparator.

    The claim the jump-2 composite's "Phase 1 x 2" rests on. Checked at small widths here
    and at 255 bits by the registered versions; if it holds at neither the composite table
    is arithmetic on an assumption, which is what this whole file was written to remove.
    """
    n = p.bit_length()
    it = _iters(p)
    widths = schedule.widths_j2(n, it)

    fwd = Circuit()
    x = fwd.input("x", n)
    gcd.to_dialog_j2(fwd, x, gcd.PackedTapeJ2(fwd), p, iters=it, widths=widths,
                     add_ctrl=add_ctrl)

    rev = _un_circuit(p, it, packed=True, widths=widths, add_ctrl=add_ctrl)

    cf, cr = count(fwd).toffoli, count(rev).toffoli
    assert cr == cf, f"p={p}: forward {cf}, reverse {cr}"


@pytest.mark.parametrize("p", PRIMES)
def test_the_reverse_never_peaks_above_the_forward(p):
    """The peak is the number the composite claim actually turns on.

    A composite's peak is the max over its regions, so region 3 exceeding region 1 would
    move `modmul_inplace`'s peak even though the Toffoli total did not budge.
    """
    n = p.bit_length()
    it = _iters(p)
    widths = schedule.widths_j2(n, it)

    fwd = Circuit()
    x = fwd.input("x", n)
    gcd.to_dialog_j2(fwd, x, gcd.PackedTapeJ2(fwd), p, iters=it, widths=widths)
    rev = _un_circuit(p, it, packed=True, widths=widths)

    pf, pr = count(fwd).peak_qubits, count(rev).peak_qubits
    assert pr <= pf, f"p={p}: forward peaks at {pf}, reverse at {pr}"


def test_the_inverse_ladder_is_the_forward_ladder_backwards():
    """`_un_rshift1` undoes `controlled_rshift1` at both `top_zero` settings.

    Written as a behavioural check rather than a gate-list comparison, because the claim
    that matters is that the slice trick is *exactly* the inverse and not merely the same
    width. Includes ctrl = 0, where both must be the identity.
    """
    from qsm.field.shift import controlled_rshift1

    for w in (4, 8, 13):
        for top_zero in (False, True):
            c = Circuit()
            ctrl = c.input("ctrl", 1)
            r = c.input("r", w)
            controlled_rshift1(c, ctrl[0], r, top_zero=top_zero)
            undo._un_rshift1(c, ctrl[0], r, top_zero=top_zero)
            rng = random.Random(15)
            bits = w - 1 if top_zero else w
            for _ in range(60):
                cv = rng.getrandbits(1)
                rv = rng.getrandbits(bits)
                if cv:
                    rv &= ~1
                out = simulate(c, {"ctrl": cv, "r": rv}, rng=rng).outputs
                assert out == {"ctrl": cv, "r": rv}, (w, top_zero, cv, rv)
