"""The division: Phase 2 at jump 2, run backwards.

Everything `test_inversion_unreplay.py` establishes about reversing a replay holds here
unchanged, so those arguments are not repeated. What is new is that jump 2's extra
structure -- the conditional doubling -- has to reverse too, and the whole question of
whether jump 2 is still a win backwards turns on what that costs.

It costs nothing. The surcharge for reversing is `2` Toffoli per ADDITION, and jump 2's
entire point is that it does 261 additions instead of 399, so **the division is its
multiply plus 522 rather than plus 798** -- the reversal got cheaper by exactly the amount
the jump saved. The conditional shift ladder contributes zero in both directions, because
`moddouble_ctrl_inplace` erases its reduction flag with the output's PARITY and a Toffoli
rather than with a measurement. Had it measured, every one of the 261 ladders would have
needed a comparator to reverse and jump 2 would have been a worse trade backwards than
forwards. `test_field_modhalve.py` pins the gate-level half of that; this file pins that
it survives 261 iterations of composition.

The ordering inside `halve^s` looks like it ought to matter and does not -- the gated halve
IS the ungated one when its control fires and the identity when it does not, so the two
compose either way round. The ordering that does matter is the cswap against the subtract,
which is the reverse of the forward's add-then-swap and is wrong on every input if written
the other way. One test covers both halves of that.

Correctness is checked against `un_apply_dialog_jump_circuit` rather than the mathematical
inverse, for the same reason the forward tests use `apply_dialog_jump_circuit`: the loop
runs lazily and leaves the congruence class at test widths. Exhaustive identity tests run
at f = 1 primes (31, 127), where the lazy domain closes and there is nothing to excuse.
"""

from __future__ import annotations

import functools
import random

import pytest

from qsm import params
from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl
from qsm.field.add_chunked import gidney_add_ctrl_chunked
from qsm.field.modarith import modhalve, modhalve_ctrl_inplace, modsub_lazy_ctrl
from qsm.field.swap import cswap
from qsm.inversion import pack, replay, schedule, unreplay
from qsm.inversion.gcd import encode_dialog_j2
from qsm.inversion.reference import (
    apply_dialog_jump_circuit,
    apply_dialog_jump_inverse,
    to_dialog_jump as to_dialog_jump_ref,
    un_apply_dialog_jump_circuit,
)

#: The same test widths the jump-2 forward uses.
PRIMES = [29, 61, 127, 251]
#: f = 2^n - p = 1: the lazy domain closes exactly, so the reverse is a true inverse over
#: the whole input space and an identity test needs no excuses. See the jump-1 file.
CLOSED = [31, 127]

_chunked_vent = functools.partial(gidney_add_ctrl_chunked, chunk=3, cmp_bits=3)
_chunked_vent.__name__ = "gidney_add_ctrl_chunked3_vent"      # type: ignore[attr-defined]

ADDERS = [gidney_add_ctrl, cdkm_add_ctrl, _chunked_vent]


def _iters(p: int) -> int:
    return schedule.iterations_j2(p.bit_length())


def _encode(tape: list[tuple[int, int, int]], packed: bool) -> int:
    if packed:
        return pack.pack_dialog_j2(tape, boundary=True)
    return encode_dialog_j2(tape)


def _un_circuit(p: int, iters: int, packed: bool = True, **kw) -> Circuit:
    n = p.bit_length()
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(iters) if packed else 3 * iters)
    a, b = c.input("a", n), c.input("b", n)
    a2, b2 = unreplay.un_apply_dialog_j2(c, d, a, b, iters, p, packed=packed, **kw)
    c.rebind("a", a2)
    c.rebind("b", b2)          # the unconditional modhalve relabels b once per iteration
    return c


def _round_trip_circuit(p: int, iters: int, **kw) -> Circuit:
    n = p.bit_length()
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(iters))
    a, b = c.input("a", n), c.input("b", n)
    a, b = replay.apply_dialog_j2(c, d, a, b, iters, p, **kw)
    a, b = unreplay.un_apply_dialog_j2(c, d, a, b, iters, p, **kw)
    c.rebind("a", a)
    c.rebind("b", b)
    return c


# --- values ----------------------------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "flat"])
def test_the_reverse_matches_the_circuit_reference_exhaustively(p: int, packed: bool):
    """Every jump-2 tape from every x, undone from the forward's own output.

    Driven from the image and not from arbitrary (a, b): the reverse is only total there,
    because `modsub_lazy_ctrl`'s re-derived flag borrows out of the register on a `b` below
    F. The reference raises `ValueError` at exactly those points and both counts are kept,
    so the excused set cannot quietly grow into the whole test.
    """
    n, it = p.bit_length(), _iters(p)
    c = _un_circuit(p, it, packed=packed)
    checked = excused = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        d = _encode(tape, packed)
        for y in (0, 1, 2, x, p - 1):
            av, bv = apply_dialog_jump_circuit(tape, y, 0, p, n)
            try:
                want = un_apply_dialog_jump_circuit(tape, av, bv, p, n)
            except ValueError:
                excused += 1
                continue
            r = simulate(c, {"d": d, "a": av, "b": bv}, rng=random.Random(x * 71 + y))
            assert (r.outputs["a"], r.outputs["b"]) == want, f"p={p} x={x} y={y}"
            assert r.outputs["d"] == d, f"p={p} x={x}: the tape did not come back"
            checked += 1
    assert checked > 5 * excused, f"p={p}: {excused} excused against {checked} checked"


@pytest.mark.parametrize("p", CLOSED)
def test_the_replay_and_its_reverse_compose_to_the_identity(p: int):
    """The whole point, exhaustively, where the lazy domain closes.

    `d` comes back too, which is the precondition for anything downstream: a composite has
    to run Phase 1 backwards afterwards, and can only do that if Phase 2 left the tape
    alone in both directions.
    """
    n, it = p.bit_length(), _iters(p)
    c = _round_trip_circuit(p, it)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        d = pack.pack_dialog_j2(tape, boundary=True)
        for y in range(p):
            r = simulate(c, {"d": d, "a": y, "b": 0}, rng=random.Random(x * 31 + y))
            assert (r.outputs["a"], r.outputs["b"]) == (y, 0), f"p={p} x={x} y={y}"
            assert r.outputs["d"] == d
            assert r.phase == 0, "an exact round trip must not leak phase"


@pytest.mark.parametrize("p", CLOSED)
def test_the_whole_tape_reversed_is_the_divide(p: int):
    """(p, z) -> (z/x mod p, 0), at the jump the design actually runs.

    This is the operation the point layer needs and the one that did not exist before: the
    registered division was only ever built at jump 1, and no version of Phase 2 in the
    design runs at jump 1. The emptied register again comes back holding the LAZY zero `p`
    rather than `|0>` -- the loop never canonicalises -- so a composite must consume it
    with the reverse of Phase 1 rather than free it.
    """
    n, it = p.bit_length(), _iters(p)
    c = _un_circuit(p, it)
    for x in range(1, p):
        tape, u, v = to_dialog_jump_ref(p, x, it)
        assert (u, v) == (1, 0), f"p={p} x={x}: the tape did not converge"
        d = pack.pack_dialog_j2(tape, boundary=True)
        for z in range(1, p):
            r = simulate(c, {"d": d, "a": p, "b": z}, rng=random.Random(x * 17 + z))
            assert r.outputs["a"] == (z * pow(x, -1, p)) % p, f"p={p} x={x} z={z}"
            assert r.outputs["b"] == 0


@pytest.mark.parametrize("p", PRIMES)
def test_the_reference_inverse_is_the_congruent_division(p: int):
    """`apply_dialog_jump_inverse` really divides, before any circuit is involved.

    The circuit is checked against `un_apply_dialog_jump_circuit`, which is a statement
    about gates and wraps; this is the statement about arithmetic that the gate-level
    reference is supposed to be a lazy realisation of. Both are needed -- agreeing with a
    reference that computed the wrong function would prove nothing.
    """
    it = _iters(p)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        for y in range(p):
            a, b = apply_dialog_jump_inverse(tape, 0, (x * y) % p, p)
            assert (a, b) == (y, 0), f"p={p} x={x} y={y}: got ({a}, {b})"


@pytest.mark.parametrize("p", PRIMES)
def test_the_two_readers_compute_the_same_circuit(p: int):
    """Packing is a storage decision going backwards too, base 5 included."""
    n, it = p.bit_length(), _iters(p)
    flat, packed = _un_circuit(p, it, packed=False), _un_circuit(p, it, packed=True)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        for y in (1, x, p - 1):
            av, bv = apply_dialog_jump_circuit(tape, y, 0, p, n)
            try:
                un_apply_dialog_jump_circuit(tape, av, bv, p, n)
            except ValueError:
                continue
            got = [
                simulate(cc, {"d": _encode(tape, pk), "a": av, "b": bv},
                         rng=random.Random(x * 13 + y)).outputs
                for cc, pk in ((flat, False), (packed, True))
            ]
            assert got[0]["a"] == got[1]["a"], f"p={p} x={x} y={y}"
            assert got[0]["b"] == got[1]["b"], f"p={p} x={x} y={y}"


@pytest.mark.parametrize("p", CLOSED)
@pytest.mark.parametrize("add_ctrl", ADDERS, ids=lambda f: f.__name__)
def test_any_controlled_adder_drives_the_reverse(p: int, add_ctrl):
    """The loop does not depend on which adder it is given, only the cost does.

    The chunked adder is included because its boundary vent is the one approximation in the
    set that is never exact at any comparison width, and a reversal built on it still has to
    be an inverse of the forward built on it.
    """
    it = _iters(p)
    c = _round_trip_circuit(p, it, add_ctrl_fn=add_ctrl)
    for x in range(1, p, max(1, p // 8)):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        r = simulate(c, {"d": pack.pack_dialog_j2(tape, boundary=True), "a": 1, "b": 0},
                     rng=random.Random(x))
        assert (r.outputs["a"], r.outputs["b"]) == (1, 0)


@pytest.mark.parametrize("p", CLOSED)
def test_the_cswap_must_precede_the_subtract_though_the_halvings_may_be_in_any_order(
    p: int,
):
    """Which of the reverse loop's orderings is load-bearing, and which is free.

    Reversing a loop means reversing its body, and the body has three parts. Only one of
    the orderings actually constrains anything, and it is worth knowing which before
    someone rearranges the loop for readability:

    The two halvings **commute, trivially and at every width.** `modhalve_ctrl_inplace`
    with its control set is not merely equivalent to `modhalve` -- it is the same lazy
    function, gate for gate -- and with its control clear it is the identity. So the pair is
    either `h.h` or `h`, whichever order they are written in. There is nothing to get wrong
    and no test can be written that would catch it; the ordering here is chosen to mirror
    the forward's, not because it has to be.

    The **cswap against the subtract does not commute**, and that is the whole ordering
    content of the reverse. Forwards the loop adds and then swaps, so backwards it must
    swap and then subtract; writing them the other way round is wrong on roughly a third of
    all inputs -- measured 90 of 90 at p = 31 and 378 of 378 at p = 127 -- because the
    subtract would be reading the operands through the wrong labels.

    That it fails on *every* input at these widths is what makes this cheap to check. It is
    also why nothing upstream would have caught a subtler version of the same mistake: both
    orders are phase-clean, ancilla-clean and cost identical gates.
    """
    n, it = p.bit_length(), _iters(p)

    def build(swapped: bool) -> Circuit:
        c = Circuit()
        d = c.input("d", pack.packed_bits_j2(it))
        a, b = c.input("a", n), c.input("b", n)
        a, b = replay.apply_dialog_j2(c, d, a, b, it, p)
        reader = unreplay.GroupReaderJ2(c, d, it)
        for i in range(it):
            x, y, z = reader.triple(i)
            if i:
                cc, b0, b1 = x, y, z
                modhalve_ctrl_inplace(c, cc, b, p)
                b = modhalve(c, b, p)
            else:
                s0, s1, b0 = x, y, z
                b1 = b0                          # iteration 0 swaps on its only bit
                modhalve_ctrl_inplace(c, s1, b, p)
                modhalve_ctrl_inplace(c, s0, b, p)
            if swapped:
                modsub_lazy_ctrl(c, b0, a, b, p)
                cswap(c, b1, a, b)
            else:
                cswap(c, b1, a, b)
                modsub_lazy_ctrl(c, b0, a, b, p)
            reader.done(i)
        reader.close()
        c.rebind("a", a)
        c.rebind("b", b)
        return c

    right, wrong = build(False), build(True)
    checked = bad = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        d = pack.pack_dialog_j2(tape, boundary=True)
        for y in (1, 2, p - 1):
            args = {"d": d, "a": y, "b": 0}
            rg = simulate(right, args, rng=random.Random(x * 7 + y))
            assert (rg.outputs["a"], rg.outputs["b"]) == (y, 0), f"p={p} x={x} y={y}"
            checked += 1
            try:
                rw = simulate(wrong, args, rng=random.Random(x * 7 + y))
            except DirtyAncillaError:
                bad += 1
                continue
            bad += (rw.outputs["a"], rw.outputs["b"]) != (y, 0)
    assert bad == checked, \
        f"p={p}: the swapped order was right on {checked - bad} of {checked} inputs"


# --- cost ------------------------------------------------------------------
@pytest.mark.parametrize("add_ctrl", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("p", PRIMES)
def test_the_reverse_costs_two_toffoli_per_ITERATION_more(p: int, add_ctrl):
    """+2 per iteration, and at jump 2 an iteration is an addition -- so 522, not 798.

    The capitalisation is the whole content of the test. The surcharge rides on the
    comparator inside `modsub_lazy_ctrl`, which runs once per addition, and jump 2's
    iteration count IS its addition count. Everything else reverses free:

        cswap                       its own inverse
        moddouble  -> modhalve      the parity read the other way, one CX
        moddouble_ctrl_inplace
                   -> modhalve_ctrl_inplace
                                    the same parity, gated with a Toffoli both ways

    The third line is what had to be built for this file to exist, and it is the line that
    could have gone the other way: a gated forward that erased by measurement would have
    cost a comparator per ladder to reverse, i.e. `2 * 261` MORE on top, and jump 2 would
    have been worth less backwards than forwards.

    Width costs at most one qubit, and often nothing. The materialised flag is live only
    inside the subtraction, so it raises the global peak only when the subtraction is the
    peak-setting region; `test_the_extra_qubit_is_the_comparators_and_it_usually_hides`
    pins which of the two happens and why.
    """
    n, it = p.bit_length(), _iters(p)

    def one(build):
        c = Circuit()
        d = c.input("d", pack.packed_bits_j2(it))
        a, b = c.input("a", n), c.input("b", n)
        build(c, d, a, b, it, p, add_ctrl_fn=add_ctrl)
        return count(c)

    fwd, rev = one(replay.apply_dialog_j2), one(unreplay.un_apply_dialog_j2)
    assert rev.toffoli - fwd.toffoli == 2 * it, \
        f"p={p}: forward {fwd.toffoli}, reverse {rev.toffoli}, {it} iterations"
    assert rev.peak_qubits - fwd.peak_qubits in (0, 1), \
        f"p={p}: forward peaks at {fwd.peak_qubits}, reverse at {rev.peak_qubits}"


def test_the_extra_qubit_is_the_comparators_and_it_usually_hides():
    """Where the reverse's one extra live qubit comes from, and when it costs anything.

    `cmp_lt_phase` writes its answer into the phase and needs no qubit to hold it;
    `cmp_lt`, which the reverse must call because it has to re-derive a predicate the
    forward measured away, writes it onto a wire. So the reverse is exactly one qubit wider
    **during the subtraction**, and nowhere else.

    Whether that surfaces depends entirely on whether the subtraction is the region that
    sets the peak, and the comparison width is the lever. Measured at 255 bits with CDKM,
    whose adder allocates almost nothing and so leaves the comparator exposed:

        exact comparator (k = 255)   1379 -> 1380    the comparator IS the peak
        k = 40                       1379 -> 1379    it is not

    Every registered operating point truncates, so the division is free in width at all six
    of them -- which is the only reason a point addition can schedule it beside anything
    else. It is a fact about the tuning and not about the construction, and it would come
    back the moment someone set `cmp_bits=None` to chase the failure rate down.
    """
    n, it = params.U, schedule.iterations_j2(params.U)

    def peak(build, **kw):
        c = Circuit()
        d = c.input("d", pack.packed_bits_j2(it))
        a, b = c.input("a", n), c.input("b", n)
        build(c, d, a, b, it, params.PRIME, add_ctrl_fn=cdkm_add_ctrl, **kw)
        return count(c).peak_qubits

    exact = (peak(replay.apply_dialog_j2), peak(unreplay.un_apply_dialog_j2))
    assert exact[1] - exact[0] == 1, f"exact comparator: {exact}"

    k = {"cmp_bits": params.CMP_BITS}
    trunc = (peak(replay.apply_dialog_j2, **k), peak(unreplay.un_apply_dialog_j2, **k))
    assert trunc[1] == trunc[0], f"k = {params.CMP_BITS}: {trunc}"


@pytest.mark.parametrize(
    "fwd,rev",
    [("inv.apply_dialog_j2.v1",
      "inv.un_apply_dialog_j2.v1"),
     ("inv.apply_dialog_j2_trunc.v1",
      "inv.un_apply_dialog_j2_trunc.v1"),
     ("inv.apply_dialog_j2_cdkm_trunc.v1",
      "inv.un_apply_dialog_j2_cdkm_trunc.v1"),
     ("inv.apply_dialog_j2_chunked_trunc.v1",
      "inv.un_apply_dialog_j2_chunked_trunc.v1"),
     ("inv.apply_dialog_j2_threaded_tuned_trunc.v1",
      "inv.un_apply_dialog_j2_threaded_tuned_trunc.v1"),
     ("inv.apply_dialog_j2_threaded_tuned_narrow_trunc.v1",
      "inv.un_apply_dialog_j2_threaded_tuned_narrow_trunc.v1")],
)
def test_the_division_costs_the_multiply_plus_522_at_255_bits(fwd: str, rev: str):
    """The claim at real width, on every registered pair.

    522 is 2 x 261 and it is 0.14% of the settled Phase 2's 373,484 -- so for a composite
    the division and the in-place multiply are the same price to three significant figures,
    which is what both baselines report and why the design never builds a separate
    inverter. The jump-1 file makes the same claim with 798; the difference between the two
    numbers is 276 Toffoli of pure jump-2 dividend that only shows up backwards.
    """
    from qsm.bench import registry

    f = count(registry.get_builder(fwd)().circuit)
    r = count(registry.get_builder(rev)().circuit)
    assert r.toffoli - f.toffoli == 2 * schedule.iterations_j2(params.U)
    assert r.peak_qubits == f.peak_qubits


def test_the_settled_division_fits_the_peak_budget():
    """The result the whole exercise was for, stated as a number.

    `inv.un_apply_dialog_j2_threaded_tuned_trunc.v1` is the reverse of the settled Phase 2,
    and it fits `PEAK_BUDGET` at 1,173 -- so the design has a division that a point addition
    could actually schedule. No jump-1 reverse does, and none can: `d + a + b` alone is
    1,175 there.
    """
    from qsm.bench import registry

    r = count(registry.get_builder(
        "inv.un_apply_dialog_j2_threaded_tuned_trunc.v1")().circuit)
    assert r.peak_qubits <= params.PEAK_BUDGET, f"peak {r.peak_qubits}"
    assert r.toffoli == 374006, f"toffoli {r.toffoli}"


# --- the reader ------------------------------------------------------------
@pytest.mark.parametrize("iters", [1, 2, 3, 12, 13, 14, 18, 261])
def test_the_group_reader_costs_exactly_what_the_packed_reader_costs(iters: int):
    """Reading the base-5 tape forwards is free, as reading the base-3 one was.

    The argument is tighter here than at jump 1. `PackedReaderJ2` folds only at `j == 0`,
    which is the END of its backwards pass through a group -- so it too holds all three
    triples across the group, and the two readers issue an identical sequence of takes,
    unfolds, releases and folds. Only the order the triples are handed out in differs, and
    order costs no gates.

    Both carry 5 working qubits against the base-3 readers' 3, which is the same two-qubit
    tax `PackedReaderJ2` documents; it comes off the 56 the shorter tape saves, which is
    the entire qubit case for jump 2.
    """
    def measure(cls, order):
        c = Circuit()
        d = c.input("d", pack.packed_bits_j2(iters))
        r = cls(c, d, iters)
        for i in order:
            r.triple(i)
            r.done(i)
        fresh = len(r._fresh)
        r.close()
        res = count(c)
        return fresh, res.peak_qubits - len(d), res.toffoli

    assert measure(unreplay.GroupReaderJ2, range(iters)) == \
        measure(replay.PackedReaderJ2, reversed(range(iters)))


@pytest.mark.parametrize("iters", [12, 13, 14])
def test_the_group_reader_returns_the_tape_untouched(iters: int):
    """Unfold-then-refold is the identity on |d>, boundary digit included.

    Iteration 0 uses a different alphabet from the rest -- four reachable `(s0, s1, b0)`
    states instead of the five `(c, b0, b1)` ones -- so the reader has to agree with the
    codec about which digit is which, in this direction as in the other.
    """
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(iters))
    reader = unreplay.GroupReaderJ2(c, d, iters)
    for i in range(iters):
        reader.triple(i)
        reader.done(i)
    reader.close()
    for x in range(1, 40, 3):
        tape, _, _ = to_dialog_jump_ref(251, x, iters)
        v = pack.pack_dialog_j2(tape, boundary=True)
        r = simulate(c, {"d": v}, rng=random.Random(x))
        assert r.outputs["d"] == v
        assert r.phase == 0


# --- the analytic model ----------------------------------------------------
def test_un_replay_j2_fail_prob_is_the_forwards_number():
    """Same terms, same split, same total; the promotion is in kind, not in size.

    Including the split itself -- doubling terms on `2 * iters`, addition terms on `iters`
    -- because that is the thing a reversal could plausibly have disturbed and does not.
    """
    it = schedule.iterations_j2(params.U)
    for kw in ({}, {"cmp_bits": None}, {"double_window": None, "carry_window": None},
               {"chunk": 43, "chunk_cmp_bits": 32}):
        assert params.un_replay_j2_fail_prob(it, **kw) \
            == params.replay_j2_fail_prob(it, **kw)


def test_the_jump_two_division_is_more_accurate_than_the_jump_one_one():
    """Backwards as forwards: fewer additions is fewer chances to fail.

    The reversal preserves each term exactly, so whatever the forward gained the reverse
    keeps. Stated as an assertion because the reverse promotes two of those terms from
    phase to value, and a reader could reasonably wonder whether the promotion costs
    accuracy. It does not -- it costs only quietness.
    """
    assert params.un_replay_j2_fail_prob(schedule.iterations_j2(params.U)) \
        < params.un_replay_fail_prob(schedule.iterations(params.U))


@pytest.mark.parametrize("p", [29, 61])
@pytest.mark.parametrize("cmp_bits", [None, 3], ids=["exact", "trunc"])
def test_the_reverse_never_leaks_phase_it_raises_instead(p: int, cmp_bits):
    """Why the jump-2 reverses are not in `test_harness_phase_policy.PHASE_TOLERANT`.

    The forward earns its tolerance by erasing the addition's overflow with a measurement.
    Nothing in the reverse measures -- `modsub_lazy_ctrl` materialises the flag and frees
    it, and both halvings recover theirs from a parity -- so a disagreement strands an
    ancilla and the free raises rather than venting.

    The conditional ladder is the new thing to check here and it is on the right side of
    the line: its flag is a parity, recovered exactly at every width, so it neither vents
    nor strands. If it had been erased by measurement this test would fail on the phase
    count and `un_replay_j2_fail_prob` would need a term of its own.
    """
    n, it = p.bit_length(), _iters(p)
    c = _un_circuit(p, it, cmp_bits=cmp_bits)
    leaks = raises = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        d = pack.pack_dialog_j2(tape, boundary=True)
        for y in range(p):
            av, bv = apply_dialog_jump_circuit(tape, y, 0, p, n)
            try:
                r = simulate(c, {"d": d, "a": av, "b": bv}, rng=random.Random(x * 97 + y))
            except DirtyAncillaError:
                raises += 1
                continue
            leaks += r.phase != 0
    assert leaks == 0, f"p={p} k={cmp_bits}: {leaks} phase leaks"
    if cmp_bits is not None:
        assert raises > 0, "k = 3 must be narrow enough to strand the flag somewhere"


@pytest.mark.parametrize("p", CLOSED)
def test_a_reverse_built_with_a_different_window_fails_loudly(p: int):
    """The windows have to match, and mismatching them is not a silent wrong answer.

    Same statement as at jump 1, repeated here because jump 2 adds a second consumer of
    `double_window` per iteration and a mismatch there is a different failure from a
    mismatch in the comparator. The comparator is checked, because it is the one that has
    to re-derive a measured predicate; a `double_window` mismatch is a value error of the
    ordinary kind, priced by `moddouble_lazy_input_carry_fail_prob`.
    """
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(it))
    a, b = c.input("a", n), c.input("b", n)
    a, b = replay.apply_dialog_j2(c, d, a, b, it, p)
    a, b = unreplay.un_apply_dialog_j2(c, d, a, b, it, p, cmp_bits=1)
    c.rebind("a", a)
    c.rebind("b", b)

    loud = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        for y in (1, 2, p - 1):
            try:
                r = simulate(c, {"d": pack.pack_dialog_j2(tape, boundary=True),
                                 "a": y, "b": 0},
                             rng=random.Random(x * 7 + y))
            except DirtyAncillaError:
                loud += 1
                continue
            assert (r.outputs["a"], r.outputs["b"]) == (y, 0), \
                f"p={p} x={x} y={y}: a mismatched window returned a wrong value silently"
    assert loud > 0, "k = 1 must be narrow enough to disagree somewhere at this width"
