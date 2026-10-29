"""Phase 2 at jump 2: the Bezout replay of a three-signal tape.

Everything `test_inversion_replay.py` says about the lazy domain and the vented erasure
applies here unchanged, so those checks are repeated against the jump-2 references rather
than re-argued. What is new is the shape of the loop, and three things about it need
holding down rather than asserting in prose:

**Iteration 0 is different in kind.** Its two doublings are both conditional, because `x`
may have been even, and its `b1` is not a wire at all -- `u = p > v` outright at the top of
Phase 1, so `b1 = b0` and the replay's cswap reads the same qubit the addition did. The
tape codec has to agree with the loop about that on both sides.

**Base 5 has no free digit 0.** At jump 1 digit 0's pair *is* the accumulator's bottom two
qubits and the fold is a single CX. At jump 2 the three signal qubits are not the binary
form of any legal digit, so digit 0 folds arithmetically like the rest and the triple must
be disjoint from the accumulator. Getting that wrong is not a wrong answer, it is a
`DirtyAncillaError` -- which is how it was in fact found.

**The doubling count is invariant under the jump.** Jump 2 drops 138 additions, 138 swaps
and 138 comparators and drops exactly zero doublings, because the `s` sum to the divstep
count either way. That is a claim about the error budget as much as about the gates, and
it is checked in both places.
"""

from __future__ import annotations

import functools
import random

import pytest

from qsm import params
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl
from qsm.field.add_chunked import gidney_add_ctrl_chunked
from qsm.field.modarith import moddouble, moddouble_ctrl_inplace
from qsm.inversion import pack, replay, schedule
from qsm.inversion.gcd import decode_dialog_j2, encode_dialog_j2
from qsm.inversion.reference import (
    apply_dialog_jump_circuit,
    apply_dialog_jump_faults,
    to_dialog as to_dialog_ref,
    to_dialog_jump as to_dialog_jump_ref,
)

PRIMES = [29, 61, 127, 251]

_chunked_exact = functools.partial(gidney_add_ctrl_chunked, chunk=3)
_chunked_vent = functools.partial(gidney_add_ctrl_chunked, chunk=3, cmp_bits=3)
_chunked_exact.__name__ = "gidney_add_ctrl_chunked3"          # type: ignore[attr-defined]
_chunked_vent.__name__ = "gidney_add_ctrl_chunked3_vent"      # type: ignore[attr-defined]

ADDERS = [gidney_add_ctrl, cdkm_add_ctrl, _chunked_exact, _chunked_vent]


def _iters(p: int) -> int:
    return schedule.iterations_j2(p.bit_length())


def _replay_circuit(p: int, iters: int, packed: bool = True, **kw) -> Circuit:
    n = p.bit_length()
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(iters) if packed else 3 * iters)
    a = c.input("a", n)
    b = c.input("b", n)
    a2, b2 = replay.apply_dialog_j2(c, d, a, b, iters, p, packed=packed, **kw)
    c.rebind("a", a2)
    c.rebind("b", b2)          # the unconditional moddouble relabels b once per iteration
    return c


def _encode(tape: list[tuple[int, int, int]], packed: bool) -> int:
    if packed:
        return pack.pack_dialog_j2(tape, boundary=True)
    return encode_dialog_j2(tape)


# --- the tape codec --------------------------------------------------------
@pytest.mark.parametrize("p", PRIMES + [1021])
def test_both_codecs_round_trip_every_reachable_tape(p: int):
    """Neither encoding is lossy, and neither stores what it can derive.

    Past iteration 0 the shift count is recovered as `1 + c` because `s >= 1` is an
    invariant -- odd minus odd is even. At iteration 0 `s` may be 0, so it spends a second
    qubit there and drops `b1` instead, which is free because `b1 = b0`.
    """
    it = _iters(p)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        assert decode_dialog_j2(encode_dialog_j2(tape), it) == tape, f"flat p={p} x={x}"
        got = pack.unpack_dialog_j2(pack.pack_dialog_j2(tape, boundary=True), it,
                                    boundary=True)
        assert got == tape, f"packed p={p} x={x}"


@pytest.mark.parametrize("p", PRIMES + [1021, 8191])
def test_iteration_zero_never_needs_the_b1_it_does_not_store(p: int):
    """The saving the boundary digit is built on, over every reachable tape.

    `u` starts as the prime and `v` as `x < p`, so the first comparison is decided before
    the loop runs. Four states survive at iteration 0 -- `s` in {0,1,2} against `b0` -- and
    they still fit one base-5 digit, which is why the 609-bit tape stands.
    """
    it = _iters(p)
    seen = set()
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        s, b0, b1 = tape[0]
        assert b1 == b0, f"p={p} x={x}: iteration 0 needed a b1 of its own"
        assert s <= 2, f"p={p} x={x}: {s} shifts at iteration 0"
        seen.add((s, b0, b1))
    assert len(seen) <= 5, f"p={p}: {len(seen)} boundary states will not fit a base-5 digit"


# --- values ----------------------------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "flat"])
def test_replay_matches_the_circuit_reference_exhaustively(p: int, packed: bool):
    """Every tape from every x, replayed against several accumulators.

    Against `apply_dialog_jump_circuit` and not `..._lazy`, for the reason
    `test_inversion_replay.py` gives: at these widths the lazy domain is open and the
    circuit wraps where the congruent reference does not.
    """
    n, it = p.bit_length(), _iters(p)
    c = _replay_circuit(p, it, packed=packed)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        d = _encode(tape, packed)
        for y in (0, 1, 2, x, p - 1):
            r = simulate(c, {"d": d, "a": y, "b": 0}, rng=random.Random(x * 71 + y))
            want = apply_dialog_jump_circuit(tape, y, 0, p, n)
            assert (r.outputs["a"], r.outputs["b"]) == want, f"p={p} x={x} y={y}"
            assert r.outputs["d"] == d, f"p={p} x={x}: the tape did not come back"


@pytest.mark.parametrize("p", PRIMES)
def test_the_two_readers_compute_the_same_circuit(p: int):
    """Packing is a storage decision, not an arithmetic one -- base 5 included."""
    it = _iters(p)
    flat = _replay_circuit(p, it, packed=False)
    packed = _replay_circuit(p, it, packed=True)
    for x in range(1, p):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        for y in (1, x, p - 1):
            got = [
                simulate(cc, {"d": _encode(tape, pk), "a": y, "b": 0},
                         rng=random.Random(x * 13 + y))
                for cc, pk in ((flat, False), (packed, True))
            ]
            assert got[0].outputs["a"] == got[1].outputs["a"], f"p={p} x={x} y={y}"
            assert got[0].outputs["b"] == got[1].outputs["b"], f"p={p} x={x} y={y}"


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("add_ctrl", ADDERS, ids=lambda f: f.__name__)
def test_either_controlled_adder_drives_the_replay(p: int, add_ctrl):
    """The loop does not depend on which adder it is given, only the cost."""
    n, it = p.bit_length(), _iters(p)
    c = _replay_circuit(p, it, add_ctrl_fn=add_ctrl)
    for x in range(1, p, max(1, p // 16)):
        tape, _, _ = to_dialog_jump_ref(p, x, it)
        r = simulate(c, {"d": _encode(tape, True), "a": 1, "b": 0}, rng=random.Random(x))
        assert (r.outputs["a"], r.outputs["b"]) == apply_dialog_jump_circuit(tape, 1, 0, p, n)


@pytest.mark.parametrize("p", PRIMES)
def test_the_whole_tape_is_the_multiply(p: int):
    """(y, 0) -> (0, x*y mod p), the same direction the jump-1 replay computes."""
    n, it = p.bit_length(), _iters(p)
    c = _replay_circuit(p, it)
    for x in range(1, p):
        tape, u, v = to_dialog_jump_ref(p, x, it)
        assert (u, v) == (1, 0), f"p={p} x={x}: the tape did not converge"
        for y in (1, 2, p - 1):
            if any(apply_dialog_jump_faults(tape, y, 0, p, n).values()):
                continue                       # the lazy domain broke; not a value claim
            r = simulate(c, {"d": _encode(tape, True), "a": y, "b": 0},
                         rng=random.Random(x * 29 + y))
            assert r.outputs["a"] % p == 0, f"p={p} x={x} y={y}"
            assert r.outputs["b"] % p == x * y % p, f"p={p} x={x} y={y}"


# --- phase -----------------------------------------------------------------
def test_phase_survives_exactly_when_the_lazy_domain_fails_to_close():
    """The addition's erasure is vented, so its only failure mode is invisible to values.

    Both directions, as at jump 1: a run the fault predictor calls clean must leave no
    phase, and a phase must sometimes appear or the declared tolerance is untested. The
    conditional doublings contribute nothing here by construction -- `moddouble_ctrl_inplace`
    erases its flag with a Toffoli rather than a measurement, so it is exact.
    """
    dirty = clean = leaked = 0
    for p in PRIMES:
        n, it = p.bit_length(), _iters(p)
        c = _replay_circuit(p, it)
        for x in range(1, p):
            tape, _, _ = to_dialog_jump_ref(p, x, it)
            for y in (1, 2, x, p - 1):
                r = simulate(c, {"d": _encode(tape, True), "a": y, "b": 0},
                             rng=random.Random(x * 101 + y))
                if apply_dialog_jump_faults(tape, y, 0, p, n)["erasure"]:
                    dirty += 1
                    leaked += bool(r.phase)
                else:
                    clean += 1
                    assert r.phase == 0, f"p={p} x={x} y={y}: unexplained phase"
    assert clean and dirty, f"one-sided sweep: {clean} clean, {dirty} lazy"
    assert leaked, "no phase ever leaked, so the declared tolerance is untested"


# --- the conditional doubling ----------------------------------------------
@pytest.mark.parametrize("p", [29, 61, 127, 251, 1021])
def test_the_controlled_doubling_is_the_unconditional_one_or_the_identity(p: int):
    """`moddouble_ctrl_inplace` inherits `moddouble`'s domain exactly, and no worse.

    Compared against `moddouble` itself rather than against `2a mod p`, deliberately. The
    unconditional double is already only congruent on `a < p`; asking the controlled one to
    be better than the circuit it gates would be testing the wrong thing, and would hide
    the one property that matters -- that `ctrl = 0` is the exact identity.

    That branch is where the gated erasure lives. `moddouble` erases its overflow flag with
    a bare CX because the result's parity IS the flag, which holds only because the low bit
    started at zero. Under a control with `ctrl = 0` nothing shifted, `a[0]` is an arbitrary
    bit of the operand, and a bare CX would corrupt on half of all inputs.
    """
    n = p.bit_length()

    ref = Circuit()
    ref.rebind("b", moddouble(ref, ref.input("b", n), p))

    ctl = Circuit()
    ctl.rebind("b", moddouble_ctrl_inplace(ctl, ctl.input("c", 1)[0], ctl.input("b", n), p))

    for av in range(1 << n):
        want = simulate(ref, {"b": av}, rng=random.Random(av)).outputs["b"]
        on = simulate(ctl, {"c": 1, "b": av}, rng=random.Random(av))
        off = simulate(ctl, {"c": 0, "b": av}, rng=random.Random(av))
        assert on.outputs["b"] == want, f"p={p} a={av}: ctrl=1 is not moddouble"
        assert (off.outputs["b"], off.outputs["c"]) == (av, 0), f"p={p} a={av}: not identity"
        assert off.phase == 0, f"p={p} a={av}: ctrl=0 leaked phase"


def test_the_controlled_doubling_costs_one_ladder_more_than_the_free_one():
    """The whole price of jump 2 in Phase 2, at the real width: `n + 1` Toffoli on top.

    The unconditional halving in Phase 1 is a relabel and the unconditional doubling here
    is too. What jump 2 buys -- one iteration in three -- is paid for by making the SECOND
    of each pair conditional, and a conditional shift can never be a relabel, because a
    circuit cannot make a decision about data.

    The `n + 1` is `n` for the ladder plus 1 for the erasure, and neither is roundable.
    The ladder runs over the WIDENED register: `moddouble` shifts into an `n + 1`-bit view
    so that the overflow lands on a fresh qubit, and `controlled_lshift1` on `w` qubits is
    `w - 1` Fredkins, so this is `n`, not `n - 1`. There is no `top_zero` twin to trim it
    the way Phase 1's ladder is trimmed -- dropping a Fredkin needs a zero at both ends and
    `r[0]` is a genuine low bit of a field element. The +1 is the gated parity erasure
    replacing `moddouble`'s bare CX, which is a Toffoli precisely so the `ctrl = 0` branch
    is the identity.
    """
    n = params.U
    free = Circuit()
    moddouble(free, free.input("b", n), params.PRIME)

    gated = Circuit()
    moddouble_ctrl_inplace(gated, gated.input("c", 1)[0], gated.input("b", n),
                           params.PRIME)

    assert count(gated).toffoli - count(free).toffoli == n + 1
    # 261 iterations of it, which is what the 138 dropped additions have to pay for.
    assert (n + 1) * schedule.iterations_j2(n) == 66816


# --- the reader's working space -------------------------------------------
@pytest.mark.parametrize(
    "iters,carried,peak",
    [(1, 3, 5), (2, 4, 8), (3, 5, 11), (4, 5, 11), (9, 5, 11), (17, 5, 11), (261, 5, 11)],
)
def test_the_packed_reader_costs_five_carried_qubits_and_eleven_at_its_peak(
    iters: int, carried: int, peak: int
):
    """Five, not three -- and the two extra are exactly what base 5 costs to read.

    At jump 1 digit 0's pair is the accumulator's own bottom two qubits, so the reader
    carries three across all 133 groups. At jump 2 there is no free digit: every digit
    including the last folds arithmetically into qubits of `r` that must be |0> on entry,
    so its triple has to come from the pool as well.

    Those two qubits are the difference between the 56 the shorter tape saves and the 54
    the replay's peak actually falls by (1,223 to 1,169) -- along with the wider `cmp_ge_const`
    borrow that reading a base-5 digit needs, 11 against 7.
    """
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(iters))
    reader = replay.PackedReaderJ2(c, d, iters)
    for i in reversed(range(iters)):
        reader.triple(i)
        reader.done(i)
    fresh = len(reader._fresh)
    reader.close()
    assert fresh == carried, f"{iters} iterations carried {fresh} qubits, not {carried}"
    assert count(c).peak_qubits == len(d) + peak


@pytest.mark.parametrize("iters", [7, 8, 9])
def test_the_reader_returns_the_tape_untouched(iters: int):
    """Unfold-then-refold is the identity on |d>, boundary digit included."""
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(iters))
    reader = replay.PackedReaderJ2(c, d, iters)
    for i in reversed(range(iters)):
        reader.triple(i)
        reader.done(i)
    reader.close()
    for x in range(1, 40, 3):
        tape, _, _ = to_dialog_jump_ref(251, x, iters)
        v = pack.pack_dialog_j2(tape, boundary=True)
        r = simulate(c, {"d": v}, rng=random.Random(x))
        assert r.outputs["d"] == v
        assert r.phase == 0


# --- what the jump does and does not remove --------------------------------
@pytest.mark.parametrize("p", PRIMES + [1021, 8191])
def test_the_jump_removes_additions_and_swaps_but_never_a_halving(p: int):
    """The invariant behind `replay_j2_fail_prob`'s split, over every reachable tape.

    Jump 2 runs two thirds of the iterations, so two thirds of the additions, swaps and
    comparators. The halvings -- the divsteps themselves -- do not move: batching divsteps
    does not perform fewer of them, so the doublings they dualise into do not fall either.
    A claim of the form "jump 2 is 261/399 of the work" is therefore wrong about the
    largest single term in Phase 2.

    Stated as `+ 1` rather than as an equality because the two loops shift at opposite ends
    of an iteration: `to_dialog` subtracts and then halves, `to_dialog_jump` halves and then
    subtracts. So the halving that finally zeroes `v` is logged by the jump-1 iteration
    that produced it and by the jump-2 iteration AFTER it, which is already idle. The
    offset is exactly one, at every prime and every `x`, and it is a bookkeeping boundary
    rather than a saved divstep.
    """
    n = p.bit_length()
    it1, it2 = schedule.iterations(n), schedule.iterations_j2(n)
    assert it2 < it1
    for x in range(1, p):
        t1, _, _ = to_dialog_ref(p, x, it1)
        t2, _, _ = to_dialog_jump_ref(p, x, it2)
        assert len(t2) == it2 < len(t1)
        # Jump 1 halves exactly once an iteration, so its count to convergence is the index
        # of its last active entry, plus one.
        conv1 = max((i for i, e in enumerate(t1) if any(e)), default=-1) + 1
        last2 = max((i for i, (_, b0, _) in enumerate(t2) if b0), default=-1)
        assert sum(s for s, _, _ in t2[:last2 + 1]) + 1 == conv1, f"p={p} x={x}"


def test_the_cost_model_pays_for_doubling_slots_not_for_divsteps():
    """522 against 399, and the direction of the approximation is the point.

    A jump-2 iteration has two doubling SLOTS -- one unconditional, one gated -- and the
    circuit pays gates for both whether or not the second one moves anything. The expected
    number of live doublings is 399, the same as jump 1; the number the gates cost is 522.
    `replay_j2_fail_prob` charges the worst case, so the model is conservative by 0.4 bits
    and never optimistic, which is the only direction an error here is allowed to run.
    """
    it1, it2 = schedule.iterations(params.U), schedule.iterations_j2(params.U)
    assert (it1, it2) == (399, 261)
    assert 2 * it2 == 522 > it1


def test_the_error_budget_splits_the_same_way():
    """`replay_j2_fail_prob` scales the per-iteration terms and not the doubling term.

    If the doubling term were scaled by 261/399 too, the model would be quietly claiming a
    saving the circuit does not make. Checked by construction rather than by inspection:
    with every per-iteration term switched off, the two totals must be equal.
    """
    it1 = schedule.iterations(params.U)
    it2 = schedule.iterations_j2(params.U)
    off = dict(carry_window=None, cmp_bits=None)

    # Doublings only: 2*it2 = 522 against it1 = 399, which is the conservative worst case
    # (the expected count is exactly it1). So the model must be no cheaper at jump 2.
    d1 = params.replay_fail_prob(it1, **off) - it1 * params.modadd_lazy_vent_phase_prob()
    d2 = params.replay_j2_fail_prob(it2, **off) - it2 * params.modadd_lazy_vent_phase_prob()
    assert d2 > d1, "the jump-2 model must not discount doublings it still performs"
    assert d2 / d1 == pytest.approx(2 * it2 / it1, rel=1e-9)

    # ...while the terms that ARE per-iteration fall by exactly 261/399.
    v1 = it1 * params.modadd_lazy_vent_phase_prob()
    v2 = it2 * params.modadd_lazy_vent_phase_prob()
    assert v2 / v1 == pytest.approx(it2 / it1, rel=1e-12)

    # The whole truncated budget still comes out below Phase 1's own tail, as at jump 1.
    assert params.replay_j2_fail_prob(it2) < params.gcd_j2_iteration_fail_prob()
