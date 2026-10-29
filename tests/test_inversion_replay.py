"""Phase 2 as a circuit: the Bezout replay, both tape layouts.

Three things need showing and only one of them is "does it compute the right numbers".

The replay runs in the LAZY [0, 2^u) domain -- 399 doublings fed back into their own
output, with no canonicalisation anywhere -- so at test widths it is constantly outside
the range the mathematical reference lives in. `apply_dialog_lazy` is the function the
circuit is *designed* to compute and `apply_dialog_circuit` is the one its gates actually
compute; at p = 29 they differ on 9/32 of the doublings, so these tests compare against
the second and `test_inversion_reference.py` pins the first.

And the replay's flag erasure is vented by MEASUREMENT, which means a wrong predicate
costs a relative phase rather than a wrong value or a dirty ancilla. Nothing in an output
comparison can see that. `apply_dialog_faults` predicts it per concrete input, and
`test_phase_survives_exactly_when_the_lazy_domain_fails_to_close` holds the circuit to
that prediction in both directions -- which is the test that found the missing control on
`cmp_lt_phase` in the first place.
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
from qsm.field.modarith import modadd_lazy_ctrl
from qsm.inversion import pack, replay, schedule
from qsm.inversion.reference import (
    apply_dialog_circuit,
    apply_dialog_faults,
    to_dialog as to_dialog_ref,
)

#: Small enough to simulate, wide enough that the lazy band [p, 2^n) is not everything.
PRIMES = [29, 61, 127, 251]
#: The three corners of the controlled-adder Pareto the replay chooses from. The chunked
#: pair is cut at 3 bits so that a 5-to-8-bit test prime actually crosses a boundary --
#: at chunk = 64 these primes would all be a single block and the variant would be
#: `gidney_add_ctrl` under another name.
_chunked_exact = functools.partial(gidney_add_ctrl_chunked, chunk=3)
_chunked_vent = functools.partial(gidney_add_ctrl_chunked, chunk=3, cmp_bits=3)
_chunked_exact.__name__ = "gidney_add_ctrl_chunked3"          # type: ignore[attr-defined]
_chunked_vent.__name__ = "gidney_add_ctrl_chunked3_vent"      # type: ignore[attr-defined]

ADDERS = [gidney_add_ctrl, cdkm_add_ctrl, _chunked_exact, _chunked_vent]


def _iters(p: int) -> int:
    return schedule.iterations(p.bit_length())


def _replay_circuit(p: int, iters: int, packed: bool = True, **kw) -> Circuit:
    n = p.bit_length()
    c = Circuit()
    d = c.input("d", pack.packed_bits(iters) if packed else 2 * iters)
    a = c.input("a", n)
    b = c.input("b", n)
    a2, b2 = replay.apply_dialog(c, d, a, b, iters, p, packed=packed, **kw)
    c.rebind("a", a2)
    c.rebind("b", b2)          # moddouble relabels b once per iteration
    return c


def _encode(tape: list[tuple[int, int]], packed: bool) -> int:
    if packed:
        return pack.pack_dialog(tape)
    return sum((b0 | (s << 1)) << (2 * i) for i, (b0, s) in enumerate(tape))


# --- values ----------------------------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "flat"])
def test_replay_matches_the_circuit_reference_exhaustively(p: int, packed: bool):
    """Every tape from every x, replayed against several accumulators."""
    n, it = p.bit_length(), _iters(p)
    c = _replay_circuit(p, it, packed=packed)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        d = _encode(tape, packed)
        for y in (0, 1, 2, x, p - 1):
            r = simulate(c, {"d": d, "a": y, "b": 0}, rng=random.Random(x * 71 + y))
            want = apply_dialog_circuit(tape, y, 0, p, n)
            assert (r.outputs["a"], r.outputs["b"]) == want, f"p={p} x={x} y={y}"
            assert r.outputs["d"] == d, f"p={p} x={x}: the tape did not come back"


@pytest.mark.parametrize("p", PRIMES)
def test_the_two_readers_compute_the_same_circuit(p: int):
    """Packing is a storage decision, not an arithmetic one."""
    n, it = p.bit_length(), _iters(p)
    flat = _replay_circuit(p, it, packed=False)
    packed = _replay_circuit(p, it, packed=True)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
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
    """As in Phase 1: the loop does not depend on which adder it is given, only the cost."""
    n, it = p.bit_length(), _iters(p)
    c = _replay_circuit(p, it, add_ctrl_fn=add_ctrl)
    for x in range(1, p, max(1, p // 16)):
        tape, _, _ = to_dialog_ref(p, x, it)
        r = simulate(c, {"d": pack.pack_dialog(tape), "a": 1, "b": 0},
                     rng=random.Random(x))
        assert (r.outputs["a"], r.outputs["b"]) == apply_dialog_circuit(tape, 1, 0, p, n)


@pytest.mark.parametrize("p", PRIMES)
def test_the_whole_tape_is_the_multiply(p: int):
    """(y, 0) -> (0, x*y mod p): the direction the composite depends on.

    Settled by measurement rather than assumed -- frenchie's `ApplyBitVector` docstring and
    `IPModMul.dummy_classical_function` state opposite conventions, so one of the two is
    written against the reversed circuit. Reduced mod p on the way out because the circuit
    answers in the lazy domain.
    """
    n, it = p.bit_length(), _iters(p)
    c = _replay_circuit(p, it)
    for x in range(1, p):
        tape, u, v = to_dialog_ref(p, x, it)
        assert (u, v) == (1, 0), f"p={p} x={x}: the tape did not converge"
        for y in (1, 2, p - 1):
            r = simulate(c, {"d": pack.pack_dialog(tape), "a": y, "b": 0},
                         rng=random.Random(x * 29 + y))
            if any(apply_dialog_faults(tape, y, 0, p, n).values()):
                continue                       # the lazy domain broke; not a value claim
            assert r.outputs["a"] % p == 0, f"p={p} x={x} y={y}"
            assert r.outputs["b"] % p == x * y % p, f"p={p} x={x} y={y}"


# --- phase -----------------------------------------------------------------
def test_phase_survives_exactly_when_the_lazy_domain_fails_to_close():
    """The erasure is vented, so its only failure mode is invisible to the values.

    Both directions matter. A clean run must leave no phase -- that is what caught
    `cmp_lt_phase` being applied without the addition's control, which put a coin-flip
    phase on ~half of all *correct* iterations at every width including 255. And a phase
    must appear sometimes, or the tolerance the registered versions declare would be
    untested; the lazy band is only `2^n - p` wide, so that half needs the whole sweep
    rather than one prime.
    """
    dirty = clean = leaked = 0
    for p in PRIMES:
        n, it = p.bit_length(), _iters(p)
        c = _replay_circuit(p, it)
        for x in range(1, p):
            tape, _, _ = to_dialog_ref(p, x, it)
            for y in (1, 2, x, p - 1):
                r = simulate(c, {"d": pack.pack_dialog(tape), "a": y, "b": 0},
                             rng=random.Random(x * 101 + y))
                if apply_dialog_faults(tape, y, 0, p, n)["erasure"]:
                    dirty += 1
                    leaked += bool(r.phase)
                else:
                    clean += 1
                    assert r.phase == 0, f"p={p} x={x} y={y}: unexplained phase"
    assert clean and dirty, f"one-sided sweep: {clean} clean, {dirty} with a lazy accumulator"
    assert leaked, "no phase ever leaked, so the declared tolerance is untested"


def test_the_vent_needs_the_addition_s_control():
    """`modadd_lazy_ctrl` with ctrl = 0 must be the identity, phase included.

    The regression for the bug above, isolated to one gate: with ctrl = 0 the accumulator
    is untouched, so the measured flag is 0 while `b < a` is whatever the operands say.
    Ungated, the vent's two phases stop cancelling on half of all measurement outcomes,
    which is invisible until something tracks them. Many seeds, because the outcome that
    exposes it is the random one.
    """
    n, p = 5, 29
    c = Circuit()
    ctrl = c.input("c", 1)
    a = c.input("a", n)
    b = c.input("b", n)
    modadd_lazy_ctrl(c, ctrl[0], a, b, p)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"c": 0, "a": av, "b": bv}, rng=random.Random(av * 32 + bv))
            assert (r.outputs["a"], r.outputs["b"]) == (av, bv)
            assert r.phase == 0, f"a={av} b={bv}: ctrl=0 leaked phase"


@pytest.mark.parametrize("p", [29, 61])
def test_the_control_is_what_makes_it_clean_not_the_seed(p: int):
    """Sweep the measurement outcomes for one input rather than one outcome per input."""
    n = p.bit_length()
    c = Circuit()
    ctrl = c.input("c", 1)
    a = c.input("a", n)
    b = c.input("b", n)
    modadd_lazy_ctrl(c, ctrl[0], a, b, p)
    av, bv = p - 1, 0                      # b < a, so an ungated vent has something to say
    for seed in range(64):
        r = simulate(c, {"c": 0, "a": av, "b": bv}, rng=random.Random(seed))
        assert r.phase == 0, f"p={p} seed={seed}: ctrl=0 leaked phase"


# --- the reader's working space -------------------------------------------
@pytest.mark.parametrize(
    "iters,carried,peak",
    [(1, 0, 0), (2, 2, 5), (3, 3, 7), (12, 3, 7), (13, 3, 7), (14, 3, 7), (18, 3, 7),
     (399, 3, 7)],
)
def test_the_packed_reader_costs_three_carried_qubits_and_seven_at_its_peak(
    iters: int, carried: int, peak: int
):
    """Not three per group -- each unfold hands back the bits the last digit vacated.

    digit 2 takes 2 fresh and releases 1; digit 1 takes that 1 plus 1 fresh and releases
    2; digit 0's pair *is* the accumulator's bottom two qubits. Three, then, and the same
    three across all 133 groups.

    But **three carried is not seven at the peak**, and the peak is the number the budget
    cares about: `fold_digit` reads its digit back with `cmp_ge_const` on the 5-qubit
    accumulator, and that borrows four carry ancillas while it runs. They are transient
    and they do not overlap the adder, so the replay's peak is
    `d + a + b + max(7, the arithmetic's own working set)` rather than a sum -- but at
    255 bits `d + a + b` is 665 + 255 + 255 = PEAK_BUDGET exactly, so seven is already
    seven over before anything is added up.
    """
    c = Circuit()
    d = c.input("d", pack.packed_bits(iters))
    reader = replay.PackedReader(c, d, iters)
    for i in reversed(range(iters)):
        reader.pair(i)
        reader.done(i)
    fresh = len(reader._fresh)
    reader.close()
    assert fresh == carried, f"{iters} iterations carried {fresh} qubits, not {carried}"
    assert count(c).peak_qubits == len(d) + peak


@pytest.mark.parametrize("iters", [12, 13, 14])
def test_the_reader_returns_the_tape_untouched(iters: int):
    """Unfold-then-refold is the identity on |d>, which is what lets `to_dialog` run back."""
    c = Circuit()
    d = c.input("d", pack.packed_bits(iters))
    reader = replay.PackedReader(c, d, iters)
    for i in reversed(range(iters)):
        reader.pair(i)
        reader.done(i)
    reader.close()
    for x in range(1, 40, 3):
        tape, _, _ = to_dialog_ref(251, x, iters)
        v = pack.pack_dialog(tape)
        r = simulate(c, {"d": v}, rng=random.Random(x))
        assert r.outputs["d"] == v
        assert r.phase == 0


# --- the analytic model ----------------------------------------------------
def test_replay_fail_prob_counts_the_vent_phase():
    """The phase term is real and it belongs in the total, however small it is at 255 bits."""
    it = schedule.iterations(params.U)
    vent = it * params.modadd_lazy_vent_phase_prob()
    exact = params.replay_fail_prob(it, double_window=None, carry_window=None,
                                    cmp_bits=None)
    assert exact >= vent
    # ...and it is not what dominates: the truncations are ~210 bits above it.
    trunc = params.replay_fail_prob(it)
    assert trunc > 1e3 * exact


def test_the_truncations_are_the_binding_terms_not_the_lazy_domain():
    """Widening the windows is the only lever; the lazy domain is 210 bits below them."""
    it = schedule.iterations(params.U)
    lazy = it * (params.moddouble_lazy_overflow_prob()
                 + params.modadd_lazy_vent_phase_prob())
    assert lazy < 2.0 ** -240
    assert params.replay_fail_prob(it) < params.gcd_iteration_fail_prob()


def test_the_chunked_adder_is_free_at_64_bits_and_ruinous_at_32():
    """Why the registered chunked replay is cut at 64, asserted rather than asserted-in-prose.

    The boundary vent is wrong at 2^-(chunk+1) no matter how wide the comparison is -- it
    drops the carry-in -- and 399 iterations amplify that by 8.6 bits. So the parameter
    that decides whether the chunked adder is usable here is `chunk`, and the peak saving
    a narrower block would buy is simply not available: 32 would more than double the
    replay's total failure probability, and 16 would dominate `gcd_iteration_fail_prob`
    itself, which is the rate the whole design is already living with.
    """
    it = schedule.iterations(params.U)
    base = params.replay_fail_prob(it)

    at64 = params.replay_fail_prob(it, chunk=64, chunk_cmp_bits=64)
    assert at64 / base < 1.000001, "64-bit blocks must vanish under the existing terms"

    at32 = params.replay_fail_prob(it, chunk=32, chunk_cmp_bits=32)
    assert at32 > 2 * base, "32-bit blocks are not free and the model must say so"

    at16 = params.replay_fail_prob(it, chunk=16, chunk_cmp_bits=16)
    assert at16 > params.gcd_iteration_fail_prob(), "16 would outrank Phase 1's own tail"

    # And it is `chunk`, not the comparison width, that sets the term: widening the
    # compare past the block buys exactly nothing.
    assert params.replay_fail_prob(it, chunk=32, chunk_cmp_bits=200) == at32
