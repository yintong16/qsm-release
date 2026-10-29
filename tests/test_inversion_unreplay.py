"""Phase 2 backwards: the division, and what reversing a replay actually costs.

The headline claim is a subtraction: `inv.un_apply_dialog_*` is its forward exactly plus
`2 * iters` Toffoli, at the same peak, for every adder. 798 against ~570,000 at 255 bits.
That number is not an average over the loop -- it is two Toffoli per iteration, every
iteration, and both of them are in one place: `modadd_lazy_ctrl` erased its overflow flag
by MEASURING it, and a measurement has no inverse, so `modsub_lazy_ctrl` has to rebuild the
predicate with a real `cmp_lt` instead of a phase comparison. The doubling contributes
nothing, because its flag was erased by a parity and a parity reverses for one CX.

Correctness is checked against `un_apply_dialog_circuit`, not against the mathematical
inverse, for the same reason `test_inversion_replay.py` uses `apply_dialog_circuit`: the
loop runs lazily and at test widths it leaves the congruence class constantly. Where the
forward wrapped, the reverse cannot unwrap -- so the exhaustive identity tests run at
f = 1 primes (31, 127), where the lazy domain closes exactly and there is nothing to
excuse. At 255 bits `params.un_replay_fail_prob` puts the whole non-closure at 2^-243.2.
"""

from __future__ import annotations

import random

import pytest

from qsm import params
from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl
from qsm.inversion import pack, replay, schedule, unreplay
from qsm.inversion.reference import (
    apply_dialog_circuit,
    to_dialog as to_dialog_ref,
    un_apply_dialog_circuit,
)

#: The same test widths the forward replay uses.
PRIMES = [29, 61, 127, 251]
#: f = 2^n - p = 1: no doubling non-closure and no borrow band, so the lazy domain is
#: closed and the reverse is an exact inverse over the WHOLE input space. The only widths
#: where an identity test can be exhaustive without excusing anything.
CLOSED = [31, 127]

ADDERS = [gidney_add_ctrl, cdkm_add_ctrl]


def _iters(p: int) -> int:
    return schedule.iterations(p.bit_length())


def _encode(tape: list[tuple[int, int]], packed: bool) -> int:
    if packed:
        return pack.pack_dialog(tape)
    return sum((b0 | (s << 1)) << (2 * i) for i, (b0, s) in enumerate(tape))


def _un_circuit(p: int, iters: int, packed: bool = True, **kw) -> Circuit:
    n = p.bit_length()
    c = Circuit()
    d = c.input("d", pack.packed_bits(iters) if packed else 2 * iters)
    a, b = c.input("a", n), c.input("b", n)
    a2, b2 = unreplay.un_apply_dialog(c, d, a, b, iters, p, packed=packed, **kw)
    c.rebind("a", a2)
    c.rebind("b", b2)          # modhalve relabels b once per iteration
    return c


def _round_trip_circuit(p: int, iters: int, **kw) -> Circuit:
    n = p.bit_length()
    c = Circuit()
    d = c.input("d", pack.packed_bits(iters))
    a, b = c.input("a", n), c.input("b", n)
    a, b = replay.apply_dialog(c, d, a, b, iters, p, **kw)
    a, b = unreplay.un_apply_dialog(c, d, a, b, iters, p, **kw)
    c.rebind("a", a)
    c.rebind("b", b)
    return c


# --- values ----------------------------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "flat"])
def test_the_reverse_matches_the_circuit_reference_exhaustively(p: int, packed: bool):
    """Every tape from every x, undone from the forward's own output.

    Driven from the image rather than from arbitrary (a, b), because the reverse is only
    total there -- `modsub_lazy_ctrl`'s re-derived flag borrows out of the register on a
    `b` below F, and the circuit frees a dirty ancilla. The reference raises `ValueError`
    at exactly those inputs; at these widths the forward's non-closure can walk into them,
    and both are counted so neither can quietly become the whole test.
    """
    n, it = p.bit_length(), _iters(p)
    c = _un_circuit(p, it, packed=packed)
    checked = excused = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        d = _encode(tape, packed)
        for y in (0, 1, 2, x, p - 1):
            av, bv = apply_dialog_circuit(tape, y, 0, p, n)
            try:
                want = un_apply_dialog_circuit(tape, av, bv, p, n)
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

    `d` comes back too. That is not decoration: `modmul_inplace` has to run Phase 1
    backwards afterwards, and it can only do that if the replay left the tape alone.
    """
    n, it = p.bit_length(), _iters(p)
    c = _round_trip_circuit(p, it)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        d = pack.pack_dialog(tape)
        for y in range(p):
            r = simulate(c, {"d": d, "a": y, "b": 0}, rng=random.Random(x * 31 + y))
            assert (r.outputs["a"], r.outputs["b"]) == (y, 0), f"p={p} x={x} y={y}"
            assert r.outputs["d"] == d
            assert r.phase == 0, "an exact round trip must not leak phase"


@pytest.mark.parametrize("p", CLOSED)
def test_the_whole_tape_reversed_is_the_divide(p: int):
    """(p, z) -> (z/x mod p, 0). The operation Layer 1's slope needs.

    The forward tape maps (y, 0) to (p, x*y); running it the other way divides by the same
    x, at the forward's price plus a constant. This is what both baselines mean when they
    say the division is free -- frenchie calls `IPModMul.inverse()`, ecdsa.fail flips a
    `Direction` enum -- and it is why `qsm` never builds a separate inverter.

    Note the `p` rather than the `0`, which is not cosmetic and is the one thing Step 4c
    has to design around. The emptied register comes back holding the LAZY zero, because
    the replay never canonicalises (doc 05: nothing in the design produces a canonical
    value, and paying 763 x 399 Toffoli to fix that would cost more than the whole loop).
    `x = y = 0` is the only input that leaves a true `|0>`, so `modmul_inplace` cannot
    simply `free()` that register -- it would raise. Pinned below.
    """
    n, it = p.bit_length(), _iters(p)
    c = _un_circuit(p, it)
    for x in range(1, p):
        tape, u, v = to_dialog_ref(p, x, it)
        assert (u, v) == (1, 0), f"p={p} x={x}: the tape did not converge"
        d = pack.pack_dialog(tape)
        for z in range(1, p):
            r = simulate(c, {"d": d, "a": p, "b": z}, rng=random.Random(x * 17 + z))
            assert r.outputs["a"] == (z * pow(x, -1, p)) % p, f"p={p} x={x} z={z}"
            assert r.outputs["b"] == 0


@pytest.mark.parametrize("p", CLOSED)
def test_the_emptied_register_holds_the_lazy_zero_not_a_free_able_one(p: int):
    """A precondition for Step 4c, measured rather than hoped for.

    `apply_dialog(d, y, 0)` returns `(p, x*y)` for every nonzero product, and `p` is a
    perfectly good representative of 0 that is nonetheless `|1...1>` in the register. So
    the composite has to consume that register with the reverse rather than release it,
    and any `modmul_inplace` that tries to free it will hit `DirtyAncillaError` on every
    input but one.
    """
    n, it = p.bit_length(), _iters(p)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        for y in range(p):
            av, bv = apply_dialog_circuit(tape, y, 0, p, n)
            assert av == (0 if y == 0 else p), f"p={p} x={x} y={y}: a came back {av}"
            assert bv == (x * y) % p


@pytest.mark.parametrize("p", PRIMES)
def test_the_two_readers_compute_the_same_circuit(p: int):
    """Packing is a storage decision going backwards too."""
    n, it = p.bit_length(), _iters(p)
    flat, packed = _un_circuit(p, it, packed=False), _un_circuit(p, it, packed=True)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        for y in (1, x, p - 1):
            av, bv = apply_dialog_circuit(tape, y, 0, p, n)
            try:
                un_apply_dialog_circuit(tape, av, bv, p, n)
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
def test_either_controlled_adder_drives_the_reverse(p: int, add_ctrl):
    """The loop does not depend on which adder it is given, only the cost does."""
    it = _iters(p)
    c = _round_trip_circuit(p, it, add_ctrl_fn=add_ctrl)
    for x in range(1, p, max(1, p // 8)):
        tape, _, _ = to_dialog_ref(p, x, it)
        r = simulate(c, {"d": pack.pack_dialog(tape), "a": 1, "b": 0},
                     rng=random.Random(x))
        assert (r.outputs["a"], r.outputs["b"]) == (1, 0)


# --- cost ------------------------------------------------------------------
@pytest.mark.parametrize("add_ctrl", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("p", PRIMES)
def test_the_reverse_costs_two_toffoli_per_iteration_more(p: int, add_ctrl):
    """The surcharge, and that it is the comparator rather than anything that scales.

    `cswap` is its own inverse and `modhalve` is `moddouble`'s parity trick read the other
    way, so neither contributes. What does is `cmp_lt` replacing `cmp_lt_phase`: the value
    form has to materialise the top of the carry chain and then write it, which is two
    Toffoli at every width -- pinned independently in
    `test_field_modhalve.py::test_the_reverse_costs_two_toffoli_more_than_the_forward_at_every_width`.

    Width is the more interesting half, because the replay is already over `PEAK_BUDGET`
    on its own state and a reversal that widened it would sit on a different Pareto point
    from the multiply. It does not widen -- EXCEPT with the exact comparator behind CDKM,
    where it costs one qubit. That is not a second effect, it is the same one seen from
    the side: `cmp_lt` holds one more live qubit at its widest than `cmp_lt_phase` does,
    and whether that shows up in the peak depends on whether the comparator is the
    peak-setting region. Behind Gidney it never is -- the `n-1` carry ladder is wider. It
    is behind CDKM, whose adder carries a single ancilla; measured at 255 bits, exact CDKM
    is the one combination that pays (fwd 1433 -> rev 1434), and truncating the comparator
    to `CMP_BITS` moves the peak back into the register state and the qubit comes back
    (1223 -> 1223). No registered version pairs exact arithmetic with CDKM, so this costs
    the design nothing; it is pinned because it is the kind of thing that would otherwise
    surface as a surprise the first time a variant did.
    """
    n, it = p.bit_length(), _iters(p)

    def one(build):
        c = Circuit()
        d = c.input("d", pack.packed_bits(it))
        a, b = c.input("a", n), c.input("b", n)
        build(c, d, a, b, it, p, add_ctrl_fn=add_ctrl)
        return count(c)

    fwd, rev = one(replay.apply_dialog), one(unreplay.un_apply_dialog)
    assert rev.toffoli - fwd.toffoli == 2 * it, \
        f"p={p}: forward {fwd.toffoli}, reverse {rev.toffoli}, {it} iterations"
    assert rev.peak_qubits - fwd.peak_qubits == (1 if add_ctrl is cdkm_add_ctrl else 0), \
        f"p={p}: forward peaks at {fwd.peak_qubits}, reverse at {rev.peak_qubits}"


@pytest.mark.parametrize(
    "fwd,rev",
    [("inv.apply_dialog.v1", "inv.un_apply_dialog.v1"),
     ("inv.apply_dialog_trunc.v1", "inv.un_apply_dialog_trunc.v1"),
     ("inv.apply_dialog_cdkm_trunc.v1", "inv.un_apply_dialog_cdkm_trunc.v1")],
)
def test_the_division_costs_the_multiply_plus_798_at_255_bits(fwd: str, rev: str):
    """The same claim at real width, on the registered versions doc 10 quotes.

    798 is 2 x 399 and it is 0.14% of the cheapest replay. So for the composite the
    division and the in-place multiply are the same price to three significant figures --
    which is what both baselines report, and the reason `moddiv_inplace` is not a separate
    construction.
    """
    from qsm.bench import registry

    f = count(registry.get_builder(fwd)().circuit)
    r = count(registry.get_builder(rev)().circuit)
    assert r.toffoli - f.toffoli == 2 * schedule.iterations(params.U)
    assert r.peak_qubits == f.peak_qubits


# --- the reader ------------------------------------------------------------
@pytest.mark.parametrize("iters", [1, 2, 3, 12, 13, 14, 18, 399])
def test_the_group_reader_costs_exactly_what_the_packed_reader_costs(iters: int):
    """Reading the tape forwards is free, which is not obvious and had to be measured.

    `PackedReader` gets a coincidence: it wants the iterations in the order 2, 1, 0 and
    that is the only order `unfold_digit` can peel a base-3 group in. Going forwards there
    is no coincidence to have -- digit 0 is the low digit and nothing reaches it until the
    two above are out -- so `GroupReader` unfolds the whole group up front and serves the
    pairs in the order the loop wants.

    The reason that costs nothing is that the forward reader was already holding all three
    pairs anyway; it folds them back in one go at the end of the group, exactly as this
    one does. So carried qubits, transient peak and gate count all coincide, and only the
    order of the `pair()` calls differs.
    """
    def measure(cls, order):
        c = Circuit()
        d = c.input("d", pack.packed_bits(iters))
        r = cls(c, d, iters)
        for i in order:
            r.pair(i)
            r.done(i)
        fresh = len(r._fresh)
        r.close()
        res = count(c)
        return fresh, res.peak_qubits - len(d), res.toffoli

    assert measure(unreplay.GroupReader, range(iters)) == \
        measure(replay.PackedReader, reversed(range(iters)))


@pytest.mark.parametrize("iters", [12, 13, 14])
def test_the_group_reader_returns_the_tape_untouched(iters: int):
    """Unfold-then-refold is the identity on |d>, which is what lets Phase 1 run back."""
    c = Circuit()
    d = c.input("d", pack.packed_bits(iters))
    reader = unreplay.GroupReader(c, d, iters)
    for i in range(iters):
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
def test_un_replay_fail_prob_is_the_forwards_number():
    """Same five terms, same total; two of them promoted from phase to value.

    Worth an assertion rather than a comment, because the promotion is easy to read as a
    cost. It is not: the reverse's error rate is identical, and what changes is only how
    a failure presents -- and it presents more loudly, since a stranded overflow qubit is
    an exception where a relative phase is nothing at all.
    """
    it = schedule.iterations(params.U)
    for kw in ({}, {"cmp_bits": None}, {"double_window": None, "carry_window": None}):
        assert params.un_replay_fail_prob(it, **kw) == params.replay_fail_prob(it, **kw)


@pytest.mark.parametrize("p", [29, 61])
@pytest.mark.parametrize("cmp_bits", [None, 3], ids=["exact", "trunc"])
def test_the_reverse_never_leaks_phase_it_raises_instead(p: int, cmp_bits):
    """Why `inv.un_apply_dialog_*` is NOT in `test_harness_phase_policy.PHASE_TOLERANT`.

    The forward earns its phase tolerance by erasing the overflow flag with a measurement:
    a wrong predicate then costs a relative phase and nothing else. Backwards there is no
    measurement to make -- the flag has to be materialised for the constant subtract, so
    `modsub_lazy_ctrl` rebuilds it with `cmp_lt` and frees it. A disagreement therefore
    strands the ancilla, and the free raises.

    So the reversal does not merely preserve the forward's error rate
    (`test_un_replay_fail_prob_is_the_forwards_number`) -- it changes what a failure looks
    like, from the quietest failure mode available to the loudest. Measured over the whole
    forward image at widths where the forward leaks on most shots: 0 phase, everywhere.

    p = 29 and 61 rather than something wide, because the property is structural -- no
    measurement in the reverse, so nothing to vent -- and a second small width is
    confirmation, not evidence. It was checked exhaustively at p = 251 too (0 leaks over
    all 62,750 image points, exact and truncated); that pair cost 135 s of a 6-minute
    suite, which is not what it was worth.
    """
    n, it = p.bit_length(), _iters(p)
    c = _un_circuit(p, it, cmp_bits=cmp_bits)
    leaks = raises = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        d = pack.pack_dialog(tape)
        for y in range(p):
            av, bv = apply_dialog_circuit(tape, y, 0, p, n)
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

    The reverse re-derives the flag the forward measured away. Truncate the two
    comparisons differently and they answer differently, which subtracts F where nothing
    was added -- and strands the overflow qubit at |1>, so the free raises. A reversal is
    only an inverse of the exact circuit it was built against.
    """
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    d = c.input("d", pack.packed_bits(it))
    a, b = c.input("a", n), c.input("b", n)
    a, b = replay.apply_dialog(c, d, a, b, it, p)
    a, b = unreplay.un_apply_dialog(c, d, a, b, it, p, cmp_bits=1)
    c.rebind("a", a)
    c.rebind("b", b)

    loud = 0
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        for y in (1, 2, p - 1):
            try:
                r = simulate(c, {"d": pack.pack_dialog(tape), "a": y, "b": 0},
                             rng=random.Random(x * 7 + y))
            except DirtyAncillaError:
                loud += 1
                continue
            assert (r.outputs["a"], r.outputs["b"]) == (y, 0), \
                f"p={p} x={x} y={y}: a mismatched window returned a wrong value silently"
    assert loud > 0, "k = 1 must be narrow enough to disagree somewhere at this width"
