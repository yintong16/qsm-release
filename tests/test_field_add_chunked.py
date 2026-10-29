"""Blocked Gidney addition.

Three things need pinning here, and only the first is the adder itself.

1. The BLOCK primitive. `_gidney_add_block` opens up both ends of the Gidney ladder, so
   its carry-in and carry-out have to be exhaustively right -- a chunked adder built on a
   block that is wrong only when c_in = 1 would look perfect on half its inputs.

2. The BOUNDARY identity. c_out = [S < A + c_in] is the claim that makes the whole
   construction possible, and the `+ c_in` is the part that is easy to drop (the vented
   variant drops it on purpose). It is tested as arithmetic, on its own, before any
   circuit uses it.

3. That the intermediate carries are the ones we think they are. A chunked adder whose
   blocks are individually wrong in compensating ways still produces the right total, so
   checking only the sum would not distinguish a working construction from a lucky one.
   `boundary_carries` is checked against the block primitive directly.

Phase policy: the exact variant must be phase-clean on EVERY measurement outcome. The
vented one must not -- its whole point is to trade a dirty ancilla for a phase, and a
test that saw zero phase there would be telling us the vent never fires.
"""

import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import (
    _carry_out_block,
    _gidney_add_block,
    cdkm_add_ctrl,
    gidney_add,
    gidney_add_ctrl,
    gidney_add_ctrl_threaded,
)
from qsm.field.add_chunked import (
    boundary_carries,
    gidney_add_chunked,
    gidney_add_chunked_ref,
    gidney_add_ctrl_chunked,
    gidney_add_ctrl_chunked_threaded,
)
from qsm.params import CMP_BITS, chunked_add_boundary_fail_prob

#: Chunk widths that actually chunk, paired with register widths that make the last
#: block both full (n % chunk == 0) and partial (n % chunk != 0). The ragged case is
#: where an off-by-one in the block bounds shows up.
#:
#: Kept to n <= 7 because these are swept EXHAUSTIVELY over both operands: n = 12 would
#: be 16.7M simulations, which is not thoroughness, it is a hung test run. Widths where
#: exhaustion is out of reach are sampled instead (`SAMPLED_SHAPES`), and the two sets
#: together cover both block-count and raggedness.
SHAPES = [(4, 2), (5, 2), (6, 2), (6, 3), (7, 2), (6, 5)]

#: Wider shapes, randomly sampled rather than exhausted. `TRIALS` is the sample size.
SAMPLED_SHAPES = [(8, 4), (9, 4), (12, 4), (12, 5), (16, 6), (17, 8)]
TRIALS = 400


def _n_measurements(c: Circuit) -> int:
    from qsm.core import ops
    return sum(1 for op in c.ops if op.kind == ops.MEASURE)


def _all_meas_vectors(c: Circuit, cap: int = 10):
    """Every measurement-outcome pattern, or [None] (random) when there are too many."""
    k = _n_measurements(c)
    if k > cap:
        return [None]
    return [[(v >> j) & 1 for j in range(k)] for v in range(1 << k)]


# ---------------------------------------------------------------------------
# 1. The block primitive: both ends open.
# ---------------------------------------------------------------------------
def _block_circuit(n: int, na: int, with_cin: bool, with_cout: bool) -> Circuit:
    c = Circuit()
    a = c.input("a", na)
    b = c.input("b", n)
    cin = c.input("cin", 1) if with_cin else None
    cout = c.input("cout", 1) if with_cout else None
    _gidney_add_block(c, a, b, cin=cin[0] if cin else None,
                      cout=cout[0] if cout else None)
    return c


@pytest.mark.parametrize("n,na", [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4),
                                  (5, 3), (5, 5), (4, 0)])
@pytest.mark.parametrize("with_cin", [False, True])
@pytest.mark.parametrize("with_cout", [False, True])
def test_block_is_exact_on_every_input_and_every_outcome(n, na, with_cin, with_cout):
    if na == 0 and not with_cin:
        pytest.skip("b += 0 is a no-op with nothing to check")
    c = _block_circuit(n, na, with_cin, with_cout)
    meas = _all_meas_vectors(c)
    for av in range(1 << na):
        for bv in range(1 << n):
            for cv in (0, 1) if with_cin else (0,):
                for ov in (0, 1) if with_cout else (0,):
                    total = av + bv + cv
                    want = {"a": av, "b": total % (1 << n)}
                    if with_cin:
                        want["cin"] = cv
                    if with_cout:
                        want["cout"] = ov ^ (total >> n)
                    vals = {"a": av, "b": bv}
                    if with_cin:
                        vals["cin"] = cv
                    if with_cout:
                        vals["cout"] = ov
                    for mv in meas:
                        r = simulate(c, vals, measurements=mv) if mv is not None \
                            else simulate(c, vals, rng=random.Random(av * 97 + bv))
                        assert r.outputs == want, (n, na, vals)
                        assert r.phase == 0, f"phase kickback on {vals}"


def test_block_with_both_ends_closed_is_gidney_add():
    """The generalisation must not have changed the primitive it generalises."""
    for n in (1, 2, 5, 17, 64, 255):
        old, new = Circuit(), Circuit()
        _gidney_add_block(new, new.input("a", n), new.input("b", n))
        gidney_add(old, old.input("a", n), old.input("b", n))
        assert [(o.kind, o.qubits, o.cond) for o in old.ops] == \
               [(o.kind, o.qubits, o.cond) for o in new.ops], f"op stream differs at n={n}"


# ---------------------------------------------------------------------------
# 2. The boundary identity, as arithmetic.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w", [1, 2, 3, 4, 5])
def test_carry_out_equals_sum_less_than_operand_plus_carry_in(w):
    """c_out = [S < A + c_in] -- exactly, with no tie left over.

    Dropping the `+ c_in` breaks precisely on S == A, which is the residual the vented
    variant declares and pays for. Both halves are asserted here so the declaration is
    grounded in the arithmetic rather than in a comment.
    """
    ties_that_need_cin = 0
    for A in range(1 << w):
        for B in range(1 << w):
            for cin in (0, 1):
                total = A + B + cin
                S = total % (1 << w)
                cout = total >> w
                assert cout == int(S < A + cin), (w, A, B, cin)
                if int(S < A) != cout:
                    assert S == A, "dropping c_in may only break on a tie"
                    ties_that_need_cin += 1
    # A tie needing c_in is S == A == B + c_in - 2^w * c_out, i.e. exactly the 2^w
    # inputs with B = 2^w - 1, c_in = 1.
    assert ties_that_need_cin == (1 << w)


def _carry_out_circuit(n: int, na: int, with_cin: bool) -> Circuit:
    c = Circuit()
    a = c.input("a", na)
    b = c.input("b", n)
    cin = c.input("cin", 1) if with_cin else None
    flag = c.input("flag", 1)
    _carry_out_block(c, a, b, cin[0] if cin else None, flag[0])
    return c


@pytest.mark.parametrize("n,na", [(1, 1), (2, 2), (3, 2), (4, 4), (5, 3), (4, 0)])
@pytest.mark.parametrize("with_cin", [False, True])
def test_carry_out_block_restores_its_operands(n, na, with_cin):
    if na == 0 and not with_cin:
        pytest.skip("no operand and no carry-in: nothing carries out")
    c = _carry_out_circuit(n, na, with_cin)
    meas = _all_meas_vectors(c)
    for av in range(1 << na):
        for bv in range(1 << n):
            for cv in (0, 1) if with_cin else (0,):
                for fv in (0, 1):
                    vals = {"a": av, "b": bv, "flag": fv}
                    if with_cin:
                        vals["cin"] = cv
                    want = dict(vals)
                    want["flag"] = fv ^ ((av + bv + cv) >> n)
                    for mv in meas:
                        r = simulate(c, vals, measurements=mv) if mv is not None \
                            else simulate(c, vals, rng=random.Random(av * 31 + bv))
                        assert r.outputs == want, (n, na, vals)
                        assert r.phase == 0


# ---------------------------------------------------------------------------
# 3. The chunked adder.
# ---------------------------------------------------------------------------
def _chunked_circuit(n: int, chunk: int, cmp_bits=None, na: int | None = None) -> Circuit:
    c = Circuit()
    a = c.input("a", n if na is None else na)
    b = c.input("b", n)
    gidney_add_chunked(c, a, b, chunk=chunk, cmp_bits=cmp_bits)
    return c


@pytest.mark.parametrize("n,chunk", SHAPES)
def test_exact_chunked_add_is_exhaustively_correct_and_phase_clean(n, chunk):
    c = _chunked_circuit(n, chunk)
    meas = _all_meas_vectors(c)
    for av in range(1 << n):
        for bv in range(1 << n):
            want = {"a": av, "b": gidney_add_chunked_ref(av, bv, n)}
            for mv in meas:
                r = simulate(c, {"a": av, "b": bv}, measurements=mv) if mv is not None \
                    else simulate(c, {"a": av, "b": bv},
                                  rng=random.Random(av * 131 + bv))
                assert r.outputs == want, (n, chunk, av, bv)
                assert r.phase == 0, f"exact boundary must not vent phase: {av}+{bv}"


@pytest.mark.parametrize("n,chunk", SAMPLED_SHAPES)
def test_exact_chunked_add_is_correct_at_widths_too_big_to_exhaust(n, chunk):
    """Same claim as above, sampled, at widths with more blocks than exhaustion allows.

    Includes the operands exhaustion would never reach: all-ones (every block carries
    out), and the boundary-straddling near-misses where a block sums to exactly 2^w-1.
    """
    c = _chunked_circuit(n, chunk)
    rng = random.Random(0xB10C ^ (n << 8) ^ chunk)
    full = (1 << n) - 1
    edges = [(0, 0), (full, 1), (1, full), (full, full), (full, 0), (0, full)]
    cases = edges + [(rng.getrandbits(n), rng.getrandbits(n)) for _ in range(TRIALS)]
    for av, bv in cases:
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs == {"a": av, "b": (av + bv) % (1 << n)}, (n, chunk, av, bv)
        assert r.phase == 0, f"exact boundary must not vent phase: {av}+{bv}"


@pytest.mark.parametrize("n,chunk,na", [(8, 4, 3), (8, 4, 4), (8, 4, 5), (9, 4, 1),
                                        (9, 3, 6), (7, 2, 2)])
def test_a_shorter_operand_is_zero_extended_for_free(n, chunk, na):
    """A short `a` must still work when whole blocks have no operand bits at all.

    Those blocks are pure carry propagation -- the degenerate MAJ(0, b_i, c_i) branch --
    and they are exactly what a constant add exercises.
    """
    c = _chunked_circuit(n, chunk, na=na)
    for av in range(1 << na):
        for bv in range(1 << n):
            r = simulate(c, {"a": av, "b": bv}, rng=random.Random(av * 7 + bv))
            assert r.outputs == {"a": av, "b": (av + bv) % (1 << n)}, (n, chunk, na)
            assert r.phase == 0


def test_a_chunk_at_least_as_wide_as_the_register_is_plain_gidney_add():
    for n, chunk in [(8, 8), (8, 9), (8, 64)]:
        plain, chunked = Circuit(), Circuit()
        gidney_add(plain, plain.input("a", n), plain.input("b", n))
        gidney_add_chunked(chunked, chunked.input("a", n), chunked.input("b", n),
                           chunk=chunk)
        assert [(o.kind, o.qubits) for o in plain.ops] == \
               [(o.kind, o.qubits) for o in chunked.ops], (n, chunk)


@pytest.mark.parametrize("n,chunk", [(8, 4), (9, 4), (12, 3)])
def test_boundary_carries_are_what_the_reference_says(n, chunk):
    """The intermediate carries, not just the total.

    Reconstructed block by block from `_gidney_add_block`'s own carry-out, so this fails
    if the blocks are individually wrong in ways that cancel in the final sum -- which a
    check on the sum alone cannot distinguish from a working construction.
    """
    c = Circuit()
    a = c.input("a", chunk)
    b = c.input("b", chunk)
    cin = c.input("cin", 1)
    cout = c.input("cout", 1)
    _gidney_add_block(c, a, b, cin=cin[0], cout=cout[0])

    rng = random.Random(0xCA44 ^ n ^ (chunk << 8))
    full = (1 << n) - 1
    cases = [(0, 0), (full, 1), (1, full), (full, full)]
    cases += [(rng.getrandbits(n), rng.getrandbits(n)) for _ in range(TRIALS)]
    for av, bv in cases:
        want = boundary_carries(av, bv, n, chunk)
        got, carry = [], 0
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            if hi == n:
                break
            mask = (1 << (hi - lo)) - 1
            r = simulate(c, {"a": (av >> lo) & mask, "b": (bv >> lo) & mask,
                             "cin": carry, "cout": 0}, rng=rng)
            assert r.phase == 0
            carry = r.outputs["cout"]
            got.append(carry)
        assert got == want, (n, chunk, av, bv)
        assert len(got) == len(want) > 0, "this shape has no boundary to check"


# ---------------------------------------------------------------------------
# 4. The vented variant: right values, declared phase.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,chunk,k", [(6, 3, 2), (8, 4, 2), (8, 4, 3), (9, 3, 2)])
def test_vented_chunked_add_still_computes_the_right_value(n, chunk, k):
    """The vent may be wrong; it may never be wrong about the SUM.

    That is the entire reason to erase the carry by measurement instead of by CX: a bad
    predicate lands in the phase, where a rate bound can see it, rather than in a freed
    ancilla, where nothing can. Exhaustive over both operands with random outcomes, so
    the wrong-predicate inputs are certainly included -- at k = 2 they are ~12% of them.
    """
    c = _chunked_circuit(n, chunk, cmp_bits=k)
    rng = random.Random(0xADD)
    for av in range(1 << n):
        for bv in range(1 << n):
            r = simulate(c, {"a": av, "b": bv}, rng=rng)
            assert r.outputs == {"a": av, "b": (av + bv) % (1 << n)}, (n, chunk, k)


@pytest.mark.parametrize("n,chunk,k", [(8, 4, 2), (12, 4, 3), (12, 6, 3)])
def test_vent_phase_rate_matches_the_analytic_bound(n, chunk, k):
    """Measured phase rate against `chunked_add_boundary_fail_prob`.

    The bound is a union bound over blocks, so it over-counts; what it must NOT do is
    under-count. Both sides are asserted -- a rate of zero would mean the vent is not
    being exercised and the test is vacuous, which is the failure mode that hid a real
    bug in `mul.accumulate`.
    """
    bound = chunked_add_boundary_fail_prob(n, chunk, k)
    assert 0.0 < bound < 1.0, "test the bound where it can actually bound something"
    c = _chunked_circuit(n, chunk, cmp_bits=k)
    rng = random.Random(0x5EED)
    trials, phased = 1200, 0
    for _ in range(trials):
        av, bv = rng.getrandbits(n), rng.getrandbits(n)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs["b"] == (av + bv) % (1 << n)
        phased += int(r.phase != 0)
    observed = phased / trials
    assert observed <= bound, f"observed {observed} exceeds bound {bound}"
    assert observed >= 0.1 * bound, f"observed {observed} too far below {bound}: vacuous?"


def test_dropping_the_carry_in_is_the_vents_irreducible_error():
    """Even at k = chunk the vent is wrong at 2^-(chunk+1). Widening k cannot fix it.

    This is the claim that justifies keeping a separate exact construction rather than
    just raising `cmp_bits` -- so it is asserted, not asserted-in-a-docstring.
    """
    n, chunk = 8, 4
    wide = chunked_add_boundary_fail_prob(n, chunk, k=chunk)
    wider = chunked_add_boundary_fail_prob(n, chunk, k=chunk + 40)
    assert wide == wider > 0.0
    assert chunked_add_boundary_fail_prob(n, chunk, k=None) == wide


# ---------------------------------------------------------------------------
# 4b. The controlled chunked adder.
# ---------------------------------------------------------------------------
def _cchunked_circuit(n: int, chunk: int, cmp_bits=None, na: int | None = None) -> Circuit:
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    a = c.input("a", n if na is None else na)
    b = c.input("b", n)
    gidney_add_ctrl_chunked(c, ctrl[0], a, b, chunk=chunk, cmp_bits=cmp_bits)
    return c


@pytest.mark.parametrize("n,chunk", SHAPES)
@pytest.mark.parametrize("cmp_bits", [None, 2])
def test_controlled_chunked_add_is_exhaustively_correct_at_both_control_values(
    n, chunk, cmp_bits
):
    """Values, over every operand pair and both controls, for both boundary variants.

    The per-block mask is the part that could plausibly go wrong: `t` is allocated and
    freed once per block, and `_unand` consumes it by measurement, so a block whose mask
    outlived its `_gidney_add_block` would corrupt the next block's carry-in rather than
    its own sum -- which only a sweep that reaches a carry crossing two boundaries sees.
    """
    c = _cchunked_circuit(n, chunk, cmp_bits=cmp_bits)
    rng = random.Random(0xC0DE ^ (n << 4) ^ chunk)
    for ctrl in (0, 1):
        for av in range(1 << n):
            for bv in range(1 << n):
                want = (av + bv) % (1 << n) if ctrl else bv
                r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv}, rng=rng)
                assert r.outputs == {"ctrl": ctrl, "a": av, "b": want}, \
                    (n, chunk, cmp_bits, ctrl, av, bv)


@pytest.mark.parametrize("n,chunk", SHAPES)
def test_the_exact_controlled_boundary_is_phase_clean_on_every_outcome(n, chunk):
    """`cmp_bits=None` re-masks and XORs the carry back: no vent, so no phase, ever."""
    c = _cchunked_circuit(n, chunk)
    meas = _all_meas_vectors(c)
    for ctrl in (0, 1):
        for av in range(1 << n):
            for bv in range(1 << n):
                vals = {"ctrl": ctrl, "a": av, "b": bv}
                for mv in meas:
                    r = simulate(c, vals, measurements=mv) if mv is not None \
                        else simulate(c, vals, rng=random.Random(av * 131 + bv))
                    assert r.phase == 0, f"exact boundary vented a phase on {vals}"


@pytest.mark.parametrize("n,chunk,k", [(6, 3, 2), (8, 4, 2), (7, 2, 2)])
def test_the_vent_never_phases_when_the_control_is_off(n, chunk, k):
    """The bug this construction is built around, asserted directly.

    With ctrl = 0 nothing is added and every boundary carry is |0>, but the vent still
    MEASURES it -- and `circ.measure` is an X-basis measurement, so the outcome is
    uniformly random regardless. An ungated `cmp_lt_phase` would therefore apply
    (-1)^(m . [S < A]) on a coin flip, leaving a relative phase on half of all correct
    ctrl = 0 runs at every width. Gating the predicate on `ctrl` is what removes it, and
    it costs one Toffoli. Swept over every operand pair and many outcomes, because the
    error is invisible to any comparison of values.
    """
    c = _cchunked_circuit(n, chunk, cmp_bits=k)
    for av in range(1 << n):
        for bv in range(1 << n):
            for seed in range(4):
                r = simulate(c, {"ctrl": 0, "a": av, "b": bv},
                             rng=random.Random(seed * 7919 + av * 97 + bv))
                assert r.outputs == {"ctrl": 0, "a": av, "b": bv}
                assert r.phase == 0, f"phase leaked at ctrl=0 on {av}+{bv}, seed {seed}"


@pytest.mark.parametrize("n,chunk,k", [(8, 4, 2), (12, 4, 3), (12, 6, 3)])
def test_the_controlled_vent_phases_at_the_same_rate_as_the_uncontrolled_one(n, chunk, k):
    """With ctrl = 1 the mask IS `a`, so the boundary predicate is the same predicate.

    Asserted against the same bound the uncontrolled vent is held to. The lower half of
    the assertion matters more than the upper: a controlled vent that never fired would
    pass a one-sided check while quietly having lost the trade it exists to make.
    """
    bound = chunked_add_boundary_fail_prob(n, chunk, k)
    assert 0.0 < bound < 1.0
    c = _cchunked_circuit(n, chunk, cmp_bits=k)
    rng = random.Random(0x5EED)
    trials, phased = 1200, 0
    for _ in range(trials):
        av, bv = rng.getrandbits(n), rng.getrandbits(n)
        r = simulate(c, {"ctrl": 1, "a": av, "b": bv}, rng=rng)
        assert r.outputs["b"] == (av + bv) % (1 << n)
        phased += int(r.phase != 0)
    observed = phased / trials
    assert observed <= bound, f"observed {observed} exceeds bound {bound}"
    assert observed >= 0.1 * bound, f"observed {observed} too far below {bound}: vacuous?"


@pytest.mark.parametrize("n,chunk,na", [(8, 4, 3), (8, 4, 4), (9, 4, 1), (7, 2, 2)])
def test_a_shorter_operand_still_zero_extends_under_control(n, chunk, na):
    """Blocks past the end of `a` have no mask to build; they must still carry.

    Both branches of the mask-or-not conditional are exercised here, in both the adder
    and the boundary kill, which is the pair most likely to disagree with each other.
    """
    c = _cchunked_circuit(n, chunk, na=na)
    for ctrl in (0, 1):
        for av in range(1 << na):
            for bv in range(1 << n):
                r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv},
                             rng=random.Random(av * 7 + bv))
                want = (av + bv) % (1 << n) if ctrl else bv
                assert r.outputs == {"ctrl": ctrl, "a": av, "b": want}, (n, chunk, na)
                assert r.phase == 0


def test_a_controlled_chunk_wider_than_the_register_is_plain_gidney_add_ctrl():
    for n, chunk in [(8, 8), (8, 9), (8, 64)]:
        plain, chunked = Circuit(), Circuit()
        gidney_add_ctrl(plain, plain.input("ctrl", 1)[0],
                        plain.input("a", n), plain.input("b", n))
        gidney_add_ctrl_chunked(chunked, chunked.input("ctrl", 1)[0],
                                chunked.input("a", n), chunked.input("b", n), chunk=chunk)
        assert [(o.kind, o.qubits) for o in plain.ops] == \
               [(o.kind, o.qubits) for o in chunked.ops], (n, chunk)


@pytest.mark.parametrize("n,chunk", [(8, 4), (9, 4), (12, 3)])
def test_the_controlled_boundary_carries_are_the_masked_operands_carries(n, chunk):
    """With ctrl = 1 the live carries are `boundary_carries(a, b)`; with ctrl = 0 they
    are all zero, because the mask is zero and `b` is untouched.

    Checking the sum alone cannot tell a working construction from blocks that are wrong
    in compensating ways, so the intermediate state is pinned too -- here by reading the
    reference rather than the circuit, since the boundary qubits are freed before the
    circuit ends.
    """
    for av, bv in [(0, 0), ((1 << n) - 1, 1), (1, (1 << n) - 1),
                   ((1 << n) - 1, (1 << n) - 1)]:
        assert boundary_carries(0, bv, n, chunk) == [0] * len(
            boundary_carries(av, bv, n, chunk)), "ctrl=0 masks every operand bit to zero"
        assert boundary_carries(av, bv, n, chunk) == \
            [((av & ((1 << hi) - 1)) + (bv & ((1 << hi) - 1))) >> hi
             for hi in range(chunk, n, chunk)]


# ---------------------------------------------------------------------------
# The same construction with the control threaded rather than masked (doc 11).
# `chunk + blocks` ancillas instead of `2*chunk + blocks`, which is what lets the
# replay run a Gidney-priced adder inside PEAK_BUDGET at all.
# ---------------------------------------------------------------------------


def _tchunked_circuit(n: int, chunk: int, cmp_bits=None, na: int | None = None) -> Circuit:
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    a = c.input("a", n if na is None else na)
    b = c.input("b", n)
    gidney_add_ctrl_chunked_threaded(c, ctrl[0], a, b, chunk=chunk, cmp_bits=cmp_bits)
    return c


@pytest.mark.parametrize("n,chunk", SHAPES)
@pytest.mark.parametrize("cmp_bits", [None, 2])
def test_threaded_chunked_add_is_exhaustively_correct_at_both_control_values(
    n, chunk, cmp_bits
):
    """The masked version's sweep, against the same reference.

    The failure mode this is hunting is specific to threading: the carry chain here is
    computed from the UNMASKED operands whatever `ctrl` says, so a `cout` that was not
    gated would hand a true `a + b` carry to the next block on a ctrl = 0 run. That
    corrupts the neighbour, not the block, so it is only visible once a sweep reaches a
    carry that crosses a boundary with the control off.
    """
    c = _tchunked_circuit(n, chunk, cmp_bits=cmp_bits)
    rng = random.Random(0x7EAD ^ (n << 4) ^ chunk)
    for ctrl in (0, 1):
        for av in range(1 << n):
            for bv in range(1 << n):
                want = (av + bv) % (1 << n) if ctrl else bv
                r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv}, rng=rng)
                assert r.outputs == {"ctrl": ctrl, "a": av, "b": want}, \
                    (n, chunk, cmp_bits, ctrl, av, bv)


@pytest.mark.parametrize("n,chunk", SHAPES)
def test_the_exact_threaded_boundary_is_phase_clean_on_every_outcome(n, chunk):
    """No vent, so no phase -- including through the gated carry-out's scratch qubit."""
    c = _tchunked_circuit(n, chunk)
    meas = _all_meas_vectors(c)
    for ctrl in (0, 1):
        for av in range(1 << n):
            for bv in range(1 << n):
                vals = {"ctrl": ctrl, "a": av, "b": bv}
                for mv in meas:
                    r = simulate(c, vals, measurements=mv) if mv is not None \
                        else simulate(c, vals, rng=random.Random(av * 131 + bv))
                    assert r.phase == 0, f"exact threaded boundary vented a phase on {vals}"


@pytest.mark.parametrize("n,chunk,k", [(6, 3, 2), (8, 4, 2), (7, 2, 2)])
def test_the_threaded_vent_never_phases_when_the_control_is_off(n, chunk, k):
    """Same gating bug as the masked version, and it is *sharper* here.

    There the mask being zero makes every boundary carry |0> by construction; here the
    carry-out is only |0> because `_gidney_add_block_ctrl_threaded` spends a Toffoli
    gating it. Drop that Toffoli and this test fails on values, not just on phase.
    """
    c = _tchunked_circuit(n, chunk, cmp_bits=k)
    for av in range(1 << n):
        for bv in range(1 << n):
            for seed in range(4):
                r = simulate(c, {"ctrl": 0, "a": av, "b": bv},
                             rng=random.Random(seed * 7919 + av * 97 + bv))
                assert r.outputs == {"ctrl": 0, "a": av, "b": bv}
                assert r.phase == 0, f"phase leaked at ctrl=0 on {av}+{bv}, seed {seed}"


@pytest.mark.parametrize("n,chunk,k", [(8, 4, 2), (12, 4, 3), (12, 6, 3)])
def test_the_threaded_vent_phases_at_the_same_rate_as_the_masked_one(n, chunk, k):
    """Threading changes the ancilla count, not the approximation: same bound applies."""
    bound = chunked_add_boundary_fail_prob(n, chunk, k)
    assert 0.0 < bound < 1.0
    c = _tchunked_circuit(n, chunk, cmp_bits=k)
    rng = random.Random(0x5EED)
    trials, phased = 1200, 0
    for _ in range(trials):
        av, bv = rng.getrandbits(n), rng.getrandbits(n)
        r = simulate(c, {"ctrl": 1, "a": av, "b": bv}, rng=rng)
        assert r.outputs["b"] == (av + bv) % (1 << n)
        phased += int(r.phase != 0)
    observed = phased / trials
    assert observed <= bound, f"observed {observed} exceeds bound {bound}"
    assert observed >= 0.1 * bound, f"observed {observed} too far below {bound}: vacuous?"


@pytest.mark.parametrize("n,chunk,na", [(8, 4, 3), (8, 4, 4), (9, 4, 1), (7, 2, 2)])
def test_a_shorter_operand_still_zero_extends_when_threaded(n, chunk, na):
    """Zero-extension works but is not free here; `modadd_lazy_ctrl` needs na = n-1."""
    c = _tchunked_circuit(n, chunk, na=na)
    for ctrl in (0, 1):
        for av in range(1 << na):
            for bv in range(1 << n):
                r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv},
                             rng=random.Random(av * 7 + bv))
                want = (av + bv) % (1 << n) if ctrl else bv
                assert r.outputs == {"ctrl": ctrl, "a": av, "b": want}, (n, chunk, na)
                assert r.phase == 0


def test_a_threaded_chunk_wider_than_the_register_is_plain_threaded_add():
    for n, chunk in [(8, 8), (8, 9), (8, 64)]:
        plain, chunked = Circuit(), Circuit()
        gidney_add_ctrl_threaded(plain, plain.input("ctrl", 1)[0],
                                 plain.input("a", n), plain.input("b", n))
        gidney_add_ctrl_chunked_threaded(
            chunked, chunked.input("ctrl", 1)[0],
            chunked.input("a", n), chunked.input("b", n), chunk=chunk)
        assert [(o.kind, o.qubits) for o in plain.ops] == \
               [(o.kind, o.qubits) for o in chunked.ops], (n, chunk)


@pytest.mark.parametrize("n,chunk", [(64, 16), (128, 32), (255, 38), (255, 32), (12, 6)])
def test_threading_trades_one_toffoli_per_boundary_for_chunk_minus_one_qubits(n, chunk):
    """The reason the whole construction exists, at the widths the inversion uses.

    `2*chunk + blocks` -> `chunk + blocks + 1`, so the saving is `chunk - 1`, not `chunk`:
    threading has to land the block's ungated carry on a scratch qubit for the one Toffoli
    that gates it into `cout`, and the mask never needed that because zeroing the mask
    zeroed the whole chain. One qubit back out of `chunk` -- pinned here so the docstrings
    that quote `chunk + blocks` stay honest.

    **And unlike the unchunked pair, this one is a trade, not a dominance.** Threading a
    single block is strictly better -- same `2n - 1` Toffoli for `n` fewer qubits, which is
    what `test_cadd_threaded_strictly_dominates_the_masked_gidney` pins. Chunking spends
    that back at the boundaries: the gated carry-out is 2 Toffoli where the mask's was 1,
    so the threaded chunked adder costs exactly `blocks - 1` MORE Toffoli. At n=255 /
    chunk=38 that is 6 Toffoli for 37 qubits, which is why the inversion takes it -- but it
    is a space-for-gates trade and the earlier version of this test asserted the wrong sign.
    """
    masked = count(_cchunked_circuit(n, chunk, cmp_bits=CMP_BITS))
    threaded = count(_tchunked_circuit(n, chunk, cmp_bits=CMP_BITS))
    boundaries = -(-n // chunk) - 1
    assert masked.peak_qubits - threaded.peak_qubits == chunk - 1, (n, chunk)
    assert threaded.toffoli - masked.toffoli == boundaries, (n, chunk)


@pytest.mark.parametrize("n,chunk,na", [(64, 16, 63), (255, 38, 254), (128, 32, 100)])
def test_a_narrow_operand_costs_the_threaded_chunked_adder_one_toffoli_per_missing_bit(
        n, chunk, na):
    """Above `na` the mask has nothing to add and the threading still has a sum to write.

    `gidney_add_ctrl_chunked` stops paying at `na`; the threaded form pays `2n - 1`
    whatever `na` is, because `s_i = b_i ^ c_i` above `na` needs its own `ccx(ctrl, c_i,
    b_i)`. So the gap over the masked version is `boundaries + (n - na)`, and at
    `na = n - 1` -- what `modarith.modadd_lazy_ctrl` passes when it widens `b` by an
    overflow bit -- that is still only one Toffoli more than the equal-width case.
    """
    masked = count(_cchunked_circuit(n, chunk, cmp_bits=CMP_BITS, na=na))
    threaded = count(_tchunked_circuit(n, chunk, cmp_bits=CMP_BITS, na=na))
    boundaries = -(-n // chunk) - 1
    assert threaded.toffoli - masked.toffoli == boundaries + (n - na), (n, chunk, na)
    assert masked.peak_qubits - threaded.peak_qubits == chunk - 1, (n, chunk, na)


# ---------------------------------------------------------------------------
# 5. What it costs at the real width.
# ---------------------------------------------------------------------------
def _cost(circ: Circuit):
    r = count(circ)
    return r.toffoli, circ.peak_qubits


def test_chunking_lowers_peak_and_raises_toffoli_at_255_bits():
    """The trade, measured, in both directions -- and narrower is worse on BOTH counts
    than the standalone-volume optimum, which is why chunk is a real parameter."""
    n = 255
    plain = Circuit()
    gidney_add(plain, plain.input("a", n), plain.input("b", n))
    base_tof, base_peak = _cost(plain)
    assert (base_tof, base_peak) == (254, 764)

    seen = []
    for chunk, exp_tof, exp_peak in [(128, 382, 638), (64, 446, 576), (32, 478, 548)]:
        tof, peak = _cost(_chunked_circuit(n, chunk))
        assert peak < base_peak, (chunk, peak, base_peak)
        assert tof > base_tof, (chunk, tof, base_tof)
        assert (tof, peak) == (exp_tof, exp_peak), (chunk, tof, peak)
        seen.append((tof, peak))
    # Monotone in both, in opposite directions: narrower blocks cost Toffoli and buy peak.
    assert [t for t, _ in seen] == sorted(t for t, _ in seen)
    assert [p for _, p in seen] == sorted((p for _, p in seen), reverse=True)


def test_venting_buys_back_toffoli_but_not_peak():
    """The vent is a Toffoli-only saving: the block's carries are live either way."""
    n, chunk = 255, 64
    exact_tof, exact_peak = _cost(_chunked_circuit(n, chunk))
    vent_tof, vent_peak = _cost(_chunked_circuit(n, chunk, cmp_bits=CMP_BITS))
    assert vent_tof < exact_tof, (vent_tof, exact_tof)
    assert vent_peak == exact_peak
    assert (vent_tof, vent_peak) == (371, 576)


def test_no_chunking_beats_the_plain_adder_on_standalone_volume_except_wide_and_vented():
    """Standalone, chunking is a LOSS almost everywhere -- and this is the honest headline.

    Peak x Toffoli for a lone 255-bit adder: only the widest vented block (one boundary,
    truncated) comes out ahead, and only by a few percent. The peak saving is real but
    the Toffoli it costs is bigger. What this does NOT settle is the composite, where the
    adder's peak may or may not be the circuit's peak; see
    docs/decisions/08-chunked-addition.md.
    """
    n = 255
    plain = Circuit()
    gidney_add(plain, plain.input("a", n), plain.input("b", n))
    bt, bp = _cost(plain)
    base = bt * bp

    losers = [(32, None), (64, None), (128, None), (32, CMP_BITS), (64, CMP_BITS)]
    for chunk, k in losers:
        tof, peak = _cost(_chunked_circuit(n, chunk, cmp_bits=k))
        assert tof * peak > base, (chunk, k, tof * peak, base)

    tof, peak = _cost(_chunked_circuit(n, 128, cmp_bits=CMP_BITS))
    assert tof * peak < base, (tof * peak, base)
    assert tof * peak / base > 0.9, "and the one win is small"


def test_the_controlled_chunked_adder_is_a_real_middle_point_between_the_two_corners():
    """The Pareto the dialog replay has to choose from, pinned at 255 bits.

    Every chunking is strictly between the corners on BOTH axes -- that much is the
    construction working. Whether it WINS is a separate question, and the answer splits
    on the boundary variant, which is the result worth pinning:

      vented  wins. The two corners sit at almost the same volume (519k, 522k) despite
              being a factor of two apart on each axis, and the vented middle undercuts
              both by up to 22%.
      exact   loses, everywhere. Re-masking each boundary block costs `chunk` Toffoli,
              which is most of what a block's addition cost in the first place.

    What none of this settles is the REPLAY, where the peak is set by whichever of the
    adder and `moddouble` needs more headroom, not by the adder alone.
    """
    n = 255
    gd_tof, gd_peak = _cost(_cadd(gidney_add_ctrl, n))
    cd_tof, cd_peak = _cost(_cadd(cdkm_add_ctrl, n))
    assert (gd_tof, gd_peak) == (509, 1020)
    assert (cd_tof, cd_peak) == (1020, 512)
    best_corner = min(gd_tof * gd_peak, cd_tof * cd_peak)

    for chunk, k, exp, wins in [(32, None, (957, 581), False), (32, 32, (733, 581), True),
                                (64, None, (893, 641), False), (64, 64, (701, 641), True),
                                (128, None, (765, 767), False),
                                (128, 128, (637, 767), True)]:
        tof, peak = _cost(_cchunked_circuit(n, chunk, cmp_bits=k))
        assert (tof, peak) == exp, (chunk, k, tof, peak)
        assert gd_tof < tof < cd_tof, (chunk, k, tof)
        assert cd_peak < peak < gd_peak, (chunk, k, peak)
        assert (tof * peak < best_corner) is wins, (chunk, k, tof * peak, best_corner)


def test_the_controlled_boundary_costs_one_toffoli_for_the_control_and_no_more():
    """Gating the vent's predicate is one Toffoli per boundary, not a re-masked block.

    That is the whole reason `_kill_boundary_carry_ctrl` reads the UNMASKED operand: the
    exact variant has to rebuild the mask (2 * chunk Toffoli per boundary, since masking
    and unmasking a slice is `len(A)` each way and only the mask direction is paid), the
    vented one does not. Both are measured against their uncontrolled counterparts so the
    control's price is isolated from the boundary's.
    """
    n, chunk, k = 255, 64, CMP_BITS
    boundaries = (n + chunk - 1) // chunk - 1
    assert boundaries == 3

    # Vented: mask (255) + gate the predicate (1 per boundary), and nothing else.
    plain_v, _ = _cost(_chunked_circuit(n, chunk, cmp_bits=k))
    ctrl_v, _ = _cost(_cchunked_circuit(n, chunk, cmp_bits=k))
    assert ctrl_v - plain_v == n + boundaries

    # Exact: the same mask, plus a re-mask of each boundary block (chunk Toffoli each).
    plain_e, _ = _cost(_chunked_circuit(n, chunk))
    ctrl_e, _ = _cost(_cchunked_circuit(n, chunk))
    assert ctrl_e - plain_e == n + boundaries * chunk


def _cadd(fn, n: int) -> Circuit:
    c = Circuit()
    fn(c, c.input("ctrl", 1)[0], c.input("a", n), c.input("b", n))
    return c
