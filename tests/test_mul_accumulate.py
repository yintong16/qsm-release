"""Reduce-as-you-accumulate: the three claims that hold the schedule up.

  1. Complementing all u bits conjugates modular addition into modular SUBTRACTION.
     `N(w) = 2^u - 1 - w == F - 1 - w (mod p)`, so `N(N(z)+y) == z - y`. The `F-1`
     cancels, which is why the sign flip is Clifford-only. If this were wrong the rows
     would compute a value off by a multiple of `F-1` -- a data-dependent error that a
     spot check would miss, hence exhaustive.

  2. The aggregate correction is CLASSICAL. Rows contribute `2x_i - 1`, so the schedule
     computes `y*(2x - (2^u - 1))` and the repair is `+(F-1)*y` then halve. This is the
     entire payment for removing 255 controls; if the constant is wrong the circuit is
     wrong on every input, but only at full width, where nothing is exhaustible.

  3. Modular halving is the EXACT INVERSE of modular doubling on the whole lazy range,
     including each one's non-closure -- the wrap and the borrow are the same bijection
     failure seen from opposite ends of the register, and they compose to the identity.
     This is what makes `_add_18y`'s restoring halvings cost nothing in the error budget,
     and it is the only reason the correction is affordable. Halving is NOT closed (an
     earlier version of this file claimed it was): exact halving on [0, 2^u) is not even
     injective, since z and z+p halve to values p apart, so some non-closure is forced.

Widths are tiny and every (x, y) is enumerated. The lesson from `test_mul_product.py`
applies here too: verify the algebra against a schedule-replaying reference, never
against `x*y % p`, or the reference silently absolves the laziness being measured.

And one lesson of this file's own, which cost a shipped bug. These tests used to simulate
with `ZEROS = [0] * 200000`, a fixed all-zero measurement vector, and assert `phase == 0`
after every run. That assertion was VACUOUS: a vented erasure conditions its correction on
the measured bit, so with `m = 0` throughout it contributes no phase whether its predicate
was right or wrong. Under it, an early `modmul_acc` leaked phase on ~25% of inputs -- a
first row that handed `modadd_lazy` a complemented fresh |0>, i.e. `2^u - 1`, violating its
`b < p` precondition on EVERY shot -- and 83 green tests said nothing. The benchmark, which
draws measurements at random, caught it immediately.

So: measurements here are RANDOM, and phase is checked by a rate bound rather than by
`== 0`. It cannot be zero -- `modadd_lazy`'s vented predicate is genuinely wrong when the
accumulator lands in `[p, 2^u)`, which at u=11 is percent-scale and at u=255 is 2^-250.8.
`test_vent_phase_bound_actually_bounds` is what now stands where the vacuous assertion did.
"""

import functools
import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit
from qsm.field.add import gidney_add
from qsm.field.add_chunked import gidney_add_chunked
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.modarith import modsub_direct
from qsm.mul.accumulate import (ACC_OVERFLOW, ACC_TRUNC, FLAT_DIAG_CHUNK,
                                MULADD_TRUNC,
                                TRI_OVERFLOW, TRI_TRUNC, TRISUB_OVERFLOW,
                                TRISUB_TRUNC, _lazy_double,
                                _lazy_halve, diag_split, modhalve,
                                modhalve_inplace, modmul_acc, modmul_acc_ref,
                                modmuladd, modmuladd_ref, modsqr_acc,
                                modsqr_acc_ref, modsqr_tri, modsqr_tri_ref,
                                modsqrsub_inplace, modsqrsub_inplace_ref,
                                scaling_chain_prob, schedule_counts, tri_schedule_counts,
                                tri_truncation_prob, truncation_prob,
                                vent_phase_prob)
from qsm.params import (moddouble_lazy_overflow_prob, modhalve_underflow_prob,
                        modhalve_underflows, narrow_row_carry_fail_prob)

# (u, F, p) with p = 2^u - F odd. Not all prime -- the schedule's algebra does not
# need primality, only that p is odd (so 2 is invertible) and F is odd.
PARAMS = [(5, 3, 29), (5, 5, 27), (6, 3, 61), (6, 5, 59), (7, 19, 109), (8, 5, 251)]


def _sim(circ, vals, rng, phase_clean=True):
    """Simulate with RANDOM measurement outcomes drawn from `rng`; return the outputs.

    `phase_clean` says which contract the circuit is under. Erasures done by `_unand`
    restore exactly and cannot leak, so those circuits assert `phase == 0` and the
    assertion has force. Circuits containing a VENTED erasure (anything built out of
    `modadd_lazy`) legitimately leak at `f/2^u` per add and must pass `False`; their phase
    is bounded by `test_vent_phase_bound_actually_bounds` instead. Never re-introduce a
    blanket `phase == 0` here under a fixed measurement vector -- see the module docstring
    for what that hid.
    """
    r = simulate(circ, vals, rng=rng)
    if phase_clean:
        assert r.phase == 0, f"leftover phase kickback on {vals}"
    return r.outputs


# --- claim 1: the complement conjugates addition into subtraction ----------
@pytest.mark.parametrize("u,f,p", PARAMS)
def test_complement_is_negation_mod_p(u, f, p):
    """N(w) == F-1-w (mod p), and N is an involution. The row's sign flip in one line."""
    for w in range(1 << u):
        n_w = (1 << u) - 1 - w
        assert n_w % p == (f - 1 - w) % p
        assert (1 << u) - 1 - n_w == w                      # involution


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_conjugated_add_is_subtraction(u, f, p):
    """N(N(z) + y) == z - y (mod p) for EVERY lazy z and y, not just canonical ones."""
    for z in range(1 << u):
        for y in range(p):
            comp = (1 << u) - 1
            inner = (comp - z + y)                       # exact; laziness is separate
            assert (comp - inner) % p == (z - y) % p


# --- claim 3: halving inverts doubling exactly, wrap included --------------
@pytest.mark.parametrize("u,f,p", PARAMS)
def test_halve_and_double_are_mutually_inverse_bijections(u, f, p):
    """Both directions, every lazy value. The round trip is clean even where each leg is
    non-congruent, which is what lets `_add_18y` double and halve a scratch copy without
    adding a single term to the error budget."""
    for z in range(1 << u):
        assert _lazy_halve(_lazy_double(z, u, f), u, f) == z, f"halve(double(z)) at z={z}"
        assert _lazy_double(_lazy_halve(z, u, f), u, f) == z, f"double(halve(z)) at z={z}"
        assert 0 <= _lazy_halve(z, u, f) < (1 << u)


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_both_directions_are_non_closed_and_by_the_same_count(u, f, p):
    """Doubling wraps off the top; halving borrows off the bottom; for odd f the two sets
    have the same size, and each is the other's image. Measured, not asserted by symmetry.
    """
    wrapped = [z for z in range(1 << u) if (2 * z) % p != _lazy_double(z, u, f) % p]
    borrowed = [z for z in range(1 << u)
                if (2 * _lazy_halve(z, u, f)) % p != z % p]

    assert len(wrapped) == f // 2, f"expected floor(F/2)={f // 2}, got {len(wrapped)}"
    assert all(z >= (1 << u) - f / 2 for z in wrapped)

    assert borrowed == [z for z in range(1 << u) if modhalve_underflows(z, u, f)]
    assert borrowed == [z for z in range(1, f, 2)], "the odd residues below F"
    assert len(borrowed) == len(wrapped), "the same 9, for odd f"

    # each set is exactly the other's image under the map that misbehaves on it
    assert sorted(_lazy_double(z, u, f) for z in wrapped) == sorted(borrowed)
    assert sorted(_lazy_halve(z, u, f) for z in borrowed) == sorted(wrapped)


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_the_two_analytic_rates_coincide(u, f, p):
    """`scaling_chain_prob` charges one number per scaling; that is only legitimate
    because the two rates are equal, which holds for odd f and is asserted in the code."""
    assert moddouble_lazy_overflow_prob(u, f) == modhalve_underflow_prob(u, f)
    assert modhalve_underflow_prob(u, f) == ((f - 1) // 2) / float(1 << u)


# --- the halving circuit --------------------------------------------------
def _halve_circ(u, p, inplace):
    c = Circuit()
    a = c.input("a", u)
    if inplace:
        modhalve_inplace(c, a, p)
    else:
        c.rebind("a", modhalve(c, a, p))
    return c


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("inplace", [False, True])
def test_modhalve_circuit_exhaustive(u, f, p, inplace):
    """Every lazy input, both register conventions, phase- and ancilla-clean."""
    circ = _halve_circ(u, p, inplace)
    rng = random.Random(11)
    for a in range(1 << u):
        out = _sim(circ, {"a": a}, rng)          # phase_clean: no vent, only `_unand`s
        assert out["a"] == _lazy_halve(a, u, f), \
            f"u={u} F={f} inplace={inplace} a={a}: {out['a']} != {_lazy_halve(a, u, f)}"


def test_modhalve_inplace_costs_no_toffoli_beyond_the_constant_add():
    """The shift is a relabelling; `_inplace` pays only 2u CNOTs on top of it."""
    shifted = count(_halve_circ(16, (1 << 16) - 19, inplace=False))
    inplace = count(_halve_circ(16, (1 << 16) - 19, inplace=True))
    assert inplace.toffoli == shifted.toffoli
    assert inplace.clifford - shifted.clifford == 2 * 16


# --- claims 1+2 together: the full schedule -------------------------------
def _mul_circ(u, p):
    c = Circuit()
    x, y, out = c.input("x", u), c.input("y", u), c.input("out", u)
    c.rebind("out", modmul_acc(c, x, y, out, p))
    return c


def _sqr_circ(u, p):
    c = Circuit()
    x, out = c.input("x", u), c.input("out", u)
    c.rebind("out", modsqr_acc(c, x, out, p))
    return c


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_modmul_acc_exhaustive(u, f, p):
    """Every (x, y) in [0,p)^2: the circuit equals the schedule replay, ALWAYS, and equals
    `x*y mod p` exactly when no scaling left the register.

    At these widths the non-closure fires on 2-30% of inputs, which is what makes the test
    informative -- at 255 bits it fires with probability 2^-243.8 and would never be seen.
    Asserting plain congruence here would either fail or force the widths up until the
    hazard became invisible; tracing it tests the failure MODEL as well as the circuit."""
    circ = _mul_circ(u, p)
    rng = random.Random(3)
    for x in range(p):
        for y in range(p):
            got = _sim(circ, {"x": x, "y": y, "out": 0}, rng, phase_clean=False)
            trace = []
            want = modmul_acc_ref(x, y, p, u, trace)
            assert got["out"] == want, f"u={u} F={f} x={x} y={y}: {got['out']} != {want}"
            assert got["x"] == x and got["y"] == y, "operands must be preserved"
            if not trace:
                assert got["out"] % p == (x * y) % p, \
                    f"u={u} F={f} x={x} y={y}: not congruent to x*y with a clean trace"


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_modsqr_acc_exhaustive(u, f, p):
    """The square reuses `x`'s own bits as controls -- legal only if rows never write
    to their control. If they did, the addend would be corrupted mid-row and this
    would fail on every input with more than one set bit."""
    circ = _sqr_circ(u, p)
    rng = random.Random(4)
    for x in range(p):
        trace = []
        got = _sim(circ, {"x": x, "out": 0}, rng, phase_clean=False)
        want = modsqr_acc_ref(x, p, u, trace)
        assert got["out"] == want, f"u={u} F={f} x={x}: {got['out']} != {want}"
        assert got["x"] == x, "operand must be preserved"
        if not trace:
            assert got["out"] % p == (x * x) % p, f"u={u} F={f} x={x}: not x^2"


@pytest.mark.parametrize("u,f,p", PARAMS[:4])
def test_modmuladd_into_a_live_accumulator(u, f, p):
    """`z += x*y` for nonzero z: the pre-scaling by 2^-(u-2) must exactly undo the
    loop's u-1 doublings and the final halving.

    `zv` starts at `f`, not 0, because that is `modmuladd`'s documented precondition --
    see `test_modmuladd_into_a_zero_accumulator_is_the_precondition_violation`."""
    c = Circuit()
    x, y, z = c.input("x", u), c.input("y", u), c.input("z", u)
    c.rebind("z", modmuladd(c, x, y, z, p))
    rng = random.Random(5)
    for xv in range(0, p, max(1, p // 7)):
        for yv in range(0, p, max(1, p // 7)):
            for zv in range(f, p, max(1, p // 5)):
                trace = []
                got = _sim(c, {"x": xv, "y": yv, "z": zv}, rng, phase_clean=False)
                assert got["z"] == modmuladd_ref(xv, yv, zv, p, u, trace)
                if not trace:
                    assert got["z"] % p == (zv + xv * yv) % p


#: Widths for the two phase-rate tests, chosen exactly as `test_truncation_bound_actually
#: _bounds` chooses its own: large enough that `vent_phase_prob` -- a union bound over ~u
#: adds at `f/2^u` each -- lands BELOW 1, so `observed <= bound` says something. At the
#: PARAMS widths it is 0.5-2.0 and every such assertion would be vacuous.
#:
#: They differ because the two tests measure OPPOSITE things and each needs the other's
#: blind spot. The vent residual scales as `f/2^u`, so seeing it at all needs a large `f`
#: -- at f=3 it is 0.5% and 400 trials return zero perfectly often. The zero-accumulator
#: defect does NOT scale: it is ~26% at every width measured (0.243, 0.250, 0.268, 0.278,
#: 0.290, 0.267 at the six (u,f) below), being a precondition violated on every shot. So
#: separating it from the residual needs a SMALL `f`, where the gap is 9-26x rather than
#: the 1.3x that f=19 leaves at u=10. That the defect is flat in `f` and the residual is
#: linear in `f` is the evidence that they are two phenomena and not one mis-modelled one.
VENT_PARAMS = [(10, 19, (1 << 10) - 19), (11, 19, (1 << 11) - 19),
               (12, 19, (1 << 12) - 19)]
PRECONDITION_PARAMS = [(11, 3, (1 << 11) - 3), (12, 3, (1 << 12) - 3),
                       (11, 5, (1 << 11) - 5)]


@pytest.mark.parametrize("u,f,p", PRECONDITION_PARAMS)
def test_modmuladd_into_a_zero_accumulator_is_the_precondition_violation(u, f, p):
    """The bug this module shipped, reduced to one call, and pinned as a contract.

    `_signed_row` complements the accumulator before adding, so `modadd_lazy` receives
    `2^u - 1 - z`, which exceeds `p` exactly when `z <= f - 1`. For a running accumulator
    that is `f/2^u`. For a FRESH one it is certain, every shot -- and the leak is pure
    phase, so no value comparison anywhere can see it.

    Both halves are asserted, because both are the point. The values stay right, which is
    why this hid; and the phase rate is an order above what `vent_phase_prob` allows, which
    is why it is a bug rather than the declared residual. The fix is not to make
    `modmuladd` handle it: it is to use `modmul_acc`, whose first row adds into a known
    |0> and so collapses to a copy plus a sign -- correct AND 510 Toffoli cheaper at 255
    bits. `modmuladd` keeps the precondition because its `z` is quantum and arbitrary.
    """
    c = Circuit()
    x, y, z = c.input("x", u), c.input("y", u), c.input("z", u)
    c.rebind("z", modmuladd(c, x, y, z, p))

    rng = random.Random(19)
    dirty = 0
    trials = 400
    for _ in range(trials):
        xv, yv = rng.randrange(p), rng.randrange(p)
        r = simulate(c, {"x": xv, "y": yv, "z": 0}, rng=rng)
        assert r.outputs["z"] == modmuladd_ref(xv, yv, 0, p, u, []), \
            "the violation is invisible to values -- if this fires, something else broke"
        dirty += bool(r.phase)

    rate = dirty / trials
    bound = vent_phase_prob(schedule_counts(u, p, zero_target=False)[0], u, f)
    assert bound < 1.0, "pick a wider width: a bound above 1 asserts nothing"
    assert rate > 4 * bound, (
        f"u={u} F={f}: a ZERO accumulator leaked phase on only {rate:.3f} of inputs, "
        f"within 4x the per-add vent budget {bound:.3f}. Either `_signed_row` grew a "
        "zero-case guard (delete this test and the precondition with it) or the vent "
        "stopped working (much worse -- the erasure now leaves a dirty ancilla instead).")


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_the_zero_row_narrows_the_operand_contract_to_canonical(u, f, p):
    """The price of the cheap first row, pinned exactly. It is not free and should not be
    remembered as free.

    `_signed_row` adds with `modadd_lazy`, which accepts a LAZY `a`, so the old top row
    tolerated any `y < 2^u`. `_first_row_from_zero` computes `~(y + f - 1)`, which is
    `p - y` only while `y <= p`; above that the constant add wraps and the row lands on
    `f - y (mod p)`. So the 510-Toffoli saving is bought with an operand precondition.

    The dangerous set is exactly `(p, 2^u)` -- `f - 1` values, 18 of 2^255 at real width --
    and only on the negating branch. Asserted in both directions so that neither a silent
    widening nor a silent narrowing can pass. Doc 05 section 4 is what supplies the
    precondition: canonicalisation happens at the fold, for exactly this class of reason.
    """
    from qsm.mul.accumulate import _first_row_from_zero_ref

    non_congruent = {(y, bit)
                     for y in range(1 << u) for bit in (0, 1)
                     if _first_row_from_zero_ref(y, bit, u, f) % p != (y if bit else -y) % p}

    assert non_congruent == {(y, 0) for y in range(p + 1, 1 << u)}, \
        f"u={u} F={f}: the row's failure set moved"
    assert len(non_congruent) == f - 1, \
        f"expected f-1={f - 1} bad operands, got {len(non_congruent)}"


@pytest.mark.parametrize("u,f,p", VENT_PARAMS)
def test_vent_phase_bound_actually_bounds(u, f, p):
    """The assertion that replaces the vacuous `phase == 0`, and the one that would have
    caught the shipped bug: `modmul_acc`'s phase rate must fit the declared budget.

    `vent_phase_prob` counts one `f/2^u` per `modadd_lazy` -- the chance its accumulator
    lands in `[p, 2^u)` and the vented erasure is done against a false predicate. Nothing
    else in this schedule can leak: `moddouble`, `modhalve` and the constant adds erase
    with `_unand`, which restores exactly.

    Measured ratios of observed to bound: 0.27, 0.27, 0.25 here, and 0.33/0.23/0.27 at the
    f=3,5 widths. A union bound over non-independent events must over-count, and it does so
    by a stable factor of ~4 rather than a wild one -- which is what makes the 2^-242.2 it
    reports at 255 bits meaningful rather than decorative. It is also the SAME factor the
    truncation bound over-counts by (0.27-0.43), from the same cause.

    The old first row broke this by 10-26x (it violated the precondition on every shot
    rather than on `f/2^u` of them), so this test fails loudly on its return. The lower
    bound is the other half: if phase went to zero, the vent has been turned off and the
    failure mode has silently become a dirty ancilla, which is strictly worse.
    """
    circ = _mul_circ(u, p)
    rng = random.Random(29)
    dirty, trials = 0, 900
    for _ in range(trials):
        vals = {"x": rng.randrange(p), "y": rng.randrange(p), "out": 0}
        r = simulate(circ, vals, rng=rng)
        assert r.outputs["out"] == modmul_acc_ref(vals["x"], vals["y"], p, u), \
            "values must not depend on measurement outcomes at all"
        dirty += bool(r.phase)

    bound = vent_phase_prob(schedule_counts(u, p)[0], u, f)
    observed = dirty / trials
    assert bound < 1.0, "pick a wider width: a bound above 1 asserts nothing"
    assert observed <= bound, \
        f"u={u} F={f}: phase on {observed:.4f} of inputs EXCEEDS the budget {bound:.4f}"
    assert observed >= 0.05 * bound, \
        (f"u={u} F={f}: phase on {observed:.4f} is >20x below the budget {bound:.4f}. "
         "Zero would mean the vent is off and the erasure now dirties an ancilla.")


# --- the claim that pays for the whole design -----------------------------
@pytest.mark.parametrize("u,f,p", PARAMS)
def test_the_aggregate_correction_constant_is_F_minus_1(u, f, p):
    """`(2^u - 1) mod p == F - 1`. If this is not the constant, the design collapses:
    the correction would be a dense multi-hundred-bit multiply and would cost more
    than the 255 controls it buys back."""
    assert ((1 << u) - 1) % p == (f - 1) % p


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_rows_compute_the_signed_aggregate(u, f, p):
    """The rows alone (no correction) compute y*(2x - (2^u - 1)) mod p."""
    from qsm.mul.accumulate import _rows_ref
    for x in range(p):
        for y in range(p):
            trace = []
            got = _rows_ref(x, y, 0, u, f, trace)
            if trace:
                continue                      # a doubling left the register; see below
            assert got % p == (y * (2 * x - ((1 << u) - 1))) % p, \
                f"u={u} F={f} x={x} y={y}: rows do not match the signed aggregate"


def test_the_row_never_writes_to_its_control():
    """The square's correctness rests on this: the sign layers READ the control and never
    modify it, so `x` can drive the controls and be the addend at the same time.

    Checked on `_signed_row` alone rather than on the whole square. The wider circuit does
    transiently X the addend -- `modadd_lazy`'s comparator complements it and restores it
    -- so a whole-circuit scan would report a write that is not one. What must hold is
    that the ROW's own control-conditioned layers leave the control untouched, and that
    `x` comes back intact end to end, which the exhaustive square test asserts directly.
    """
    from qsm.mul.accumulate import _signed_row
    u, p = 6, (1 << 6) - 3
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    y, z = c.input("y", u), c.input("z", u)
    before = len(c.ops)
    _signed_row(c, y, z, ctrl[0], p, gidney_add, None, None)

    written = set()
    for op in c.ops[before:]:
        if op.kind in (ops.X, ops.CX, ops.CCX):
            written.add(op.qubits[-1])
    assert ctrl[0] not in written, "the row writes to its own control"


# --- resource shape at real width -----------------------------------------
def test_peak_stays_under_three_registers_plus_a_carry_ladder():
    """The design's whole claim is space. Three live registers plus one Gidney carry
    ladder and nothing else -- 4u+1 measured at u=64. `mul.modmul.v1` needs ~7u for the
    same job because it materialises a 2u-bit product and a fold.

    The bound is tight on purpose: the correction used to copy `y` into a scratch register
    and cost a fifth u, and this assertion is what would catch that coming back."""
    u = 64
    p = (1 << u) - 19
    res = count(_mul_circ(u, p))
    assert res.peak_qubits <= 4 * u + 4, \
        f"peak {res.peak_qubits} > 4u = {4 * u}: a scratch register crept in"


# --- claim 4: the declared failure rates match what the circuit actually runs ----
def _count_primitives(u, p, build):
    """Count the modadd/moddouble/modhalve calls a build actually makes.

    `schedule_counts` is a closed form; this is the ground truth it is checked against.
    The whole point of deriving the counts rather than writing them down was that the
    accounting could not drift from the circuit -- which is only true if something checks.
    """
    import qsm.mul.accumulate as A

    tally = {"add": 0, "dbl": 0, "halve": 0, "const": 0}
    originals = {n: getattr(A, n) for n in
                 ("modadd_lazy", "moddouble", "moddouble_inplace",
                  "modhalve", "modhalve_inplace", "add_const_inplace")}
    keys = {"modadd_lazy": "add", "moddouble": "dbl", "moddouble_inplace": "dbl",
            "modhalve": "halve", "modhalve_inplace": "halve",
            # the zero-target first row: a controlled `+(F-1)` in place of a whole
            # `modadd_lazy`. `sub_const_inplace` is deliberately NOT tallied -- that one
            # belongs to `modhalve` and is already counted as a halving.
            "add_const_inplace": "const"}

    def wrap(name):
        fn, key = originals[name], keys[name]

        def wrapped(*a, **kw):
            tally[key] += 1
            return fn(*a, **kw)
        return wrapped

    for n in originals:
        setattr(A, n, wrap(n))
    try:
        build(Circuit(), u, p)
    finally:
        for n, fn in originals.items():
            setattr(A, n, fn)
    return tally["add"], tally["dbl"], tally["halve"], tally["const"]


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_schedule_counts_match_the_circuit(u, f, p):
    """The closed form equals the call counts, at every width, for all three entry points.

    Four numbers now, not three: `modmul_acc`/`modsqr_acc` know their target starts at |0>,
    so their top row is a controlled constant add rather than a full `modadd_lazy`. That
    trades one `n_add` for one `n_const` -- and it is the reason `schedule_counts` takes a
    `zero_target` flag at all, since `modmuladd` cannot make the same trade.
    """
    def build_mul(c, u, p):
        x, y, o = c.input("x", u), c.input("y", u), c.input("out", u)
        modmul_acc(c, x, y, o, p)

    assert _count_primitives(u, p, build_mul) == schedule_counts(u, p)

    def build_sqr(c, u, p):
        x, o = c.input("x", u), c.input("out", u)
        modsqr_acc(c, x, o, p)

    assert _count_primitives(u, p, build_sqr) == schedule_counts(u, p)

    # `modmuladd` is the same schedule plus u-2 pre-halvings, which is exactly the
    # offset the MULADD_* constants apply.
    def build_madd(c, u, p):
        x, y, z = c.input("x", u), c.input("y", u), c.input("z", u)
        modmuladd(c, x, y, z, p)

    n_add, n_dbl, n_halve, n_const = schedule_counts(u, p, zero_target=False)
    assert n_const == 0, "a live accumulator cannot use the constant-add first row"
    assert _count_primitives(u, p, build_madd) == (n_add, n_dbl, n_halve + u - 2, 0)

    # and the trade itself: one `modadd_lazy` becomes one controlled constant add
    assert schedule_counts(u, p) == (n_add - 1, n_dbl, n_halve, 1)


@pytest.mark.parametrize("u,f,L,k,n", [(11, 3, 9, 7, 2000), (12, 3, 10, 8, 2000),
                                       (11, 5, 9, 7, 2000)])
def test_truncation_bound_actually_bounds(u, f, L, k, n):
    """Truncate hard enough to SEE the failures, then check the union bound bounds them.

    The widths are chosen so the BOUND ITSELF IS BELOW 1. That is the whole point of this
    test and the reason it is not run at u=6: the schedule makes ~2u truncated calls, so
    at any width small enough to enumerate, a per-call rate large enough to observe makes
    the union bound saturate above 1 and `observed <= bound` becomes vacuously true. Here
    the bound is 0.06-0.16 and the assertion has force. Sampled rather than exhaustive for
    the same reason -- u=11 has 4M pairs.

    Measured ratios: 0.27, 0.32, 0.43. The bound over-counts, as a union bound of
    non-independent events with cancelling errors must, and by a factor of 2-4 rather than
    a wild one -- which is what makes the 2^-31.9 it reports at real width meaningful.

    Note what is NOT asserted: that the truncated circuit equals the exact one. It does
    not, and cannot -- that is the entire content of the number being bounded."""
    p = (1 << u) - f
    exact, trunc = Circuit(), Circuit()
    x, y, o = exact.input("x", u), exact.input("y", u), exact.input("out", u)
    exact.rebind("out", modmul_acc(exact, x, y, o, p))
    xt, yt, ot = trunc.input("x", u), trunc.input("y", u), trunc.input("out", u)
    trunc.rebind("out", modmul_acc(trunc, xt, yt, ot, p, carry_window=L,
                                   cmp_bits=k, window=L))

    rng = random.Random(7)
    bad = 0
    for _ in range(n):
        vals = {"x": rng.randrange(p), "y": rng.randrange(p), "out": 0}
        if (_sim(trunc, vals, rng, phase_clean=False)["out"]
                != _sim(exact, vals, rng, phase_clean=False)["out"]):
            bad += 1

    bound = truncation_prob(*schedule_counts(u, p), L, k, f)
    assert bound < 1.0, "pick a wider window: a bound above 1 asserts nothing"
    observed = bad / n
    assert observed <= bound, \
        f"u={u} F={f} L={L} k={k}: observed {observed:.4f} EXCEEDS bound {bound:.4f}"
    assert observed >= 0.05 * bound, \
        f"u={u} F={f} L={L} k={k}: bound {bound:.4f} is >20x the observed {observed:.4f}"


def test_the_truncation_term_dominates_the_declared_rate_at_real_width():
    """Two error sources, 210 orders apart, and the registered number must be the sum.

    The first version of this module reported ACC_OVERFLOW alone for the truncated
    circuits -- 2^-243.8 beside a name saying `_trunc`, which is off by 212 orders in the
    optimistic direction. This is the assertion that would have caught it."""
    import math
    from qsm.params import ERROR_BUDGET, F, PRIME, U
    adds, doubles, halves, consts = schedule_counts(U, PRIME)
    assert ACC_OVERFLOW == scaling_chain_prob(doubles, 1) + vent_phase_prob(adds)
    assert ACC_TRUNC == ACC_OVERFLOW + truncation_prob(adds, doubles, halves, consts)
    assert -33 < math.log2(ACC_TRUNC) < -30, \
        f"truncation term moved: 2^{math.log2(ACC_TRUNC):.1f}"
    # Non-closure scales with F/2^U; a fixed 1e60 gap is specific to 25519.
    assert ACC_TRUNC > 2 ** (U - F.bit_length() - ERROR_BUDGET - 2) * ACC_OVERFLOW, \
        "truncation must dominate; if it does not, the accounting lost a term"
    # and every registered `_trunc` version must carry the truncated figure
    from qsm.bench.registry import _REGISTRY
    for name in ("mul.modmul_acc_trunc.v1", "mul.modsqr_acc_trunc.v1"):
        assert _REGISTRY[name]().meta["fail_analytic"] == ACC_TRUNC, name
    assert _REGISTRY["mul.modmuladd_acc_trunc.v1"]().meta["fail_analytic"] == MULADD_TRUNC


# --- the triangle ---------------------------------------------------------
#
# `modsqr_acc` runs U full-width rows because every bit of the operand drives one. The
# triangle runs rows whose addend is the `j`-bit PREFIX of the operand, which is the thing
# doc 07 recorded as open. Everything below exists to pin the three claims that makes:
# the identity, the dropped carry, and that the narrowing survives contact with the
# reduction.

@pytest.mark.parametrize("u", [3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_the_triangle_identity(u):
    """`x^2 == S + 2^u x - D` over the INTEGERS, for every `x` below `2^u`.

    `S = sum_j s_j A_j 2^j` with `A_j = x mod 2^j` and `s_j = 2x_j - 1`; `D = sum_j x_j
    2^(2j)`. No modulus anywhere -- this is the algebra the circuit implements, and if it
    is wrong no amount of correct reduction saves it. The `-D` is the price of writing the
    row addend as the free prefix slice `A_j` rather than the `2A_j + 2^j` the plan
    proposed, which collides with itself at bit `j` and still needs a `D`.
    """
    for x in range(1 << u):
        s = sum((2 * ((x >> j) & 1) - 1) * (x & ((1 << j) - 1)) << j for j in range(u))
        d = sum(((x >> j) & 1) << (2 * j) for j in range(u))
        assert x * x == s + (x << u) - d, f"u={u} x={x}"


@pytest.mark.parametrize("u", [5, 6, 7, 8, 9, 10, 11])
def test_diag_split_reassembles_D_and_both_halves_fit(u):
    """`D = D_lo + 2^u D_hi`, and each half fits in `u` bits so each can be one
    `modsub_direct` against a register built out of `x`'s own qubits.

    `D_hi < 2^(u-1)` is the stronger half and is load-bearing twice over: it is why the
    pre-loop `modsub_direct` can never hit its own `a - b > p` failure set, and it is why
    `D_hi` can be injected before the loop and scaled by the loop's own doublings instead
    of being multiplied by `F`.
    """
    for x in range(1 << u):
        lo, hi = diag_split(x, u)
        d = sum(((x >> j) & 1) << (2 * j) for j in range(u))
        assert lo + (hi << u) == d, f"u={u} x={x}"
        assert lo < (1 << u) and hi < (1 << (u - 1)), f"u={u} x={x}: half does not fit"


def _tri_circ(u, p, ripple):
    c = Circuit()
    x, out = c.input("x", u), c.input("out", u)
    c.rebind("out", modsqr_tri(c, x, out, p, ripple=ripple))
    return c


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("ripple", [None, 2, 3])
def test_modsqr_tri_exhaustive(u, f, p, ripple):
    """Every nonzero `x`: circuit equals schedule replay, and a clean trace means `x^2`.

    Note what `ripple` does to the trace: at `ripple=2` a third of the inputs come back
    non-congruent at these widths, because the dropped carry fires at `2^-(ripple+1)` a
    row and there are `u` rows. That is the approximation working, not a defect -- the
    test is that the REFERENCE predicts exactly which inputs those are.

    `x` starts at 0, and that is deliberate: `modsqr_tri` has no operand precondition, a
    claim `test_modsqr_tri_takes_zero_without_a_precondition` makes separately and this
    one keeps honest for every width and every ripple setting.
    """
    circ = _tri_circ(u, p, ripple)
    rng = random.Random(13)
    for x in range(p):
        trace = []
        got = _sim(circ, {"x": x, "out": 0}, rng, phase_clean=False)
        want = modsqr_tri_ref(x, p, u, ripple, trace)
        assert got["out"] == want, f"u={u} F={f} ripple={ripple} x={x}: {got['out']} != {want}"
        assert got["x"] == x, "operand must be preserved -- the pads are read, never written"
        if not trace:
            assert got["out"] % p == (x * x) % p, \
                f"u={u} F={f} ripple={ripple} x={x}: clean trace but not x^2"


@pytest.mark.parametrize("u,f,p", PRECONDITION_PARAMS)
def test_modsqr_tri_takes_zero_without_a_precondition(u, f, p):
    """`x = 0` is clean, and the reason is not the one it looks like.

    This test exists because the opposite was asserted first and was wrong. The triangle
    opens with `z = 2(x - D_hi) mod p`, which at `x = 0` is IDENTICALLY zero -- and a
    complemented zero accumulator under `_signed_row` is precisely the `modmuladd` defect
    that `test_modmuladd_into_a_zero_accumulator_is_the_precondition_violation` pins at
    ~25% of shots. It does not fire here, because at `x = 0` the row addends `A_j` are the
    zero register too, and `modadd_lazy` erases with `result < a`, which is EXACT when
    `a = 0`. The dangerous conjunction is a small accumulator with a nonzero addend, and
    both come from the same `x`, so the triangle cannot assemble it at the top row.

    Asserted as an exact zero, not as a bound. If a future edit gives the accumulator an
    independent source -- injecting a live `acc`, say, as a fused subtract-square would --
    this becomes a real precondition and this test is where it announces itself.
    """
    circ = _tri_circ(u, p, None)
    rng = random.Random(17)
    dirty = 0
    for _ in range(400):
        r = simulate(circ, {"x": 0, "out": 0}, rng=rng)
        assert r.outputs["out"] == 0, "0^2 = 0"
        dirty += bool(r.phase)
    assert dirty == 0, \
        f"u={u} F={f}: x=0 leaked phase on {dirty}/400 shots; the precondition is back"


@pytest.mark.parametrize("u,f,p", VENT_PARAMS)
def test_tri_vent_phase_bound_actually_bounds(u, f, p):
    """The triangle's phase residual is the ordinary vent budget and nothing more.

    Same assertion as `test_vent_phase_bound_actually_bounds` makes for `modmul_acc`, and
    it is what says the previous test is a statement about `x = 0` rather than about a
    circuit that never leaks at all. `_narrow_row` contributes nothing: it is a bare
    `gidney_add` conjugated by Cliffords, erased by `_unand`, with no vent to be wrong.
    So the count charged is the FULL rows plus the two `modsub_direct` calls.
    """
    circ = _tri_circ(u, p, None)
    rng = random.Random(31)
    dirty, trials = 0, 500
    for _ in range(trials):
        x = rng.randrange(p)
        r = simulate(circ, {"x": x, "out": 0}, rng=rng)
        assert r.outputs["out"] == modsqr_tri_ref(x, p, u), \
            "values must not depend on measurement outcomes at all"
        dirty += bool(r.phase)

    bound = vent_phase_prob(tri_schedule_counts(u, None)[0] + 2, u, f)
    observed = dirty / trials
    assert bound < 1.0, "pick a wider width: a bound above 1 asserts nothing"
    assert observed <= bound, \
        f"u={u} F={f}: phase on {observed:.4f} of inputs EXCEEDS the budget {bound:.4f}"
    assert observed >= 0.05 * bound, \
        f"u={u} F={f}: phase on {observed:.4f} is >20x below {bound:.4f}; the vent is off"


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("ripple", [2, 3, 4])
def test_the_narrow_row_rate_bounds_the_narrow_rows(u, f, p, ripple):
    """`narrow_row_carry_fail_prob` must bound what the narrow rows actually lose.

    Counted on the reference, whose `("ripple", ...)` entries are one per dropped carry,
    against `narrow_rows * 2^-(ripple+1)`. The closed form assumes the top `ripple` bits
    of the accumulator are uniform and the low add carries half the time; both are
    approximations, and the test asserts only the direction that matters -- that the bound
    is not exceeded -- plus a floor, because a rate of zero would mean the narrowing is
    not actually dropping anything and the whole saving is imaginary.
    """
    narrow = tri_schedule_counts(u, ripple)[1]
    bound = narrow_row_carry_fail_prob(ripple)
    if narrow * (p - 1) * bound < 20:
        pytest.skip("fewer than 20 expected drops: the floor would be noise, not a bound")
    dropped = trials = 0
    for x in range(1, p):
        trace = []
        modsqr_tri_ref(x, p, u, ripple, trace)
        dropped += sum(1 for t in trace if t[0] == "ripple")
        trials += 1

    observed = dropped / (trials * narrow)
    assert observed <= bound, \
        f"u={u} F={f} ripple={ripple}: {observed:.4f} per row EXCEEDS {bound:.4f}"
    assert observed >= 0.05 * bound, \
        (f"u={u} F={f} ripple={ripple}: {observed:.4f} per row is >20x below {bound:.4f}. "
         "A zero here means the rows are not narrow and the triangle saves nothing.")


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("ripple", [None, 2, 3])
def test_tri_schedule_counts_match_the_circuit(u, f, p, ripple):
    """The closed form equals the calls the build actually makes, both row kinds.

    `n_dbl` is `u`, not `u-1`: the pre-loop doubling that turns the loop's `u-1` into the
    `2^u` the identity needs is a real `moddouble` and is counted. The two
    `modsub_direct`s are deliberately not in `tri_schedule_counts` -- `tri_truncation_prob`
    adds them separately so their provenance stays visible -- so they are not asserted
    here either.
    """
    import qsm.mul.accumulate as A

    tally = {"narrow": 0}
    orig_narrow = A._narrow_row

    def counted(*a, **kw):
        tally["narrow"] += 1
        return orig_narrow(*a, **kw)

    A._narrow_row = counted
    try:
        n_add, n_dbl, n_halve, n_const = _count_primitives(
            u, p, lambda c, u, p: modsqr_tri(c, c.input("x", u), c.input("out", u),
                                             p, ripple=ripple))
    finally:
        A._narrow_row = orig_narrow

    full, narrow, dbl = tri_schedule_counts(u, ripple)
    assert (n_add, tally["narrow"], n_dbl) == (full, narrow, dbl), \
        f"u={u} ripple={ripple}: schedule drifted from the circuit"
    assert (n_halve, n_const) == (0, 0), \
        "the triangle has no halvings and no zero-target constant row"


def test_the_triangle_is_where_the_saving_is():
    """The claim the module is for: narrowed rows beat full ones, measured, at real width.

    Three assertions, because three separate things could regress. The triangle must beat
    the square it replaces; it must beat it by a wide margin rather than a rounding error,
    since a narrow win would not justify the extra `u/2` qubits of diagonal pad; and the
    EXACT triangle must come out level with the square, because `ripple=None` is the same
    arithmetic with none of the narrowing.

    "Level" is 0.9-1.0x rather than 1.0x exactly, and the small edge is real and accounted
    for: the triangle has no `_add_correction` and no final `modhalve`, and pays two
    `modsub_direct` calls instead. That is ~4% at u=128 and ~1% at u=255. What the bound
    rules out is the exact form being where the saving lives -- if `ripple=None` ever
    approached the narrowed number, the narrowing would not be doing the work this module
    says it is.
    """
    import math
    from qsm.params import ERROR_BUDGET
    u, p = 128, (1 << 128) - 159
    # This toy prime is fixed: derive its window from 159, not the ambient field's F.
    carry_window = math.ceil(ERROR_BUDGET + math.log2((1 << u) - p))
    kw = dict(carry_window=carry_window, cmp_bits=ERROR_BUDGET, window=carry_window)

    acc = Circuit()
    xa, oa = acc.input("x", u), acc.input("out", u)
    modsqr_acc(acc, xa, oa, p, **kw)

    tri = Circuit()
    xt, ot = tri.input("x", u), tri.input("out", u)
    modsqr_tri(tri, xt, ot, p, ripple=carry_window, **kw)

    exact = Circuit()
    xe, oe = exact.input("x", u), exact.input("out", u)
    modsqr_tri(exact, xe, oe, p, ripple=None, **kw)

    n_acc, n_tri, n_exact = (count(c).toffoli for c in (acc, tri, exact))
    assert n_tri < n_acc, f"the triangle ({n_tri}) does not beat the square ({n_acc})"
    assert n_tri < 0.75 * n_acc, \
        f"the triangle wins by only {n_acc / n_tri:.2f}x; 128 qubits of pad is not worth that"
    assert 0.9 * n_acc <= n_exact <= n_acc, \
        f"the EXACT triangle ({n_exact}) is not level with the square ({n_acc}): the " \
        "saving is mis-attributed to the narrowing"


def test_the_triangles_declared_rates_are_the_ones_registered():
    """Same trap as `mul.modmul_acc_trunc.v1`'s, one layer up: an exact-schedule number
    reported beside a truncated circuit is off by two hundred orders."""
    import math
    from qsm.params import ERROR_BUDGET, F, U
    assert TRI_OVERFLOW == scaling_chain_prob(U, 0) + vent_phase_prob(U + 1)
    assert -35 < math.log2(TRI_TRUNC) < -31, \
        f"the triangle's truncation term moved: 2^{math.log2(TRI_TRUNC):.1f}"
    assert TRI_TRUNC > 2 ** (U - F.bit_length() - ERROR_BUDGET - 2) * TRI_OVERFLOW

    from qsm.bench.registry import _REGISTRY
    assert _REGISTRY["mul.modsqr_tri.v1"]().meta["fail_analytic"] == TRI_OVERFLOW
    assert _REGISTRY["mul.modsqr_tri_trunc.v1"]().meta["fail_analytic"] == TRI_TRUNC
    assert TRI_TRUNC == TRI_OVERFLOW + tri_truncation_prob()


# --- the fused subtract ----------------------------------------------------
#
# A square is wanted subtracted from a running value, never on its own, and doing that out
# of `modsqr_tri` costs the square twice plus a 255-qubit scratch. `modsqrsub_inplace`
# fuses it for one extra scaling chain. Three things could be wrong and each gets a test:
# the negated rows (claim 1 run backwards), the pre-halvings meeting the loop's doublings,
# and the control, which cannot be applied to the rows and so is applied to the operand.

def _trisub_circ(u, p, ripple, ctrl=False):
    c = Circuit()
    cq = c.input("c", 1) if ctrl else None
    x, acc = c.input("x", u), c.input("acc", u)
    c.rebind("acc", modsqrsub_inplace(c, x, acc, p, ripple=ripple,
                                      ctrl=cq[0] if ctrl else None))
    return c


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("ripple", [None, 2, 3])
def test_modsqrsub_inplace_exhaustive(u, f, p, ripple):
    """Every `x`, every lazy `acc`: circuit equals the schedule replay.

    Sampled over `acc` rather than enumerated, because the pair space is `p * 2^u` and the
    circuit is `u` halvings longer than the triangle. `acc` is drawn over the FULL lazy
    range `[0, 2^u)` and not `[0, p)`, which is the contract the point layer will call it
    under -- nothing upstream canonicalises -- and it is the half of the input space a
    `randrange(p)` would have left untested.
    """
    circ = _trisub_circ(u, p, ripple)
    rng = random.Random(19)
    for x in range(p):
        for _ in range(3):
            acc = rng.randrange(1 << u)
            trace = []
            got = _sim(circ, {"x": x, "acc": acc}, rng, phase_clean=False)
            want = modsqrsub_inplace_ref(x, acc, p, u, ripple, 1, trace)
            assert got["acc"] == want, \
                f"u={u} F={f} ripple={ripple} x={x} acc={acc}: {got['acc']} != {want}"
            assert got["x"] == x, "operand must be preserved"
            if not trace:
                assert got["acc"] % p == (acc - x * x) % p, \
                    f"u={u} F={f} ripple={ripple} x={x} acc={acc}: clean trace, wrong value"


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_the_fused_rows_really_do_subtract(u, f, p):
    """The exact form is congruent to `acc - x^2` on every input whose trace is clean.

    Separate from the test above because that one would pass if the sign were backwards
    and the reference were backwards with it -- both were written in the same sitting from
    the same `negate` flag. This one goes through `(acc - x*x) % p` and nothing else, and
    it is the only assertion here that would notice `-S` having become `+S`.

    Every input is tried, and the ones with a dirty trace are counted rather than skipped
    silently: at these widths the scaling chain is percent-scale, and a test that quietly
    excused most of its inputs would be worth very little. The floor is a quarter of the
    trials, which the tightest of these params clears with room -- (7, 19) runs 545 and
    keeps 180, because `2u = 14` scaling steps at `9/128` apiece is where a small F stops
    being a rounding error. Measured, not guessed: the other five keep 59-77%.
    """
    circ = _trisub_circ(u, p, None)
    rng = random.Random(23)
    clean = trials = 0
    for x in range(p):
        for acc in (0, 1, p - 1, (1 << u) - 1, (x * x) % p):
            trace = []
            modsqrsub_inplace_ref(x, acc, p, u, None, 1, trace)
            got = _sim(circ, {"x": x, "acc": acc}, rng, phase_clean=False)
            trials += 1
            if trace:
                continue
            clean += 1
            assert got["acc"] % p == (acc - x * x) % p, \
                f"u={u} F={f} x={x} acc={acc}: the fused rows are not subtracting"
    assert clean > trials // 4, \
        f"u={u} F={f}: only {clean} of {trials} inputs had a clean trace; the test is " \
        "close to vacuous and the scaling chain has got worse"


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("ripple", [None, 3])
def test_modsqrsub_inplace_at_ctrl_0_is_the_identity(u, f, p, ripple):
    """`ctrl = 0` must be the EXACT identity on `acc`, not merely congruent to it.

    This is the assertion the naive control would fail. `_signed_row`'s sign is a Clifford
    conjugation, so gating the rows gives `acc + x^2` at `ctrl = 0` rather than `acc` --
    the circuit gates the OPERAND instead, which zeroes every `A_j`, every row control and
    both diagonal halves at once. With nothing between them the `u` pre-halvings meet the
    loop's `u` doublings, and `_lazy_halve`/`_lazy_double` are mutually inverse on the
    whole lazy range, so the identity is exact including the wrap. Hence `==` on the full
    `[0, 2^u)` range and not `% p`, and `ripple` is swept because a narrow row with a zero
    addend must also be a no-op.

    `c` and `x` are checked too: the gate is `_and`/`_unand`, which restores both exactly.
    """
    circ = _trisub_circ(u, p, ripple, ctrl=True)
    rng = random.Random(29)
    for acc in range(1 << u):
        x = rng.randrange(p)
        got = _sim(circ, {"c": 0, "x": x, "acc": acc}, rng, phase_clean=False)
        assert got["acc"] == acc, \
            f"u={u} F={f} ripple={ripple} x={x}: ctrl=0 moved acc {acc} -> {got['acc']}"
        assert (got["x"], got["c"]) == (x, 0), "the gate must restore both its inputs"


@pytest.mark.parametrize("u,f,p", PARAMS)
def test_modsqrsub_inplace_at_ctrl_1_is_the_uncontrolled_circuit(u, f, p):
    """And `ctrl = 1` must agree with the ungated build value for value, on the nose.

    The two are different circuits -- one carries `u` extra qubits and `2u` extra Toffoli
    -- so agreement is a statement about `_and` being a faithful copy of `x` and about the
    gated register being read in exactly the places the ungated `x` is.
    """
    gated, plain = _trisub_circ(u, p, None, ctrl=True), _trisub_circ(u, p, None)
    rng = random.Random(31)
    for x in range(p):
        acc = rng.randrange(1 << u)
        g = _sim(gated, {"c": 1, "x": x, "acc": acc}, rng, phase_clean=False)
        q = _sim(plain, {"x": x, "acc": acc}, rng, phase_clean=False)
        assert g["acc"] == q["acc"], \
            f"u={u} F={f} x={x} acc={acc}: gated {g['acc']} != ungated {q['acc']}"


@pytest.mark.parametrize("u,f,p", VENT_PARAMS)
def test_trisub_vent_phase_bound_actually_bounds(u, f, p):
    """The fused form's phase residual, against the count its rate constant charges.

    `+2` over the triangle's row count, same as `TRISUB_OVERFLOW` uses: the two diagonal
    halves are `modadd_lazy` here where the triangle had `modsub_direct`, and both vent.
    The `u` pre-halvings do not appear -- `modhalve_inplace` erases its parity with a
    Toffoli, so it can lose a VALUE (that is `modhalve_borrow_fail_prob`, which is why
    `TRISUB_TRUNC` carries `u` of them) but never a phase.
    """
    circ = _trisub_circ(u, p, None)
    rng = random.Random(37)
    dirty, trials = 0, 500
    for _ in range(trials):
        x, acc = rng.randrange(p), rng.randrange(1 << u)
        r = simulate(circ, {"x": x, "acc": acc}, rng=rng)
        assert r.outputs["acc"] == modsqrsub_inplace_ref(x, acc, p, u), \
            "values must not depend on measurement outcomes at all"
        dirty += bool(r.phase)

    bound = vent_phase_prob(tri_schedule_counts(u, None)[0] + 2, u, f)
    observed = dirty / trials
    assert bound < 1.0, "pick a wider width: a bound above 1 asserts nothing"
    assert observed <= bound, \
        f"u={u} F={f}: phase on {observed:.4f} of inputs EXCEEDS the budget {bound:.4f}"
    assert observed >= 0.05 * bound, \
        f"u={u} F={f}: phase on {observed:.4f} is >20x below {bound:.4f}; the vent is off"


def test_fusing_the_subtract_beats_doing_it_twice():
    """The reason the fused form exists, measured at real width.

    The unfused alternative is the whole square forwards, one `modsub_direct` into `acc`,
    then the whole square backwards to give the scratch back -- and a `u`-qubit scratch
    live across all of it. So the bar is `2 * tri + sub`, and the fused form's entire
    price over ONE triangle is the `u` pre-halvings. Asserted as a range, not a point: the
    upper bound says the fusion beats the honest alternative by better than 1.5x, the
    lower says it is not somehow cheaper than the square it contains, which would mean a
    row had gone missing.
    """
    from qsm.params import CARRY_WINDOW, CMP_BITS
    u, p = 128, (1 << 128) - 159
    kw = dict(carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS, window=CARRY_WINDOW)

    tri = Circuit()
    modsqr_tri(tri, tri.input("x", u), tri.input("out", u), p,
               ripple=CARRY_WINDOW, **kw)

    fused = Circuit()
    xf, af = fused.input("x", u), fused.input("acc", u)
    modsqrsub_inplace(fused, xf, af, p, ripple=CARRY_WINDOW, **kw)

    sub = Circuit()
    modsub_direct(sub, sub.input("a", u), sub.input("b", u), p, gidney_add,
                  CARRY_WINDOW, CMP_BITS)

    n_tri, n_fused, n_sub = (count(c).toffoli for c in (tri, fused, sub))
    unfused = 2 * n_tri + n_sub
    assert n_fused > n_tri, \
        f"the fused form ({n_fused}) is cheaper than the square it contains ({n_tri})"
    assert n_fused < unfused / 1.5, \
        f"fusing saves only {unfused / n_fused:.2f}x over {unfused}; it also costs a " \
        "scaling chain and should be earning more than that"


def test_the_fused_subtracts_declared_rates_are_the_ones_registered():
    """Same check as the triangle's, plus the one term that is genuinely new.

    `TRISUB_TRUNC` carries `u * modhalve_borrow_fail_prob` that `TRI_TRUNC` does not: the
    pre-halvings are a truncated `-F` apiece and each can lose a value. It is the whole
    difference between the two truncated rates and the test pins it, because a copy-paste
    of the triangle's constant would look right in a table and be short by that term.
    """
    import math
    from qsm.bench.registry import _REGISTRY
    from qsm.params import CARRY_WINDOW, F, U, modhalve_borrow_fail_prob

    assert TRISUB_OVERFLOW == scaling_chain_prob(U, U) + vent_phase_prob(U + 1)
    assert -35 < math.log2(TRISUB_TRUNC) < -31, \
        f"the fused subtract's truncation term moved: 2^{math.log2(TRISUB_TRUNC):.1f}"
    assert TRISUB_TRUNC == (TRISUB_OVERFLOW + tri_truncation_prob(U, CARRY_WINDOW)
                            + U * modhalve_borrow_fail_prob(CARRY_WINDOW, F))
    assert TRISUB_TRUNC > TRI_TRUNC, "the u pre-halvings cannot be free in the budget"

    for name, want in (("mul.modsqrsub_inplace.v1", TRISUB_OVERFLOW),
                       ("mul.modsqrsub_inplace_trunc.v1", TRISUB_TRUNC),
                       ("mul.modsqrsub_inplace_ctrl_trunc.v1", TRISUB_TRUNC)):
        assert _REGISTRY[name]().meta["fail_analytic"] == want, f"{name} drifted"


# --- the flattened diagonal ------------------------------------------------
#
# The triangle's peak is NOT where its cost is. At U=255 the global 894 is held for 4,114
# ops out of 958,864 -- the two diagonal subtractions, each carrying a 128-qubit |0> pad on
# top of a full-width carry chain -- while the 254 rows that are 99.1% of the circuit run at
# 811. `diag_add_fn` gives those two additions a chunked adder and nothing else.
#
# Two claims, and they pull in opposite directions, so both need pinning. It must compute
# the SAME function (an exact chunking is the same sum, so the reference and the failure
# rate are unchanged -- if that slipped, the versions would be sharing a rate they had no
# right to). And it must actually MOVE the peak, or it is 256 Toffoli for nothing.

def _flat_kw(chunk=FLAT_DIAG_CHUNK):
    return dict(diag_add_fn=functools.partial(gidney_add_chunked, chunk=chunk,
                                              cmp_bits=None))


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("ripple", [None, 3])
@pytest.mark.parametrize("chunk", [2, 3])
def test_flattening_the_diagonal_changes_no_value(u, f, p, ripple, chunk):
    """Exhaustive: the flat circuit's output is the unflattened one's, bit for bit.

    Against the OTHER CIRCUIT, not against `modsqr_tri_ref`. Both agreeing with the
    reference would be the weaker statement -- the reference is a schedule replay and
    would absolve a chunking that changed the schedule in some way the replay also
    modelled. Here the two circuits must be the same function of `x`, and the chunk sizes
    are small enough (2, 3) that a 5-to-8-bit register crosses several boundaries.
    """
    plain, flat = _tri_circ(u, p, ripple), Circuit()
    xf, of = flat.input("x", u), flat.input("out", u)
    flat.rebind("out", modsqr_tri(flat, xf, of, p, ripple=ripple, **_flat_kw(chunk)))

    for x in range(p):
        a = simulate(plain, {"x": x, "out": 0}, rng=random.Random(5))
        b = simulate(flat, {"x": x, "out": 0}, rng=random.Random(5))
        assert a.outputs["out"] == b.outputs["out"], \
            f"u={u} chunk={chunk} ripple={ripple} x={x}: " \
            f"flat gave {b.outputs['out']}, plain gave {a.outputs['out']}"
        assert b.outputs["x"] == x, f"u={u} x={x}: operand not preserved"


@pytest.mark.parametrize("u,f,p", PARAMS)
@pytest.mark.parametrize("chunk", [2, 3])
def test_flattening_the_fused_subtract_changes_no_value(u, f, p, chunk):
    """The same claim for `modsqrsub_inplace`, over the full lazy accumulator range.

    `acc` is drawn from `[0, 2^u)` and not `[0, p)`: the fused form accepts a lazy
    accumulator, and a chunked boundary is exactly the kind of thing that could behave
    differently on the wrap. Three draws per `x` keeps this the same size as the
    unflattened exhaustive test.
    """
    rng = random.Random(91)
    plain = _trisub_circ(u, p, ripple=3)
    flat = Circuit()
    xf, af = flat.input("x", u), flat.input("acc", u)
    flat.rebind("acc", modsqrsub_inplace(flat, xf, af, p, ripple=3, **_flat_kw(chunk)))

    for x in range(p):
        for _ in range(3):
            acc = rng.randrange(1 << u)
            v = {"x": x, "acc": acc}
            a = simulate(plain, dict(v), rng=random.Random(5))
            b = simulate(flat, dict(v), rng=random.Random(5))
            assert a.outputs["acc"] == b.outputs["acc"], \
                f"u={u} chunk={chunk} x={x} acc={acc}: " \
                f"flat gave {b.outputs['acc']}, plain gave {a.outputs['acc']}"


def test_flattening_actually_moves_the_peak_and_barely_moves_the_toffoli():
    """At real width: -83/+256 for 25519, -55/+512 for secp256k1, both squares.

    Pin the width-dependent differences, not either circuit's total gate count.
    What this defends is the SHAPE of the trade, which is the whole argument for
    registering a second version: if flattening ever started costing percent-level Toffoli,
    or stopped moving the peak, the `_flat` versions would have no reason to exist and this
    would say so rather than the bench quietly drifting.
    """
    from qsm.params import CARRY_WINDOW, CMP_BITS, PRIME, U

    u, p = U, PRIME
    kw = dict(carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS, window=CARRY_WINDOW,
              ripple=CARRY_WINDOW)

    def build(fn, flat):
        c = Circuit()
        extra = _flat_kw() if flat else {}
        if fn is modsqr_tri:
            fn(c, c.input("x", u), c.input("out", u), p, **kw, **extra)
        else:
            fn(c, c.input("x", u), c.input("acc", u), p, **kw, **extra)
        return count(c)

    for fn in (modsqr_tri, modsqrsub_inplace):
        a, b = build(fn, False), build(fn, True)
        # The unflattened diagonal carries ceil(U/2) pad bits; after flattening
        # the row's CARRY_WINDOW determines the peak. Each exact chunk boundary
        # costs FLAT_DIAG_CHUNK Toffoli on each of two (U+1)-bit diagonal adds.
        peak_saving = (U + 1) // 2 - CARRY_WINDOW
        boundary_cost = 2 * (U // FLAT_DIAG_CHUNK) * FLAT_DIAG_CHUNK
        assert b.peak_qubits == a.peak_qubits - peak_saving, \
            f"{fn.__name__}: peak went {a.peak_qubits} -> {b.peak_qubits}"
        assert b.toffoli - a.toffoli == boundary_cost, \
            f"{fn.__name__}: flattening cost {b.toffoli - a.toffoli} Toffoli"
        # Historical 25519 trade is -83/+256; secp256k1 is -55/+512.
        volume_limit = 0.92 if U == 255 else 0.95
        assert b.circuit_volume < a.circuit_volume * volume_limit, \
            f"{fn.__name__}: volume only {a.circuit_volume / b.circuit_volume:.3f}x better"


def test_the_peak_really_is_a_transient_in_the_unflattened_square():
    """The premise: 99% of the ops run well below the global peak. Measured, not assumed.

    This is the fact that makes flattening worth 256 Toffoli, and it is invisible in
    `Resources`, which reports only the max. If the rows ever rose to meet the diagonals,
    `circuit_volume` would stop over-charging and the `_flat` versions would be pointless
    -- so the premise gets its own test rather than living in a docstring.
    """
    from qsm.params import CARRY_WINDOW, CMP_BITS, PRIME, U

    c = Circuit()
    modsqr_tri(c, c.input("x", U), c.input("out", U), PRIME,
               carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS, window=CARRY_WINDOW,
               ripple=CARRY_WINDOW)

    live, trace = 0, []
    for op in c.ops:
        if op.kind == ops.ALLOC:
            live += len(op.qubits)
        elif op.kind == ops.FREE:
            live -= len(op.qubits)
        elif op.kind == ops.MEASURE:
            trace.append(live)
            live -= 1
            continue
        trace.append(live)

    peak = max(trace)
    at_peak = sum(1 for n in trace if n == peak)
    assert at_peak / len(trace) < 0.01, \
        f"the peak is held for {at_peak}/{len(trace)} ops -- no longer a transient"
    assert peak - max(trace[len(trace) // 2:len(trace) // 2 + 1000]) >= 80, \
        "the middle of the circuit is no longer far below the peak"


def test_the_flat_versions_share_their_parents_reference_and_rate():
    """A `_flat` version declares the SAME `fail_analytic` as its parent. That is a claim.

    It rests on `cmp_bits=None` making the chunking exact -- the boundary carry is killed
    by a real comparator rather than vented -- so no new approximation enters. Registering
    a flat version with a vented boundary would need a new rate, and this test is what
    turns that into a failure instead of a silently optimistic table.
    """
    from qsm.bench.registry import _REGISTRY

    for flat, parent in (("mul.modsqr_tri_trunc_flat.v1", "mul.modsqr_tri_trunc.v1"),
                         ("mul.modsqrsub_inplace_trunc_flat.v1",
                          "mul.modsqrsub_inplace_trunc.v1"),
                         ("mul.modsqrsub_inplace_ctrl_trunc_flat.v1",
                          "mul.modsqrsub_inplace_ctrl_trunc.v1")):
        a, b = _REGISTRY[flat]().meta, _REGISTRY[parent]().meta
        assert a["fail_analytic"] == b["fail_analytic"], \
            f"{flat} declares {a['fail_analytic']}, parent {parent} declares " \
            f"{b['fail_analytic']} -- an exact chunking cannot change the rate"
