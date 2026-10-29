"""Phase 1 as a circuit, against the classical reference it was written from.

The reference is already exhaustively tested (`test_inversion_reference.py`); the job here
is to show the circuit computes the same tape, that the "free" shift really is free, and
that the two approximations -- the shrinking window and the truncated comparator -- fail
the way the design says they do rather than silently.
"""

from __future__ import annotations

import math
import random

import pytest

from qsm import params
from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl
from qsm.inversion import schedule
from qsm.inversion.gcd import (
    PooledTape,
    decode_dialog,
    dialog_extra_bits,
    encode_dialog,
    gcd_loop,
    to_dialog,
)
from qsm.inversion.reference import dialog_widths, to_dialog as to_dialog_ref

PRIMES = [29, 61, 127, 251]
ADDERS = [gidney_add_ctrl, cdkm_add_ctrl]


def _iters(p: int) -> int:
    return schedule.iterations(p.bit_length())


def _to_dialog_circuit(p: int, **kw) -> tuple[Circuit, int]:
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    x = c.input("x", n)
    d = c.input("d", 2 * it)
    to_dialog(c, x, d, p, iters=it, **kw)
    return c, it


def _pooled_circuit(p: int, widths: list[int], **kw) -> tuple[Circuit, int]:
    """The same loop with the tape living in the slots the window releases."""
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    x = c.input("x", n)
    c.input("d", 0)
    d = to_dialog(c, x, PooledTape(c), p, iters=it, widths=widths, **kw)
    c.rebind("d", d)
    return c, it


def _to_dialog_circuit_255(cmp_bits: int | None) -> tuple[Circuit, int]:
    n = params.U
    it = schedule.iterations(n)
    c = Circuit()
    x = c.input("x", n)
    to_dialog(c, x, c.input("d", 2 * it), params.PRIME, iters=it,
              cmp_bits=cmp_bits, widths=schedule.widths(n, it))
    return c, it


def _perfect_widths(p: int) -> list[int]:
    """The tightest window that never clips, measured over every x. Independent of the
    padding constant, so it tests the mechanism rather than `GCD_WIDTH_PAD_VAR`."""
    it = _iters(p)
    best = [1] * it
    for x in range(1, p):
        for i, w in enumerate(dialog_widths(p, x, it)):
            best[i] = max(best[i], w)
    return best


# --- correctness -----------------------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
def test_to_dialog_matches_the_reference_exhaustively(p: int):
    """Every x in [1,p) produces the reference tape, and consumes x doing it."""
    c, it = _to_dialog_circuit(p)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        r = simulate(c, {"x": x, "d": 0}, rng=random.Random(x))
        assert r.outputs["d"] == encode_dialog(tape), (
            f"p={p} x={x}: tape {decode_dialog(r.outputs['d'], it)} != {tape}"
        )
        assert r.outputs["x"] == 0, "x is consumed, not merely zeroed"
        assert r.phase == 0, f"p={p} x={x}: dirty phase"


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("add_ctrl", ADDERS, ids=lambda f: f.__name__)
def test_either_controlled_adder_drives_the_loop(p: int, add_ctrl):
    """The loop does not depend on which subtract it is given; only the cost does."""
    c, it = _to_dialog_circuit(p, add_ctrl=add_ctrl)
    for x in range(1, p, max(1, p // 24)):
        if x == 0:
            continue
        tape, _, _ = to_dialog_ref(p, x, it)
        r = simulate(c, {"x": x, "d": 0}, rng=random.Random(x))
        assert r.outputs["d"] == encode_dialog(tape)
        assert r.phase == 0


@pytest.mark.parametrize("p", PRIMES)
def test_the_loop_really_ends_at_u_equals_one_and_v_equals_zero(p: int):
    """Read the working registers directly instead of inferring it from a clean free."""
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    x = c.input("x", n)
    u = c.input("u", n)
    uf, vf = gcd_loop(c, u, x, c.input("d", 2 * it), iters=it)
    c.rebind("u", uf)
    c.rebind("x", vf)
    for x0 in range(1, p):
        r = simulate(c, {"x": x0, "u": p, "d": 0}, rng=random.Random(x0))
        assert (r.outputs["u"], r.outputs["x"]) == (1, 0), f"p={p} x={x0}"


@pytest.mark.parametrize("p", PRIMES)
def test_a_budget_that_is_too_short_fails_loudly(p: int):
    """The iteration budget is probabilistic, so its failure mode matters.

    Cutting the loop short leaves v != 0, and `to_dialog` frees v -- so an under-run
    surfaces as DirtyAncillaError, never as a plausible wrong tape.
    """
    n = p.bit_length()
    short = max(1, _iters(p) // 2)
    c = Circuit()
    x = c.input("x", n)
    to_dialog(c, x, c.input("d", 2 * short), p, iters=short)
    failures = 0
    for x0 in range(1, p):
        try:
            simulate(c, {"x": x0, "d": 0}, rng=random.Random(x0))
        except DirtyAncillaError:
            failures += 1
    assert failures > 0, "a half-length run must be caught, not silently accepted"


# --- the shift ------------------------------------------------------------
def test_the_shift_costs_nothing():
    """v >>= 1 is a relabelling: the loop emits no gate for it.

    Priced by comparing the loop against its own parts -- one CNOT, one controlled
    compare, one cswap, one controlled subtract per iteration and nothing else.
    """
    from qsm.field.compare import cmp_lt
    from qsm.field.swap import cswap

    n, it = 16, 5
    c = Circuit()
    u, v, d = c.input("u", n), c.input("v", n), c.input("d", 2 * it)
    gcd_loop(c, u, v, d, iters=it)

    parts = Circuit()
    pu, pv, pd = parts.input("u", n), parts.input("v", n), parts.input("d", 2 * it)
    for i in range(it):
        parts.cx(pv[0], pd[2 * i])
        cmp_lt(parts, pu, pv, pd[2 * i + 1], ctrl=pd[2 * i])
        cswap(parts, pd[2 * i + 1], pu, pv)
        for q in pv:
            parts.x(q)
        gidney_add_ctrl(parts, pd[2 * i], pu, pv)
        for q in pv:
            parts.x(q)

    assert count(c).toffoli == count(parts).toffoli
    assert count(c).peak_qubits == count(parts).peak_qubits


# --- the shrinking window --------------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
def test_the_tightest_non_clipping_window_gives_the_same_tape(p: int):
    """Narrowing to the measured envelope is exact -- it discards only zeros."""
    widths = _perfect_widths(p)
    c, it = _to_dialog_circuit(p, widths=widths)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        r = simulate(c, {"x": x, "d": 0}, rng=random.Random(x))
        assert r.outputs["d"] == encode_dialog(tape), f"p={p} x={x}"
        assert r.phase == 0


@pytest.mark.parametrize("p", PRIMES)
def test_the_narrow_window_is_cheaper(p: int):
    widths = _perfect_widths(p)
    wide, _ = _to_dialog_circuit(p)
    narrow, _ = _to_dialog_circuit(p, widths=widths)
    assert count(narrow).toffoli < count(wide).toffoli


@pytest.mark.parametrize("p", PRIMES)
def test_a_window_one_bit_too_tight_is_caught(p: int):
    """A clip corrupts the value, and the corrupted value is what gets freed.

    This is the assurance the padding constant rests on: `GCD_WIDTH_PAD_VAR` buys margin
    against a failure that reports itself, not against one that returns a wrong tape.
    """
    widths = [max(1, w - 1) for w in _perfect_widths(p)]
    c, _ = _to_dialog_circuit(p, widths=widths)
    caught = 0
    for x in range(1, p):
        try:
            simulate(c, {"x": x, "d": 0}, rng=random.Random(x))
        except DirtyAncillaError:
            caught += 1
    assert caught > 0, f"p={p}: a clipping window went entirely unnoticed"


# --- the tape in the freed bits -------------------------------------------
@pytest.mark.parametrize("p", PRIMES)
def test_the_pooled_tape_gives_the_same_tape(p: int):
    """Hosting the tape inside u and v is relabelling, so the answer cannot move.

    The pooled circuit reads its tape off a scattered register -- some wires from u, some
    from what x used to be, some freshly allocated -- and bit 2i still has to be iteration
    i's b0. If the release accounting were off by one this would return a permuted tape,
    not an error.
    """
    widths = _perfect_widths(p)
    c, it = _pooled_circuit(p, widths)
    for x in range(1, p):
        tape, _, _ = to_dialog_ref(p, x, it)
        r = simulate(c, {"x": x, "d": 0}, rng=random.Random(x))
        assert r.outputs["d"] == encode_dialog(tape), f"p={p} x={x}"
        assert r.outputs["x"] == 0
        assert r.phase == 0


@pytest.mark.parametrize("p", PRIMES)
def test_pooling_costs_no_gates_and_saves_qubits(p: int):
    """Identical Toffoli and depth, strictly lower peak. A gate difference is a bug."""
    widths = _perfect_widths(p)
    flat, _ = _to_dialog_circuit(p, widths=widths)
    pooled, _ = _pooled_circuit(p, widths)
    assert count(pooled).toffoli == count(flat).toffoli
    assert count(pooled).depth == count(flat).depth
    assert count(pooled).peak_qubits < count(flat).peak_qubits


@pytest.mark.parametrize("p", PRIMES)
def test_a_clipping_window_is_still_caught_when_the_tape_lives_in_the_window(p: int):
    """Pooling raises the stakes on the width schedule, so re-check the failure mode.

    A slot is handed to the tape on the claim that the values no longer reach it. If a
    clip makes that false the loop overwrites tape it has already written, and the values
    it frees are wrong too -- which is what has to stay loud.
    """
    widths = [max(1, w - 1) for w in _perfect_widths(p)]
    c, _ = _pooled_circuit(p, widths)
    caught = 0
    for x in range(1, p):
        try:
            simulate(c, {"x": x, "d": 0}, rng=random.Random(x))
        except DirtyAncillaError:
            caught += 1
    assert caught > 0, f"p={p}: a clipping window scribbled on the tape unnoticed"


def test_dialog_extra_bits_predicts_the_top_up_the_pool_actually_allocates():
    """The closed form is what sizes the replay's register, so it has to be the truth."""
    n = params.U
    it = schedule.iterations(n)
    W = schedule.widths(n, it)
    c = Circuit()
    x = c.input("x", n)
    tape = PooledTape(c)
    d = to_dialog(c, x, tape, params.PRIME, iters=it, widths=W,
                  cmp_bits=schedule.compare_bits(n))
    assert len(d.ids) == len(set(d.ids)) == 2 * it, "the tape reused a wire"
    assert tape.allocated == dialog_extra_bits(n, it, W)
    # ... and the released half really is the window's, twice over: 2*(n - w_final).
    assert 2 * it - tape.allocated == 2 * (n - W[-1])

    # Pre-supplying exactly that many leaves nothing to allocate.
    c2 = Circuit()
    x2 = c2.input("x", n)
    eager = PooledTape(c2, c2.input("d", dialog_extra_bits(n, it, W)))
    to_dialog(c2, x2, eager, params.PRIME, iters=it, widths=W,
              cmp_bits=schedule.compare_bits(n))
    assert eager.allocated == 0


def test_allocating_the_top_up_on_demand_is_what_gets_phase_1_under_budget():
    """The two curves peak at opposite ends of the loop; that is the whole mechanism.

    Reserving the top-up register up front is correct and costs the same gates -- it just
    holds 312 qubits through the wide iterations, where the adder's carry ladder needs
    them. This asserts the ordering claim in `PooledTape`'s docstring, and that the
    on-demand build is the one that fits.
    """
    n = params.U
    it = schedule.iterations(n)
    W = schedule.widths(n, it)
    peaks = {}
    for label, reserve in (("eager", True), ("lazy", False)):
        c = Circuit()
        x = c.input("x", n)
        extra = c.input("d", dialog_extra_bits(n, it, W)) if reserve else None
        to_dialog(c, x, PooledTape(c, extra), params.PRIME, iters=it, widths=W,
                  cmp_bits=schedule.compare_bits(n))
        peaks[label] = count(c).peak_qubits
    assert peaks["lazy"] < peaks["eager"]
    assert peaks["lazy"] <= params.PEAK_BUDGET, peaks
    assert peaks["eager"] > params.PEAK_BUDGET, peaks


# --- the truncated comparator ---------------------------------------------
#
# These run the *classical* trajectory rather than the circuit: the effect being measured
# is a property of the GCD's path, the circuit is exercised on it above, and 255-bit
# simulation is too slow to sample a rate with.
def _trajectory(x: int, W: list[int]):
    """Yield (window, u, v) for every iteration at which the comparison is used."""
    p = params.PRIME
    u, v = p, x
    for w in W:
        if v & 1:
            yield w, u, v
        b0 = v & 1
        if b0 and u > v:
            u, v = v, u
        if b0:
            v -= u
        v >>= 1


def _lead(seed: int, N: int, pad: int | None = None) -> float:
    """log2 E[sum over live slots of 2^slack] -- the constant that prices any width."""
    n = params.U
    it = schedule.iterations(n)
    W = schedule.widths(n, it) if pad is None else schedule.widths(n, it, pad=pad)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(N):
        for w, u, v in _trajectory(rng.randrange(1, params.PRIME), W):
            total += 2.0 ** (w - max(u.bit_length(), v.bit_length()))
    return math.log2(total / N)


def test_the_plain_cmp_bits_width_is_unusable_inside_the_window():
    """CMP_BITS bits of a padded window are padding, and padding always ties.

    This is the finding that sets GCD_CMP_MARGIN. It is asserted rather than described
    because the natural thing to write -- `cmp_lt(..., k=CMP_BITS)` -- is silently wrong
    here in a way the per-call formula in `params.cmp_fail_prob` does not show.
    """
    n = params.U
    W = schedule.widths(n, schedule.iterations(n))
    rng = random.Random(5)
    ties_naive = ties_designed = live = 0
    designed = schedule.compare_bits(n)
    for _ in range(40):
        for w, u, v in _trajectory(rng.randrange(1, params.PRIME), W):
            live += 1
            for k, hit in ((params.CMP_BITS, "n"), (designed, "d")):
                if w <= k:
                    continue
                lo, mask = w - k, (1 << w) - 1
                tied = ((u & mask) >> lo) == ((v & mask) >> lo)
                if hit == "n":
                    ties_naive += tied
                else:
                    ties_designed += tied
    assert live > 4000, "expected ~150 usable comparisons per run"
    assert ties_naive > live // 4, (
        f"only {ties_naive}/{live} ties at k=CMP_BITS -- the effect being guarded "
        f"against has gone away, so GCD_CMP_MARGIN needs re-deriving"
    )
    assert ties_designed == 0, (
        f"{ties_designed} ties at the designed width {designed}; that is a ~2^-45 event"
    )


def test_the_comparator_margin_is_measured_against_the_pad_not_the_width():
    """log2 E[.] moves one-for-one with the pad, which is why the constant is a margin.

    Same seed on both sides, so the trajectories are identical and the relation is exact
    rather than statistical -- to within 1e-4. The residual is the `min(..., n)` clamp in
    `schedule.width`: the first few windows are already at full width and do not move when
    the pad does, so they contribute the same slack to both sides.
    """
    base = _lead(1, 400)
    for delta in (-10, +10):
        shifted = _lead(1, 400, pad=schedule.width_pad() + delta)
        assert abs((shifted - base) - delta) < 1e-3, (
            f"pad {delta:+d} moved the constant by {shifted - base:.3f}, not {delta}"
        )


def test_the_measured_lead_constant_still_holds():
    """Re-measure GCD_CMP_LEAD_LOG2, cheaply.

    Tolerance is 3 bits, not 0.1: the expectation is dominated by the rare runs that get
    far ahead of the width schedule, so a few-hundred-sample estimate wanders by ~2 bits.
    `params` records the converged N = 150,000 figure; this only has to catch a change of
    algorithm, not sharpen the constant.
    """
    got = _lead(11, 400) - schedule.width_pad()
    assert abs(got - params.GCD_CMP_LEAD_LOG2) < 3.0, got


def test_the_designed_comparator_is_far_below_the_iteration_budget():
    """The comparator must not be what limits an inversion; the budget already is."""
    cmp_p = schedule.compare_fail_prob()
    assert cmp_p < 2 ** -40
    assert cmp_p < schedule.iteration_fail_prob() * 2 ** -25


def test_truncating_the_comparator_is_worth_real_gates():
    """~13% of Phase 1, which is why the width is tuned rather than switched off."""
    exact, _ = _to_dialog_circuit_255(None)
    trunc, _ = _to_dialog_circuit_255(schedule.compare_bits())
    saved = count(exact).toffoli - count(trunc).toffoli
    assert saved > 20000, saved
    assert count(trunc).toffoli < count(exact).toffoli


def test_the_truncated_255_bit_loop_agrees_with_the_reference():
    """The circuit itself, at the real width -- a few samples, because it is slow."""
    c, it = _to_dialog_circuit_255(schedule.compare_bits())
    rng = random.Random(3)
    for _ in range(4):
        x = rng.randrange(1, params.PRIME)
        tape, _, _ = to_dialog_ref(params.PRIME, x, it)
        r = simulate(c, {"x": x, "d": 0}, rng=rng)
        assert r.outputs["d"] == encode_dialog(tape)
        assert r.outputs["x"] == 0
        assert r.phase == 0
