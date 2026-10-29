"""The classical reference, exhaustively — before a single qubit is allocated.

Everything here is plain integer arithmetic. Its job is to pin down the facts the
circuits will be built against: that the loop terminates inside the budget, which
composition is the multiply and which is the division, and how wide the registers have
to be.
"""

import math
import random

import pytest

from qsm import params
from qsm.inversion import schedule
from qsm.inversion.reference import (
    DIALOG_STATES,
    apply_dialog,
    apply_dialog_inverse,
    dialog_matrix,
    dialog_widths,
    moddiv_inplace_ref,
    modmul_inplace_ref,
    to_dialog,
    un_dialog,
)

SMALL_PRIMES = [29, 61, 127, 251, 1021, 8191]
# Tests that sweep x *and* y are quadratic; run those exhaustively only where that is
# cheap, and sample y elsewhere. The x sweep stays exhaustive everywhere.
EXHAUSTIVE_PRIMES = [29, 61, 127, 251]


def _iters(p: int) -> int:
    return schedule.iterations(p.bit_length())


def _ys(p: int, seed: int) -> list[int]:
    """Every y for a small prime, a fixed sample for a larger one."""
    if p in EXHAUSTIVE_PRIMES:
        return list(range(p))
    rng = random.Random(seed)
    return [0, 1, p - 1] + [rng.randrange(p) for _ in range(60)]


# --- Phase 1 ---------------------------------------------------------------
@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_phase1_terminates_within_budget(p: int):
    """Every x in [1,p) must reach (u,v) == (1,0) inside `iterations(n)`."""
    it = _iters(p)
    for x in range(1, p):
        tape, u, v = to_dialog(p, x, it)
        assert (u, v) == (1, 0), f"p={p} x={x}: ended at (u,v)=({u},{v}) after {it}"
        assert len(tape) == it


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_phase1_only_ever_logs_the_three_valid_states(p: int):
    """(b0, b0&b1) == (0,1) is impossible; the 3-state compression depends on it."""
    it = _iters(p)
    for x in range(1, p):
        tape, _, _ = to_dialog(p, x, it)
        for st in tape:
            assert st in DIALOG_STATES, f"p={p} x={x}: impossible state {st}"


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_phase1_never_outgrows_p(p: int):
    """max(u,v) <= p throughout, so n-bit registers hold the values themselves."""
    n = p.bit_length()
    it = _iters(p)
    for x in range(1, p):
        assert max(dialog_widths(p, x, it)) <= n, f"p={p} x={x} exceeded {n} bits"


# --- Direction: which composition is the multiply? -------------------------
@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_replay_of_the_full_tape_reconstructs_bezout(p: int):
    """Over the integers the tape maps (1,0) -> (p, x). That is what fixes the direction.

    `dialog_matrix(tape)` with no reduction is the integer replay; its first column is
    (p, x), so mod p the first row vanishes and feeding (y, 0) puts y*x in the second
    slot. Nothing else about the convention needs to be taken on faith.
    """
    it = _iters(p)
    for x in range(1, p):
        tape, _, _ = to_dialog(p, x, it)
        m00, _, m10, _ = dialog_matrix(tape)
        assert (m00, m10) == (p, x), f"p={p} x={x}: first column {(m00, m10)}"


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_forward_replay_is_the_multiply(p: int):
    """apply_dialog(tape(x), y, 0) == (0, x*y mod p) for every x, y."""
    it = _iters(p)
    ys = _ys(p, 11)
    for x in range(1, p):
        tape, _, _ = to_dialog(p, x, it)
        for y in ys:
            a, b = apply_dialog(tape, y, 0, p)
            assert a == 0, f"p={p} x={x} y={y}: a={a}, expected 0"
            assert b == x * y % p, f"p={p} x={x} y={y}: got {b}"


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_reverse_replay_is_the_division(p: int):
    """The same tape run backwards divides -- so the division is free, given the multiply."""
    it = _iters(p)
    ys = _ys(p, 12)
    for x in range(1, p):
        tape, _, _ = to_dialog(p, x, it)
        xinv = pow(x, -1, p)
        for y in ys:
            a, b = apply_dialog_inverse(tape, 0, y, p)
            assert b == 0, f"p={p} x={x} y={y}: b={b}, expected 0"
            assert a == y * xinv % p, f"p={p} x={x} y={y}: got {a}"


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_replay_and_its_inverse_round_trip(p: int):
    it = _iters(p)
    rng = random.Random(p)
    for _ in range(200):
        x = rng.randrange(1, p)
        a0, b0 = rng.randrange(p), rng.randrange(p)
        tape, _, _ = to_dialog(p, x, it)
        a, b = apply_dialog(tape, a0, b0, p)
        assert apply_dialog_inverse(tape, a, b, p) == (a0, b0)


@pytest.mark.parametrize("p", EXHAUSTIVE_PRIMES)
def test_inplace_helpers_agree_with_plain_modular_arithmetic(p: int):
    """The user-facing entry points, on the small primes where x*y is cheap to sweep."""
    for x in range(1, p):
        xinv = pow(x, -1, p)
        for y in range(0, p, max(1, p // 32)):
            assert modmul_inplace_ref(p, x, y) == x * y % p
            assert moddiv_inplace_ref(p, x, y) == y * xinv % p


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_inversion_is_the_y_equals_one_case(p: int):
    """Inversion is the y = 1 case of the division, ALGORITHMICALLY.

    True of the reference and false of the circuit, which is the whole of doc 12: the
    truncated `-F` in `un_apply_dialog_j2` has a danger set that a constant numerator sits
    in and stays in. Nothing is wrong with this assertion -- it is about the algorithm, and
    the algorithm does invert. It is kept next to that warning on purpose.
    """
    for x in range(1, p):
        assert moddiv_inplace_ref(p, x, 1) == pow(x, -1, p)


# --- Schedule --------------------------------------------------------------
def test_iteration_and_dialog_sizes_match_the_published_design():
    """399 iterations and 665 dialog qubits at n = 255 -- frenchie's numbers exactly."""
    assert schedule.iterations(255) == 399
    assert schedule.dialog_bits(399) == 665
    assert 399 % params.GCD_ITER_BLOCK == 0


def test_replay_peak_is_exactly_the_budget_at_the_default_knob():
    """d + x + y = 665 + 255 + 255 = 1,175 = PEAK_BUDGET, with zero left for ancillas.

    This is the whole reason `GCD_ITER_VAR` cannot simply be raised to a 2^-40 setting,
    and the reason the replay has to use a 1-ancilla adder. It is asserted so that a
    future change to either knob fails here with the tradeoff spelled out.
    """
    peak = schedule.dialog_bits() + 2 * params.U
    assert peak == params.PEAK_BUDGET


def test_iteration_budget_is_the_worst_knob_in_params():
    """~2^-15, against 2^-40 for every arithmetic truncation. Documented, not accidental."""
    p_iter = schedule.iteration_fail_prob()
    assert 2 ** -16 < p_iter < 2 ** -14
    assert p_iter > params.cmp_fail_prob() * 2 ** 20
    assert p_iter > params.carry_fail_prob() * 2 ** 20


@pytest.mark.parametrize("n", [8, 13, 64, 255])
def test_width_schedule_is_monotone_and_bracketed(n: int):
    ws = schedule.widths(n)
    assert ws[0] == n, "the first iteration needs the full width"
    assert all(a >= b for a, b in zip(ws, ws[1:])), "widths must not grow"
    assert all(0 <= w <= n for w in ws)
    assert all(w % params.GCD_WIDTH_STEP == 0 or w == n for w in ws)


def test_width_schedule_covers_the_real_values():
    """The measurement the padding is chosen from: no clipping over many random runs.

    A clip is not a graceful failure -- the freed high bits hold the tape, so an
    overflowing u/v corrupts the dialog as well as the value. Kept small enough to run
    in the suite; `params` records the 798,000-slot version.
    """
    p, n = params.PRIME, params.U
    it = schedule.iterations(n)
    sched = schedule.widths(n, it)
    rng = random.Random(4)
    for _ in range(40):
        real = dialog_widths(p, rng.randrange(1, p), it)
        for i, (w, allowed) in enumerate(zip(real, sched)):
            assert w <= allowed, f"iteration {i}: value needs {w} bits, window is {allowed}"


def test_width_pad_matches_frenchies_constant():
    assert schedule.width_pad(255) == 37


def test_dialog_slices_tile_the_register_exactly():
    it = schedule.iterations()
    seen = set()
    for i in range(it):
        lo, hi = schedule.dialog_slice(i)
        assert hi - lo == 5
        seen.add((lo, hi))
    assert len(seen) == it // params.GCD_ITER_BLOCK
    assert max(hi for _, hi in seen) == schedule.dialog_bits(it)


def test_gcd_iterations_scales_with_the_knob():
    """Raising GCD_ITER_VAR buys accuracy and costs peak; both directions must move."""
    base = schedule.iterations(255, var=params.GCD_ITER_VAR)
    hi = schedule.iterations(255, var=4.3)
    assert hi > base
    assert schedule.iteration_fail_prob(255, hi) < 2 ** -40
    assert schedule.dialog_bits(hi) + 2 * params.U > params.PEAK_BUDGET


def test_measured_iteration_statistics_still_hold():
    """GCD_SHIFT_RATE and GCD_ITER_SD are measurements; re-measure them, cheaply."""
    p, n = params.PRIME, params.U
    rng = random.Random(2026)
    N = 400
    xs = []
    for _ in range(N):
        # count real iterations by running a generous budget and finding where v hits 0
        x = rng.randrange(1, p)
        u, v, k = p, x, 0
        while v:
            b0 = v & 1
            if b0 and u > v:
                u, v = v, u
            if b0:
                v -= u
            v >>= 1
            k += 1
        xs.append(k)
    mean = sum(xs) / N
    sd = math.sqrt(sum((t - mean) ** 2 for t in xs) / (N - 1))
    assert abs(mean / n - params.GCD_SHIFT_RATE) < 0.01, mean / n
    assert abs(sd / math.sqrt(n) - params.GCD_ITER_SD) < 0.10, sd / math.sqrt(n)


# --- the reverse pass ------------------------------------------------------
@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_un_dialog_reconstructs_both_the_prime_and_the_input(p: int):
    """|d> -> |p, x>, exhaustively over x. The composite depends on this exactly.

    `to_dialog` consumes `x`, so the in-place multiply has nothing to give back at the
    end unless the tape can be run backward -- and the tape is all that is left. Both
    halves are asserted: `x` is the answer the composite needs, and `p` coming back is
    the check that the reverse pass tracked the true (u, v) state rather than merely
    landing on a plausible value for x.
    """
    it = schedule.iterations(p.bit_length())
    for x in range(1, p):
        tape, u, v = to_dialog(p, x, it)
        assert (u, v) == (1, 0), f"p={p} x={x}: forward pass did not converge"
        assert un_dialog(tape) == (p, x), f"p={p} x={x}"


@pytest.mark.parametrize("p", EXHAUSTIVE_PRIMES)
def test_the_reverse_pass_is_the_replay_loop_without_the_modulus(p: int):
    """The structural claim, asserted as an identity rather than left in a docstring.

    `un_dialog` is `apply_dialog`'s loop with `mod p` deleted: double, conditionally add,
    conditionally swap. Running the replay's own body over the integers must therefore
    reproduce it step for step -- which is what says the composite is three runs of ONE
    loop shape and not three separate circuits.
    """
    it = schedule.iterations(p.bit_length())
    for x in range(1, p):
        tape, _, _ = to_dialog(p, x, it)
        a, b = 1, 0                                   # the state to_dialog leaves behind
        for b0, s in reversed(tape):
            b = 2 * b                                 # moddouble, without the modulus
            if b0:
                b = a + b                             # modadd_lazy, without the modulus
            if s:
                a, b = b, a                           # the same cswap
        assert (a, b) == un_dialog(tape) == (p, x), f"p={p} x={x}"


@pytest.mark.parametrize("p", EXHAUSTIVE_PRIMES)
def test_the_reverse_pass_needs_no_comparator(p: int):
    """Phase 1 derives `b1` with a truncated `u > v`; the reverse pass reads it.

    That asymmetry is why uncomputing Phase 1 is CHEAPER than Phase 1, which is not the
    usual relationship. Pinned by construction: `un_dialog` is fed nothing but the tape,
    so if it needed a comparison it could not compute one -- and it still reproduces the
    exact (u, v) trajectory the forward pass took, comparisons and all.
    """
    it = schedule.iterations(p.bit_length())
    for x in range(1, p):
        tape, _, _ = to_dialog(p, x, it)
        # Forward trajectory, recorded.
        fwd, u, v = [], p, x
        for _ in range(it):
            fwd.append((u, v))
            b0 = v & 1
            s = b0 & (1 if u > v else 0)
            if s:
                u, v = v, u
            if b0:
                v -= u
            v >>= 1
        # Backward trajectory, from the tape alone, must retrace it.
        bwd, a, b = [], 1, 0
        for b0, s in reversed(tape):
            b <<= 1
            if b0:
                b += a
            if s:
                a, b = b, a
            bwd.append((a, b))
        assert bwd == fwd[::-1], f"p={p} x={x}: the reverse pass left the forward path"


@pytest.mark.parametrize("p", EXHAUSTIVE_PRIMES)
def test_the_reverse_pass_registers_grow_where_the_forward_ones_shrink(p: int):
    """The width schedule has to run backward too, and that is a real cost asymmetry.

    Phase 1's largest single gate saving is that `u` and `v` shrink as the loop runs. The
    reverse pass sees the same widths in the opposite order, so it cannot reuse the
    forward schedule as written -- `schedule.width(iters - 1 - i, n)` is the mapping, and
    this pins that it is the right one.
    """
    it = schedule.iterations(p.bit_length())
    for x in range(1, p):
        tape, _, _ = to_dialog(p, x, it)
        fwd = dialog_widths(p, x, it)
        bwd, a, b = [], 1, 0
        for b0, s in reversed(tape):
            b <<= 1
            if b0:
                b += a
            if s:
                a, b = b, a
            bwd.append(max(a, b).bit_length())
        assert bwd[::-1] == fwd, f"p={p} x={x}"
        assert bwd[-1] == p.bit_length(), "the last backward step is the widest"
