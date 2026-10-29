"""The GCD budget constants, checked against behaviour rather than against themselves.

Three claims are pinned here, and each one was wrong in an earlier draft of `params.py`:

  * the jump-1 and jump-2 budgets are matched on SIGMAS. `GCD_ITER_VAR` multiplies
    sqrt(n), not the sd, so reusing it across jumps hands jump-2 7.7 sigma against
    jump-1's 4.02 and quietly credits jump-2 with 42 qubits it did not earn.
  * the width schedule carries its own failure rate, and it is EXPONENTIAL. A Gaussian
    fitted to the same mean and sd is 21 bits optimistic, which is the difference between
    a term that matters and a term that can be dropped.
  * the width schedule's worst case is exactly `rate*(n-1)` and grows linearly in n,
    while its pad grows as sqrt(n) -- so no choice of `GCD_WIDTH_PAD_VAR` makes it exact.

The simulation-backed figures come from 200,000 uniform x at n = 255 (excess tail) and
60,000 (iteration counts); the small-n statements below are exhaustive.
"""

from __future__ import annotations

import math
import random

import pytest

from qsm import params


def _widths(p: int, x: int, jump: int) -> list[int]:
    """bitlen(max(u, v)) at the top of each iteration of the jump-J loop."""
    u, v, i, out = p, x, 0, []
    while v:
        out.append(max(u, v).bit_length())
        for j in range(jump):
            if j == 0 and i > 0:
                v >>= 1
            elif v % 2 == 0 and v:
                v >>= 1
            else:
                break
        if v & 1:
            if u > v:
                u, v = v, u
            v -= u
        i += 1
    return out


def _iters(p: int, x: int, jump: int) -> int:
    return len(_widths(p, x, jump))


def _iters_classic(p: int, x: int) -> int:
    """The ORIGINAL loop ordering -- subtract, then shift -- which is what 2n-1 bounds.

    Not interchangeable with `_iters(p, x, 1)`. The jump form hoists the shift to the top
    of the iteration, so the two split the same sequence of operations into iterations
    differently and their counts differ by one. Comparing a bound proved for one against a
    count produced by the other is off by exactly that.
    """
    u, v, i = p, x, 0
    while v:
        if v & 1:
            if u > v:
                u, v = v, u
            v -= u
        v >>= 1
        i += 1
    return i


# --- the budgets are matched on sigmas --------------------------------------
def test_the_two_jump_budgets_sit_at_the_same_place_in_their_own_tails():
    """Not at the same `var`, which is a different thing and worth 42 qubits.

    `GCD_ITER_VAR = 2.4` is a multiplier on sqrt(n). Jump-2's sd is 0.310*sqrt(n) against
    jump-1's 0.597, so the same var is 7.7 sigma for one and 4.02 for the other -- 279
    iterations instead of 261, at 2^-48 instead of 2^-16. Comparing tape sizes across that
    would credit jump-2 with qubits it did not earn.
    """
    assert params.GCD_ITER_Z == pytest.approx(4.017, abs=0.01)

    def z_of(iters, rate, sd, n=params.U):
        return (iters - rate * n) / (sd * math.sqrt(n))

    z1 = z_of(params.gcd_iterations(), params.GCD_SHIFT_RATE, params.GCD_ITER_SD)
    z2 = z_of(params.gcd_j2_iterations(), params.GCD_J2_SHIFT_RATE, params.GCD_J2_ITER_SD)
    # Both budgets round up to a whole block, so they cannot land on exactly the same z.
    assert abs(z1 - z2) < 0.35, f"budgets are not tail-matched: {z1:.2f} vs {z2:.2f} sigma"
    assert (params.gcd_iterations(), params.gcd_j2_iterations()) == (399, 261)


def test_the_jump_two_tape_fits_under_the_peak_budget_with_room_to_spare():
    """609 + 255 + 255 = 1119 against 1175. The 56 is what the whole exercise is for.

    Split honestly: 25 of it is jump-1's own packing being suboptimal (5-in-3 = 1.6667
    bits/iteration where 8-in-5 = 1.6000 would do, i.e. 640), and only 31 is jump-2's.
    """
    d1, d2 = params.gcd_dialog_bits(), params.gcd_j2_dialog_bits()
    assert (d1, d2) == (665, 609)
    assert d1 + 2 * params.U == params.PEAK_BUDGET, "jump-1 sits at the budget exactly"
    assert d2 + 2 * params.U == 1119
    assert params.PEAK_BUDGET - (d2 + 2 * params.U) == 56

    denser_j1 = math.ceil(params.gcd_iterations() / 5) * 8
    assert denser_j1 == 640, "8 bits per 5 iterations is base 3's real optimum"
    assert d1 - denser_j1 == 25 and denser_j1 - d2 == 31


@pytest.mark.parametrize("jump,rate,sd", [(1, "GCD_SHIFT_RATE", "GCD_ITER_SD"),
                                          (2, "GCD_J2_SHIFT_RATE", "GCD_J2_ITER_SD")])
def test_the_iteration_rate_and_sd_match_a_fresh_sample(jump, rate, sd):
    """The constants are re-derived from a run with a different seed than they were fit on.

    A tolerance of 1% on the rate and 5% on the sd, at 3,000 samples -- loose enough that
    it is testing the constant and not the seed.
    """
    p = (1 << params.U) - params.F
    rng = random.Random(31337)
    xs = [rng.randrange(1, p) for _ in range(3000)]
    its = [_iters(p, x, jump) for x in xs]
    m = sum(its) / len(its)
    s = (sum((i - m) ** 2 for i in its) / len(its)) ** 0.5
    assert m / params.U == pytest.approx(getattr(params, rate), rel=0.01)
    assert s / math.sqrt(params.U) == pytest.approx(getattr(params, sd), rel=0.05)


def test_jump_two_takes_two_thirds_the_iterations_and_sheds_width_proportionally_faster():
    """The width rate is the iteration ratio applied to the jump-1 rate, to 0.1%.

    Which is the expected relationship and not an independent fact: the same total bits
    are shed over 2/3 as many iterations. Recorded because the earlier note claimed the
    rate had to be measured because it could not be rescaled -- the per-index SCHEDULE
    cannot (an iteration eats a variable number of halvings), but the rate plainly can.
    """
    ratio = params.GCD_J2_SHIFT_RATE / params.GCD_SHIFT_RATE
    assert ratio == pytest.approx(2 / 3, abs=0.005)
    assert params.GCD_WIDTH_RATE / ratio == pytest.approx(params.GCD_J2_WIDTH_RATE, rel=0.002)


# --- the width schedule is a failure mode, not a margin ---------------------
def test_the_width_clip_is_a_declared_failure_term_and_not_a_rounding_error():
    """2^-18.6 sits 4.7 bits under the iteration budget, so it moves the total by ~9%.

    The point of the assertion is that it is neither negligible (like the comparator at
    2^-45) nor dominant. A total that omits it is wrong by more than its own last digit.
    """
    it = params.gcd_iteration_fail_prob()
    wc = params.gcd_width_clip_fail_prob()
    cmp_ = params.gcd_compare_fail_prob()
    assert math.log2(wc) == pytest.approx(-18.6, abs=0.2)
    assert cmp_ < wc < it, "the width clip is the middle term, not the largest or smallest"
    assert (it + wc + cmp_) / it == pytest.approx(1.09, abs=0.02)
    # ...and jump-2's is better, which is why its total is better despite a similar budget.
    assert math.log2(params.gcd_width_clip_fail_prob(jump=2)) == pytest.approx(-25.3, abs=0.2)


def test_a_gaussian_width_tail_would_be_twenty_bits_optimistic():
    """The excess is a max over a walk. Fitting the familiar shape is worth 21 bits here.

    Mean 7.21 and sd 4.24 at n = 255 put the pad 7.0 sd out, which a Gaussian prices at
    2^-39.8 against the measured-and-fitted 2^-18.6. This is the reason
    `gcd_width_clip_fail_prob` is exponential, and it is asserted rather than commented
    because picking the tail shape by habit is exactly the mistake it guards.
    """
    pad = math.ceil(params.GCD_WIDTH_PAD_VAR * math.sqrt(params.U))
    assert pad == 37
    mean, sd = 7.21, 4.24                       # measured, N = 200,000 at n = 255
    gauss = math.log2(0.5 * math.erfc((pad - mean) / sd / math.sqrt(2)))
    real = math.log2(params.gcd_width_clip_fail_prob(pad))
    assert gauss == pytest.approx(-39.8, abs=0.5)
    assert real - gauss > 19, f"gaussian is only {real - gauss:.1f} bits off, expected ~21"


@pytest.mark.parametrize("n", [14, 16, 18, 20])
def test_the_worst_case_width_excess_is_rate_times_n_minus_one_and_clips_the_pad(n):
    """Exhaustive at small n: the worst input is 2^(n-1), and it overshoots by rate*(n-1).

    `u` sits at `p` for n-1 iterations while the schedule shrinks under it, so the excess
    is exact rather than statistical. It grows LINEARLY in n while the pad grows as
    sqrt(n), which is why no setting of `GCD_WIDTH_PAD_VAR` makes the window exact and why
    a 2n-1 iteration budget buys nothing unless the window is widened to match.
    """
    p = (1 << n) - params.F
    rate = params.GCD_WIDTH_RATE
    worst, at = max(
        ((b - (n - rate * i), x)
         for x in range(1, p) for i, b in enumerate(_widths(p, x, 1))),
        key=lambda t: t[0],
    )
    assert at == 1 << (n - 1), f"n={n}: worst input is {at}, expected 2^(n-1)"
    assert worst == pytest.approx(rate * (n - 1), abs=1e-9)
    assert worst > math.ceil(params.GCD_WIDTH_PAD_VAR * math.sqrt(n)), "must clip its pad"


def test_jump_two_does_not_attain_the_two_n_minus_one_bound_that_jump_one_attains():
    """So a provably-exact jump-2 saves nothing: both must still budget 2n-1 = 509.

    The bound itself holds for both -- the unconditional halving drops
    bitlen(u)+bitlen(v) by at least one per iteration. Jump-1 attains it, at x = 2^(n-1).
    Jump-2 does not come close, and its worst case is CLIMBING with n (1.20n at n = 10 to
    1.35n at n = 20), so small-n enumeration cannot say where it lands. Jump-2's win is
    therefore entirely in the probabilistic regime, which is a limitation worth failing a
    test over if it ever silently stops being true.
    """
    seen = []
    for n in range(10, 19):
        p = (1 << n) - params.F
        m1 = max(_iters_classic(p, x) for x in range(1, p))
        m2 = max(_iters(p, x, 2) for x in range(1, p))
        assert m1 == 2 * n - 1, f"n={n}: jump-1 max {m1}, bound {2*n-1}"
        assert m2 < 2 * n - 1, f"n={n}: jump-2 unexpectedly attained the bound"
        assert _iters_classic(p, 1 << (n - 1)) == 2 * n - 1, "jump-1's worst input is 2^(n-1)"
        # No claim is made about WHICH input is jump-2's worst. It moves around with n
        # (2^(n-2) at some odd n, an irregular value at others, ties at n = 10), and
        # reading a pattern off a scan of nine values is how the extrapolation this test
        # exists to forbid gets made in the first place.
        seen.append(m2 / n)
    assert seen[-1] > seen[0], "jump-2's worst case is still climbing; do not extrapolate"
