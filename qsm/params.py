"""Field parameters and approximation knobs, in one place so they can be retuned.

The field is p = 2^U - F  (pseudo-Mersenne). F is the whole reason modular reduction is
cheap here: since B - p = (B - 2^U) + F, and subtracting 2^U is a single X gate on bit U,
*adding F is the entire arithmetic cost of a reduction*. Small F => cheap reduction.

    Curve25519   p = 2^255 - 19            F = 19          (5 bits)
    secp256k1    p = 2^256 - 2^32 - 977    F = 2^32 + 977  (33 bits)

That 28-bit gap in F is the concrete field-level advantage 25519 has over secp256k1, and
it shows up directly in CARRY_WINDOW below.

Approximation knobs
-------------------
Shor's algorithm only needs the point-addition circuit to succeed with large constant
probability, so both arithmetic truncations below are deliberate. Neither leaves dirty
ancillas: a truncated circuit is still an exact unitary, it just implements a slightly
different function. The cost is a little amplitude on wrong answers, not entangled
garbage.

    CMP_BITS      compare only the top k bits of two registers.
                  Wrong only when the top k bits TIE (prob 2^-k), and given a tie the
                  discarded low bits decide it 50/50, so
                      P = 2^-(k+1)

    CARRY_WINDOW  add F into only the low L bits, dropping the carry past bit L-1.
                  Wrong exactly when (B mod 2^L) + F >= 2^L, i.e. when B's low part is
                  one of the top F values in [0, 2^L). That is F values out of 2^L:
                      P = F / 2^L                        (exact, not asymptotic)

To hit P = 2^-40 the second needs L = 40 + log2(F): **45 bits for us, 73 for secp256k1**.
The 28-bit gap is `log2(2^32+977) - log2(19)` and is the whole field-level advantage: at
any target accuracy, secp256k1's window is 28 bits -- and so ~28 Toffoli -- wider.

**ecdsa.fail uses the same rule, `L = budget + F_BITLEN`, and has moved its budget.**
Both of its shipped constants are real, at different dates:

    trailmix_port/rfold_mbu.rs:42   (Jun 2026)  RFOLD_WINDOW = 73 = 40 + 33  -> 2^-40
                              :48               COMPARE_TOPK = 64            -> 2^-64
    trailmix_ludicrous/arith.rs:452 (Aug 2026)  LSBS = 20 + F_BITLEN = 53    -> 2^-21
                               :454             MSBS = PAD = 19              -> ~2^-19

Their own comment at rfold_mbu.rs:36 spells the first one out: "at most
RFOLD_WINDOW-33 = 40 bits ... <= 2^-40 per call, within Shor's budget". So 73 was never
our arithmetic attributed to them -- it was their constant at a 2^-40 budget, the same
budget we chose, and it is the independent confirmation that `L = budget + log2(F)` is
the right law. They have since traded accuracy for gates, down to ~2^-20 at HEAD.

The consequence for citation: **73 vs our 45 is the like-for-like pair** (both 2^-40),
and the gap is 28 = 33 - 5 = log2(F) difference. Comparing our numbers to their *current*
HEAD figures instead compares a 2^-40 circuit to a 2^-20 one. `python -m qsm.audit curves`
prints both budgets so the comparison is always stated.

Each extra bit costs about one Toffoli and halves the failure probability, so these are
cheap to raise. Retune here; nothing else hard-codes them.

Which field is live
-------------------
`QSM_FIELD` picks one of `FIELDS` below at import time, defaulting to 25519. Both windows
are derived from `ERROR_BUDGET`, matching the field-operation truncation target under
the stated input models. This does not equalize GCD-tail probabilities or total circuit
error, and it does not establish those models for arbitrary quantum input distributions.
"""

from __future__ import annotations

import math
import os

# --- the field -------------------------------------------------------------
# Selected at import time by the QSM_FIELD environment variable, because every module
# below binds `U`/`F`/`PRIME` with `from ..params import ...` and a mid-run swap would
# leave half the process on each prime. One process, one field.
#
#     python -m qsm.bench 'point.*'                     -> 25519
#     QSM_FIELD=secp256k1 python -m qsm.bench 'point.*' -> 2^256 - 2^32 - 977
#
# secp256k1 is here for ONE reason: it is the prime every published baseline uses, so
# running our own construction over it is the only way to say how much of a margin is
# the circuit and how much is the field. See docs/decisions/15-prime-port.md.
FIELDS = {
    "25519":     dict(u=255, f=19),
    "secp256k1": dict(u=256, f=(1 << 32) + 977),
}

FIELD = os.environ.get("QSM_FIELD", "25519")
if FIELD not in FIELDS:
    raise SystemExit(f"QSM_FIELD={FIELD!r}: pick one of {sorted(FIELDS)}")

U = FIELDS[FIELD]["u"]      # p = 2^U - F
F = FIELDS[FIELD]["f"]      # pseudo-Mersenne deficit
PRIME = (1 << U) - F

# Odd F is a standing assumption, not an accident of 19: `modhalve`'s `(f-1)/2`, the
# parity argument behind `moddouble_lazy_overflow_prob`'s floor(f/2), and the
# "2a - p is always odd" step all break for even F. Both fields here satisfy it.
assert F & 1, f"{FIELD}: F must be odd (the lazy/halve analysis assumes it)"

# --- approximation knobs ---------------------------------------------------
# This target applies to the per-operation truncation models, not the composed circuit
# or arbitrary input states. GCD iteration/width tails have separate tuning and assumptions.
ERROR_BUDGET = 40

CMP_BITS = ERROR_BUDGET     # top-k truncated comparison -> 2^-(k+1); no F dependence
CARRY_WINDOW = math.ceil(ERROR_BUDGET + math.log2(F))   # truncated +F -> F/2^L
# A square's narrow row drops an integer carry, not an F-correction: 2^-(ripple+1).
# Earlier tuning reused CARRY_WINDOW here, over-provisioning large-F fields.
SQUARE_RIPPLE = ERROR_BUDGET

# --- the peak budget -------------------------------------------------------
# Historical comparison point for the tuned space-efficient line, not a correctness
# constraint or a ceiling on conservative arithmetic. Source: the original
# ecdsafail-challenge/README.md "Reference numbers":
#
#     Challenge initial circuit                 3,942,753 Toffoli   2,715 qubits
#     Google's private low-qubit Pareto point   2,700,000           1,175   <- this one
#     Google's private low-gate Pareto point    2,100,000           1,425
#
# Resource regression tests track which registered variants exceed this fixed comparison
# point. Canonical conservative variants are measured separately without this ceiling.
#
# Their score is Toffoli x peak, and the scoring rule is why peak alone is the wrong thing
# to minimise (doc 05 section 1) -- but it is the right thing to BOUND, because peak is the
# one resource a scheduler cannot trade away by running longer.
#
# `tests/test_peak_budget.py` fails loudly when a registered version crosses it. Raise
# this only against a newly published number, with the citation, never to make a test pass.
PEAK_BUDGET = 1175
PEAK_BUDGET_SOURCE = ("ecdsafail-challenge/README.md, Reference numbers: "
                      "Google's private low-qubit Pareto point, 1,175 peak qubits")


# --- failure probabilities (analytic; sampling cannot resolve these) --------
def cmp_fail_prob(
    k: int | None = CMP_BITS,
    n: int | None = None,
    informative_bits: int | None = None,
) -> float:
    """P(a top-k truncated register comparison is wrong).

        P = 2^-(k_eff + 1)

    where k_eff is the number of compared bits that actually carry entropy. Wrong only
    on a tie (prob 2^-k_eff), and a tie is resolved by the discarded low bits, so it is
    right half the time even then.

    `informative_bits` overrides k_eff, and modular addition is exactly why it exists:
    it compares values < 2^U inside a (U+1)-bit register, so the TOP window bit is 0 for
    both operands and contributes nothing. k_eff = k - 1, giving 2^-k -- twice the naive
    figure. Measured across several widths; see
    tests/test_field_compare.py::test_operand_range_costs_one_window_bit.

    Correlation between the operands (modadd compares (a+b) mod p against a) turns out
    NOT to matter measurably -- it is the operand *range* that costs the bit. A
    comparison against a *constant* is a different calculation again; see
    `cmp_ge_const_fail_prob`.
    """
    if k is None or (n is not None and k >= n):
        return 0.0
    k_eff = k if informative_bits is None else informative_bits
    if k_eff <= 0:
        return 0.5
    return 2.0 ** -(k_eff + 1)


def cmp_ge_const_fail_prob(c: int, n: int, k: int | None = CMP_BITS) -> float:
    """P(a top-k truncated comparison `B >= c` is wrong), for uniform n-bit B.

    On a tie in the top k bits the circuit answers ">=", so it errs exactly when the low
    parts say otherwise:  P = P(tie) * P(B_low < c_low) = c_low / 2^n, with
    c_low = c mod 2^(n-k).

    Note the strong dependence on c's *low* bits. For c = p = 2^U - F the low part is
    nearly maximal, so this is a poor way to test `B >= p`; comparing against 2^U instead
    (i.e. just reading bit U) is both free and far more accurate -- see
    `msb_reduce_fail_prob`.
    """
    if k is None or k >= n:
        return 0.0
    return (c % (1 << (n - k))) / float(1 << n)


def chunked_add_boundary_fail_prob(
    n: int = U, chunk: int = 64, k: int | None = CMP_BITS
) -> float:
    """P(a VENTED chunked adder gets at least one boundary carry wrong), per addition.

    `add_chunked.gidney_add_chunked` kills block j's carry-out with the identity
    c_out = [S_j < A_j + c_in]. The vented variant drops the `+ c_in` and truncates the
    comparison to the top k bits of the block, so it is wrong exactly on a tie the
    discarded information would have broken:

        P = (blocks - 1) * 2^-(k_eff + 1),   k_eff = min(k, chunk)

    Two things to notice. The `min` is not decoration: at k >= chunk the comparison is
    full-width and yet STILL wrong at rate 2^-(chunk+1), because dropping c_in leaves
    S_j == A_j undecided. Truncation is not the only approximation in the vent, so there
    is no k that buys exactness -- that is what the `cmp_bits=None` variant is for, and
    it uses a different construction (`_carry_out_block`) rather than a wider compare.

    And the linear factor in the block count is why chunking narrow is not free twice
    over: more blocks means both more Toffoli and more chances to be wrong.
    """
    blocks = (n + chunk - 1) // chunk
    if blocks <= 1:
        return 0.0
    k_eff = chunk if k is None else min(k, chunk)
    return (blocks - 1) * 2.0 ** -(k_eff + 1)


def carry_fail_prob(L: int = CARRY_WINDOW, f: int = F) -> float:
    """P(the truncated `+f` carries out of the low L bits), for uniform input.

    Exactly f of the 2^L possible low parts overflow (the top f values), so P = f / 2^L.

    This is the UNCONDITIONAL rate -- correct when the `+f` always runs, as in `modneg`
    and the standalone constant adders. If the `+f` is CONTROLLED on a reduction flag
    (`modadd_lazy`, `modsub_direct`) it only runs on the reduction branch and the rate
    halves: use `controlled_carry_fail_prob`. Using this one there over-states the error
    by exactly 2x.
    """
    return f / float(1 << L)


def controlled_carry_fail_prob(L: int = CARRY_WINDOW, f: int = F) -> float:
    """P(a truncated `+f` fails) when the `+f` is CONTROLLED on the reduction flag.

        P = P(flag) * f / 2^L = f / 2^(L+1)

    `modadd_lazy` and `modsub_direct` add f only when the operation actually needs a
    reduction -- the overflow (or borrow) bit gates the constant adder. That branch is
    taken with probability -> 1/2 for uniform operands, so half of all calls cannot lose
    a carry because they never add anything.

    For 25519 at L = 45: 19/2^46 = **2^-41.75**, not the 19/2^45 = 2^-40.75 that a bare
    `carry_fail_prob` gives. Measured ratio to the unconditional formula, f = 19:

        u = 14 -> 0.500    u = 16 -> 0.489    u = 18 -> 0.502    u = 20 -> 0.496

    This is the same P(flag) factor `moddouble_carry_fail_prob` already carries; it was
    derived there and never applied here, so every controlled rate in the project was
    quoted 2x high until `qsm.audit.rates` measured it. The error is conservative -- the
    circuits are better than advertised -- but it is still wrong.

    NOT for `moddouble` (which gets a further even-limb factor) and NOT for `modneg` or
    the standalone constant adders (whose `+c` is unconditional).
    """
    if L is None:
        return 0.0
    return f / float(1 << (L + 1))


def narrow_row_carry_fail_prob(ripple: int | None = CARRY_WINDOW) -> float:
    """P(a `mul.accumulate._narrow_row` loses its carry out of the window).

        P = P(the top `ripple` bits of the accumulator are all 1) * P(the low add carries)
          = 2^-ripple * ~1/2  =  2^-(ripple+1)

    The row adds a `j`-bit addend into bits `0 .. j+ripple-1` of a `u`-bit accumulator and
    lets the carry out of the top of THAT window fall off. To escape, the carry has to be
    generated in the low `j` bits (probability -> 1/2 for a uniform addend) and then
    propagate through `ripple` accumulator bits that are all 1.

    Independent of `j`, which is why the triangle's rows can all be priced at one number.
    It is also independent of `f`: unlike every other truncation in this project the lost
    carry is not a `+f`, it is the row's own sum, so there is no `f/2^L` and no reduction
    branch -- the row does not reduce at all. See `mul.accumulate._narrow_row`.

    A union bound, and measurably a loose one at small width -- the low-`j` rows carry
    less often than 1/2 because their addend is a short prefix of `x`. Loose in the safe
    direction; `python -m qsm.audit rates` reports the measured side.
    """
    if ripple is None:
        return 0.0
    return 1.0 / float(1 << (ripple + 1))


# --- the dialog-GCD iteration budget ---------------------------------------
# The binary GCD that drives division and in-place multiplication runs a DATA-DEPENDENT
# number of iterations, but a quantum circuit is a fixed gate sequence: we have to pick
# one count up front and accept a failure probability on the tail. That makes the
# iteration budget an approximation knob exactly like CMP_BITS and CARRY_WINDOW -- and,
# uncomfortably, it is the WORST one in this file by a wide margin. See doc 10.
#
# Measured at n = 255 over 60,000 uniform x in [1, p), p = 2^255 - 19:
#
#     mean = 360.42 = 1.41339 * n        sd = 9.54 = 0.5974 * sqrt(n)
#
# The 1.4134 is the classical binary-GCD constant (2/log2(3+2*sqrt(2)) ~ 1.4141 is the
# usual analytic value; the small deficit is the `u odd` variant we run). The sd scaling
# as sqrt(n) is what makes the tail term sqrt(n) rather than linear.
#
# The measured upper tail is HEAVIER than Gaussian in the range that matters:
#
#     budget   slack/sd   empirical (60k)   gaussian
#        384       2.47   2^-7.4            2^-7.2
#        390       3.10   2^-10.3           2^-10.0
#        396       3.73   2^-12.2           2^-13.4
#        399       4.05   2^-13.9           2^-15.2
#
# so `gcd_iteration_fail_prob` is an OPTIMISTIC estimate above ~3.5 sd and must be
# treated as such. Empirically the log-probability falls ~0.43 per extra iteration in
# that range; the Gaussian falls faster because its log-tail is quadratic.
GCD_SHIFT_RATE = 1.41339     # mean iterations per bit of n   (measured, N = 60,000)
GCD_ITER_SD = 0.5974         # sd of the iteration count, per sqrt(n)  (measured)
GCD_ITER_VAR = 2.4           # tail multiplier on sqrt(n): budget = rate*n + var*sqrt(n)
GCD_ITER_BLOCK = 3           # iterations per compression block; the budget is a multiple

# GCD_ITER_VAR = 2.4 gives 399 iterations at n = 255, which is frenchie's constant
# (`ITERATIONS_VAR = 2.4` in ecdlp/fullreg/gcd_functions.py) and therefore the
# like-for-like setting for the comparison in doc 10. It is NOT the 2^-40 setting:
# 2^-40 needs var ~ 4.3 by the Gaussian estimate and more in truth, i.e. ~432-450
# iterations. **We cannot afford that at v1**: the dialog register costs 5 qubits per 3
# iterations, so +33 iterations is +55 qubits on top of a replay that already holds
# 665 + 255 + 255 = 1,175 = PEAK_BUDGET exactly. Raising this knob is therefore a
# genuine Pareto move (peak vs failure), not a free tightening -- which is why it gets a
# name here rather than being buried as a literal.


# The other half of the schedule: `u` and `v` SHRINK as the loop runs, so the adders and
# comparators can run on a narrowing window instead of the full 255 bits. This roughly
# halves Phase 1's gate count and is the single largest saving in it.
#
# Measured at n = 255 over 4,000 uniform x, tracking bitlen(max(u, v)) per iteration:
# the envelope tracks `n - 0.7075*i` with a slack that peaks at 31.5 bits, and the
# frenchie window `ceil(n - 0.7075*i + 37)` clipped 0 times in 798,000 iteration-slots.
#
# **Zero clips in a sample is not a zero rate, and this window is a SECOND probabilistic
# failure mode, not a safe engineering margin.** Two things establish that:
#
#   * On the worst-case trajectory it clips outright. At x = 2^(n-1) the register `u`
#     sits at p for n-1 iterations while the schedule shrinks under it, so the worst
#     excess is exactly 0.7075*(n-1) -- verified against every n from 10 to 21, e.g.
#     14.15 bits at n = 21. That grows LINEARLY in n while the pad grows as sqrt(n), so
#     at n = 255 the worst case overshoots a 37-bit pad by ~180 bits. It first clips at
#     n = 14. A budget of 2n-1 iterations therefore buys nothing on its own: the window
#     has to be widened to match, and `docs/decisions/10-inversion.md` prices that.
#   * The rate at the pad is 2^-18.6, not the 2^-39.8 a Gaussian would claim. The excess
#     is a maximum over a random walk, so its tail is EXPONENTIAL, and fitting the wrong
#     shape understates this by 21 bits. See `gcd_width_clip_fail_prob`.
#
# 2^-18.6 sits only ~4.7 bits below the iteration budget, so it is a real contributor to
# the total (~5%), not the free margin the zero-clip sample suggests -- and unlike the
# comparator at 2^-45.3, it cannot be dropped from the accounting.
#
# 0.7075 = GCD_SHIFT_RATE/2 to three places, and that is not a coincidence: one bit
# leaves `v` per iteration and the loop takes GCD_SHIFT_RATE iterations per bit, so the
# *pair* sheds half a bit per iteration.
GCD_WIDTH_RATE = 0.7075      # bits of max(u,v) shed per iteration   (measured)
GCD_WIDTH_PAD_VAR = 2.3      # window pad = ceil(var * sqrt(n)) = 37 at n = 255
GCD_WIDTH_STEP = 2           # windows are rounded up to a multiple of this

# Tail of the width excess `max_i [ bitlen(max(u,v)) - (n - rate*i) ]`, fitted on the
# five decades a 200,000-sample run can see (p90 through p99.999) at n = 255. The fit is
# log2 P(excess > e) = -slope*e + lead, and it is mildly CONSERVATIVE: the quantile
# spacing decelerates, so the local slope at the far end gives 2^-19.5 (J=1) and 2^-26.5
# (J=2) where the whole-range fit gives 2^-18.6 and 2^-25.3. The conservative figure is
# the one used.
GCD_WIDTH_CLIP_SLOPE = 0.6606     # bits of log2-probability per bit of pad  (measured)
GCD_WIDTH_CLIP_LEAD = 5.80        # intercept of the same fit                (measured)
GCD_J2_WIDTH_CLIP_SLOPE = 0.8684  # jump-2's excess is tighter: sd 3.31 vs 4.24
GCD_J2_WIDTH_CLIP_LEAD = 6.81


# The third GCD knob: how many bits the loop's comparator looks at. It is coupled to the
# window pad above, and the naive figure for it is wrong by nineteen bits.
#
# Inside a shrinking window, a truncated comparator is NOT priced at `2^-(k+1)`. The
# window sits `slack = width(i) - bitlen(max(u,v))` bits above the values by construction,
# so the top `slack` of the compared bits are zero in both operands and only `k - slack`
# of them are informative. Comparing the window's top CMP_BITS = 40 bits therefore
# compares padding against padding for most of the loop -- measured, that costs a wrong
# tape or a dirty free on roughly two of every three random 255-bit inputs.
#
# The right quantity is measured once and prices every comparator width, because it does
# not depend on one. Summing the per-slot failure over the slots that can actually deviate
# (`b0 = 1`, and the window still wider than the comparator):
#
#     P(wrong tape) = sum_live 2^-(k - slack + 1) = 2^-(k+1) * E[ sum_live 2^slack ]
#
# and the expectation is a property of the GCD's trajectory alone. Measured at n = 255,
# 150,000 uniform x, default schedule:  log2 E[sum 2^slack] = 55.7 (stable to ~1 bit from
# N = 30,000 upward; it is dominated by the rare runs that get far ahead of the schedule,
# max slack seen 67 against a mean of 42.4 and an sd of 4.6).
#
# Widening the pad shifts every slack by the same amount, so the constant moves one-for-one
# with it -- verified: log2 E[.] = 46.59 / 56.59 / 66.59 at pad = 27 / 37 / 47, to two
# decimals. What is invariant is therefore the MARGIN k - pad, which is why that, and not
# k, is the constant named here.
#
#     log2 P(wrong tape) = GCD_CMP_LEAD_LOG2 - 1 - GCD_CMP_MARGIN
#
# Measured cost at n = 255: ~262 Toffoli per bit of margin, against 32,040 Toffoli
# (13.4% of Phase 1) for dropping truncation altogether. Buying accuracy here is cheap,
# so the margin is set well past 2^-40 rather than at it. frenchie's 40 + 37 = 77 is
# margin 40, i.e. 2^-22.3 -- fine against a 2^-15.2 iteration budget, but not the 2^-41
# the per-call formula suggests.
GCD_CMP_LEAD_LOG2 = 18.7     # log2 E[sum 2^slack], less the pad   (measured, N = 150,000)
GCD_CMP_MARGIN = 63          # comparator bits above the window pad -> 77 + 23 = 100


# --- jump-2 -----------------------------------------------------------------------
# The loop can eat up to J halvings per iteration for the price of ONE compare, cswap and
# controlled subtract, by moving the shift to the TOP of the iteration -- where `v` is
# known even, because odd - odd = even -- and making only the second shift conditional.
# Iteration 0 is the exception and needs both conditional, since `x` may be even.
#
# What that buys, and what it does not. All measured at n = 255 rather than assumed:
#
#   * iterations fall to 0.6669x, which is the derived 2/3: trailing zeros after a
#     subtraction are Geom(1/2) on {1,2,...}, so E[tz] = 2 against E[ceil(tz/2)] = 4/3.
#   * gates fall by 11-17%, unconditionally but far short of the 1.53x folklore. Per
#     steady iteration J=1 costs `3w + k + 1` and J=2 costs `4w + k - 1` -- fitted with no
#     residual at n in {32,64,128,255}, k in {5,20,w} -- the difference being the one
#     controlled shift ladder that J=1 gets free as a relabel. Two thirds of the second
#     against the first is (8w+2k-2)/(9w+3k+3), which is below 1 for every w and k. The
#     `w` terms do NOT cancel: the controlled subtract is 2w, not w. Measured end to end
#     at n=255, same schedule and comparator both sides: 239,159 -> 198,436, a ratio of
#     0.830 against the 0.833 the formula predicts at full comparator width.
#   * the tape shrinks, and that is the whole point. The alphabet is 2J+1 = 5 states, not
#     4J: `shifts < J` means the chain stopped on an odd v, which forces b0 = 1, and
#     b0 = 0 forces b1 = 0. 5^3 = 125 fits in 7 bits, so 2.3333 bits/iteration against an
#     entropy bound of 2.3219 -- within 0.5% of optimal.
#
# 609 qubits at 261 iterations against 665 today, so `d + a + b = 609 + 255 + 255 = 1119`
# leaves 56 under PEAK_BUDGET on a design that currently sits at exactly 1175 with none.
# Of that 56, 25 is really an independent J=1 win: 5-in-3 is not base 3's best packing,
# 8-in-5 is (1.6000 bits/iteration against 1.6667), which is 640. The incremental figure
# for jump-2 against the best jump-1 is 31 qubits, and both are worth taking.
#
# That 56 is a claim about the REPLAY, and only the replay -- it is where the whole tape
# is held at once alongside both operands. Phase 1's own peak barely moves (measured 1112
# packed at J=1 against 1107 at J=2) because Phase 1 peaks early, when the window is full
# width and the tape has consumed almost nothing; the tape reaches its full size only at
# the end, by which point the window has gone. Phase 1 numbers are not evidence for it.
#
# MEASURED at the replay, which is where it had to be measured: 1,223 -> 1,169 on the CDKM
# version, so 54 of the 56 and not the whole of it -- the base-5 reader carries five qubits
# where base 3 carries three, and its digit read-back borrows a wider window. 1,169 is six
# under PEAK_BUDGET, and it is the FIRST replay in the design that fits one. Gidney's
# ladder (509 live carries) and 64-bit chunks (130) both want more than the 56 on offer, so
# they stay over at 1,635 and 1,255; the peak case for jump 2 is specifically a case for
# CDKM at jump 2, not for jump 2 at any adder.
#
# The gate win survives into Phase 2 and is larger there than in Phase 1, on matched pairs
# differing only in the jump (n = 255, same adder, same windows):
#
#     exact Gidney       618,450 -> 550,703   0.890
#     truncated Gidney   365,085 -> 330,158   0.904
#     truncated CDKM     570,171 -> 464,312   0.814
#
# The spread is the point and it is not noise. Phase 2 pays `n + 1` per iteration for the
# conditional doubling -- 66,816 Toffoli over 261 iterations -- and gets back 138 additions,
# 138 swaps and 138 comparators. How good that trade is depends entirely on what an
# addition costs, which is why the win is widest on CDKM (the dearest adder) and narrowest
# on truncated Gidney (the cheapest). Nowhere is it close to 261/399 = 0.654.
#
# J = 3 is WORSE, for a packing reason rather than an iteration one. It needs 0.573x the
# iterations, but base 7 wants a group of 11 to approach its entropy bound and sits at a
# flat 3.0 bits/iteration at any group small enough to decode, which leaves its tape
# longer than J=1's. Base 5 at group 3 is the reason the answer is J = 2 and not more.
#
# One trap worth naming, because it silently flatters jump-2 by 42 qubits: GCD_ITER_VAR
# multiplies sqrt(n), NOT the sd. Jump-2's sd is 0.310*sqrt(n) where jump-1's is 0.597, so
# reusing var = 2.4 provisions 7.7 sigma instead of 4.02 -- 279 iterations, not 261. The
# budgets are therefore matched on SIGMAS here, via GCD_ITER_Z.
GCD_ITER_Z = GCD_ITER_VAR / GCD_ITER_SD   # 4.017 sigma; what var = 2.4 actually buys J=1
GCD_J2_SHIFT_RATE = 0.94263  # mean iterations per bit of n at J=2  (measured, N = 60,000)
GCD_J2_ITER_SD = 0.31031     # sd of the iteration count, per sqrt(n)       (measured)
GCD_J2_ITER_BLOCK = 3        # 5^3 = 125 <= 2^7: three iterations pack into seven bits
GCD_J2_DIALOG_BITS = 7       # ...bits per block, against 5 for the 3-state J=1 alphabet

# The width rate is the J=1 rate divided by the iteration ratio, and the measurement says
# so: 0.7075 / 0.6669 = 1.0609 against 1.0620 measured, a 0.1% match. The same total bits
# are shed over 2/3 as many iterations. What does NOT survive the rescale is the per-index
# schedule itself -- an iteration eats a variable number of halvings, so `dialog_widths`
# cannot be resampled and `dialog_widths_jump` exists to build the schedule directly.
GCD_J2_WIDTH_RATE = 1.0620   # bits of max(u,v) shed per iteration   (measured, N = 2,000)


def gcd_iterations(
    n: int = U,
    var: float = GCD_ITER_VAR,
    rate: float = GCD_SHIFT_RATE,
    block: int = GCD_ITER_BLOCK,
) -> int:
    """Iterations of the binary-GCD dialog loop for an n-bit modulus.

        ceil((rate*n + var*sqrt(n)) / block) * block

    Rounded up to a whole number of compression blocks so the dialog register packs
    exactly. n = 255 with the defaults gives **399**.
    """
    raw = rate * n + var * (n ** 0.5)
    blocks = int(-(-raw // block))          # ceil, in integer arithmetic
    return blocks * block


def gcd_iteration_fail_prob(n: int = U, iters: int | None = None) -> float:
    """P(the dialog loop has not terminated within `iters` iterations). OPTIMISTIC.

    Gaussian tail on the measured mean and sd:

        z = (iters - GCD_SHIFT_RATE*n) / (GCD_ITER_SD*sqrt(n))
        P = erfc(z/sqrt(2)) / 2

    The real tail is heavier than this beyond ~3.5 sd (see the table above), by about
    1.3 bits at the default budget, so read the result as a LOWER bound on the failure
    rate. It is quoted rather than sampled because the interesting region is far past
    what sampling can reach.

    At n = 255, iters = 399 this returns ~2^-15.2 against a measured ~2^-13.9. Either
    way it is ~26 orders of magnitude worse than every other knob in this file, and so
    it -- not CMP_BITS, not CARRY_WINDOW -- is what sets the accuracy of an inversion.
    """
    import math

    if iters is None:
        iters = gcd_iterations(n)
    z = (iters - GCD_SHIFT_RATE * n) / (GCD_ITER_SD * (n ** 0.5))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def gcd_width_clip_fail_prob(
    pad: int | None = None, n: int = U, jump: int = 1
) -> float:
    """P(the shrinking u/v window is too narrow at some iteration of the run).

        2 ** (lead - slope*pad)

    The window is a probabilistic device, not a margin -- see the note above -- and this
    is the failure rate it carries. It is the term `gcd_iteration_fail_prob` and
    `gcd_compare_fail_prob` between them do NOT cover, so a total that adds only those
    two is short by this much.

    Exponential, because the excess is a maximum over a random walk. A Gaussian fitted to
    the same mean and sd says 2^-39.8 at the default pad against a measured-and-fitted
    2^-18.6, so the shape is worth 21 bits and picking it by habit is not safe.

    At n = 255, pad = 37 this returns ~2^-18.6 for J = 1 and ~2^-25.3 for J = 2 -- the
    jump-2 window is tighter because it sheds width faster relative to its own variance.
    Whole-run figures, already maximised over iterations; do not scale by the loop length.
    """
    if pad is None:
        import math

        pad = math.ceil(GCD_WIDTH_PAD_VAR * (n ** 0.5))
    slope, lead = (
        (GCD_J2_WIDTH_CLIP_SLOPE, GCD_J2_WIDTH_CLIP_LEAD) if jump == 2
        else (GCD_WIDTH_CLIP_SLOPE, GCD_WIDTH_CLIP_LEAD)
    )
    return 2.0 ** (lead - slope * pad)


def gcd_j2_iterations(
    n: int = U,
    z: float = GCD_ITER_Z,
    rate: float = GCD_J2_SHIFT_RATE,
    sd: float = GCD_J2_ITER_SD,
    block: int = GCD_J2_ITER_BLOCK,
) -> int:
    """Iterations of the jump-2 dialog loop for an n-bit modulus. **261** at n = 255.

    Parametrised by SIGMAS, not by `GCD_ITER_VAR`. That constant multiplies sqrt(n), not
    the sd, so reusing it here would buy jump-2 2.4/0.31031 = 7.7 sigma where it buys
    jump-1 4.02 -- 279 iterations at 2^-48 instead of 261 at 2^-16. That is not jump-2
    being safer, it is jump-2 being handed a different budget, and it would have flattered
    the tape comparison by 42 qubits. `z` defaults to whatever `GCD_ITER_VAR` implies for
    jump-1, so the two are compared at the same tail position.

    At matched z = 4.02 the measured overrun is 2^-15.9 (1 event in 60,000 above 260.2)
    against jump-1's 2^-13.9, so jump-2 is still mildly the safer of the two -- but on one
    event, which is not a real difference.

    The worst case is a different matter. The provable bound is still 2n-1, by the same
    potential argument (the unconditional halving drops bitlen(u)+bitlen(v) by at least
    one per iteration), but unlike jump-1 it is NOT attained: exhaustively, the worst x
    runs 1.125n at n = 8 rising to 1.350n at n = 20, and is still climbing. Nor does the
    worst INPUT settle down -- it is 2^(n-1) for jump-1 at every n, but for jump-2 it moves
    with n and ties at n = 10 -- so jump-1's worst case cannot be reused to bound jump-2's.
    Small-n enumeration therefore cannot say where jump-2's true worst case lands, and a
    provably-exact jump-2 must still budget 2n-1 = 509. **Jump-2's win is entirely in the
    probabilistic regime**; at exactness it saves nothing.
    """
    raw = rate * n + z * sd * (n ** 0.5)
    blocks = int(-(-raw // block))
    return blocks * block


def gcd_j2_iteration_fail_prob(n: int = U, iters: int | None = None) -> float:
    """P(the jump-2 dialog loop has not terminated within `iters` iterations).

    Gaussian on the measured jump-2 mean and sd, with the same caveat as the jump-1
    version -- except that here the measurement runs the other way. At n = 255,
    iters = 261 the sample gives 2^-15.9 against a Gaussian 2^-14.9, i.e. the empirical
    tail is LIGHTER, on one event in 60,000. That is a single-event difference well inside
    Poisson noise and is not evidence of either shape; it is recorded so the asymmetry
    with the jump-1 note above is not read as a finding.
    """
    import math

    if iters is None:
        iters = gcd_j2_iterations(n)
    z = (iters - GCD_J2_SHIFT_RATE * n) / (GCD_J2_ITER_SD * (n ** 0.5))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def gcd_j2_dialog_bits(
    iters: int | None = None,
    block: int = GCD_J2_ITER_BLOCK,
    bits: int = GCD_J2_DIALOG_BITS,
) -> int:
    """Qubits the compressed jump-2 dialog tape occupies. **609** at 261 iterations.

    Five states per iteration (see the jump-2 note above), 5^3 = 125 in 7 bits. Against
    the 665 the jump-1 tape costs today this is 56 qubits, but only 31 of those are
    jump-2's: the other 25 come from jump-1's own packing being suboptimal at 5-in-3
    where 8-in-5 would do (640).
    """
    if iters is None:
        iters = gcd_j2_iterations()
    return -(-iters // block) * bits


def gcd_compare_fail_prob(margin: int = GCD_CMP_MARGIN) -> float:
    """P(the dialog loop logs a wrong tape because its comparator was truncated).

        2 ** (GCD_CMP_LEAD_LOG2 - 1 - margin)

    `margin` is the comparator width MINUS the window pad; see the note above for why
    that difference, and not the width itself, is the thing with a fixed price. The
    default margin of 63 -- a 100-bit comparator against a 37-bit pad -- gives ~2^-45.3,
    which is 30 bits below the iteration budget and so does not move the total.

    This is a whole-run figure, already summed over the ~153 iterations per run whose
    comparison can matter. Do not multiply it by the iteration count again.
    """
    return 2.0 ** (GCD_CMP_LEAD_LOG2 - 1 - margin)


def gcd_dialog_bits(
    iters: int | None = None, block: int = GCD_ITER_BLOCK, bits: int = 5
) -> int:
    """Qubits the compressed dialog tape occupies.

    Each iteration logs (b0, b0 & b1), which cannot be (0,1) -- b0=0 forces the AND to
    0 -- so an iteration has 3 states, not 4, and `block` of them have 3^block. With
    block = 3 that is 27 states in 5 bits instead of 6: **665 qubits at 399 iterations
    rather than 798**, and the replay does not fit under PEAK_BUDGET without it.

    Packing denser is possible (3^5 = 243 < 2^8 gives 1.600 bits/iteration against
    5/3 = 1.667, i.e. 640 qubits) but costs a wider incremental encoder; measure before
    adopting. The information-theoretic floor is 399*log2(3) = 633.
    """
    if iters is None:
        iters = gcd_iterations()
    return -(-iters // block) * bits


def replay_fail_prob(
    iters: int | None = None,
    double_window: int | None = CARRY_WINDOW,
    carry_window: int | None = CARRY_WINDOW,
    cmp_bits: int | None = CMP_BITS,
    u: int = U,
    f: int = F,
    chunk: int | None = None,
    chunk_cmp_bits: int | None = None,
) -> float:
    """P(a full dialog replay is wrong), summed over all `iters` iterations.

    Unlike Phase 1 -- whose whole error budget is one iteration-count tail -- Phase 2
    fails five independent ways *per iteration* (six with a chunked adder), and every one
    of them scales linearly with the loop length. At 255 bits and 399 iterations:

        moddouble non-closure   iters * floor(f/2) / 2^u        2^-243.2
        moddouble carry window  iters * floor(f/2) / 2^L        2^-33.2
        modadd carry window     iters * f / 2^(L+1)             2^-32.4
        modadd erasure window   iters * 2^-(cmp_bits+1)         2^-32.4
        modadd vent PHASE       iters * f / 2^u                 2^-242.2
        adder boundary vent     iters * (blocks-1) * 2^-(c+1)   0 unless chunked
        -------------------------------------------------------------------
        total (default windows)                                 2^-30.9

    The sixth term is zero unless `chunk` is set, and it is the reason the chunked
    controlled adder cannot be cut narrow inside this loop. `chunked_add_boundary_fail_prob`
    is never exact however wide the comparison is -- it drops the carry-in -- so the
    effective width is `chunk`, not `chunk_cmp_bits`, and 399 additions amplify it by 8.6
    bits: chunk = 64 gives 2^-54.4 and disappears under the table above, chunk = 32 gives
    2^-22.4 and becomes the single largest term in it.

    The last term is the only one in `qsm` that costs a phase rather than a value, and it
    is the price of running lazily: `modadd_lazy`'s vent re-derives its overflow flag as
    `b < a`, which needs the accumulator below `p`, and the replay's accumulator is a lazy
    representative 399 times over (`modadd_lazy_vent_phase_prob`). Invisible to any check
    that does not simulate the measurements, and NOT negligible at test widths -- at
    p = 29 the lazy band is 3 of 32, so a small-prime replay leaks phase on most shots.
    `reference.apply_dialog_faults` counts it per concrete input for exactly that reason.

    Three observations, because the shape of this list is the design's actual accuracy
    story and it is not the one the per-call figures suggest.

    **The 399 costs 8.6 bits and it costs them three times.** Each per-call rate is around
    2^-41, which is comfortable; the loop turns all three into ~2^-32.4, which is not
    obviously so. Any per-call figure quoted for this circuit is 8.6 bits optimistic.

    **It is still not the binding term.** `gcd_iteration_fail_prob` is 2^-15.2 at the same
    parameters, so the replay's entire truncation budget sits ~16 bits below the iteration
    count Phase 1 already spends. Widening these windows buys nothing until the iteration
    budget moves; that is the same conclusion the comparator analysis reached in Phase 1,
    and it is why all three windows keep their Layer-0 defaults here rather than being
    retuned for the loop.

    **The non-closure is free and the loop is why.** `moddouble` wants a canonical input
    and the replay cannot supply one (doc 05: no fold in the design produces one), so the
    loop runs lazily and pays `moddouble_lazy_overflow_prob` 399 times over. Even so it
    lands at 2^-243.2, i.e. 210 bits below the truncations it sits next to -- which is the
    measurement that says canonicalising between doublings, at ~763 Toffoli x 399, would
    be over half the replay's Toffoli budget spent on the smallest term in this table.

    The `+f` terms use the *controlled* rates: both constant additions are gated on a
    reduction flag, so half of all calls add nothing and cannot lose a carry.
    """
    if iters is None:
        iters = gcd_iterations(u)
    per = moddouble_lazy_overflow_prob(u, f)
    per += moddouble_lazy_input_carry_fail_prob(double_window, f)
    if carry_window is not None:
        per += controlled_carry_fail_prob(carry_window, f)
    per += cmp_fail_prob(cmp_bits)
    per += modadd_lazy_vent_phase_prob(u, f)
    if chunk is not None:
        per += chunked_add_boundary_fail_prob(u, chunk, chunk_cmp_bits)
    return iters * per


def replay_j2_fail_prob(
    iters: int | None = None,
    double_window: int | None = CARRY_WINDOW,
    carry_window: int | None = CARRY_WINDOW,
    cmp_bits: int | None = CMP_BITS,
    u: int = U,
    f: int = F,
    chunk: int | None = None,
    chunk_cmp_bits: int | None = None,
) -> float:
    """P(a jump-2 replay is wrong). NOT `replay_fail_prob` at 261 -- the terms split.

    A jump-2 iteration does one add, one swap and `s` doublings, so the five per-iteration
    terms of `replay_fail_prob` no longer share a multiplier:

        the two moddouble terms scale with the number of DOUBLINGS, which is the number
        of divsteps -- unchanged by the jump, because jump 2 batches halvings, it does not
        remove any. Bounded here by `2 * iters` (every iteration taking its maximum),
        which is 522 against an expected 399: conservative by 0.4 bits.

        the three modadd terms scale with the number of ITERATIONS, 261 against 399,
        because that is what jump 2 actually reduces.

    So the jump-2 replay is *more* accurate than the jump-1 one at identical windows, by
    about half a bit -- and the reason is worth keeping in view, because it is the same
    reason the gate saving is smaller than the iteration ratio: what jump 2 removes is
    additions and comparisons, never doublings. Every term that survives is a doubling
    term, in the error budget exactly as in the Toffoli count.

    `moddouble_ctrl_inplace` errs only when its control fires; charging it as though it
    always does is where the 0.4 bits of conservatism sits. Tighten it only against a
    measured `s` distribution, not against the 1.529 average, since the rate is not linear
    in a distribution's mean.
    """
    if iters is None:
        iters = gcd_j2_iterations(u)
    doublings = 2 * iters                      # worst case; see above
    total = doublings * (
        moddouble_lazy_overflow_prob(u, f)
        + moddouble_lazy_input_carry_fail_prob(double_window, f)
    )
    per = cmp_fail_prob(cmp_bits) + modadd_lazy_vent_phase_prob(u, f)
    if carry_window is not None:
        per += controlled_carry_fail_prob(carry_window, f)
    if chunk is not None:
        per += chunked_add_boundary_fail_prob(u, chunk, chunk_cmp_bits)
    return total + iters * per


def un_replay_fail_prob(*args, **kw) -> float:
    """P(a full REVERSE dialog replay is wrong). Numerically `replay_fail_prob`, exactly.

    The reverse runs `cswap; modsub_lazy_ctrl; modhalve` where the forward runs
    `moddouble; modadd_lazy_ctrl; cswap`, and the five per-iteration terms map across
    one-for-one:

        moddouble non-closure  ->  the same wrap, which modhalve cannot undo
        moddouble carry window ->  modhalve carry window   (identical, see below)
        modadd carry window    ->  modsub carry window     (the same controlled +-F)
        modadd erasure window  ->  the comparator, now materialised
        modadd vent PHASE      ->  the same disagreement, now a VALUE

    so this function delegates rather than restating them. The two window terms coincide
    because `modhalve_carry_fail_prob` and `moddouble_lazy_input_carry_fail_prob` are the
    same expression -- the danger band sits at the bottom of the window instead of the top,
    and it is the same floor(F/2) values wide.

    **The number is the same; the failure is not.** Two of these five are phase-only
    forward and neither is backward. The comparator is materialised to drive the `-F`, so
    a truncation disagreement is arithmetic; and the vent's proxy, which forward leaked as
    an unobservable relative phase, backward strands the overflow qubit at |1>. That last
    one at least fails loudly -- the free raises rather than returning a wrong value
    (`test_field_modhalve.py::test_truncating_the_comparator_fails_loudly_backwards_not_silently`).

    Which is why the composite is priced with this called twice, not once: `moddiv_inplace`
    runs the reverse replay, `modmul_inplace` runs the forward, and both then pay Phase 1
    in both directions.
    """
    return replay_fail_prob(*args, **kw)


def un_replay_j2_fail_prob(*args, **kw) -> float:
    """P(a REVERSE jump-2 replay is wrong). Numerically `replay_j2_fail_prob`, exactly.

    The same one-for-one map `un_replay_fail_prob` sets out, applied to a term split that
    the reversal does not disturb. The reverse of `add^b0; swap^b1; double^s` is
    `halve^s; swap^b1; sub^b0`, so the doubling terms still ride on the DOUBLINGS and the
    addition terms still ride on the ITERATIONS -- 2 * 261 and 261 respectively, with the
    same 0.4 bits of conservatism in the first.

    The one thing worth checking rather than assuming is that the extra structure jump 2
    buys -- the conditional doubling -- reverses without a new term, and it does:
    `modhalve_ctrl_inplace` recovers its flag from the output's PARITY, exact at every
    width, exactly as the ungated `modhalve` does. Its error set is
    `modhalve_corrupts`, which is `moddouble_lazy_input_carry_fail_prob`'s band read from
    the other end and the same floor(F/2) values wide, so it is already the term this
    function inherits rather than an addition to it. A gated forward whose flag had been
    MEASURED would have broken that, which is why the forward spends a Toffoli on the
    parity erasure instead of a measurement.

    So jump 2's whole surcharge over jump 1 is direction-independent in the error budget
    just as it is in the gate count, and this delegates.
    """
    return replay_j2_fail_prob(*args, **kw)


# --- failure predicates: WHICH inputs break a truncation, not just how often -------
#
# The probabilities above assume UNIFORM operands. Real callers do not always supply
# those -- an iterative subroutine starts from structured state, and structured values
# have structured low parts. These predicates let a caller replay its ACTUAL value
# sequence and measure, instead of inheriting the uniform assumption.
#
# They are for auditing, never for circuits: a quantum circuit is a fixed gate sequence
# over a superposition and cannot branch on data. Legal uses are (a) compile-time
# branching on algorithmic position, and (b) compile-time checks of classically-known
# constants such as lookup-table entries.
def carry_truncation_fails(
    x: int, window: int | None = CARRY_WINDOW, f: int = F
) -> bool:
    """True if the truncated `+f` gets `x` wrong -- the low `window` bits carry out.

    The danger set is the top `f` residues mod 2^window, i.e. values sitting just BELOW
    a power of two. Small constants are safe; `2^k - small` is not.
    """
    if window is None:
        return False
    return (x % (1 << window)) + f >= (1 << window)


def borrow_truncation_fails(
    x: int, window: int | None = CARRY_WINDOW, f: int = F
) -> bool:
    """True if the truncated `-f` gets `x` wrong -- the low `window` bits borrow out.

    The mirror of `carry_truncation_fails`, and the danger set is the mirror too: the
    BOTTOM `f` residues mod 2^window, i.e. values sitting just ABOVE a power of two.
    Small constants are exactly what is unsafe here, where for the `+f` they were the
    safe case.

    That flip is why a numerator can be safe through `apply_dialog_j2` and unsafe through
    `un_apply_dialog_j2`. The forward's accumulator is built by DOUBLING, which shifts
    zeros into the low limb and so walks away from a danger set that sits just below a
    power of two. The reverse's is built by HALVING, which pulls the top bits down, and a
    small value has nothing up there to pull. See
    `reference.un_apply_dialog_jump_trunc_faults`, which counts the events on a real
    trajectory rather than assuming the uniform rate.
    """
    if window is None:
        return False
    return (x % (1 << window)) < f


def cmp_truncation_blind(a: int, b: int, n: int, k: int | None = CMP_BITS) -> bool:
    """True if a top-`k` comparison of two `n`-bit values is BLIND (top k bits tie).

    Blind is not the same as wrong: on a tie the discarded low bits decide the answer,
    so roughly half of blind cases still come out right. This predicate marks EXPOSURE.
    """
    if k is None or k >= n:
        return False
    lo = n - k
    return (a >> lo) == (b >> lo)


def audit_carry_truncation(
    values, window: int | None = CARRY_WINDOW, f: int = F
) -> dict:
    """Replay a real value sequence and compare its failure rate to the uniform model.

    This is what turns a guessed hybrid cut-off into a measured one: Schrottenloher's
    `ITER_CAN_BE_Q = 50` (gcd.py:134) has no stated derivation, but the first iteration
    whose operands land in the bad set can simply be counted.
    """
    vals = list(values)
    failures = sum(1 for x in vals if carry_truncation_fails(x, window, f))
    return {
        "total": len(vals),
        "failures": failures,
        "measured_rate": failures / len(vals) if vals else 0.0,
        "uniform_rate": carry_fail_prob(window, f) if window is not None else 0.0,
    }


def msb_reduce_fail_prob(u: int = U, f: int = F) -> float:
    """P(using `bit u` in place of `B >= p` gives an under-reduced result), for B
    uniform over the FULL (u+1)-bit range.

    Testing bit u answers `B >= 2^u` rather than `B >= 2^u - f`; the two differ exactly
    on the f values in [p, 2^u), so for that premise the rate is f / 2^(u+1).

    This is not an incorrect *result* so much as a redundant representation: the output
    may land in [p, 2^u) instead of [0, p). Downstream circuits must tolerate it.

    CAUTION: `modadd_lazy` is NOT in this premise -- its intermediate B = a+b is uniform
    over [0, 2p), not [0, 2^(u+1)). Use `lazy_unreduced_prob` for that; it is 2x larger.
    """
    return f / float(1 << (u + 1))


def moddouble_carry_fail_prob(
    L: int = CARRY_WINDOW, f: int = F, u: int = U, p: int | None = None
) -> float:
    """P(the truncated `+f` in `moddouble` carries out of the low L bits).

    NOT the generic `carry_fail_prob`. Doubling is roughly **2x safer for free**, for two
    compounding reasons:

      (a) the `+f` only fires when the reduction flag is set, i.e. `a >= 2^(u-1)`, which
          is about half of [0, p);
      (b) the value being added to is a SHIFTED one, so its bit 0 is 0 and its low L bits
          are always EVEN. Only `floor(f/2)` of the `f` dangerous residues just below
          `2^L` are even, so most of the danger set is unreachable.

        P = P(flag) * floor(f/2) / 2^(L-1),     P(flag) = (p - 2^(u-1)) / p ~ 1/2

    which is ~`floor(f/2) / 2^L`. For 25519 at L=45 that is 9/2^45 = 2^-41.8, against
    modadd's 19/2^45 = 2^-40.8. Verified at five (u, f, L) combinations in
    tests/test_field_moddouble.py.

    A reminder that these rates are properties of (circuit, input distribution) and not
    of the window alone -- the same truncation is safer here purely because doubling
    cannot produce an odd low limb.
    """
    if L is None:
        return 0.0
    if p is None:
        p = (1 << u) - f
    p_flag = (p - (1 << (u - 1))) / p
    return p_flag * (f // 2) / float(1 << (L - 1))


def moddouble_lazy_input_carry_fail_prob(
    L: int = CARRY_WINDOW, f: int = F
) -> float:
    """`moddouble_trunc`'s carry-truncation rate for a LAZY input, uniform on [0, 2^u).

    Same circuit as `moddouble_carry_fail_prob`, different input distribution -- which is
    the whole point of keeping two of these. Over the full lazy range the fold flag is set
    with probability exactly 1/2, rather than the `(p - 2^(u-1))/p` that a canonical input
    gives. Conditioned on the flag, the shifted low limb is uniform over the 2^(L-1) EVEN
    L-bit residues, of which exactly floor(f/2) lie in the danger band [2^L - f, 2^L):

        P = (1/2) * floor(f/2) / 2^(L-1) = floor(f/2) / 2^L

    Exact, not asymptotic, and measured so at eight (u, f, L) combinations including
    f = 19 -- see `test_lazy_inputs_shift_the_truncation_rate`. Slightly ABOVE the
    canonical-input rate (0.1406 vs 0.1393 at u=11, f=19, L=6), because a canonical input
    trips the flag slightly less than half the time.

    For Curve25519 at L = 45 this is 9/2^45 = 2^-41.8.
    """
    if L is None:
        return 0.0
    return (f // 2) / float(1 << L)


def modhalve_carry_fail_prob(
    L: int = CARRY_WINDOW, f: int = F, u: int = U, p: int | None = None
) -> float:
    """P(the truncated `-f` in `modhalve` borrows out of the low L bits).

    The mirror of `moddouble_carry_fail_prob`, and it lands on the same count from the
    other side. Halving subtracts rather than adds, so the danger band is at the BOTTOM of
    the window instead of the top, and the parity is odd instead of even:

      (a) the `-f` only fires when the recovered flag is set, i.e. when the input is ODD;
      (b) the value being subtracted from is then odd, so its low L bits are odd, and only
          `floor(f/2)` of the `f` dangerous residues {0, ..., f-1} are odd.

        P = P(odd) * floor(f/2) / 2^(L-1)

    For a uniform CANONICAL input, `P(odd) = (p-1)/(2p)` -- p is odd, so [0, p) holds one
    fewer odd value than even. That is a different constant from doubling's
    `(p - 2^(u-1))/p`, which is why this is its own helper rather than an alias, but both
    are just below 1/2 and both formulas collapse to ~`floor(f/2)/2^L`.

    For a uniform LAZY input the flag is set with probability exactly 1/2 and the rate is
    exactly `floor(f/2)/2^L` -- numerically identical to
    `moddouble_lazy_input_carry_fail_prob`, which is a genuine symmetry rather than a
    coincidence: the same window, the same constant, the same half of the residues
    unreachable, only the direction of the carry reversed.

    For 25519 at L = 45 this is 9/2^45 = 2^-41.8, the same as doubling's.
    """
    if L is None:
        return 0.0
    if p is None:
        p = (1 << u) - f
    p_flag = (p - 1) / (2 * p)
    return p_flag * (f // 2) / float(1 << (L - 1))


def moddouble_lazy_overflow_prob(u: int = U, f: int = F) -> float:
    """P(`moddouble` overflows its register because its INPUT was lazily reduced).

    `moddouble` is the one Layer-0 circuit that is **not closed under the lazy contract**.
    Given a lazy input `a` in [p, 2^u), the reduced result `2a - p` can exceed `2^u - 1`
    and wrap. It happens exactly when

        2a - p >= 2^u   <=>   a >= 2^u - f/2

    i.e. on the top `floor(f/2)` values of the register: 9 of 2^255 here, `2^-251.8`.

    **The numerator is floor(f/2) = 9, not f = 19**, which is worth spelling out because
    the overflow band above 2^u is f wide and invites the 19. `2a` is even and `p` is odd,
    so `2a - p` is ALWAYS odd, and only the odd half of that band is reachable:

        reachable:    2^u + 1, 3, 5, ..., 17      <- 9
        unreachable:  2^u + 2, 4, 6, ..., 16      <- even, cannot be produced

    Same thing from the input side: `a >= 2^u - 9.5` with `a` integral gives `a >= 2^u - 9`
    -- the doubling halves the threshold. Hence

        log2(9) - 255 = 3.1699 - 255 = -251.83     ->  2^-251.8
        log2(19) - 255 = 4.2479 - 255 = -250.75    ->  2^-250.8, which is
                                                       `lazy_unreduced_prob`, NOT this.

    The two sit one bit apart and are both about laziness, so they are easy to confuse.
    This is also the same parity fact that puts `floor(f/2)` in
    `moddouble_carry_fail_prob` -- there the addend is even, here the result is odd. A
    `19` in either is a bug; the `9` is the doubling's signature.

    Compare `modadd`, where canonical+lazy is closed and only lazy+lazy has a hole (at
    `2^-502.6`). Doubling has a single operand, so there is no canonical partner to lean
    on -- a doubling CHAIN feeds lazy into lazy at every step, and pays this per step.

    Negligible at real width (a ladder of ~2^8 doublings totals ~2^-243.8), but it is a
    genuine term and the first place the lazy contract does not compose. Documented in
    docs/decisions/03-approximate-arithmetic.md.
    """
    return (f // 2) / float(1 << u)


def moddouble_overflows(x: int, u: int = U, f: int = F) -> bool:
    """True if `moddouble(x)` overflows its register -- the non-closure predicate.

    Overflow needs `2x - p >= 2^u`, i.e. `x >= 2^u - f/2`: the top `floor(f/2)` values.

    **A canonical input can never trigger this**, and the proof is one line: overflow
    needs `x >= 2^u - f/2`, canonical means `x < p = 2^u - f`, and `f > f/2`, so
    `x < 2^u - f < 2^u - f/2`. The entire failure set lives in the lazy band `[p, 2^u)`,
    and within that only its top half.

    So the mitigation is a scheduling constraint, not a circuit change: **anything that
    guarantees a canonical input makes `moddouble` exact.**

    CORRECTION (2026-08-27): this used to add "a multiplication's reduction produces
    canonical output, so a double immediately after a multiply is always safe". That is
    **false**. The `x = x_lo + F*x_hi` fold lands in `[0, 2^u)` -- max exactly
    `2^u - 1 = p + f - 1` -- which is the LAZY contract, not the canonical one. Verified
    by closed form and by exhaustive brute force over all products at five widths. A
    multiply therefore supplies no free canonical value; reaching `[0, p)` needs the same
    full-width comparison as anywhere else (~250 Toffoli, since testing `>= p` on a value
    below `2^u` means testing that the top u - log2(f) bits are all ones).

    So a double after a multiply is exposed like any other, and the available mitigations
    are `moddouble_lazy_wide`, or paying for canonicalisation at the multiply, or
    accepting 2^-251.8.

    For auditing a real value sequence, as with `carry_truncation_fails`. Never for
    circuits -- a quantum register has no single value to test.
    """
    return x >= (1 << u) - f / 2


def modhalve_underflow_prob(u: int = U, f: int = F) -> float:
    """P(`modhalve` returns a NON-CONGRUENT result because its input was lazy).

    `modhalve` is the exact inverse of `moddouble`, so it inherits a non-closure of the
    same size at the other end of the register. Doubling loses the value that would land
    at `2^u`; halving loses the value that would land below `0`:

        z odd:  (z - f)/2 + 2^(u-1)      -- exact only while `z >= f`

    For odd `z < f` the constant subtraction borrows past the top of the register, and the
    result comes back too large by exactly `2^(u-1)`. The danger set is therefore the odd
    residues below `f`,

        {1, 3, 5, ..., f-2}      -- (f-1)/2 = 9 values, so 9/2^255 = 2^-251.8

    the SAME 9 as `moddouble_lazy_overflow_prob` -- `(f-1)/2 == floor(f/2)` for odd f, and
    the two sets are images of one another under the double/halve bijection. The failure
    mode is also the same shape: a clean ancilla and a wrong representative, not a dirty
    ancilla. Reversibility forces the choice, because exact halving on the full lazy range
    is not injective: `z` and `z + p` are congruent and halve to values `p` apart, so
    (f-1)/2 pairs collide. Something has to give, and a non-congruent representative is
    cheaper than the extra qubit and full-width borrow chain the other branch would cost.

    THE PROVENANCE ESCAPE IS THE USEFUL PART. A halving that is undoing a doubling can
    never trip this, because the doubling's image contains no odd value below f. So in
    `mul/accumulate.py` the four halvings inside the `18y` correction are free of it and
    only the halvings applied to a live accumulator pay.
    """
    return ((f - 1) // 2) / float(1 << u)


def modhalve_underflows(x: int, u: int = U, f: int = F) -> bool:
    """True if `modhalve(x)` returns a non-congruent result. Mirror of `moddouble_overflows`.

    Underflow needs the controlled `-f` to borrow out of the register, i.e. `x` odd and
    `x < f`. A CANONICAL input does not save you here -- unlike doubling, whose whole
    failure set lives in the lazy band `[p, 2^u)`, halving's failure set is at the BOTTOM
    of the range and every one of its 9 members is canonical. The mitigation is provenance
    of a different kind: `x` must be in the image of `moddouble`, which is exactly
    "not (odd and below f)".

    For auditing a real value sequence, never for circuits.
    """
    return bool(x & 1) and x < f


def modhalve_borrow_fail_prob(L: int | None = CARRY_WINDOW, f: int = F) -> float:
    """P(the truncated `-f` in `modhalve` borrows out of the low L bits).

    The mirror of `moddouble_lazy_input_carry_fail_prob`, and it lands on the same number
    for the same parity reason -- read from the other end:

      (a) the `-f` only fires when the input is ODD, i.e. on half of any range;
      (b) conditioned on that, the low L bits are an odd residue, and a borrow needs
          `x mod 2^L < f`. Only the ODD residues below f are reachable, so the danger set
          is {1, 3, ..., f-2} rather than all f of them.

        P = (1/2) * ((f-1)/2) / 2^(L-1) = ((f-1)/2) / 2^L

    For 25519 at L = 45 that is 9/2^45 = 2^-41.8, numerically identical to the doubling's
    `floor(f/2)/2^L` because `(f-1)/2 == floor(f/2)` for odd f. That coincidence is not
    luck: doubling's addend is even and cannot reach the odd half of its danger band,
    halving's minuend is odd and cannot reach the even half of its. Same 9, third context
    -- after `moddouble_lazy_overflow_prob` and `modhalve_underflow_prob`.

    Unlike those two, this one is NOT escapable by provenance. The non-closure spares a
    halving that undoes a doubling; a lost borrow does not care where the value came from,
    because the failure is in the low limb rather than at the ends of the range. So
    `mul/accumulate.py`'s restoring halvings are free of the non-closure but pay this.
    """
    if L is None:
        return 0.0
    return ((f - 1) // 2) / float(1 << L)


def modneg_corrupts(x: int, u: int = U, f: int = F) -> bool:
    """True if `modneg(x)` returns a NON-CONGRUENT result -- the unrepairable class.

    `modneg` computes `~(x + F - 1)`, which is `p - x` only while `x + F - 1` stays
    inside the register. That holds for `x <= p` and fails above it:

        corrupts  <=>  x > p          -- exactly the top F-1 values of [0, 2^u)

    `x = p` is fine (it returns 0, and p == 0 mod p). Verified exhaustively at four
    widths in tests/test_field_modneg.py.

    **This is worse than laziness and must not be confused with it.** A lazy output is
    congruent -- a single final `% p` repairs it. This is not congruent: for u=11, f=19,
    `modneg(2030)` returns 18 where -2030 mod p is 2028. No downstream reduction recovers
    that, so it is in the same class as a truncation error, not the same class as the
    lazy contract.

    It propagates: `modsub.v1` (the sandwich) calls `modneg` on `a`, so a lazy `a` gives
    a non-congruent difference. `modsub_direct.v2` does NOT call `modneg` and is immune
    (measured unaffected even with every operand lazy) -- which is a reason to prefer the
    direct construction beyond its 84-Toffoli saving.

    Frequency if a lazy schedule feeds it: (F-1)/2^u = 18/2^255 = 2^-250.9, so it does
    not move the budget. It is a scheduling constraint, like `moddouble_overflows`:
    anything that guarantees a canonical operand makes `modneg` exact.

    For auditing a real value sequence, never for circuits.
    """
    return x > (1 << u) - f


def modhalve_corrupts(b: int, u: int = U, f: int = F) -> bool:
    """True if `modhalve(b)` returns a NON-CONGRUENT result -- the unrepairable class.

        corrupts  <=>  b is odd and b < F

    `modhalve` recovers the fold decision from the parity and then subtracts F under it.
    That subtraction borrows out of the register exactly when `b < F`, and the borrow is
    discarded, so the halved value is off by a multiple of 2^(u-1) rather than merely
    unreduced. Same class as `modneg_corrupts`: not repairable by a final `% p`.

    The dual of `moddouble_overflows`, and the two do not overlap in practice, because
    `moddouble`'s odd outputs are `2a - p` for a in [2^(u-1), p) -- that is, exactly
    [F, p). Anything `modhalve` is asked to undo therefore satisfies the domain by
    construction, which is why this is a precondition and not a check.

    Frequency if an unrelated lazy value is fed in: floor(F/2)/2^u = 9/2^255 = 2^-251.8.

    For auditing a real value sequence, never for circuits.
    """
    return bool(b & 1) and b < f


def canonical_modadd_dirties(b: int, u: int = U, f: int = F) -> bool:
    """True if `modadd`/`modadd_cmp` leave a DIRTY ANCILLA on accumulator value `b`.

        dirties  <=>  b >= p

    Note this constrains only the ACCUMULATOR. The preserved operand `a` may be lazy
    with no ill effect -- measured across four widths, the dirty set is exactly
    `{(a, b) : b >= p}` regardless of `a`. So the docstring precondition "requires
    a < p and b < p" is stronger than the circuit actually needs.

    **This is the worst failure class in Layer 0.** The canonical `modadd` erases its
    reduction flag by recomputing the comparison from the result (step 6, "flag ^= NOT
    B[n]"), and that argument assumes `b_orig < p`. With a lazy accumulator the recompute
    disagrees and the flag does not return to |0>: the freed qubit comes back |1>, which
    is entangled garbage rather than a wrong number. A value-only test cannot see it --
    the returned sum can still be right.

    `modadd_lazy.v4` does NOT have this problem with one lazy operand (it is closed for
    canonical+lazy; only lazy+lazy has a hole, at ~2^-501.6), which is one more reason
    the lazy construction is the default.

    Frequency: F/2^u = 2^-250.8 per call if a lazy schedule feeds the accumulator.

    For auditing a real value sequence, never for circuits.
    """
    return b >= (1 << u) - f


def modsub_direct_corrupts(a: int, b: int, u: int = U, f: int = F) -> bool:
    """True if `modsub_direct(a, b)` returns a non-congruent difference.

        corrupts  <=>  a - b > p

    The circuit subtracts once and adds `p` back on a borrow, so it can only repair a
    difference no more negative than `-p`. A canonical `a < p` makes that automatic
    (`a - b <= a < p`), which is why the precondition is usually written `a < p` -- but
    the real condition is the weaker `a - b <= p`, and it tolerates a lazy `a` provided
    `b` is not tiny.

    That conjunction is what makes this circuit the robust one. `modneg` needs only a
    single lazy operand to corrupt (2^-250.9); this needs a lazy `a` AND a `b` below
    `a - p` at the same time. The failure set has exactly f(f-1)/2 pairs, so

        P = f(f-1)/2 / 2^2u = 171/2^510 = 2^-502.6 for 25519

    the same order as `modadd_lazy`'s lazy+lazy hole, and 250 orders below `modneg`'s.
    Verified exhaustively at three widths (the set equals {(a,b) : a-b > p} exactly).

    For auditing a real value sequence, never for circuits.
    """
    return a - b > (1 << u) - f


def modsub_direct_corrupt_prob(u: int = U, f: int = F) -> float:
    """P(`modsub_direct` corrupts) for uniform operands over the full lazy range."""
    return (f * (f - 1) / 2) / float(1 << (2 * u))


def lazy_unreduced_prob(u: int = U, f: int = F) -> float:
    """P(a `modadd_lazy` output is unreduced, i.e. lands in [p, 2^u)), for a, b uniform
    in [0, p).

        P = f / 2^u          (to within 1.5% at every width tested)

    Derivation. The output is unreduced exactly when a+b lands in [p, 2^u) -- below that
    no reduction is triggered and the sum is already reduced; at or above 2^u the
    circuit subtracts p and lands in [f, p). So

        P = #{(a,b) in [0,p)^2 : a+b in [p, 2^u)} / p^2.

    Each of those f values of the sum is hit by ~p pairs, giving f*p/p^2 = f/p ~ f/2^u.

    Note the 2x against `msb_reduce_fail_prob`: the sum is uniform-ish over [0, 2p), NOT
    over the full (u+1)-bit range, so the window [p, 2^u) is a fraction of 2^u rather
    than of 2^(u+1). For 2^255-19 this is 19/2^255 = 2^-250.75.

    Verified exhaustively at five widths in tests/test_field_modadd_lazy.py.
    """
    return f / float(1 << u)


def modadd_lazy_vent_phase_prob(u: int = U, f: int = F) -> float:
    """P(`modadd_lazy(vent=True)`'s erasure leaks PHASE) for a LAZY accumulator `b`.

    The other half of `lazy_unreduced_prob`, and the reason that function's "not an error"
    is only true one step at a time. `modadd_lazy` erases its overflow flag by measuring it
    and re-deriving it as `b < a`, which is sound because a reduction happened iff the sum
    fell below `a` -- and THAT step needs `b_0 < p`. Hand it an accumulator in `[p, 2^u)`
    and the sum `a + b_0 - p` can still be `>= a`, so the predicate says "no reduction"
    while the measured flag said "reduction". The two no longer cancel and a relative phase
    survives.

        P = f / 2^u          the accumulator lands in [p, 2^u)

    (Times ~1/2 for the measurement outcome, not included: the ERASURE is wrong either
    way, and the surviving phase is what a union bound should count.)

    A value error would be caught by any output comparison. This one is invisible to every
    check that does not simulate the measurements -- which is exactly how it went unnoticed
    in `mul/accumulate.py` until the benchmark ran it under random measurement vectors.

    NOT a rate to reach for when the accumulator's laziness is structural rather than
    incidental: `_signed_row` complements the accumulator, so `2^u-1-z >= p` whenever
    `z <= f-1`, and a `z` that is DETERMINISTICALLY small (a fresh |0> target, say) makes
    the erasure wrong on EVERY shot rather than on 2^-250.8 of them -- ~26% of inputs then
    keep a surviving phase (measured 0.243-0.290 across six (u, f), flat in the width,
    because a certainty does not scale with the register). See
    `mul.accumulate._first_row_from_zero`.
    """
    return f / float(1 << u)
