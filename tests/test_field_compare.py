import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.params import CMP_BITS, cmp_fail_prob, cmp_ge_const_fail_prob
from qsm.field.compare import carry_out, carry_out_const, cmp_ge_const, cmp_lt

PRIME = (1 << 255) - 19


def _meas_vectors(c: Circuit):
    k = sum(1 for op in c.ops if op.kind == ops.MEASURE)
    if k > 8:
        return [None]
    return [[(m >> j) & 1 for j in range(k)] for m in range(1 << k)]


# --------------------------------------------------------------------------
# carry_out_const
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_carry_out_const_exhaustive(n: int):
    for cval in range(1 << n):
        circ = Circuit()
        b = circ.input("b", n)
        flag = circ.input("flag", 1)
        carry_out_const(circ, b, cval, flag[0])
        for mv in _meas_vectors(circ):
            for bv in range(1 << n):
                for fv in (0, 1):
                    r = simulate(circ, {"b": bv, "flag": fv}, measurements=mv)
                    want = fv ^ int(bv + cval >= (1 << n))
                    assert r.outputs["flag"] == want, f"n={n} {bv}+{cval} flag"
                    assert r.outputs["b"] == bv, "B must be restored"
                    assert r.phase == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_cmp_ge_const_exhaustive(n: int):
    for cval in range(1 << n):
        circ = Circuit()
        b = circ.input("b", n)
        flag = circ.input("flag", 1)
        cmp_ge_const(circ, b, cval, flag[0])
        for mv in _meas_vectors(circ):
            for bv in range(1 << n):
                for fv in (0, 1):
                    r = simulate(circ, {"b": bv, "flag": fv}, measurements=mv)
                    assert r.outputs["flag"] == fv ^ int(bv >= cval), \
                        f"n={n}: {bv} >= {cval}"
                    assert r.outputs["b"] == bv
                    assert r.phase == 0


# --------------------------------------------------------------------------
# carry_out (register operand) and cmp_lt
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_carry_out_qq_exhaustive(n: int):
    for na in range(1, n + 1):
        circ = Circuit()
        a = circ.input("a", na)
        b = circ.input("b", n)
        flag = circ.input("flag", 1)
        carry_out(circ, a, b, flag[0])
        for mv in _meas_vectors(circ):
            for av in range(1 << na):
                for bv in range(1 << n):
                    r = simulate(circ, {"a": av, "b": bv, "flag": 0}, measurements=mv)
                    assert r.outputs["flag"] == int(av + bv >= (1 << n)), \
                        f"na={na} n={n}: {av}+{bv}"
                    assert r.outputs["a"] == av and r.outputs["b"] == bv
                    assert r.phase == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_cmp_lt_exhaustive(n: int):
    for na in range(1, n + 1):
        circ = Circuit()
        a = circ.input("a", na)
        b = circ.input("b", n)
        flag = circ.input("flag", 1)
        cmp_lt(circ, a, b, flag[0])
        for mv in _meas_vectors(circ):
            for av in range(1 << na):
                for bv in range(1 << n):
                    for fv in (0, 1):
                        r = simulate(circ, {"a": av, "b": bv, "flag": fv},
                                     measurements=mv)
                        assert r.outputs["flag"] == fv ^ int(bv < av), \
                            f"na={na} n={n}: {bv} < {av}"
                        assert r.outputs["a"] == av and r.outputs["b"] == bv
                        assert r.phase == 0


# --------------------------------------------------------------------------
# full width, real prime
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# truncated (approximate) comparison
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 6])
def test_cmp_lt_truncated_is_exact_when_top_bits_differ(n: int):
    """The guarantee that makes truncation safe: a non-tie is never wrong."""
    for k in range(1, n + 1):
        circ = Circuit()
        a = circ.input("a", n)
        b = circ.input("b", n)
        flag = circ.input("flag", 1)
        cmp_lt(circ, a, b, flag[0], k=k)
        lo = n - k
        for av in range(1 << n):
            for bv in range(1 << n):
                if (av >> lo) == (bv >> lo):
                    continue                      # tie: allowed to be wrong
                r = simulate(circ, {"a": av, "b": bv, "flag": 0})
                assert r.outputs["flag"] == int(bv < av), \
                    f"n={n} k={k}: {bv} < {av} wrong despite differing top bits"
                assert r.outputs["a"] == av and r.outputs["b"] == bv
                assert r.phase == 0


@pytest.mark.parametrize("n,k", [(6, 1), (6, 2), (8, 2), (8, 3)])
def test_cmp_lt_truncated_error_rate_matches_formula(n: int, k: int):
    """Measured rate must match 2^-(k+1), not merely 'be small'.

    Exhaustive over both operands, so this is the exact rate, not an estimate.
    """
    circ = Circuit()
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=k)

    wrong = total = 0
    for av in range(1 << n):
        for bv in range(1 << n):
            r = simulate(circ, {"a": av, "b": bv, "flag": 0})
            wrong += r.outputs["flag"] != int(bv < av)
            total += 1
            assert r.outputs["a"] == av and r.outputs["b"] == bv, "operands clobbered"
            assert r.phase == 0, "truncation must NOT dirty the phase"
    rate = wrong / total
    predicted = cmp_fail_prob(k)
    # exact rate is 2^-(k+1) * (1 - 2^-(n-k)); allow that finite-width correction
    assert abs(rate - predicted) < 0.12 * predicted, \
        f"n={n} k={k}: measured {rate:.5f} vs predicted {predicted:.5f}"


@pytest.mark.parametrize("n", [6, 8])
def test_cmp_ge_const_truncated_error_rate_matches_formula(n: int):
    """Constant comparison: rate is c_low/2^n -- it depends on c's LOW bits."""
    for k in (2, 3):
        for c in (1, 3, (1 << (n - 1)) + 1, (1 << n) - 5):
            circ = Circuit()
            b = circ.input("b", n)
            flag = circ.input("flag", 1)
            cmp_ge_const(circ, b, c, flag[0], k=k)
            wrong = sum(
                simulate(circ, {"b": bv, "flag": 0}).outputs["flag"] != int(bv >= c)
                for bv in range(1 << n)
            )
            rate = wrong / (1 << n)
            predicted = cmp_ge_const_fail_prob(c, n, k)
            assert abs(rate - predicted) < 1e-9, \
                f"n={n} k={k} c={c}: measured {rate:.5f} vs predicted {predicted:.5f}"


def test_cmp_ge_const_rate_depends_on_low_bits_not_just_k():
    """Two constants, same k, wildly different accuracy -- the trap the docstring warns
    about, and the reason modadd should test bit U instead of comparing against p."""
    n, k = 10, 3
    lo = n - k
    c_hi_low_part = (1 << lo) - 1          # low part maximal  -> worst case
    c_lo_low_part = (1 << lo) * 3          # low part zero     -> exact
    rates = []
    for c in (c_hi_low_part, c_lo_low_part):
        circ = Circuit()
        b = circ.input("b", n)
        flag = circ.input("flag", 1)
        cmp_ge_const(circ, b, c, flag[0], k=k)
        wrong = sum(
            simulate(circ, {"b": bv, "flag": 0}).outputs["flag"] != int(bv >= c)
            for bv in range(1 << n)
        )
        rates.append(wrong / (1 << n))
    assert rates[1] == 0.0, "constant with zero low part must be compared exactly"
    assert rates[0] > 0.1, "constant with maximal low part should be badly wrong"


def test_truncated_cmp_is_catastrophic_on_near_ties():
    """The load-bearing caveat: 2^-(k+1) assumes UNIFORM INDEPENDENT operands.

    Adjacent values (b = a, a+-1) agree on every high bit, so they land squarely in the
    blind spot and the comparator is wrong about half the time -- roughly 10^11 times its
    nominal rate. Recorded here (rather than quietly avoided) because any caller that
    feeds correlated operands invalidates the whole error budget.
    """
    n, k = 16, 4
    circ = Circuit()
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=k)

    rng = random.Random(9)
    def rate(pick):
        wrong = 0
        for _ in range(400):
            av = rng.randrange(1 << (n - 1))
            bv = pick(av) % (1 << n)
            r = simulate(circ, {"a": av, "b": bv, "flag": 0}, rng=rng)
            wrong += r.outputs["flag"] != int(bv < av)
        return wrong / 400

    near = rate(lambda av: rng.choice([av, av - 1, av + 1]))
    unif = rate(lambda _av: rng.getrandbits(n))
    assert near > 0.3, f"near-ties should be ~50% wrong, got {near:.3f}"
    assert unif < 3 * cmp_fail_prob(k) + 0.02, f"uniform should be ~2^-(k+1), got {unif:.3f}"
    assert near > 10 * unif


@pytest.mark.parametrize("u,k", [(12, 4), (12, 5), (14, 5), (14, 6)])
def test_operand_range_costs_one_window_bit(u: int, k: int):
    """At modadd's widths the rate is 2^-k, not 2^-(k+1).

    Both operands are < 2^u inside a (u+1)-bit register, so the TOP window bit is 0 for
    both and informs nothing: only k-1 of the k compared bits count. Also checks that
    CORRELATION is not the cause -- (a+b) mod p vs a measures the same as an independent
    pair, which is what disproved an earlier guess that correlation cost the factor.
    """
    p = (1 << u) - 19
    n = u + 1
    circ = Circuit()
    a = circ.input("a", u)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=k)

    rng = random.Random(21)
    trials = 4000
    wrong_corr = wrong_indep = 0
    for _ in range(trials):
        av, bv = rng.randrange(p), rng.randrange(p)          # independent, both < p
        r = simulate(circ, {"a": av, "b": bv, "flag": 0}, rng=rng)
        wrong_indep += r.outputs["flag"] != int(bv < av)

        av2 = rng.randrange(p)                               # modadd's real output
        bv2 = (av2 + rng.randrange(p)) % p
        r2 = simulate(circ, {"a": av2, "b": bv2, "flag": 0}, rng=rng)
        wrong_corr += r2.outputs["flag"] != int(bv2 < av2)

    corr, indep = wrong_corr / trials, wrong_indep / trials
    predicted = cmp_fail_prob(k, informative_bits=k - 1)      # == 2^-k
    assert abs(indep - predicted) < 0.30 * predicted, \
        f"u={u} k={k}: independent {indep:.5f} vs predicted {predicted:.5f}"
    assert abs(corr - predicted) < 0.30 * predicted, \
        f"u={u} k={k}: correlated {corr:.5f} vs predicted {predicted:.5f}"
    # correlation is NOT the driver: the two distributions agree
    assert abs(corr - indep) < 0.45 * predicted


def test_truncation_is_free_of_dirty_ancilla_and_phase():
    """The correction to the earlier assumption: approximate != unclean.
    A truncated circuit is still an exact unitary; only the function it computes moves."""
    n, k = 12, 3
    circ = Circuit()
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=k)
    rng = random.Random(4)
    for _ in range(200):
        # would raise DirtyAncillaError if any freed qubit were non-zero
        r = simulate(circ, {"a": rng.getrandbits(n), "b": rng.getrandbits(n),
                            "flag": 0}, rng=rng)
        assert r.phase == 0


def test_truncated_comparator_is_cheaper():
    n = 255
    def toffoli(k):
        circ = Circuit()
        a = circ.input("a", n)
        b = circ.input("b", n + 1)
        flag = circ.input("flag", 1)
        cmp_lt(circ, a, b, flag[0], k=k)
        return count(circ).toffoli

    assert toffoli(CMP_BITS) < toffoli(None) // 5      # ~40 vs ~256
    assert toffoli(40) == 40                            # one Toffoli per compared bit
    assert toffoli(None) == 256


def test_cmp_ge_p_boundary_255bit():
    """The interesting inputs are the 19 values in [p, 2^255)."""
    n = 256
    circ = Circuit()
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_ge_const(circ, b, PRIME, flag[0])
    for bv in [0, 1, PRIME - 2, PRIME - 1, PRIME, PRIME + 1,
               (1 << 255) - 1, 1 << 255, 2 * PRIME - 2]:
        r = simulate(circ, {"b": bv, "flag": 0})
        assert r.outputs["flag"] == int(bv >= PRIME), f"b={bv}"
        assert r.outputs["b"] == bv
        assert r.phase == 0


def test_cmp_lt_255bit_random_and_ties():
    circ = Circuit()
    a = circ.input("a", 255)
    b = circ.input("b", 256)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0])
    rng = random.Random(11)
    cases = [(0, 0), (1, 0), (0, 1), (PRIME - 1, PRIME - 1)]
    cases += [(rng.randrange(PRIME), rng.randrange(PRIME)) for _ in range(20)]
    for av, bv in cases:
        r = simulate(circ, {"a": av, "b": bv, "flag": 0}, rng=rng)
        assert r.outputs["flag"] == int(bv < av), f"{bv} < {av}"
        assert r.outputs["a"] == av and r.outputs["b"] == bv
        assert r.phase == 0


# --------------------------------------------------------------------------
# Controlled comparison: flag ^= ctrl & (B < a)
#
# The dialog-GCD needs `b0 & (u > v)`. Doing it by computing the predicate into an
# ancilla and ANDing afterwards would run the carry ladder twice; gating the single
# write into `flag` instead costs two Toffoli, full stop. These pin that.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_cmp_lt_ctrl_exhaustive(n: int):
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], ctrl=ctrl[0])
    for mv in _meas_vectors(circ):
        for cv in (0, 1):
            for av in range(1 << n):
                for bv in range(1 << n):
                    for fv in (0, 1):
                        r = simulate(
                            circ,
                            {"ctrl": cv, "a": av, "b": bv, "flag": fv},
                            measurements=mv,
                        )
                        assert r.outputs["a"] == av, "a not restored"
                        assert r.outputs["b"] == bv, "b not restored"
                        assert r.outputs["ctrl"] == cv, "ctrl not preserved"
                        assert r.outputs["flag"] == fv ^ (cv & int(bv < av)), \
                            (n, cv, av, bv, fv)
                        assert r.phase == 0


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_cmp_lt_ctrl_zero_is_exactly_identity(n: int):
    """ctrl=0 must leave `flag` alone -- the ladder still runs, so this is not free."""
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], ctrl=ctrl[0])
    for av in range(1 << n):
        for bv in range(1 << n):
            r = simulate(circ, {"ctrl": 0, "a": av, "b": bv, "flag": 1})
            assert (r.outputs["a"], r.outputs["b"], r.outputs["flag"]) == (av, bv, 1)
            assert r.phase == 0


@pytest.mark.parametrize("n", [8, 40, 255])
def test_cmp_lt_ctrl_costs_two_toffoli_more_than_uncontrolled(n: int):
    """The claim the module docstring makes: +2 total, not +2 per bit."""
    def build(ctrl_on: bool) -> Circuit:
        c = Circuit()
        ctrl = c.input("ctrl", 1)
        a = c.input("a", n)
        b = c.input("b", n)
        flag = c.input("flag", 1)
        cmp_lt(c, a, b, flag[0], ctrl=ctrl[0] if ctrl_on else None)
        return c

    plain = count(build(False)).toffoli
    ctrl = count(build(True)).toffoli
    assert ctrl - plain == 2, (plain, ctrl)


@pytest.mark.parametrize("n,k", [(6, 2), (8, 3)])
def test_cmp_lt_ctrl_truncated_matches_the_uncontrolled_rate(n: int, k: int):
    """Gating the flag cannot change *which* inputs the truncation gets wrong."""
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=k, ctrl=ctrl[0])

    wrong = total = 0
    for av in range(1 << n):
        for bv in range(1 << n):
            r = simulate(circ, {"ctrl": 1, "a": av, "b": bv, "flag": 0})
            total += 1
            wrong += int(r.outputs["flag"] != int(bv < av))
    predicted = cmp_fail_prob(k)
    assert abs(wrong / total - predicted) < 0.12 * predicted


def test_cmp_lt_ctrl_255bit_random():
    n = 255
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], ctrl=ctrl[0])
    rng = random.Random(5)
    for _ in range(24):
        cv, av, bv = rng.getrandbits(1), rng.randrange(PRIME), rng.randrange(PRIME)
        r = simulate(circ, {"ctrl": cv, "a": av, "b": bv, "flag": 0}, rng=rng)
        assert r.outputs["flag"] == (cv & int(bv < av))
        assert (r.outputs["a"], r.outputs["b"]) == (av, bv)
        assert r.phase == 0
