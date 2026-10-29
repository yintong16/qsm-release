import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from qsm.bench.statistics import binomial_tail, clopper_pearson_upper


def _exact_tail(k, n, p):
    return float(sum(math.comb(n, j) * p**j * (1 - p)**(n - j)
                     for j in range(k, n + 1)))


@pytest.mark.parametrize("p", [Fraction(0), Fraction(1, 1000), Fraction(1, 4),
                              Fraction(1, 2), Fraction(9, 10), Fraction(1)])
def test_tail_against_exact_rational_sum(p):
    for n in range(21):
        for k in range(n + 1):
            assert binomial_tail(k, n, float(p)) == pytest.approx(
                _exact_tail(k, n, p), rel=2e-13, abs=1e-300)


@pytest.mark.parametrize("n,k,p", [
    (1000, 1, "1e-250"), (1000, 2, "1e-100"), (10000, 3, "1e-9"),
    (1000000000, 2, "1e-12"),
])
def test_tiny_tail_avoids_cancellation_and_underflow(n, k, p):
    with localcontext() as ctx:
        ctx.prec = 80
        q = Decimal(p)
        term = Decimal(math.comb(n, k)) * q**k * (1 - q)**(n - k)
        expected = term
        for j in range(k, min(n, k + 100)):
            term *= Decimal(n - j) / (j + 1) * q / (1 - q)
            expected += term
    assert binomial_tail(k, n, float(p)) == pytest.approx(float(expected), rel=1e-11, abs=0)


def test_large_central_tail_and_complement():
    n = 100000
    mass = math.exp(math.lgamma(n + 1) - 2 * math.lgamma(n // 2 + 1) - n * math.log(2))
    assert binomial_tail(n // 2, n, 0.5) == pytest.approx((1 + mass) / 2, abs=2e-10)
    assert binomial_tail(n // 2 + 1, n, 0.5) == pytest.approx((1 - mass) / 2, abs=2e-10)
    assert binomial_tail(100, 100, 0.999999) == pytest.approx(0.999999**100)


def test_cp_endpoints_and_known_values():
    assert clopper_pearson_upper(0, 0) == 1
    assert clopper_pearson_upper(0, 1) == pytest.approx(0.95)
    assert clopper_pearson_upper(1, 1) == 1
    assert clopper_pearson_upper(1, 2) == pytest.approx(math.sqrt(0.95))
    assert clopper_pearson_upper(1, 10) == pytest.approx(0.3941633024365047)
    for n in (10, 1000, 10**12):
        assert clopper_pearson_upper(0, n) == pytest.approx(
            -math.expm1(math.log(0.05) / n), rel=1e-14)
        assert clopper_pearson_upper(n - 1, n) == pytest.approx(0.95**(1 / n))


@pytest.mark.parametrize("n", [1, 2, 5, 10, 25])
def test_cp_frequentist_coverage_from_exact_binomial_weights(n):
    bounds = [clopper_pearson_upper(k, n) for k in range(n + 1)]
    assert bounds == sorted(bounds)
    for p in [Fraction(1, 100), Fraction(1, 10), Fraction(3, 10),
              Fraction(1, 2), Fraction(9, 10), Fraction(99, 100)]:
        coverage = sum(math.comb(n, k) * p**k * (1 - p)**(n - k)
                       for k, bound in enumerate(bounds) if float(p) <= bound)
        assert float(coverage) >= 0.95 - 1e-14


def test_cp_inverts_tail_at_large_sample():
    for k, n in [(1, 100000), (700, 1000), (5000, 10000)]:
        upper = clopper_pearson_upper(k, n)
        assert k / n <= upper <= 1
        assert binomial_tail(k + 1, n, upper) == pytest.approx(0.95, abs=1e-10)


@pytest.mark.parametrize("k,n", [(-1, 10), (2, 1), (0, -1), (1.0, 10),
                                (0, 1.5), (True, 2), (0, False)])
def test_invalid_counts(k, n):
    with pytest.raises(ValueError):
        binomial_tail(k, n, 0.1)
    with pytest.raises(ValueError):
        clopper_pearson_upper(k, n)


@pytest.mark.parametrize("p", [-1, 1.1, float("nan"), float("inf"), None, "0.5", True])
def test_invalid_probability(p):
    with pytest.raises(ValueError):
        binomial_tail(0, 0, p)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0, p)


@pytest.mark.parametrize("confidence", [0, 1])
def test_confidence_must_be_interior(confidence):
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0, confidence)
