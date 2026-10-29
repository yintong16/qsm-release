"""Fixed-sample binomial inference, using only standard-library math."""

from __future__ import annotations

import math
from numbers import Real

REGRESSION_ALPHA = 0.01
CONFIDENCE = 0.95


def trial_count(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return value


def _counts(failures: int, trials: int) -> None:
    trial_count(trials, "trials")
    trial_count(failures, "failures")
    if failures > trials:
        raise ValueError("failures must not exceed trials")


def _log_tail(k: int, n: int, p: float, *, upper: bool) -> float:
    """Sum away from the mode, scaled by the endpoint mass to avoid underflow."""
    m = min(k, n - k)
    # Avoid subtracting nearly equal lgammas for sparse failures / large samples.
    log_choose = (math.fsum(math.log(n - i) - math.log(i + 1) for i in range(m))
                  if m <= 64 else
                  math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    log_mass = log_choose + k * math.log(p) + (n - k) * math.log1p(-p)

    def terms():
        term = 1.0
        yield term
        indices = range(k, n) if upper else range(k, 0, -1)
        for j in indices:
            ratio = (((n - j) / (j + 1)) * p / (1 - p) if upper else
                     (j / (n - j + 1)) * (1 - p) / p)
            term *= ratio
            yield term
            # Ratios decrease away from the mode; bound the omitted geometric tail.
            if ratio < 1 and term <= 1e-17 * (1 - ratio):
                break

    return min(0.0, log_mass + math.log(math.fsum(terms())))


def binomial_tail(failures: int, trials: int, p: float) -> float:
    """P[Binomial(trials, p) >= failures], without subtracting tiny tails from one."""
    _counts(failures, trials)
    p = probability(p, "p")
    if failures == 0 or p == 1:
        return 1.0
    if p == 0:
        return 0.0
    if failures == 1:
        return -math.expm1(trials * math.log1p(-p))
    if failures > trials * p:
        return math.exp(_log_tail(failures, trials, p, upper=True))
    return -math.expm1(_log_tail(failures - 1, trials, p, upper=False))


def clopper_pearson_upper(
    failures: int, trials: int, confidence: float = CONFIDENCE,
) -> float:
    """One-sided exact binomial upper limit (not a two-sided interval endpoint).

    Requires a fixed number of independent, identically distributed Bernoulli batches.
    A bound on 'any shot in a batch failed' also bounds single-shot failure.
    """
    _counts(failures, trials)
    confidence = probability(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if failures == trials:  # Includes no observations.
        return 1.0
    if failures == 0:
        return -math.expm1(math.log1p(-confidence) / trials)
    lo, hi = failures / trials, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if binomial_tail(failures + 1, trials, mid) < confidence:
            lo = mid
        else:
            hi = mid
    return hi
