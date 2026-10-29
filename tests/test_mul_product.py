"""Raw products: the add/subtract trick, and squaring as exactly half a multiply.

The two claims this file has to pin down, because everything in Layer 1 rests on them:

  1. `n` UNCONDITIONAL adds compute a multiply, not `n` controlled ones. The rows compute
     the wrong function on purpose; a closed-form correction in x and y alone repairs it
     for ~5n Toffoli. If that correction is wrong the circuit is silently wrong on a
     data-dependent subset, which no spot check would find -- hence exhaustive.

  2. The rows have ZERO headroom. Each adds into `wide[k : k+n+1]`, whose top bit is
     exactly the highest bit the running total can reach, so nothing may be pre-loaded
     into the accumulator. The first draft of both circuits pre-loaded a constant onto the
     zeroed register (free, in Cliffords) and was wrong for precisely this reason;
     `test_nothing_may_be_preloaded_into_the_row_accumulator` keeps that dead end measured.
"""

import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import gidney_add
from qsm.mul.product import (addsub, addsub_inverse, schoolbook,
                             schoolbook_inverse, square_symmetric,
                             square_symmetric_inverse)

WIDTHS = [2, 3, 4, 5, 6]

#: Measurement outcomes are drawn at RANDOM, not fixed at zero. Every circuit in this file
#: erases with `_unand`, which restores exactly, so `phase == 0` must hold for every
#: outcome vector -- and under an all-zero one it holds whether that is true or not, since
#: an erasure's correction is conditioned on the measured bit. That vacuity let a ~25%
#: phase leak ship from `qsm/mul/accumulate.py`; see that module's test file.
_MEAS_RNG = random.Random(0xC0FFEE)


def _sim(circ, vals, meas=None):
    if meas is not None:
        return simulate(circ, vals, measurements=meas)
    return simulate(circ, vals, rng=_MEAS_RNG)


def _mul_circ(n, inverse=False):
    c = Circuit()
    x, y = c.input("x", n), c.input("y", n)
    prod = c.input("prod", 2 * n)
    schoolbook(c, x, y, prod)
    if inverse:
        schoolbook_inverse(c, x, y, prod)
    return c


def _sqr_circ(n, inverse=False):
    c = Circuit()
    x = c.input("x", n)
    prod = c.input("prod", 2 * n)
    square_symmetric(c, x, prod)
    if inverse:
        square_symmetric_inverse(c, x, prod)
    return c


# ---------------------------------------------------------------------------
# correctness, exhaustive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", WIDTHS)
def test_schoolbook_is_exact_on_every_input_pair(n):
    c = _mul_circ(n)
    for a in range(1 << n):
        for b in range(1 << n):
            r = _sim(c, {"x": a, "y": b, "prod": 0})
            assert r.outputs == {"x": a, "y": b, "prod": a * b}, (a, b)
            assert r.phase == 0, (a, b)


@pytest.mark.parametrize("n", WIDTHS)
def test_square_is_exact_on_every_input(n):
    c = _sqr_circ(n)
    for a in range(1 << n):
        r = _sim(c, {"x": a, "prod": 0})
        assert r.outputs == {"x": a, "prod": a * a}, a
        assert r.phase == 0, a


@pytest.mark.parametrize("n", WIDTHS)
def test_the_inverses_return_the_product_register_to_zero(n):
    """The whole modular layer depends on this: a product that cannot be uncomputed
    cannot be freed, and a 2n-bit leak per multiply would dominate the qubit count."""
    for build in (_mul_circ, _sqr_circ):
        c = build(n, inverse=True)
        keys = list(c.inputs)
        for a in range(1 << n):
            vals = {k: (a if k != "prod" else 0) for k in keys}
            if "y" in vals:
                vals["y"] = (a * 7 + 3) % (1 << n)
            r = _sim(c, vals)
            assert r.outputs["prod"] == 0, (build.__name__, a, r.outputs)
            assert r.phase == 0


@pytest.mark.parametrize("n", WIDTHS)
def test_no_freed_ancilla_is_ever_dirty(n):
    """These circuits are EXACT, so a dirty free is a bug and not a budgeted failure."""
    for build in (_mul_circ, _sqr_circ):
        c = build(n)
        keys = list(c.inputs)
        for a in range(1 << n):
            vals = {k: (a if k != "prod" else 0) for k in keys}
            if "y" in vals:
                vals["y"] = ((1 << n) - 1) - a
            _sim(c, vals)          # raises DirtyAncillaError if any free is not |0>


@pytest.mark.parametrize("n", [3, 4, 5])
def test_erasures_are_phase_clean_under_every_measurement_outcome(n):
    """Gidney carries are erased by measurement, so correctness has to hold for every
    outcome vector, not just the all-zero one a single simulation happens to draw."""
    rng = random.Random(0)
    for build, vals in ((_mul_circ, {"x": 5 % (1 << n), "y": 3, "prod": 0}),
                        (_sqr_circ, {"x": 5 % (1 << n), "prod": 0})):
        c = build(n)
        for _ in range(24):
            meas = [rng.getrandbits(1) for _ in range(100000)]
            r = simulate(c, dict(vals), measurements=meas)
            assert r.phase == 0
            assert r.outputs["prod"] == (
                vals["x"] * vals.get("y", vals["x"]))


# ---------------------------------------------------------------------------
# the add/subtract row itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_addsub_adds_on_one_and_complements_on_zero(n):
    """`ctrl=1 -> +a`, `ctrl=0 -> +(2^n - a)`. The second branch is what makes the row
    unconditional; the aggregate correction exists to undo it."""
    c = Circuit()
    a = c.input("a", n)
    B = c.input("B", n + 1)
    ctrl = c.input("ctrl", 1)
    addsub(c, a, B, ctrl[0])
    for av in range(1 << n):
        for bv in range(1 << (n + 1)):
            for cv in (0, 1):
                r = _sim(c, {"a": av, "B": bv, "ctrl": cv})
                want = (bv + (av if cv else (1 << n) - av)) % (1 << (n + 1))
                assert r.outputs == {"a": av, "B": want, "ctrl": cv}


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_addsub_inverse_undoes_addsub(n):
    c = Circuit()
    a = c.input("a", n)
    B = c.input("B", n + 1)
    ctrl = c.input("ctrl", 1)
    addsub(c, a, B, ctrl[0])
    addsub_inverse(c, a, B, ctrl[0])
    for av in range(1 << n):
        for bv in range(1 << (n + 1)):
            for cv in (0, 1):
                r = _sim(c, {"a": av, "B": bv, "ctrl": cv})
                assert r.outputs == {"a": av, "B": bv, "ctrl": cv}


def test_addsub_costs_one_unconditional_add_not_two():
    """The point of the trick, stated as a number.

    ecdsa.fail's proven-floor note puts an uncontrolled n-bit add at n-1 Toffoli (tight)
    and a controlled one at a best-known 2n against a proven floor of n. `addsub` buys the
    conditionality for ZERO Toffoli: it costs exactly what the bare adder it wraps costs,
    and the 2n+3 extra gates carrying the control are all Clifford.
    """
    n = 64
    bare = Circuit()
    a0, b0 = bare.input("a", n), bare.input("B", n + 1)
    gidney_add(bare, a0, b0)
    rb = count(bare)

    c = Circuit()
    a, B, ctrl = c.input("a", n), c.input("B", n + 1), c.input("c", 1)
    addsub(c, a, B, ctrl[0])
    r = count(c)

    assert r.toffoli == rb.toffoli == n, (r.toffoli, rb.toffoli)
    assert r.clifford - rb.clifford == 2 * n + 3, r.clifford - rb.clifford


# ---------------------------------------------------------------------------
# the headroom result -- why nothing may be pre-loaded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_nothing_may_be_preloaded_into_the_row_accumulator(n):
    """A DEAD END, kept measured.

    The multiply's correction needs a `+2^n`, and pre-loading it with one X onto the
    still-zero product register looks free. It is not: after row k the running total can
    reach exactly 2^(n+k+1)-1 and row k's slice tops out at bit n+k, so the register has
    no spare bit anywhere. The pre-load pushes some row over its slice and the carry is
    lost. Paying n Toffoli for a real constant add afterwards is the fix.

    This asserts the tightness that makes it true, rather than the historical bug: the
    largest value reachable after row k needs exactly bits 0..n+k.
    """
    for k in range(n):
        # every row so far taking its `2^n` branch is the worst case
        worst = sum((1 << n) << j for j in range(k + 1))
        assert worst.bit_length() == n + k + 1     # bits 0..n+k, not one more
        assert worst < (1 << (n + k + 1))


# ---------------------------------------------------------------------------
# cost structure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [16, 32, 64])
def test_the_multiply_is_n_squared_plus_a_linear_correction(n):
    r = count(_mul_circ(n))
    assert r.toffoli == n * n + 5 * n - 1, (n, r.toffoli)


def test_squaring_is_half_a_multiply_and_the_half_is_structural():
    """Not a tuned constant: the cross terms form a triangle, so the quadratic term is
    n(n-1)/2 against n^2 at every width, and the ratio falls MONOTONICALLY toward 1/2 as
    the shared linear correction is amortised. Asserting the trend, not one number."""
    ratios = []
    for n in (16, 32, 64, 128):
        m = count(_mul_circ(n)).toffoli
        s = count(_sqr_circ(n)).toffoli
        assert s - n * (n - 1) // 2 == pytest.approx(6 * n, abs=2 * n)
        ratios.append(s / m)
    assert ratios == sorted(ratios, reverse=True), ratios      # monotone decreasing
    assert 0.5 < ratios[-1] < 0.54, ratios                     # and heading to 1/2


def test_squaring_beats_calling_the_multiply_with_equal_operands():
    """The reason `modsqr` exists as its own version rather than `modmul(x, x)`."""
    n = 128
    assert count(_sqr_circ(n)).toffoli < count(_mul_circ(n)).toffoli * 0.55
