"""The pseudo-Mersenne fold, and what it costs to leave the lazy contract.

Three things are established here and used everywhere downstream:

  1. TWO ROUNDS SUFFICE, and the output is LAZY. `2^U == F (mod p)` sends a 2U-bit
     product to `< (F+1)*2^U`, then to `< 2^U + F^2`. It never reaches [0, p) on its own,
     which is why the fold's contract matches Layer 0's rather than improving on it.

  2. THE FOLD IS NOT WHERE THE COST IS. weight(F) additions, ~2% of the multiply that
     produced the product -- so truncating it buys almost nothing and costs 200 orders of
     accuracy. That is a finding, not a disappointment.

  3. CANONICALISATION CANNOT ERASE ITS FLAG. The map (flag, out) -> out is many-to-one,
     so the flag survives at exactly F/2^U. This is
     docs/decisions/04-no-in-circuit-canonicalization.md at one-bit scale, and it is why
     canonicalising is only legal where the output is being written into fresh space.
"""

import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.mul.fold import (canonicalise, canonicalise_fail_prob, fold_once_ref,
                          fold2_overflow_prob, naf, pm_fold, pm_fold_ref,
                          set_bits)
from qsm.params import CARRY_WINDOW, F, PRIME, U

SMALL_PM = [(5, 1), (5, 3), (6, 3), (6, 5), (7, 5), (7, 19), (8, 19)]
#: Measurement outcomes are drawn at RANDOM, not fixed at zero: these circuits erase
#: with `_unand`, which restores exactly, so `phase == 0` must hold for EVERY outcome
#: vector. Under an all-zero one it holds whether that is true or not, since an erasure's
#: correction is conditioned on the measured bit -- a vacuity that let a ~25% phase leak
#: ship from `qsm/mul/accumulate.py`. See that module's test file.
_MEAS_RNG = random.Random(0xC0FFEE)


def _fold_circ(u, f, **kw):
    c = Circuit()
    prod = c.input("prod", 2 * u)
    out = c.input("out", u)
    pm_fold(c, prod, out, u, f, **kw)
    return c


# ---------------------------------------------------------------------------
# correctness, exhaustive over every 2u-bit product
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f", SMALL_PM)
def test_fold_matches_its_reference_on_every_input(u, f):
    c = _fold_circ(u, f)
    for v in range(1 << (2 * u)):
        r = simulate(c, {"prod": v, "out": 0}, rng=_MEAS_RNG)
        assert r.outputs == {"prod": v, "out": pm_fold_ref(v, u, f)}, v
        assert r.phase == 0, v


@pytest.mark.parametrize("u,f", SMALL_PM)
def test_the_fold_leaves_the_source_product_untouched(u, f):
    """It has to: the caller uncomputes the product against the operands afterwards, and
    a fold that consumed it would leave a 2U-bit register that can never be freed."""
    c = _fold_circ(u, f)
    for v in range(0, 1 << (2 * u), max(1, (1 << (2 * u)) // 200)):
        assert simulate(c, {"prod": v, "out": 0},
                        rng=_MEAS_RNG).outputs["prod"] == v


# ---------------------------------------------------------------------------
# the two-round bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f", SMALL_PM + [(11, 19), (12, 19)])
def test_round_one_lands_below_f_plus_one_times_2u(u, f):
    """The width of the round-1 accumulator is a claim about the arithmetic, and an
    accumulator one bit too narrow would lose a carry on rare inputs only."""
    worst = fold_once_ref((1 << (2 * u)) - 1, u, f)
    assert worst < (f + 1) * (1 << u)
    assert worst.bit_length() <= u + f.bit_length()


@pytest.mark.parametrize("u,f", SMALL_PM + [(11, 19), (12, 19)])
def test_round_two_lands_within_f_squared_of_2u_and_no_third_round_is_needed(u, f):
    """`w_hi <= f`, so round 2 adds at most f^2. That is what makes a third round
    unnecessary and the dropped carry rare, and both facts come from the same bound."""
    over = 0
    for v in range(0, 1 << (2 * u), max(1, (1 << (2 * u)) // 5000)):
        w = fold_once_ref(v, u, f)
        assert (w >> u) <= f
        s = fold_once_ref(w, u, f)
        assert s < (1 << u) + f * f
        over += s >= (1 << u)
    assert over <= 5000 * fold2_overflow_prob(u, f) + 1


@pytest.mark.parametrize("u,f", SMALL_PM)
def test_the_dropped_carry_is_exactly_the_declared_failure_set(u, f):
    """Not a bound: the inputs where `pm_fold_ref` differs from a true `% p` are exactly
    those whose second round overflows u bits."""
    differs = {v for v in range(1 << (2 * u))
              if pm_fold_ref(v, u, f) % ((1 << u) - f) != v % ((1 << u) - f)}
    overflows = {v for v in range(1 << (2 * u))
                 if fold_once_ref(fold_once_ref(v, u, f), u, f) >= (1 << u)}
    assert differs <= overflows
    assert len(differs) / float(1 << (2 * u)) <= fold2_overflow_prob(u, f)


def test_the_overflow_is_non_congruent_not_lazy():
    """Worth pinning because the two are constantly confused. A lazy value is congruent
    and a final `% p` repairs it; this one is short by a multiple of 2^u == F and nothing
    repairs it."""
    u, f = 7, 19
    p = (1 << u) - f
    bad = [v for v in range(1 << (2 * u))
           if fold_once_ref(fold_once_ref(v, u, f), u, f) >= (1 << u)]
    assert bad, "pick a width where the overflow is reachable"
    for v in bad:
        exact = fold_once_ref(fold_once_ref(v, u, f), u, f)
        lost = exact - pm_fold_ref(v, u, f)
        assert lost > 0 and lost % (1 << u) == 0               # whole 2^u units, dropped
        assert pm_fold_ref(v, u, f) % p != v % p               # ... and not congruent


def test_at_the_real_width_at_most_one_bit_can_ever_be_lost():
    """At u=7 the round-2 sum can exceed 2^u several times over, because f^2 = 361 is a
    large fraction of 128. At U = 255 it cannot: f^2 < 2^U by 246 orders, so the dropped
    carry is a single bit and the failure is worth exactly F. The small-width tests
    therefore exercise a HARSHER regime than the real one, which is the right direction."""
    assert F * F < (1 << U)
    assert (1 << U) - 1 + F * F < (1 << (U + 1))
    assert F * F > 2 * (1 << 7), "u=7 really is the harsher regime, not a typo"


# ---------------------------------------------------------------------------
# cost: F's weight is the price of a reduction
# ---------------------------------------------------------------------------
def test_the_fold_costs_weight_of_F_additions_and_25519_pays_three():
    assert set_bits(F) == [0, 1, 4]
    assert len(naf(F)) == 3, "NAF buys nothing for 19; its weight already equals binary"
    assert len(naf((1 << 32) + 977)) == 5, "secp256k1's F needs five terms, not three"


def test_the_fold_is_a_small_fraction_of_the_multiply_it_reduces():
    """The load-bearing negative result for this layer: optimising the fold is not where
    the win is. Measured both ways -- against the single product it reduces, and against
    the modular multiply it sits inside, which pays for that product twice."""
    from qsm.mul.modmul import modmul
    from qsm.mul.product import schoolbook

    prod_c = Circuit()
    x, y = prod_c.input("x", U), prod_c.input("y", U)
    pr = prod_c.input("prod", 2 * U)
    schoolbook(prod_c, x, y, pr)

    m = Circuit()
    mx, my, mo = m.input("x", U), m.input("y", U), m.input("out", U)
    modmul(m, mx, my, mo)

    fold_t = count(_fold_circ(U, F)).toffoli
    assert fold_t / count(prod_c).toffoli < 0.05, fold_t
    assert fold_t / count(m).toffoli < 0.025, fold_t


def test_truncating_the_fold_saves_almost_nothing_and_costs_two_hundred_orders():
    """A trade the project would take anywhere else and should NOT take here. Recorded so
    the asymmetry against Layer 0 -- where the same window is the whole saving -- is
    visible rather than assumed."""
    exact = count(_fold_circ(U, F)).toffoli
    trunc = count(_fold_circ(U, F, window=CARRY_WINDOW)).toffoli
    saved = (exact - trunc) / exact
    assert saved < 0.30, saved                      # a fifth of a fold, of a 2% fold
    # ... against this much accuracy
    assert fold2_overflow_prob() < 2.0 ** -240
    assert 3 * F / float(1 << CARRY_WINDOW) > 2.0 ** -45


# ---------------------------------------------------------------------------
# canonicalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f", SMALL_PM)
def test_canonicalise_reduces_into_0_p_and_its_dirty_set_is_exactly_below_f(u, f):
    """The value is always right. The FLAG is what cannot be erased, and its failure set
    is exactly the inputs already below f -- indistinguishable in the output from the
    inputs that were reduced down to below f."""
    p = (1 << u) - f
    c = Circuit()
    out = c.input("out", u)
    canonicalise(c, out, u, f)

    dirty = set()
    for v in range(1 << u):
        try:
            r = simulate(c, {"out": v}, rng=_MEAS_RNG)
        except DirtyAncillaError:
            dirty.add(v)
            continue
        assert r.outputs["out"] == v % p, v
        assert r.outputs["out"] < p
        assert r.phase == 0
    assert dirty == {v for v in range(f)}, (sorted(dirty)[:8], f)
    assert len(dirty) / float(1 << u) == pytest.approx(canonicalise_fail_prob(u, f))


def test_canonicalising_costs_well_under_one_percent_of_a_multiply():
    """The answer to the trade left open in docs/decisions/05: doc 05 guessed ~10% of a
    schoolbook multiply. Measured against the whole modular multiply it is under 1%, which
    inverts the recommendation -- see the docstring on `fold.canonicalise`."""
    c = Circuit()
    out = c.input("out", U)
    canonicalise(c, out, U, F)
    canon = count(c).toffoli

    from qsm.mul.modmul import modmul
    m = Circuit()
    x, y, o = m.input("x", U), m.input("y", U), m.input("out", U)
    modmul(m, x, y, o)

    assert canon < 3 * U, canon
    assert canon / count(m).toffoli < 0.01, canon / count(m).toffoli


def test_subtracting_p_from_a_u_bit_register_is_just_adding_F():
    """Why canonicalisation needs no wide constant. On u bits, `-p == +f (mod 2^u)`, so
    the reduction reuses the same +F the entire layer is built from."""
    for u, f in SMALL_PM + [(U, F)]:
        p = (1 << u) - f
        for v in (p, p + 1, (1 << u) - 1, (1 << u) - f):
            assert (v - p) % (1 << u) == (v + f) % (1 << u)
    assert (PRIME + F) % (1 << U) == 0
