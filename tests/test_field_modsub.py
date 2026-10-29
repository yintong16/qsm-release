"""modsub: b <- (b - a) mod p, two constructions.

    v1 `modsub`        modneg(a); modadd_lazy(a,b); modneg(a)   -- the sandwich
    v2 `modsub_direct` borrow flag + `+p if borrow` + vented erasure

The interesting result is that v2 is BOTH cheaper and better behaved: it returns a
canonically reduced value where the sandwich returns a lazy one. Subtraction gets for
free what addition has to be lazy to afford, because the borrow bit answers exactly the
question that matters (`is b < a?`) with no gap, while modadd's overflow bit answers
`is a+b >= 2^U?` when it needed `is a+b >= p?`.

What v2 pays instead is a phase term: its erasure predicate compares against `~a` rather
than the unavailable `p - a`, which is wrong on a window of width F, i.e. F/(2p) after
the 1/2 from the measurement outcome. That is 2^-251.75 -- below the lazy contract, and
210 orders below the truncations.
"""

import math
import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add, gidney_add
from qsm.field.modarith import (
    PRIME,
    modsub,
    modsub_direct,
    modsub_lazy_ref,
    modsub_ref,
)
from qsm.params import CARRY_WINDOW, CMP_BITS, F, U

SMALL_PM = [(6, 3, 61), (7, 5, 123), (8, 3, 253), (8, 5, 251)]


def _circ(fn, u, p, add_fn=gidney_add, carry_window=None, cmp_bits=None):
    c = Circuit()
    a = c.input("a", u)
    b = c.input("b", u)
    fn(c, a, b, p, add_fn, carry_window, cmp_bits)
    return c


# ---------------------------------------------------------------------------
# exact correctness, exhaustive, against the RIGHT reference for each version
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_sandwich_exact_matches_its_LAZY_reference(u, f, p):
    """v1 routes through modadd_lazy, so it inherits laziness. Comparing it against the
    canonical reference (or against `% p`) would hide exactly that."""
    c = _circ(modsub, u, p)
    rng = random.Random(0)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv}, rng=rng)
            assert r.outputs["b"] == modsub_lazy_ref(av, bv, p, u), f"p={p}: {bv}-{av}"
            assert r.outputs["a"] == av, "a must be restored"
            assert r.outputs["b"] < (1 << u)


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_direct_exact_is_CANONICALLY_reduced(u, f, p):
    """v2's headline property: output in [0, p), not merely [0, 2^u)."""
    c = _circ(modsub_direct, u, p)
    rng = random.Random(0)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv}, rng=rng)
            assert r.outputs["b"] == modsub_ref(av, bv, p), f"p={p}: {bv}-{av}"
            assert r.outputs["b"] < p, "direct modsub must be canonically reduced"
            assert r.outputs["a"] == av, "a must be restored"


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_the_two_versions_agree_modulo_p_but_not_on_the_nose(u, f, p):
    """Congruent, but different representatives -- which is the whole distinction."""
    c1, c2 = _circ(modsub, u, p), _circ(modsub_direct, u, p)
    rng = random.Random(1)
    differ = 0
    for av in range(p):
        for bv in range(p):
            o1 = simulate(c1, {"a": av, "b": bv}, rng=rng).outputs["b"]
            o2 = simulate(c2, {"a": av, "b": bv}, rng=rng).outputs["b"]
            assert o1 % p == o2 % p, "must be congruent"
            differ += o1 != o2
    assert differ > 0, "the sandwich should sometimes return the unreduced representative"


@pytest.mark.parametrize("add_fn", [cdkm_add, gidney_add], ids=lambda f: f.__name__)
def test_both_versions_work_with_either_adder(add_fn):
    u, p = 7, 123
    rng = random.Random(2)
    for fn, ref in [(modsub, lambda a, b: modsub_lazy_ref(a, b, p, u)),
                    (modsub_direct, lambda a, b: modsub_ref(a, b, p))]:
        c = _circ(fn, u, p, add_fn=add_fn)
        for av in range(0, p, 5):
            for bv in range(0, p, 7):
                assert simulate(c, {"a": av, "b": bv}, rng=rng).outputs["b"] == ref(av, bv)


# ---------------------------------------------------------------------------
# the erasure predicate: where v2's phase term comes from
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_direct_predicate_is_wrong_on_exactly_a_window_of_width_F(u, f, p):
    """Pure arithmetic. The exact separator is `r >= p - a`; we compare against `~a`
    because `p - a` is not in any register (materialising it IS the sandwich).

        p - a  ==  ~a - (F - 1)

    so `r > ~a` and `r >= p - a` differ on a window of width F -> rate F/p.
    """
    mask = (1 << u) - 1
    exact = strict = 0
    for av in range(p):
        for bv in range(p):
            borrow = bv < av
            r = modsub_ref(av, bv, p)
            exact += (r >= (p - av)) != borrow
            strict += (r > (~av & mask)) != borrow
    assert exact == 0, "`r >= p - a` must be an exact separator"
    rate = strict / p ** 2
    assert abs(rate - f / p) < 0.05 * (f / p), f"expected ~F/p={f/p:.5f}, got {rate:.5f}"


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_direct_phase_rate_is_F_over_2p_and_values_stay_correct(u, f, p):
    """The measurement contributes m, so only half of wrong predicates show as phase.

    CAUTION: F/p is only small when p is large. At u=6 it is 4.8%, so a small-width
    exhaustive run sees phase errors in bulk -- that is the scale, not a defect. What
    must hold at every width is that the VALUE is right.
    """
    c = _circ(modsub_direct, u, p)
    rng = random.Random(3)
    phase = 0
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv}, rng=rng)
            assert r.outputs["b"] == modsub_ref(av, bv, p), "value must always be correct"
            phase += r.phase != 0
    rate = phase / p ** 2
    predicted = f / (2 * p)
    assert abs(rate - predicted) < 0.25 * predicted, \
        f"phase {rate:.5f} vs predicted F/2p {predicted:.5f}"


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_sandwich_has_NO_phase_term_at_all(u, f, p):
    """The sandwich pays Toffoli to buy exactness: with a' = p - a materialised, the
    modadd erasure `result < a'` is exact, so nothing kicks back."""
    c = _circ(modsub, u, p)
    rng = random.Random(4)
    for av in range(p):
        for bv in range(p):
            assert simulate(c, {"a": av, "b": bv}, rng=rng).phase == 0


def test_neither_version_ever_leaves_a_dirty_ancilla():
    u, p = 8, 253
    rng = random.Random(5)
    for fn in (modsub, modsub_direct):
        for cw, k in [(None, None), (5, 4), (4, 3)]:
            c = _circ(fn, u, p, carry_window=cw, cmp_bits=k)
            for av in range(0, p, 3):
                for bv in range(0, p, 5):
                    simulate(c, {"a": av, "b": bv}, rng=rng)   # would raise


# ---------------------------------------------------------------------------
# truncated
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (8, 5, 251)])
def test_truncation_degrades_values_gracefully_in_both(u, f, p):
    """Truncation makes VALUES wrong (that is its nature); it must not make ancillas
    dirty, and the rate must fall as the window widens."""
    rng = random.Random(6)
    for fn, ref in [(modsub, lambda a, b: modsub_lazy_ref(a, b, p, u)),
                    (modsub_direct, lambda a, b: modsub_ref(a, b, p))]:
        rates = []
        for cw in (3, 5, 7):
            c = _circ(fn, u, p, carry_window=cw, cmp_bits=None)
            wrong = sum(simulate(c, {"a": av, "b": bv}, rng=rng).outputs["b"] != ref(av, bv)
                        for av in range(p) for bv in range(0, p, 3))
            rates.append(wrong)
        assert rates[0] >= rates[1] >= rates[2], f"{fn.__name__}: {rates} should decrease"


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def test_direct_beats_the_sandwich_on_cost_AND_on_contract():
    """The result worth reporting: v2 is cheaper and returns the better representation.
    Normally those trade against each other."""
    sand = count(_circ(modsub, U, PRIME, carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS))
    direct = count(_circ(modsub_direct, U, PRIME,
                         carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS))
    assert direct.toffoli < sand.toffoli, f"{direct.toffoli} vs {sand.toffoli}"
    assert direct.peak_qubits <= sand.peak_qubits


def test_modsub_is_far_below_litinski():
    """Litinski (arXiv:2306.08585) gives 6n for curve-agnostic modular subtraction."""
    direct = count(_circ(modsub_direct, U, PRIME,
                         carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS)).toffoli
    assert direct < 6 * U / 4, f"{direct} should be >4x below 6n={6 * U}"


def test_modsub_costs_the_same_as_modadd():
    """Subtraction is not more expensive than addition here -- same three steps, all
    mirrored. Worth pinning: a naive `negate, add, negate` would be 3x this."""
    from qsm.field.modarith import modadd_lazy

    c = Circuit()
    a = c.input("a", U)
    b = c.input("b", U)
    modadd_lazy(c, a, b, PRIME, gidney_add, CARRY_WINDOW, CMP_BITS, vent=True)
    add = count(c).toffoli
    subd = count(_circ(modsub_direct, U, PRIME,
                       carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS)).toffoli
    assert subd == add, f"modsub {subd} should equal modadd {add}"


def test_direct_phase_term_is_negligible_at_25519():
    """F/(2p) = 2^-251.75 -- below even the lazy contract's F/2^U."""
    rate = F / (2 * PRIME)
    assert math.log2(rate) < -251
    from qsm.params import lazy_unreduced_prob
    assert rate < lazy_unreduced_prob(U, F)
