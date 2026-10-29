"""What breaks when a Layer-0 value precondition is violated.

A precondition a circuit cannot check is folklore until something feeds it the violating
band and records the result. `moddouble` got that treatment first (predicate + lazy-input
test + scheduler rule); these are the rest of them, and two were worse than their
docstrings implied:

    modadd / modadd_cmp   a lazy ACCUMULATOR leaves a DIRTY ANCILLA -- entangled
                          garbage, which a value-only test cannot see at all
    modneg                a lazy operand gives a NON-CONGRUENT result -- unrepairable,
                          not the ordinary lazy-representative case

Reproduce the whole table with `python -m qsm.audit preconditions`.
"""

import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.simulator import simulate
from qsm.field.add import gidney_add
from qsm.field.modarith import (
    modadd, modadd_cmp, modneg, modsub, modsub_direct,
)
from qsm.core.resources import count  # noqa: F401  -- used inside tests
from qsm.params import (
    canonical_modadd_dirties, modneg_corrupts, modsub_direct_corrupts,
)

WIDTHS = [(8, 3, 253), (8, 5, 251), (10, 5, 1019), (11, 19, 2029)]


# ---------------------------------------------------------------------------
# modneg: non-congruent above p, and the predicate is exact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", WIDTHS)
def test_modneg_corrupts_exactly_above_p(u, f, p):
    c = Circuit()
    a = c.input("a", u)
    modneg(c, a, p, window=None)

    bad = []
    for av in range(1 << u):
        got = simulate(c, {"a": av}).outputs["a"]
        if got % p != (-av) % p:
            bad.append(av)

    assert bad == [x for x in range(1 << u) if x > p], \
        "the corrupting set must be exactly (p, 2^u)"
    assert all(modneg_corrupts(x, u, f) for x in bad)
    assert not any(modneg_corrupts(x, u, f) for x in range(p + 1)), \
        "a <= p must never be flagged -- a == p is fine, it yields 0"
    assert len(bad) == f - 1


@pytest.mark.parametrize("u,f,p", WIDTHS)
def test_modneg_failure_is_NOT_repairable_by_a_final_reduction(u, f, p):
    """The distinction that matters: laziness is congruent, this is not."""
    c = Circuit()
    a = c.input("a", u)
    modneg(c, a, p, window=None)
    for av in range(p + 1, 1 << u):
        got = simulate(c, {"a": av}).outputs["a"]
        assert got % p != (-av) % p, "if it were congruent, `% p` would repair it"


@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (8, 5, 251), (10, 5, 1019)])
def test_modsub_direct_corrupts_exactly_when_a_minus_b_exceeds_p(u, f, p):
    """Its requirement is `a - b <= p`, weaker than `a < p` -- and it is NOT immune.

    I first recorded this circuit as having no value precondition at all, because a
    random sweep over lazy `a` never happened to pair one with a small enough `b`. The
    failure needs both at once, which is exactly why it is 250 orders rarer than
    modneg's -- but "rarer" is not "absent".
    """
    c = Circuit()
    a, b = c.input("a", u), c.input("b", u)
    modsub_direct(c, a, b, p, gidney_add, None, None)

    bad = [(av, bv) for av in range(1 << u) for bv in range(1 << u)
           if simulate(c, {"a": av, "b": bv}).outputs["b"] % p != (bv - av) % p]

    assert sorted(bad) == sorted((av, bv) for av in range(1 << u)
                                 for bv in range(1 << u) if av - bv > p)
    assert len(bad) == f * (f - 1) // 2, "the failure set is exactly f(f-1)/2 pairs"
    assert all(modsub_direct_corrupts(av, bv, u, f) for av, bv in bad)
    assert not any(modsub_direct_corrupts(av, bv, u, f)
                   for av in range(p) for bv in range(p)), \
        "a canonical `a` makes a - b <= a < p automatic"


@pytest.mark.parametrize("u,f,p", [(10, 5, 1019), (11, 19, 2029)])
def test_the_modneg_hazard_propagates_into_the_sandwich_but_barely_into_direct(u, f, p):
    """modsub.v1 calls modneg and inherits its single-operand hazard; direct does not.

    Both are fed the same lazy `a`. The sandwich fails broadly; the direct form fails
    only on the narrow `a - b > p` corner, so with `b` drawn uniformly it essentially
    never does.
    """
    sand = Circuit()
    a1, b1 = sand.input("a", u), sand.input("b", u)
    modsub(sand, a1, b1, p, gidney_add, None, None)

    direct = Circuit()
    a2, b2 = direct.input("a", u), direct.input("b", u)
    modsub_direct(direct, a2, b2, p, gidney_add, None, None)

    rng = random.Random(5)
    sand_bad = direct_bad = 0
    for _ in range(220):
        av, bv = rng.randrange(p + 1, 1 << u), rng.randrange(f, p)   # b >= f => a-b <= p
        want = (bv - av) % p
        if simulate(sand, {"a": av, "b": bv}).outputs["b"] % p != want:
            sand_bad += 1
        if simulate(direct, {"a": av, "b": bv}).outputs["b"] % p != want:
            direct_bad += 1

    assert sand_bad > 0, "the sandwich must inherit modneg's hazard"
    assert direct_bad == 0, \
        "outside the a-b>p corner the direct form must be unaffected by a lazy a"


# ---------------------------------------------------------------------------
# canonical modadd: a lazy ACCUMULATOR dirties an ancilla
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [modadd, modadd_cmp], ids=["modadd", "modadd_cmp"])
@pytest.mark.parametrize("u,f,p", [(8, 5, 251), (10, 5, 1019), (11, 19, 2029)])
def test_canonical_modadd_dirties_on_a_lazy_ACCUMULATOR_only(fn, u, f, p):
    """The failure is a dirty ancilla, not a wrong value -- and only `b` matters.

    A value-only assertion would pass straight through this: the returned sum can be
    correct while the flag qubit comes back |1>.
    """
    c = Circuit()
    a, b = c.input("a", u), c.input("b", u)
    fn(c, a, b, p, gidney_add)

    def dirty(av, bv):
        try:
            simulate(c, {"a": av, "b": bv})
            return False
        except DirtyAncillaError:
            return True

    rng = random.Random(11)
    for _ in range(60):
        lazy_b = rng.randrange(p, 1 << u)
        canon_b = rng.randrange(p)
        lazy_a = rng.randrange(p, 1 << u)
        canon_a = rng.randrange(p)

        assert dirty(canon_a, lazy_b), "a lazy accumulator must dirty the flag"
        assert not dirty(lazy_a, canon_b), \
            "a lazy PRESERVED operand is harmless -- the precondition is only on b"
        assert not dirty(canon_a, canon_b)

    assert canonical_modadd_dirties(p, u, f)
    assert not canonical_modadd_dirties(p - 1, u, f)


@pytest.mark.parametrize("u,f,p", [(10, 5, 1019), (11, 19, 2029)])
def test_the_predicate_matches_the_circuit_exhaustively_on_the_accumulator(u, f, p):
    c = Circuit()
    a, b = c.input("a", u), c.input("b", u)
    modadd(c, a, b, p, gidney_add)
    av = 3
    for bv in range(1 << u):
        try:
            simulate(c, {"a": av, "b": bv})
            got = False
        except DirtyAncillaError:
            got = True
        assert got == canonical_modadd_dirties(bv, u, f), f"b={bv}"


# ---------------------------------------------------------------------------
# the lazy construction is the one that survives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", [(8, 5, 251), (10, 5, 1019), (11, 19, 2029)])
def test_venting_v3_moves_the_failure_out_of_the_dirty_class(u, f, p):
    """The chosen fix for the canonical-modadd hazard: reclassify, don't repair.

    Venting cannot make a lazy accumulator CORRECT -- after step 3 the "not reduced" and
    "reduced with lazy b" cases overlap in the sign bit, so no single bit separates them.
    What it does is erase the flag by measurement, so a wrong predicate leaves residual
    phase instead of an entangled dirty ancilla. Phase is the class the layer already
    budgets for; a dirty ancilla is not.
    """
    def run(vent):
        c = Circuit()
        a, b = c.input("a", u), c.input("b", u)
        modadd_cmp(c, a, b, p, gidney_add, vent=vent)
        rng = random.Random(1)
        dirty = phase = 0
        for _ in range(300):
            av, bv = rng.randrange(p), rng.randrange(p, 1 << u)   # lazy ACCUMULATOR
            try:
                r = simulate(c, {"a": av, "b": bv}, rng=random.Random(rng.random()))
                phase += r.phase
            except DirtyAncillaError:
                dirty += 1
        return dirty, phase

    unvented_dirty, _ = run(False)
    vented_dirty, vented_phase = run(True)

    assert unvented_dirty > 0, "v3 unvented must dirty on a lazy accumulator"
    assert vented_dirty == 0, "venting must remove the dirty-ancilla class entirely"
    assert vented_phase > 0, "...by converting it into phase, not by fixing the value"


@pytest.mark.parametrize("u,f,p", [(8, 5, 251), (10, 5, 1019)])
def test_venting_v3_is_free_and_still_exact_on_canonical_input(u, f, p):
    """Reclassifying must not cost correctness on the inputs that were always fine."""
    from qsm.core.resources import count

    circs = {}
    for vent in (False, True):
        c = Circuit()
        a, b = c.input("a", u), c.input("b", u)
        modadd_cmp(c, a, b, p, gidney_add, vent=vent)
        circs[vent] = c
        for av in range(0, p, max(1, p // 30)):
            for bv in range(0, p, max(1, p // 30)):
                r = simulate(c, {"a": av, "b": bv}, rng=random.Random(3))
                assert r.outputs["b"] == (av + bv) % p, f"vent={vent} {av}+{bv}"
                assert r.phase == 0, "canonical input must stay phase-clean"

    assert count(circs[True]).toffoli <= count(circs[False]).toffoli, \
        "venting should not cost Toffoli -- the top carry becomes a conditioned CZ+Z"


@pytest.mark.parametrize("u,f,p", [(10, 5, 1019), (11, 19, 2029)])
def test_the_lazy_plus_lazy_hole_is_exactly_the_accepted_one(u, f, p):
    """Decision: lazy+lazy overflow is ACCEPTED, not fixed. Pin what we accepted.

    It needs BOTH operands in a band of F values, so (F/2^U)^2 = 2^-501.6 -- some 460
    orders below the truncation error already in the budget. Recorded rather than
    guarded, so nobody later mistakes it for an unexamined case.
    """
    from qsm.field.modarith import modadd_lazy

    c = Circuit()
    a, b = c.input("a", u), c.input("b", u)
    modadd_lazy(c, a, b, p, gidney_add, None, None, vent=True)

    overflow = []
    for av in range(p, 1 << u):
        for bv in range(p, 1 << u):
            got = simulate(c, {"a": av, "b": bv}).outputs["b"]
            if got % p != (av + bv) % p:
                overflow.append((av, bv))
            assert got < (1 << u), "must never leave the register"

    assert overflow, "the hole is real -- if this stops firing the contract changed"
    assert all(av + bv - p >= (1 << u) for av, bv in overflow), \
        "every failure must be the a+b-p overflow, not some other mechanism"
    # and one lazy operand is still closed
    for av in range(p, 1 << u):
        got = simulate(c, {"a": av, "b": 3}).outputs["b"]
        assert got % p == (av + 3) % p, "canonical+lazy must stay closed"


@pytest.mark.parametrize("u,f,p", [(10, 5, 1019), (11, 19, 2029)])
def test_modadd_lazy_is_closed_for_canonical_plus_lazy(u, f, p):
    """Why v4 is the default: it tolerates the single lazy operand that v1-v3 cannot."""
    from qsm.field.modarith import modadd_lazy

    c = Circuit()
    a, b = c.input("a", u), c.input("b", u)
    modadd_lazy(c, a, b, p, gidney_add, None, None, vent=True)

    rng = random.Random(2)
    for _ in range(250):
        av, bv = rng.randrange(p), rng.randrange(p, 1 << u)   # canonical + lazy
        r = simulate(c, {"a": av, "b": bv})                   # must not raise
        assert r.outputs["b"] % p == (av + bv) % p, "must stay congruent"
        assert r.outputs["b"] < (1 << u), "must stay in the register"
