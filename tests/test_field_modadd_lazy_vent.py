"""Vented flag erasure (`modadd_lazy(..., vent=True)`, the default).

The point: with venting a dirty ancilla is structurally impossible. `vent=False` is
Algorithm 10 as published and IS exposed -- these tests pin the difference.
"""

import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add, gidney_add
from qsm.field.modarith import PRIME, modadd_lazy, modadd_lazy_ref
from qsm.params import CARRY_WINDOW, CMP_BITS, U, carry_fail_prob, cmp_fail_prob

SMALL_PM = [(4, 3, 13), (5, 3, 29), (5, 1, 31), (6, 3, 61)]


def _vented(u, p, add_fn=gidney_add, carry_window=None, cmp_bits=None):
    c = Circuit()
    a = c.input("a", u)
    b = c.input("b", u)
    modadd_lazy(c, a, b, p, add_fn,
                carry_window=carry_window, cmp_bits=cmp_bits, vent=True)
    return c


@pytest.mark.parametrize("add_fn", [cdkm_add, gidney_add], ids=lambda f: f.__name__)
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_vented_exact_matches_reference(u, f, p, add_fn):
    """With truncation off, vented must be exactly right AND phase-clean."""
    c = _vented(u, p, add_fn)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv})
            assert r.outputs["a"] == av
            assert r.outputs["b"] == modadd_lazy_ref(av, bv, p, u), f"{av}+{bv}"
            assert r.phase == 0, f"exact v5 must not leak phase ({av}+{bv})"


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_vented_agrees_with_cx_form_when_exact(u, f, p):
    """Venting only changes the erasure; untruncated the two must be indistinguishable."""
    cx = Circuit()
    modadd_lazy(cx, cx.input("a", u), cx.input("b", u), p, vent=False)
    vt = _vented(u, p)
    for av in range(p):
        for bv in range(p):
            assert (simulate(cx, {"a": av, "b": bv}).outputs
                    == simulate(vt, {"a": av, "b": bv}).outputs)


# ---------------------------------------------------------------------------
# THE POINT: truncated, a dirty ancilla is impossible
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p,L,k", [(8, 3, 253, 4, 3), (8, 3, 253, 5, 4),
                                       (10, 5, 1019, 5, 4)])
def test_vented_never_dirties_the_ancilla(u, f, p, L, k):
    """vent=False raises DirtyAncillaError at these settings; vent=True must never do so.
    Its comparator failures show up as PHASE instead, and its values stay correct
    exactly as often as the CX form's did."""
    c5 = _vented(u, p, carry_window=L, cmp_bits=k)
    rng = random.Random(31)
    trials = 4000
    wrong_value = phase_err = 0
    for _ in range(trials):
        av, bv = rng.randrange(p), rng.randrange(p)
        r = simulate(c5, {"a": av, "b": bv}, rng=rng)   # must never raise
        wrong_value += r.outputs["b"] != modadd_lazy_ref(av, bv, p, u)
        phase_err += r.phase != 0
    assert phase_err > 0, "comparator truncation should be producing phase errors"
    assert wrong_value > 0, "value truncation should still be active"
    # each mode stays inside its own standalone rate
    assert wrong_value / trials <= carry_fail_prob(L, f)
    assert phase_err / trials <= cmp_fail_prob(k)


@pytest.mark.parametrize("u,f,p,L,k", [(8, 3, 253, 4, 3), (10, 5, 1019, 5, 4)])
def test_cx_form_dirties_where_vented_does_not(u, f, p, L, k):
    """Side by side, same settings, same inputs: CX form leaves garbage, vented does not."""
    c4 = Circuit()
    modadd_lazy(c4, c4.input("a", u), c4.input("b", u), p,
                carry_window=L, cmp_bits=k, vent=False)
    c5 = _vented(u, p, carry_window=L, cmp_bits=k)

    rng = random.Random(37)
    cx_dirty = vent_dirty = 0
    for _ in range(3000):
        av, bv = rng.randrange(p), rng.randrange(p)
        try:
            simulate(c4, {"a": av, "b": bv}, rng=rng)
        except DirtyAncillaError:
            cx_dirty += 1
        try:
            simulate(c5, {"a": av, "b": bv}, rng=rng)
        except DirtyAncillaError:
            vent_dirty += 1
    assert cx_dirty > 0, "vent=False should dirty its flag at these settings"
    assert vent_dirty == 0, f"vent=True must NEVER dirty the flag, saw {vent_dirty}"


def test_the_KNOBS_are_pinned_to_their_consequences_not_to_themselves():
    """CMP_BITS could be changed 40 -> 39 and all 518 tests still passed.

    Every reference to it in the suite was either an import or an argument fed into a
    cost *inequality*, so nothing tied the knob to the accuracy it is chosen for. Same
    for CARRY_WINDOW, which was pinned only by an incidental hardcoded `45` in a doubling
    cost test. These assertions fail if either knob drifts from its target.
    """
    from qsm.params import F, U, controlled_carry_fail_prob

    # the design target: both knobs land on ~2^-40 so neither dominates the budget
    assert cmp_fail_prob(CMP_BITS, informative_bits=CMP_BITS - 1) <= 2.0 ** -40
    assert cmp_fail_prob(CMP_BITS, informative_bits=CMP_BITS - 1) > 2.0 ** -41

    # a CONTROLLED +F (modadd_lazy, modsub_direct): F/2^(L+1)
    assert controlled_carry_fail_prob(CARRY_WINDOW, F) <= 2.0 ** -41
    assert controlled_carry_fail_prob(CARRY_WINDOW, F) > 2.0 ** -42
    # an UNCONDITIONAL +F (modneg, the standalone const adders): F/2^L, one bit worse
    assert carry_fail_prob(CARRY_WINDOW, F) <= 2.0 ** -40
    assert carry_fail_prob(CARRY_WINDOW, F) > 2.0 ** -41

    # and the width really is the one the L = 40 + log2(F) rule asks for
    import math
    assert CARRY_WINDOW == math.ceil(40 + math.log2(F))
    assert 0 < CMP_BITS < U


def test_vented_25519_random():
    c = _vented(U, PRIME, carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS)
    rng = random.Random(41)
    for _ in range(24):
        av, bv = rng.randrange(PRIME), rng.randrange(PRIME)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs["a"] == av
        assert r.outputs["b"] == modadd_lazy_ref(av, bv, PRIME, U)
        assert r.outputs["b"] % PRIME == (av + bv) % PRIME
        assert r.phase == 0


@pytest.mark.parametrize("u,f,p", [(6, 3, 61), (8, 5, 251), (10, 5, 1019)])
def test_the_DEFAULT_vents__caller_passing_nothing_is_still_safe(u, f, p):
    """Every other test here passes `vent=` explicitly, so none of them exercises the
    DEFAULT -- and the default is the whole safety argument.

    Flipping `vent: bool = True` to `False` in the signature used to pass all 518 tests
    while making dirty ancillas possible again (36/1500 measured at u=10). This calls
    `modadd_lazy` the way a downstream caller will, with truncation on and nothing said
    about venting, and demands the flag still come back clean.
    """
    c = Circuit()
    a, b = c.input("a", u), c.input("b", u)
    modadd_lazy(c, a, b, p, gidney_add, carry_window=4, cmp_bits=3)   # no vent= given

    rng = random.Random(17)
    for _ in range(700):
        av, bv = rng.randrange(p), rng.randrange(p)
        try:
            simulate(c, {"a": av, "b": bv}, rng=random.Random(rng.random()))
        except DirtyAncillaError:                      # pragma: no cover - the bug
            pytest.fail(
                "modadd_lazy's DEFAULT left a dirty ancilla: the vent default is off")


@pytest.mark.parametrize("u,f,p", [(8, 5, 251), (10, 5, 1019)])
def test_venting_is_needed_EXACTLY_when_the_erasure_predicate_is_truncated(u, f, p):
    """Scopes when `vent=False` is legitimate, so the option can stay without being a trap.

    The dirtiness comes from the *erasure predicate*, not the carry window. With an exact
    `cmp_bits` the CX recompute clears the flag perfectly and `vent=False` is a genuine
    (cheaper-to-reason-about) choice. With a truncated `cmp_bits` it fails open and leaves
    entangled garbage, which is exactly what venting converts into a phase.
    """
    def dirty(carry_window, cmp_bits, vent):
        c = Circuit()
        a, b = c.input("a", u), c.input("b", u)
        modadd_lazy(c, a, b, p, gidney_add, carry_window, cmp_bits, vent=vent)
        rng = random.Random(3)
        n = 0
        for _ in range(500):
            av, bv = rng.randrange(p), rng.randrange(p)
            try:
                simulate(c, {"a": av, "b": bv}, rng=random.Random(rng.random()))
            except DirtyAncillaError:
                n += 1
        return n

    assert dirty(None, None, False) == 0, "exact everything: CX erasure is fine"
    assert dirty(4, None, False) == 0, "a truncated CARRY does not dirty the flag"
    assert dirty(None, 3, False) > 0, "a truncated ERASURE does -- this is the trap"
    assert dirty(4, 3, False) > 0
    for cw, k in ((None, 3), (4, 3)):
        assert dirty(cw, k, True) == 0, "venting must remove it in every case"


def test_vented_top_carry_costs_no_toffoli():
    """Venting also makes the top step cheaper: a conditioned CZ+Z instead of an AND."""
    v4 = Circuit()
    modadd_lazy(v4, v4.input("a", U), v4.input("b", U), PRIME,
                carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS, vent=False)
    v5 = _vented(U, PRIME, carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS)
    rc, rv = count(v4), count(v5)
    assert rv.toffoli <= rc.toffoli, f"vented {rv.toffoli} should not exceed CX {rc.toffoli}"
    assert rv.peak_qubits <= rc.peak_qubits
