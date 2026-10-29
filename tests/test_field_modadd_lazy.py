import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add, gidney_add
from qsm.field.modarith import PRIME, modadd_cmp, modadd_lazy, modadd_lazy_ref
from qsm.params import CARRY_WINDOW, CMP_BITS, F, U, carry_fail_prob, cmp_fail_prob

ADDERS = [cdkm_add, gidney_add]
# (u, f, p) with p = 2^u - f, all pseudo-Mersenne like 2^255 - 19
SMALL_PM = [(4, 3, 13), (5, 3, 29), (5, 1, 31), (6, 3, 61), (7, 5, 123)]


def _circ(u, p, add_fn=gidney_add, carry_window=None, cmp_bits=None, vent=False):
    """Defaults to vent=False: this file documents the CX-erasure form and its
    dirty-ancilla failure. The vented (default) form is covered in
    tests/test_field_modadd_vented.py."""
    c = Circuit()
    a = c.input("a", u)
    b = c.input("b", u)
    modadd_lazy(c, a, b, p, add_fn, carry_window=carry_window, cmp_bits=cmp_bits,
                vent=vent)
    return c


# ---------------------------------------------------------------------------
# exact (no truncation): the lazy algorithm itself must be exactly right
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_lazy_exact_matches_its_reference(u, f, p, add_fn):
    """Exhaustive over all reduced (a, b). With truncation off there is no error term."""
    c = _circ(u, p, add_fn)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv})
            assert r.outputs["a"] == av, f"a changed (p={p}, {av},{bv})"
            assert r.outputs["b"] == modadd_lazy_ref(av, bv, p, u), \
                f"p={p}: {av}+{bv} -> {r.outputs['b']}"
            assert r.phase == 0


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_lazy_output_is_congruent_mod_p(u, f, p):
    """The result may be lazily reduced, but it is ALWAYS congruent to a+b mod p."""
    c = _circ(u, p)
    for av in range(p):
        for bv in range(p):
            got = simulate(c, {"a": av, "b": bv}).outputs["b"]
            assert got % p == (av + bv) % p, f"p={p}: {av}+{bv} -> {got}"
            assert got < (1 << u), "result must still fit the register"


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_lazy_reduction_happens_and_is_rare(u, f, p):
    """Exactly the f residues in [p, 2^u) come out under-reduced -- no more, no less."""
    c = _circ(u, p)
    under = 0
    for av in range(p):
        for bv in range(p):
            got = simulate(c, {"a": av, "b": bv}).outputs["b"]
            under += got >= p
    total = p * p
    # every (a,b) whose plain sum lands in [p, 2^u) is under-reduced
    expected = sum(1 for av in range(p) for bv in range(p)
                   if p <= av + bv < (1 << u))
    assert under == expected, f"p={p}: {under} under-reduced vs {expected} expected"
    assert under / total < 2.0 * f / (1 << u), "under-reduction must stay rare"


def test_lazy_representation_is_NOT_closed():
    """Recorded limitation: feeding a lazily-reduced value back in can leave the flag
    ancilla DIRTY.

    Step 3 erases the flag via `result < a`, which holds iff b_0 < p. An under-reduced
    input has b_0 in [p, 2^u), so the erasure silently fails and the ancilla is freed at
    |1> -- entangled garbage, not merely a wrong number. It is rare in practice
    (F/2^U ~ 2^-250.8 for real inputs) but it is NOT structurally impossible, so the
    representation is only approximately closed. Any caller that could produce
    under-reduced values at higher rates than that must reduce before re-entering.
    """
    u, f, p = 6, 3, 61
    c = _circ(u, p)
    dirty = 0
    for av in range(p):
        for bv in range(p, 1 << u):          # deliberately under-reduced input
            try:
                got = simulate(c, {"a": av, "b": bv}).outputs["b"]
                assert got % p == (av + bv) % p, f"{av}+{bv} -> {got}"
            except DirtyAncillaError:
                dirty += 1
    assert dirty > 0, "expected under-reduced inputs to break the erasure"


def test_lazy_is_closed_for_properly_reduced_inputs():
    """The contract that IS guaranteed: reduced inputs never dirty the ancilla."""
    for u, f, p in SMALL_PM:
        c = _circ(u, p)
        for av in range(p):
            for bv in range(p):
                simulate(c, {"a": av, "b": bv})   # raises if the flag is left set


# ---------------------------------------------------------------------------
# truncated: rate must match the analytic sum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p,L,k", [(8, 3, 253, 4, 3), (8, 3, 253, 5, 4),
                                       (10, 5, 1019, 5, 4)])
def test_lazy_truncated_rate_matches_analytic(u, f, p, L, k):
    """Total failure must not exceed the analytic sum, and the two modes are separated.

    Both truncations only fire on the branch where a reduction happens (the +F is
    controlled on the overflow, and the erasure is only wrong when the flag is set), so
    each contributes about HALF its standalone rate. The analytic sum is therefore an
    upper bound here, not an equality.
    """
    c = _circ(u, p, carry_window=L, cmp_bits=k)
    rng = random.Random(13)
    trials = 4000
    wrong_value = dirty = 0
    for _ in range(trials):
        av, bv = rng.randrange(p), rng.randrange(p)
        try:
            r = simulate(c, {"a": av, "b": bv}, rng=rng)
        except DirtyAncillaError:
            dirty += 1                      # step-3 truncation: flag left set
            continue
        assert r.phase == 0, "truncation must never leak phase"
        wrong_value += r.outputs["b"] != modadd_lazy_ref(av, bv, p, u)

    total = (wrong_value + dirty) / trials
    bound = carry_fail_prob(L, f) + cmp_fail_prob(k)
    assert total <= bound, f"u={u} L={L} k={k}: {total:.5f} exceeds bound {bound:.5f}"
    assert total > 0.25 * bound, "rate implausibly low -- is truncation actually on?"
    # both modes must be present and each within its own standalone rate
    assert 0 < wrong_value / trials <= carry_fail_prob(L, f)
    assert 0 < dirty / trials <= cmp_fail_prob(k)


@pytest.mark.parametrize("u,f,p,L,k", [(8, 3, 253, 4, 3), (10, 5, 1019, 5, 4)])
def test_exact_erasure_never_dirties_the_ancilla(u, f, p, L, k):
    """The `safe` variant: truncate the VALUE but keep the erasure exact.
    Wrong answers remain possible; entangled garbage does not."""
    c = _circ(u, p, carry_window=L, cmp_bits=None)   # exact step 3
    rng = random.Random(23)
    wrong = 0
    for _ in range(2500):
        av, bv = rng.randrange(p), rng.randrange(p)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)   # must never raise
        wrong += r.outputs["b"] != modadd_lazy_ref(av, bv, p, u)
        assert r.phase == 0
    assert wrong > 0, "value truncation should still be active"
    assert wrong / 2500 <= carry_fail_prob(L, f)


def test_lazy_25519_random():
    c = _circ(U, PRIME, carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS)
    rng = random.Random(17)
    for _ in range(24):
        av, bv = rng.randrange(PRIME), rng.randrange(PRIME)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs["a"] == av
        assert r.outputs["b"] == modadd_lazy_ref(av, bv, PRIME, U)
        assert r.outputs["b"] % PRIME == (av + bv) % PRIME
        assert r.phase == 0


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def test_lazy_v4_is_far_cheaper_than_v3():
    v3 = Circuit()
    modadd_cmp(v3, v3.input("a", U), v3.input("b", U), PRIME, gidney_add)
    v4 = Circuit()
    modadd_lazy(v4, v4.input("a", U), v4.input("b", U), PRIME, gidney_add,
                carry_window=CARRY_WINDOW, cmp_bits=CMP_BITS)
    r3, r4 = count(v3), count(v4)
    assert r4.toffoli < r3.toffoli // 2, f"v4 {r4.toffoli} vs v3 {r3.toffoli}"
    assert r4.peak_qubits <= r3.peak_qubits
    assert r4.clifford < r3.clifford


def test_lazy_needs_no_flag_qubit_beyond_the_overflow():
    """v4 allocates exactly one ancilla of its own: the overflow, which IS the flag."""
    u = 32
    p = (1 << u) - 5
    c = _circ(u, p, carry_window=8, cmp_bits=6)
    # a + b + the single overflow bit; everything else is transient adder/comparator
    # scratch, so the peak can never dip below this.
    assert count(c).peak_qubits >= 2 * u + 1
