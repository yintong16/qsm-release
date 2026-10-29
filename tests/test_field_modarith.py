import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add, gidney_add
from qsm.field.modarith import PRIME, modadd, modadd_cmp

ADDERS = [cdkm_add, gidney_add]

# Small primes of the same pseudo-Mersenne shape 2^k - c, so the exhaustive tests
# exercise the same carry/borrow structure as 2^255 - 19.
SMALL_PRIMES = [(3, 5), (3, 7), (4, 13), (5, 29), (5, 31), (6, 61)]


def _modadd_circuit(
    n: int, p: int, add_fn, inplace_const: bool = True, pseudo_mersenne: bool = True
) -> Circuit:
    c = Circuit()
    a = c.input("a", n)
    b = c.input("b", n)
    modadd(c, a, b, p, add_fn,
           inplace_const=inplace_const, pseudo_mersenne=pseudo_mersenne)
    return c


@pytest.mark.parametrize("pseudo_mersenne", [False, True], ids=["naive_p", "pm_decomp"])
@pytest.mark.parametrize("inplace_const", [False, True], ids=["v1_reg", "v2_inplace"])
@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n,p", SMALL_PRIMES)
def test_modadd_exhaustive(
    add_fn, n: int, p: int, inplace_const: bool, pseudo_mersenne: bool
):
    """Every reduced (a,b) pair, both reduction branches, all four const strategies."""
    c = _modadd_circuit(n, p, add_fn, inplace_const, pseudo_mersenne)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv})
            assert r.outputs["a"] == av, f"a changed (p={p}, {av},{bv})"
            assert r.outputs["b"] == (av + bv) % p, \
                f"wrong sum (p={p}, {av}+{bv} -> {r.outputs['b']}, want {(av+bv)%p})"
            assert r.phase == 0, f"dirty phase (p={p}, {av},{bv})"


def test_modadd_hits_both_branches():
    """Guard against a test set that accidentally never triggers reduction."""
    n, p = 5, 29
    c = _modadd_circuit(n, p, gidney_add)
    # 3 + 4 = 7 < p  -> no reduction;  20 + 15 = 35 >= p -> reduction
    assert simulate(c, {"a": 3, "b": 4}).outputs["b"] == 7
    assert simulate(c, {"a": 20, "b": 15}).outputs["b"] == 35 % p == 6


def test_modadd_boundary_values():
    n, p = 6, 61
    c = _modadd_circuit(n, p, gidney_add)
    for av, bv in [(0, 0), (0, p - 1), (p - 1, 0), (p - 1, p - 1), (1, p - 1)]:
        r = simulate(c, {"a": av, "b": bv})
        assert r.outputs["b"] == (av + bv) % p, f"{av}+{bv} mod {p}"
        assert r.phase == 0


@pytest.mark.parametrize("pseudo_mersenne", [False, True], ids=["naive_p", "pm_decomp"])
@pytest.mark.parametrize("inplace_const", [False, True], ids=["v1_reg", "v2_inplace"])
@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
def test_modadd_25519_random(add_fn, inplace_const: bool, pseudo_mersenne: bool):
    """Full width, real prime."""
    c = _modadd_circuit(255, PRIME, add_fn, inplace_const, pseudo_mersenne)
    rng = random.Random(3)
    for _ in range(24):
        av, bv = rng.randrange(PRIME), rng.randrange(PRIME)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs["a"] == av
        assert r.outputs["b"] == (av + bv) % PRIME
        assert r.phase == 0


# ---------------------------------------------------------------------------
# v3: compare-then-subtract
# ---------------------------------------------------------------------------
def _modadd_v3_circuit(n: int, p: int, add_fn, inplace_const: bool = True) -> Circuit:
    c = Circuit()
    a = c.input("a", n)
    b = c.input("b", n)
    modadd_cmp(c, a, b, p, add_fn, inplace_const=inplace_const)
    return c


@pytest.mark.parametrize("inplace_const", [False, True], ids=["reg", "inplace"])
@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n,p", SMALL_PRIMES)
def test_modadd_v3_exhaustive(add_fn, n: int, p: int, inplace_const: bool):
    c = _modadd_v3_circuit(n, p, add_fn, inplace_const)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv})
            assert r.outputs["a"] == av, f"a changed (p={p}, {av},{bv})"
            assert r.outputs["b"] == (av + bv) % p, \
                f"wrong sum (p={p}, {av}+{bv} -> {r.outputs['b']})"
            assert r.phase == 0, f"dirty phase (p={p}, {av},{bv})"


@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
def test_modadd_v3_25519_random(add_fn):
    c = _modadd_v3_circuit(255, PRIME, add_fn)
    rng = random.Random(5)
    for _ in range(24):
        av, bv = rng.randrange(PRIME), rng.randrange(PRIME)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs["a"] == av
        assert r.outputs["b"] == (av + bv) % PRIME
        assert r.phase == 0


def test_modadd_v3_25519_edge_cases():
    c = _modadd_v3_circuit(255, PRIME, gidney_add)
    pm1 = PRIME - 1
    # includes the boundary values that straddle the reduction decision
    for av, bv in [(0, 0), (0, pm1), (pm1, 0), (pm1, pm1), (1, pm1),
                   (PRIME // 2, PRIME // 2), (PRIME // 2, PRIME // 2 + 1)]:
        r = simulate(c, {"a": av, "b": bv})
        assert r.outputs["b"] == (av + bv) % PRIME, f"{av}+{bv}"
        assert r.outputs["a"] == av
        assert r.phase == 0


def test_modadd_v3_agrees_with_v2_everywhere():
    """v3 must be observationally identical to the verified v2, not merely plausible."""
    for n, p in SMALL_PRIMES:
        v2 = _modadd_circuit(n, p, gidney_add)
        v3 = _modadd_v3_circuit(n, p, gidney_add)
        for av in range(p):
            for bv in range(p):
                assert (simulate(v2, {"a": av, "b": bv}).outputs
                        == simulate(v3, {"a": av, "b": bv}).outputs), \
                    f"v2/v3 disagree at p={p}, {av}+{bv}"


def test_modadd_v3_is_cheaper_than_v2():
    """The point of v3: fewer Toffoli and fewer Cliffords than v2."""
    v2 = count(_modadd_circuit(255, PRIME, gidney_add))
    v3 = count(_modadd_v3_circuit(255, PRIME, gidney_add))
    assert v3.toffoli < v2.toffoli
    assert v3.clifford < v2.clifford


def test_modadd_25519_edge_cases():
    c = _modadd_circuit(255, PRIME, gidney_add)
    pm1 = PRIME - 1
    for av, bv in [(0, 0), (pm1, pm1), (0, pm1), (pm1, 1), (PRIME // 2, PRIME // 2)]:
        r = simulate(c, {"a": av, "b": bv})
        assert r.outputs["b"] == (av + bv) % PRIME
        assert r.outputs["a"] == av
        assert r.phase == 0
