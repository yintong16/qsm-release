import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add, gidney_add
from qsm.field.constadd import add_const, sub_const
from qsm.field.sub import sub

ADDERS = [cdkm_add, gidney_add]


def _sub_circuit(add_fn, n: int) -> Circuit:
    c = Circuit()
    a = c.input("a", n)
    b = c.input("b", n)
    sub(c, a, b, add_fn)
    return c


def _all_meas_vectors(c: Circuit):
    k = sum(1 for op in c.ops if op.kind == ops.MEASURE)
    if k > 8:
        return [None]
    return [[(m >> j) & 1 for j in range(k)] for m in range(1 << k)]


@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_sub_exhaustive(add_fn, n: int):
    c = _sub_circuit(add_fn, n)
    mask = (1 << n) - 1
    for mv in _all_meas_vectors(c):
        for av in range(1 << n):
            for bv in range(1 << n):
                r = simulate(c, {"a": av, "b": bv}, measurements=mv)
                assert r.outputs["a"] == av, f"a changed (n={n}, {av},{bv})"
                assert r.outputs["b"] == (bv - av) & mask, \
                    f"wrong difference (n={n}, {bv}-{av})"
                assert r.phase == 0, f"dirty phase (n={n}, {av},{bv}, meas={mv})"


@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
def test_sub_random_255bit(add_fn):
    n = 255
    c = _sub_circuit(add_fn, n)
    rng = random.Random(1)
    mask = (1 << n) - 1
    for _ in range(64):
        av, bv = rng.getrandbits(n), rng.getrandbits(n)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs["a"] == av
        assert r.outputs["b"] == (bv - av) & mask
        assert r.phase == 0


def test_sub_borrow_wraps_to_zero():
    """a - a == 0 and 0 - 1 == all-ones: the two borrow edge cases."""
    n = 8
    c = _sub_circuit(gidney_add, n)
    r = simulate(c, {"a": 200, "b": 200})
    assert r.outputs["b"] == 0
    r = simulate(c, {"a": 1, "b": 0})
    assert r.outputs["b"] == 0xFF


@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
def test_sub_costs_same_toffoli_as_its_adder(add_fn):
    """The complement identity must not add Toffolis -- only 2n X gates."""
    n = 64
    a_circ = Circuit()
    x = a_circ.input("x", n)
    y = a_circ.input("y", n)
    add_fn(a_circ, x, y)

    s_circ = _sub_circuit(add_fn, n)
    ra, rs = count(a_circ), count(s_circ)
    assert rs.toffoli == ra.toffoli
    assert rs.peak_qubits == ra.peak_qubits
    assert rs.clifford == ra.clifford + 2 * n


# ---------------------------------------------------------------------------
# Constant addition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_add_const_exhaustive(n: int):
    mask = (1 << n) - 1
    for cval in range(1 << n):
        c = Circuit()
        b = c.input("b", n)
        add_const(c, b, cval, gidney_add)
        for mv in _all_meas_vectors(c):
            for bv in range(1 << n):
                r = simulate(c, {"b": bv}, measurements=mv)
                assert r.outputs["b"] == (bv + cval) & mask, \
                    f"n={n}: {bv}+{cval}"
                assert r.phase == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_sub_const_exhaustive(n: int):
    mask = (1 << n) - 1
    for cval in range(1 << n):
        c = Circuit()
        b = c.input("b", n)
        sub_const(c, b, cval, gidney_add)
        for bv in range(1 << n):
            r = simulate(c, {"b": bv})
            assert r.outputs["b"] == (bv - cval) & mask, f"n={n}: {bv}-{cval}"
            assert r.phase == 0


def test_add_const_zero_is_free():
    c = Circuit()
    b = c.input("b", 16)
    add_const(c, b, 0, gidney_add)
    r = count(c)
    assert r.toffoli == 0 and r.n_gates == 0


def test_pseudo_mersenne_identity_255bit():
    """The claim this module exists for: on 256 bits, b - p == (b XOR 2^255) + 19."""
    n = 256
    p = (1 << 255) - 19
    c = Circuit()
    b = c.input("b", n)
    c.x(b[255])                 # b XOR 2^255  ==  b - 2^255  (mod 2^256)
    add_const(c, b, 19, gidney_add)

    rng = random.Random(2)
    for _ in range(32):
        bv = rng.getrandbits(n)
        r = simulate(c, {"b": bv}, rng=rng)
        assert r.outputs["b"] == (bv - p) % (1 << n), f"b={bv}"
        assert r.phase == 0
