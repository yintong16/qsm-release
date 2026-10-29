import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.swap import cswap, cswap_bit


def _cswap_circuit(n: int) -> Circuit:
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    a = c.input("a", n)
    b = c.input("b", n)
    cswap(c, ctrl[0], a, b)
    return c


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_cswap_exhaustive(n: int):
    c = _cswap_circuit(n)
    for ctrl in (0, 1):
        for av in range(1 << n):
            for bv in range(1 << n):
                r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv})
                want = (bv, av) if ctrl else (av, bv)
                assert (r.outputs["a"], r.outputs["b"]) == want, (n, ctrl, av, bv)
                assert r.outputs["ctrl"] == ctrl, "ctrl not preserved"
                assert r.phase == 0


def test_cswap_random_255bit():
    n = 255
    c = _cswap_circuit(n)
    rng = random.Random(0)
    for _ in range(64):
        ctrl, av, bv = rng.getrandbits(1), rng.getrandbits(n), rng.getrandbits(n)
        r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv}, rng=rng)
        want = (bv, av) if ctrl else (av, bv)
        assert (r.outputs["a"], r.outputs["b"]) == want
        assert r.outputs["ctrl"] == ctrl
        assert r.phase == 0


def test_cswap_is_its_own_inverse():
    """Two cswaps on the same control are the identity, for either control value."""
    n = 6
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    a = c.input("a", n)
    b = c.input("b", n)
    cswap(c, ctrl[0], a, b)
    cswap(c, ctrl[0], a, b)
    for ctrlv in (0, 1):
        for av in range(1 << n):
            bv = (av * 37 + 11) & ((1 << n) - 1)
            r = simulate(c, {"ctrl": ctrlv, "a": av, "b": bv})
            assert (r.outputs["a"], r.outputs["b"]) == (av, bv)
            assert r.phase == 0


@pytest.mark.parametrize("n", [1, 8, 64, 255])
def test_cswap_cost_is_one_toffoli_per_bit(n: int):
    """The whole point of the module: n Toffoli, 2n CNOT, zero ancillas."""
    c = _cswap_circuit(n)
    res = count(c)
    assert res.toffoli == n
    assert res.peak_qubits == 2 * n + 1, "cswap must not allocate"


def test_cswap_bit_exhaustive():
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    x = c.input("x", 1)
    y = c.input("y", 1)
    cswap_bit(c, ctrl[0], x[0], y[0])
    for ctrlv in (0, 1):
        for xv in (0, 1):
            for yv in (0, 1):
                r = simulate(c, {"ctrl": ctrlv, "x": xv, "y": yv})
                want = (yv, xv) if ctrlv else (xv, yv)
                assert (r.outputs["x"], r.outputs["y"]) == want
                assert r.phase == 0
