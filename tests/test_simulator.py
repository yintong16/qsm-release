import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.simulator import simulate


def test_xor_exhaustive():
    c = Circuit()
    x = c.input("x", 4)
    y = c.input("y", 4)
    for i in range(4):
        c.cx(x[i], y[i])
    for xv in range(16):
        for yv in range(16):
            r = simulate(c, {"x": xv, "y": yv})
            assert r.outputs == {"x": xv, "y": yv ^ xv}
            assert r.phase == 0


def test_and_toffoli():
    c = Circuit()
    x = c.input("x", 1)
    y = c.input("y", 1)
    z = c.input("z", 1)
    c.ccx(x[0], y[0], z[0])
    for xv in (0, 1):
        for yv in (0, 1):
            r = simulate(c, {"x": xv, "y": yv, "z": 0})
            assert r.outputs["z"] == (xv & yv)


def test_dirty_ancilla_detected():
    c = Circuit()
    a = c.alloc(1)
    c.x(a[0])       # a = 1
    c.free(a)       # freed while |1> -> must raise during simulation
    with pytest.raises(DirtyAncillaError):
        simulate(c, {})


def _and_uncompute(correct: bool) -> Circuit:
    """AND into an ancilla, use it, then measurement-based uncompute (with/without the
    CZ phase correction)."""
    c = Circuit()
    x = c.input("x", 1)
    y = c.input("y", 1)
    z = c.input("z", 1)
    a = c.alloc(1)
    c.ccx(x[0], y[0], a[0])   # a = x & y
    c.cx(a[0], z[0])          # z ^= a
    m = c.measure(a[0])       # H-basis measurement, consumes a
    if correct:
        with c.condition(m, 1):
            c.cz(x[0], y[0])  # cancel the (-1)^((x&y)*m) kickback
    return c


def test_measurement_uncompute_clean():
    c = _and_uncompute(correct=True)
    for xv in (0, 1):
        for yv in (0, 1):
            for mv in (0, 1):
                r = simulate(c, {"x": xv, "y": yv, "z": 0}, measurements=[mv])
                assert r.outputs["z"] == (xv & yv)
                assert r.phase == 0


def test_measurement_uncompute_dirty_flagged():
    c = _and_uncompute(correct=False)
    # x=y=1, m=1 leaves a (-1)^1 phase kickback that nothing cancels.
    r = simulate(c, {"x": 1, "y": 1, "z": 0}, measurements=[1])
    assert r.phase == 1
