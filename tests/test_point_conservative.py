"""Canonical point arithmetic on nonexceptional inputs, including zero slope."""

import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.simulator import simulate
from qsm.params import PRIME, U
from qsm.point import active, curve
from qsm.point.conservative import (
    add_const_affine_conservative_inplace,
    add_const_affine_conservative_inplace_ref,
    square_sub_conservative_inplace,
)


def _build(p, xp, yp, controlled=True):
    c = Circuit()
    ctrl = c.input("c", 1) if controlled else None
    x, y = c.input("x", p.bit_length()), c.input("y", p.bit_length())
    x, y = add_const_affine_conservative_inplace(
        c, x, y, xp, yp, None if ctrl is None else ctrl[0], p)
    c.rebind("x", x)
    c.rebind("y", y)
    return c


@pytest.mark.parametrize("p", [13, 29, 31])
def test_square_subtract_all_canonical_inputs(p):
    c = Circuit()
    ctrl = c.input("c", 1)
    x, acc = c.input("x", p.bit_length()), c.input("acc", p.bit_length())
    acc = square_sub_conservative_inplace(c, x, acc, p, ctrl[0])
    c.rebind("acc", acc)
    for bit in (0, 1):
        for xv in range(p):
            for av in range(p):
                got = simulate(c, {"c": bit, "x": xv, "acc": av},
                               rng=random.Random(xv * p + av))
                assert got.outputs == {"c": bit, "x": xv, "acc": (av - bit*xv*xv) % p}
                assert got.phase == 0


@pytest.mark.parametrize("p", [13, 29])
def test_point_rational_map_all_nonexceptional_inputs(p):
    xp, yp = 2, 3
    c = _build(p, xp, yp)
    for bit in (0, 1):
        for x in range(p):
            for y in range(p):
                try:
                    expected = add_const_affine_conservative_inplace_ref(x, y, xp, yp, bit, p)
                except AssertionError:
                    continue
                got = simulate(c, {"c": bit, "x": x, "y": y},
                               rng=random.Random(bit + x * p + y))
                assert got.outputs == {"c": bit, "x": expected[0], "y": expected[1]}
                assert got.phase == 0


@pytest.mark.parametrize("p,a,b", [(13, 2, 3), (29, 2, 3), (31, 2, 3)])
def test_group_law_horizontal_chords_and_canonical_output(p, a, b):
    points = [(x, y) for x in range(p) for y in range(p)
              if curve.is_on_curve(x, y, p, a, b)]
    xp, yp = {13: (3, 6), 29: (5, 14), 31: (5, 13)}[p]
    assert (xp, yp) in points
    c = _build(p, xp, yp)
    tested = horizontal = 0
    for x, y in points:
        if x == xp:
            continue
        expected = curve.add(x, y, xp, yp, p)
        if expected is None or expected[0] == xp:
            continue
        for seed in (0, 1):
            for bit in (0, 1):
                result = simulate(c, {"c": bit, "x": x, "y": y}, rng=random.Random(seed))
                want = expected if bit else (x, y)
                assert result.outputs == {"c": bit, "x": want[0], "y": want[1]}
                assert result.phase == 0
        horizontal += y == yp
        tested += 1
    assert tested
    if p == 29:
        assert horizontal


def test_reference_retains_explicit_exceptional_domain():
    with pytest.raises(AssertionError, match="x == x_P"):
        add_const_affine_conservative_inplace_ref(2, 4, 2, 3, p=29)
    # lambda = 3, so lambda² - 5 - 2 = 2 = x_P.
    with pytest.raises(AssertionError, match="x_3 == x_P"):
        add_const_affine_conservative_inplace_ref(5, 12, 2, 3, p=29)
    assert add_const_affine_conservative_inplace_ref(5, 12, 2, 3, ctrl=0, p=29) == (5, 12)


@pytest.mark.slow
def test_full_width_point_and_control_phase_clean():
    c = _build(PRIME, active.GX, active.GY)
    points = [active.mul(k, active.GX, active.GY) for k in (2, 3, 19)]
    if U == 255:
        points.extend([
            (54612104369102466295186403110110132856877110769130571327805934252953404584066,
             active.GY + 20),
            (14442679257967034402166127258426498254760573270987925111286464008990023459040,
             active.GY + 44),
        ])
    for pt in points:
        assert pt is not None and active.is_on_curve(*pt)
        assert active.mul(active.ORDER, *pt) is None
        want = active.add(*pt, active.GX, active.GY)
        assert want is not None and pt[0] != active.GX and want[0] != active.GX
        for ctrl in (0, 1):
            got = simulate(c, {"c": ctrl, "x": pt[0], "y": pt[1]}, rng=random.Random(0))
            expected = want if ctrl else pt
            assert got.outputs == {"c": ctrl, "x": expected[0], "y": expected[1]}
            assert got.phase == 0


def test_consecutive_additions_stay_canonical():
    p, a = 13, 2
    xp, yp = 0, 4  # y² = x³ + 2x + 3 over F13.
    c = Circuit()
    x, y = c.input("x", 4), c.input("y", 4)
    for _ in range(2):
        x, y = add_const_affine_conservative_inplace(c, x, y, xp, yp, p=p)
    c.rebind("x", x)
    c.rebind("y", y)
    checked = 0
    for xv in range(p):
        for yv in range(p):
            if not curve.is_on_curve(xv, yv, p, a, 3):
                continue
            state = (xv, yv)
            for _ in range(2):
                if state is None or state[0] == xp:
                    break
                state = curve.add(*state, xp, yp, p)
                if state is None or state[0] == xp:
                    break
            else:
                got = simulate(c, {"x": xv, "y": yv}, rng=random.Random(0))
                assert got.outputs == {"x": state[0], "y": state[1]}
                assert got.phase == 0
                checked += 1
    assert checked
