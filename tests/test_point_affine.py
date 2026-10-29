"""The affine point addition: the group law, and the claim that the control is free.

Three things are being checked, and only the first is routine.

1.  The schedule computes the group law. Checked exhaustively at p = 31 and p = 127 --
    the f = 1 widths where the lazy domain closes, for the reason
    `test_inversion_inplace` gives -- by sweeping every point of a small curve against a
    fixed classical addend, and comparing against `curve.add` rather than against the
    reference. The reference replays the schedule, so a reference-only test would prove
    the schedule equals itself.

2.  `ctrl = 0` is the IDENTITY, not merely cheap. This is what makes the control nearly
    free: steps 3 and 6 are uncontrolled because `moddiv_inplace` and `modmul_inplace` are
    each other's inverse, so with nothing in between to disturb `u`, the multiply undoes
    the division and returns `v`. If that ever stops holding, the whole schedule has to be
    rebuilt around a controlled division, so it is pinned as an exact equality on every
    input rather than sampled.

3.  The control costs U Toffoli and nothing else. Six constant subtracts absorb their
    control into the CNOT layer that loads the scratch register -- no controlled adder
    anywhere -- so the only Toffoli the control buys is `modsqrsub_inplace`'s
    `t = ctrl AND x`. That is an exact arithmetic claim about two 255-bit circuits and it
    is asserted as an exact difference, not a bound.

The exceptional cases (`x = x_P`, `x_3 = x_P`) are checked as a partition, the way
`test_inversion_inplace` checks the zero operand: they raise, and nothing else does.
"""

from __future__ import annotations

import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.inversion import schedule
from qsm.params import FIELD, PRIME, U
from qsm.point.active import CURVE as curve
from qsm.point.affine import (
    add_const_affine_inplace,
    add_const_affine_inplace_ref,
    sub_const_mod,
)

#: f = 2^n - p = 1, so the lazy domain closes and the composite is exact on its whole
#: domain. Same choice and same reason as `test_inversion_inplace.CLOSED`; at f > 1 the
#: replay is already inexact at these widths and the test would be measuring the width.
CLOSED = [31, 127]

#: (p, a, b) -- small curves with enough points to sweep. `b` is chosen so the curve is
#: not singular and has a point of order > 2.
SMALL = {31: (2, 3), 127: (3, 4)}


def _points(p: int) -> list[tuple[int, int]]:
    a, b = SMALL[p]
    return [(x, y) for x in range(p) for y in range(p)
            if (y * y - (x * x * x + a * x + b)) % p == 0]


def _circuit(p: int, xp: int, yp: int, ctrl: bool) -> Circuit:
    n = p.bit_length()
    it = schedule.iterations_j2(n)
    inv_kw = dict(iters=it, widths=schedule.widths_j2(n, it))
    c = Circuit()
    cb = c.input("c", 1) if ctrl else None
    x, y = c.input("x", n), c.input("y", n)
    xo, yo = add_const_affine_inplace(
        c, x, y, xp, yp, None if cb is None else cb[0], p, inv_kw=inv_kw)
    c.rebind("x", xo)
    c.rebind("y", yo)
    return c


def _res(name: str):
    from qsm.bench import registry

    return count(registry.get_builder(name)().circuit)


# --- the curve constants ---------------------------------------------------
@pytest.mark.skipif(FIELD != "25519", reason="Wei25519-specific published constants")
def test_wei25519_constants_match_the_published_ones():
    """Derived here, checked against draft-ietf-lwig-curve-representations.

    The draft publishes the ODD representative of the base point's y; we derive the even
    one. They are negatives of each other and generate the same group, so the assertion is
    on the pair and not on our choice.
    """
    draft_a = int("19298681539552699237261830834781317975"
                  "544997444273427339909597334573241639236")
    draft_b = int("55751746669818908907645289078257140818"
                  "241103727901012315294400837956729358436")
    draft_gx = int("19298681539552699237261830834781317975"
                   "544997444273427339909597334652188435546")
    draft_gy = int("14781619447589544791020593568409986887"
                   "264606134616475288964881837755586237401")
    assert (curve.A, curve.B, curve.GX) == (draft_a, draft_b, draft_gx)
    assert curve.GY in (draft_gy, PRIME - draft_gy)
    assert curve.is_on_curve(curve.GX, curve.GY)


@pytest.mark.skipif(FIELD != "25519", reason="Wei25519-specific published constants")
def test_the_base_point_is_the_montgomery_one_shifted():
    """`x = u + A/3` is the whole birational map on x, and u = 9 is Curve25519's base."""
    assert curve.GX == (9 + curve.SHIFT) % PRIME
    assert curve.SHIFT == (curve.MONT_A * pow(3, -1, PRIME)) % PRIME


def test_the_base_point_has_the_stated_order():
    """Evaluate the published prime-subgroup order without reducing the scalar first."""
    assert curve.mul(curve.ORDER, curve.GX, curve.GY) is None
    assert curve.mul(curve.ORDER - 1, curve.GX, curve.GY) == curve.neg(curve.GX, curve.GY)
    assert curve.mul(curve.ORDER + 1, curve.GX, curve.GY) == (curve.GX, curve.GY)


def test_an_incorrect_claimed_order_does_not_kill_the_base_point(monkeypatch):
    monkeypatch.setattr(curve, "ORDER", curve.ORDER + 1)
    assert curve.mul(curve.ORDER, curve.GX, curve.GY) == (curve.GX, curve.GY)


@pytest.mark.skipif(FIELD != "25519", reason="Wei25519 has rational two-torsion; secp256k1 does not")
@pytest.mark.parametrize("k", [0, 1, 2, 3, -1, -2, -3, curve.ORDER, -curve.ORDER])
def test_two_torsion_is_not_reduced_modulo_the_odd_subgroup_order(k):
    torsion = (curve.SHIFT, 0)
    assert curve.is_on_curve(*torsion)
    assert curve.ORDER & 1
    assert curve.mul(k, *torsion) == (torsion if k & 1 else None)


# --- the group law ---------------------------------------------------------
@pytest.mark.parametrize("p", CLOSED)
def test_the_schedule_computes_the_group_law(p: int):
    """Every point of a small curve, plus a fixed classical addend, against `curve.add`.

    Compared against the GROUP LAW and not against the reference: the reference replays the
    schedule, so checking the circuit against it proves only that the schedule is what we
    wrote down. This proves it is addition.

    Compared MODULO p, because the output is lazy by contract -- `y_3 = 0` comes back as
    `p`, which is `modneg`'s documented `[1, p]` boundary reached through step 9's
    complement. Which representative the schedule lands on is pinned exactly against the
    reference in `test_the_circuit_matches_its_own_reference`; that is the right place for
    it, and putting it here too would only hide the group law behind it.
    """
    pts = _points(p)
    assert len(pts) > 4, f"p={p}: {len(pts)} points is not enough to sweep"
    xp, yp = pts[0]
    c = _circuit(p, xp, yp, ctrl=False)
    checked = lazy = 0
    for (xv, yv) in pts:
        if xv == xp or yv == yp:
            continue                     # A = +-P, or the horizontal chord: out of domain
        want = curve.add(xv, yv, xp, yp, p)
        if want[0] == xp:
            continue                                   # x_3 = x_P: a zero into the multiply
        r = simulate(c, {"x": xv, "y": yv}, rng=random.Random(xv * 31 + yv))
        got = (r.outputs["x"] % p, r.outputs["y"] % p)
        assert got == want, f"p={p} A=({xv},{yv})"
        assert curve.is_on_curve(*want, p, *SMALL[p])
        lazy += got != (r.outputs["x"], r.outputs["y"])
        checked += 1
    assert checked > 3, f"p={p}: only {checked} additions were in domain"
    # Lazy outputs are the two-torsion images and the top-F landings; at f = 1 and these
    # widths there are a handful. Pinned from above so "lazy" cannot quietly become "most".
    assert lazy <= checked // 4, f"p={p}: {lazy} of {checked} outputs were lazy"


@pytest.mark.parametrize("p", CLOSED)
def test_control_one_computes_the_group_law_and_control_zero_is_the_identity(p: int):
    """The gated step, both branches, on every point of the curve.

    `ctrl = 0` has to be the EXACT identity -- same value, no residue shuffling -- because
    that is what lets steps 3 and 6 stay uncontrolled. It holds because the multiply is the
    division's inverse and nothing in between touches `u` when the control is off.
    """
    pts = _points(p)
    xp, yp = pts[0]
    c = _circuit(p, xp, yp, ctrl=True)
    for (xv, yv) in pts:
        if xv == xp or yv == yp:
            continue
        want = curve.add(xv, yv, xp, yp, p)
        if want[0] == xp:
            continue
        rng = random.Random(xv * 17 + yv)
        on = simulate(c, {"c": 1, "x": xv, "y": yv}, rng=rng)
        assert (on.outputs["x"] % p, on.outputs["y"] % p) == want, \
            f"p={p} ctrl=1 ({xv},{yv})"
        off = simulate(c, {"c": 0, "x": xv, "y": yv}, rng=rng)
        assert (off.outputs["x"], off.outputs["y"]) == (xv, yv), \
            f"p={p} ctrl=0 ({xv},{yv}) moved to {off.outputs['x'], off.outputs['y']}"
        assert off.outputs["c"] == 0 and on.outputs["c"] == 1, "the control was disturbed"


@pytest.mark.parametrize("p", CLOSED)
def test_the_circuit_matches_its_own_reference(p: int):
    """The reference replays the schedule, so it has to agree on non-curve inputs too.

    The circuit is a rational map and never reads the curve equation. Sweeping the whole
    domain rather than the curve is what exercises `modsqrsub_inplace`'s lazy chain on
    accumulators a curve point would never produce.
    """
    n = p.bit_length()
    xp, yp = _points(p)[0]
    for ctrl in (0, 1):
        c = _circuit(p, xp, yp, ctrl=True)
        for xv in range(p):
            for yv in range(p):
                if xv == xp:
                    continue
                try:
                    want = add_const_affine_inplace_ref(xv, yv, xp, yp, ctrl, p, n)
                except AssertionError:
                    continue                           # x_3 = x_P, out of domain
                r = simulate(c, {"c": ctrl, "x": xv, "y": yv},
                             rng=random.Random(xv * 7 + yv + ctrl))
                assert (r.outputs["x"], r.outputs["y"]) == want, \
                    f"p={p} ctrl={ctrl} ({xv},{yv})"


def test_the_reference_is_the_group_law_at_full_width():
    """At 255 bits the lazy chain never fires, so the schedule IS addition on real points.

    This is the bridge the small widths cannot build: they close the lazy domain by having
    f = 1, and the real width closes it by having f/2^U = 2^-250.8. Both give exact
    addition, for opposite reasons, and only this one is the circuit we ship.
    """
    rng = random.Random(20260914)
    for _ in range(25):
        k = rng.randrange(2, curve.ORDER)
        pt = curve.mul(k, curve.GX, curve.GY)
        assert pt is not None
        want = curve.add(pt[0], pt[1], curve.GX, curve.GY)
        got = add_const_affine_inplace_ref(pt[0], pt[1], curve.GX, curve.GY,
                                           1, PRIME, U, ripple=None)
        assert got == want, f"k={k}"
        assert curve.is_on_curve(*got)
        # and the control off leaves the point exactly where it was
        assert add_const_affine_inplace_ref(pt[0], pt[1], curve.GX, curve.GY,
                                            0, PRIME, U, ripple=None) == pt


def test_the_lambda_destroy_identity_holds_on_the_real_curve():
    """`lambda * u_3 = -(y_3 + y_P)` -- why the step costs two expensive ops and not three.

    The in-place multiply at step 6 is not a convenience. It is the ONLY thing that empties
    `lambda`'s register, and it empties it by turning it into a y-coordinate. The identity
    is the line through Q and P being the line through R = Q+P and -P.
    """
    rng = random.Random(11)
    for _ in range(50):
        pt = curve.mul(rng.randrange(2, curve.ORDER), curve.GX, curve.GY)
        x3, y3 = curve.add(pt[0], pt[1], curve.GX, curve.GY)
        lam = ((pt[1] - curve.GY) * pow(pt[0] - curve.GX, -1, PRIME)) % PRIME
        u3 = (x3 - curve.GX) % PRIME
        assert (lam * u3) % PRIME == (-(y3 + curve.GY)) % PRIME


# --- the exceptional cases -------------------------------------------------
@pytest.mark.parametrize("p", CLOSED)
def test_the_exceptional_cases_raise_and_they_are_exactly_the_exceptional_ones(p: int):
    """A partition, not a bound -- a circuit that raised on everything would pass a bound.

    Both preconditions land on a zero operand into an `inv` composite, and that composite
    raises rather than returning garbage for the reason its own docstring gives: the
    emptied register lands on the wrong constant and the clearing X layer leaves a live
    qubit. So the addition law's exceptional cases surface as `DirtyAncillaError`, which is
    the behaviour Layer 4 has to schedule around.
    """
    n = p.bit_length()
    xp, yp = _points(p)[0]
    c = _circuit(p, xp, yp, ctrl=False)
    raised, expected = set(), set()
    for xv in range(p):
        for yv in range(p):
            try:
                add_const_affine_inplace_ref(xv, yv, xp, yp, 1, p, n)
            except AssertionError:
                expected.add((xv, yv))
            try:
                simulate(c, {"x": xv, "y": yv}, rng=random.Random(xv * 3 + yv))
            except DirtyAncillaError:
                raised.add((xv, yv))
    assert raised == expected, (
        f"p={p}: raised-but-fine {sorted(raised - expected)[:5]}, "
        f"fine-but-should-raise {sorted(expected - raised)[:5]}")
    # All three preconditions have to be represented, or the partition is vacuous for
    # whichever one the sweep missed. The third -- the horizontal chord -- is the one the
    # group law does not consider exceptional at all, so it is the easiest to forget.
    assert any(xv == xp for xv, _ in expected), "x = x_P was not swept"
    assert any(yv == yp and xv != xp for xv, yv in expected), "y = y_P was not swept"


# --- the cost claims -------------------------------------------------------
def test_a_controlled_constant_subtract_costs_what_an_uncontrolled_one_costs():
    """The load layer carries the control, so the Toffoli count cannot move.

    This is what removes the need for a controlled modular adder in Layer 3. Asserted at
    full width because that is where it is used, and on Toffoli because the extra CNOTs are
    real but free.
    """
    def toffoli(ctrl: bool) -> int:
        c = Circuit()
        cb = c.input("c", 1) if ctrl else None
        b = c.input("b", U)
        sub_const_mod(c, b, curve.GX, PRIME, None if cb is None else cb[0])
        return count(c).toffoli

    assert toffoli(True) == toffoli(False)


def test_a_zero_constant_subtract_is_the_exact_identity():
    """`ctrl = 0` loads 0, and `b - 0` must not shuffle the representative or kick phase.

    If `modsub_direct` canonicalised here instead of being the identity, `ctrl = 0` would
    move a lazy accumulator and the uncontrolled steps 3 and 6 would no longer cancel.
    """
    p = 31
    n = p.bit_length()
    c = Circuit()
    cb = c.input("c", 1)
    b = c.input("b", n)
    sub_const_mod(c, b, 7, p, cb[0])
    for v in range(1 << n):
        r = simulate(c, {"c": 0, "b": v}, rng=random.Random(v))
        assert r.outputs["b"] == v, f"b={v} moved to {r.outputs['b']}"


@pytest.mark.slow
def test_the_control_buys_exactly_the_squares_operand_gate():
    """The headline: gating the whole point addition costs U Toffoli.

    Steps 3 and 6 are uncontrolled by construction, the six constant subtracts take their
    control for free, and the two complement layers are CNOTs. What is left is
    `modsqrsub_inplace`'s `t = ctrl AND x`, which is U Toffoli by its own docstring. An
    exact difference, because anything else means the schedule grew a controlled adder
    somewhere it should not have.
    """
    plain = _res("point.add_const_affine_inplace.v1")
    gated = _res("point.add_const_affine_inplace_ctrl.v1")
    assert gated.toffoli - plain.toffoli == U, (
        f"control cost {gated.toffoli - plain.toffoli} Toffoli, expected {U}")


@pytest.mark.slow
def test_the_composite_is_the_sum_of_its_regions():
    """Gates add, so the composite has to equal the parts -- and the peak does not.

    Doc 10 §2 is the record of that distinction being got wrong. The regions here are two
    `inv` composites, one `modsqrsub_inplace` and six `sub_const_mod`s; if this ever
    disagrees, a region has been mis-attributed and the peak below is measuring something
    other than what it claims.
    """
    from qsm.mul.accumulate import modsqrsub_inplace
    from qsm.params import CARRY_WINDOW, CMP_BITS
    from qsm.point.affine import _CONST_KW, _SQR_KW

    plain = _res("point.add_const_affine_inplace.v1")
    div = _res("inv.moddiv_inplace.v1").toffoli
    mul = _res("inv.modmul_inplace.v1").toffoli

    c = Circuit()
    x, acc = c.input("x", U), c.input("acc", U)
    c.rebind("acc", modsqrsub_inplace(c, x, acc, PRIME, **_SQR_KW))
    sqr = count(c).toffoli

    c = Circuit()
    b = c.input("b", U)
    sub_const_mod(c, b, curve.GX, PRIME, None, **_CONST_KW)
    sub = count(c).toffoli

    assert plain.toffoli == div + mul + sqr + 6 * sub, (
        f"composite {plain.toffoli} != {div} + {mul} + {sqr} + 6*{sub} "
        f"= {div + mul + sqr + 6 * sub}")
