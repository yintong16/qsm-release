"""Window selection, residue semantics, cleanup and real composed-window circuits."""
import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.inversion.config import InversionConfig
from qsm.params import FIELD, PRIME, U
from qsm.point import active as curve
from qsm.point.affine import (
    add_const_affine_inplace, add_const_affine_inplace_ref, tuned_point_kwargs,
)
from qsm.point.conservative import add_const_affine_conservative_inplace
from qsm.point.group import scalar_mul
from qsm.scalarmul.window import (
    WindowTable, add_window_affine_inplace, add_window_affine_inplace_ref,
    build_window_case, make_window_table,
)


def _full_config(p):
    n = p.bit_length()
    iters = 2*n + 1
    return InversionConfig.tuned(
        n, p, iters=iters, widths=(n,)*iters, gcd_cmp_bits=None,
        replay_cmp_bits=None, replay_chunk=None, replay_chunk_cmp_bits=None,
        carry_window=None, double_window=None)


def _table(p=31, bits=2):
    a, b, base = (2, 3, (3, 6)) if p == 31 else (3, 4, (0, 2))
    return make_window_table(bits, base, p=p, a=a, b=b)


def _build(table):
    circ = Circuit()
    digit = circ.input("j", table.bits)
    x, y = circ.input("x", table.p.bit_length()), circ.input("y", table.p.bit_length())
    x, y = add_window_affine_inplace(circ, x, y, digit, table,
                                   config=_full_config(table.p), const_kw={}, sqr_kw={})
    circ.rebind("x", x)
    circ.rebind("y", y)
    return circ


def _fixed_chain(tables, digits):
    c = Circuit()
    controls = [c.input(f"c{k}", 1) for k in range(len(tables))]
    n = tables[0].p.bit_length()
    x, y = c.input("x", n), c.input("y", n)
    for ctrl, table, digit in zip(controls, tables, digits):
        x, y = add_const_affine_inplace(
            c, x, y, *table.operand(digit), ctrl=ctrl[0], p=table.p,
            const_kw={}, sqr_kw={}, inv_kw={"config": _full_config(table.p)})
    c.rebind("x", x)
    c.rebind("y", y)
    return c


def _points(table):
    p = table.p
    return [(x, y) for x in range(p) for y in range(p)
            if (y*y-x*x*x-table.a*x-table.b) % p == 0]


def _add(pt, qt, p, a):
    if pt is None:
        return qt
    if qt is None:
        return pt
    x, y = pt
    xx, yy = qt
    if x == xx and (y+yy) % p == 0:
        return None
    slope = (((3*x*x+a) * pow(2*y, p-2, p)) if pt == qt else
             ((yy-y)*pow(xx-x, p-2, p))) % p
    xr = (slope*slope-x-xx) % p
    return xr, (slope*(x-xr)-y) % p


def test_table_matches_independent_repeated_group_addition():
    table = _table(127, 3)
    acc = None
    for point in table.points:
        assert point == acc
        acc = _add(acc, table.points[1], table.p, table.a)
    same = WindowTable(table.p, table.a, table.b, list(table.points))
    assert same.fingerprint == table.fingerprint
    changed = WindowTable(table.p, table.a, table.b,
                          (None,) + tuple(reversed(table.points[1:])))
    assert changed.fingerprint != table.fingerprint
    assert isinstance(same.points, tuple)


@pytest.mark.parametrize("points", [(), (None,), ((3, 6), (3, 6)),
                                    (None, (3, 6), (3, 25)),
                                    (None, None), (None, (3, 7)), (None, (34, 6))])
def test_bad_tables_rejected(points):
    with pytest.raises(ValueError):
        WindowTable(31, 2, 3, points)


def test_table_rejects_infinity_from_small_order():
    with pytest.raises(ValueError, match="finite"):
        make_window_table(5, (3, 6), p=31, a=2, b=3)
    with pytest.raises(ValueError, match="explicit"):
        make_window_table(2, p=31)
    with pytest.raises(ValueError, match="positive"):
        make_window_table(0)


def test_alias_and_width_checks_precede_emission():
    table = _table()
    c = Circuit()
    j, x, y = c.input("j", 2), c.input("x", 5), c.input("y", 5)
    before = len(c.ops)
    for args in [(x, x, j), (x, y, j[:1]), (x[:4], y[:4], j)]:
        with pytest.raises(ValueError):
            add_window_affine_inplace(c, *args, table)
        assert len(c.ops) == before
    with pytest.raises(ValueError, match="match operand"):
        add_window_affine_inplace(c, x, y, j, table, config=InversionConfig.tuned(7, 127))


@pytest.mark.parametrize("bits", [1, 2])
def test_invalid_and_freed_wires_rejected_before_emission(bits):
    table = _table(bits=bits)
    c = Circuit()
    j, x, y = c.input("j", bits), c.input("x", 5), c.input("y", 5)
    freed = c.alloc(1)
    c.free(freed)
    before = len(c.ops)
    invalid = [
        (Register([999, *x.ids[1:]]), y, j),
        (x, Register([freed[0], *y.ids[1:]]), j),
        (x, y, Register([999, *j.ids[1:]])),
        (x, y, Register([True, *j.ids[1:]])),
    ]
    for xx, yy, address in invalid:
        with pytest.raises((TypeError, ValueError)):
            add_window_affine_inplace(c, xx, yy, address, table)
        assert len(c.ops) == before


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_all_small_curve_inputs_and_digits(bits):
    table = _table(31, bits)
    c = _build(table)
    tested = excluded = zero = phases = 0
    fixed = [_fixed_chain([table], [j]) for j in range(1 << bits)]
    for pt in _points(table):
        for j in range(1 << bits):
            try:
                expected = add_window_affine_inplace_ref(*pt, j, table)
            except AssertionError:
                excluded += 1
                continue
            assert expected == _add(pt, table.points[j], table.p, table.a)
            xp, yp = table.operand(j)
            raw = add_const_affine_inplace_ref(*pt, xp, yp, int(j != 0), table.p,
                                               table.p.bit_length())
            assert tuple(v % table.p for v in raw) == expected
            res = simulate(c, {"j": j, "x": pt[0], "y": pt[1]}, rng=random.Random(tested))
            assert res.outputs == {"j": j, "x": raw[0], "y": raw[1]}
            reference = simulate(fixed[j], {"c0": int(j != 0), "x": pt[0], "y": pt[1]},
                                 rng=random.Random(tested))
            assert res.measurements == reference.measurements
            assert res.phase == reference.phase
            phases += res.phase
            tested += 1
            zero += j == 0
    assert tested > 20 and excluded > 0 and zero > 0
    assert phases > 0  # Full comparisons do not remove lazy arithmetic's vented phase.


def test_one_bit_dispatch_preserves_gate_stream():
    table = _table(bits=1)
    c = _build(table)
    old = Circuit()
    j, x, y = old.input("j", 1), old.input("x", 5), old.input("y", 5)
    x, y = add_const_affine_inplace(old, x, y, *table.points[1], ctrl=j[0], p=table.p,
                                   const_kw={}, sqr_kw={}, inv_kw={"config": _full_config(table.p)})
    old.rebind("x", x)
    old.rebind("y", y)
    assert c.ops == old.ops
    assert count(c) == count(old)


def test_selected_small_inputs_against_canonical_circuit():
    table = _table()
    c = _build(table)
    checked = 0
    for j in range(4):
        oracle = Circuit()
        x, y = oracle.input("x", 5), oracle.input("y", 5)
        ctrl = oracle.input("c", 1)
        x, y = add_const_affine_conservative_inplace(oracle, x, y, *table.operand(j),
                                                    ctrl=ctrl[0], p=table.p)
        oracle.rebind("x", x)
        oracle.rebind("y", y)
        for pt in _points(table):
            try:
                add_window_affine_inplace_ref(*pt, j, table)
            except AssertionError:
                continue
            inp = {"x": pt[0], "y": pt[1]}
            got = simulate(c, dict(inp, j=j),
                           measurements=[0] * sum(op.kind == "MEASURE" for op in c.ops))
            ref = simulate(oracle, dict(inp, c=int(j != 0)), rng=random.Random(checked))
            assert tuple(got.outputs[k] % table.p for k in ("x", "y")) == tuple(
                ref.outputs[k] for k in ("x", "y"))
            assert got.phase == ref.phase == 0
            checked += 1
            if checked % 4 == 0:
                break
    assert checked == 16


def test_three_windows_compose_without_resetting_coordinates():
    p, a, b, base, bits = 127, 3, 4, (0, 2), 2
    tables = [make_window_table(bits, scalar_mul(4**k, *base, p=p, a=a), p=p, a=a, b=b)
              for k in range(3)]
    c = Circuit()
    js = [c.input(f"j{k}", bits) for k in range(3)]
    x, y = c.input("x", 7), c.input("y", 7)
    for address, table in zip(js, tables):
        x, y = add_window_affine_inplace(c, x, y, address, table,
                                       config=_full_config(p), const_kw={}, sqr_kw={})
    c.rebind("x", x)
    c.rebind("y", y)
    rng = random.Random(191)
    tested = excluded = 0
    for _ in range(100):
        digits = [rng.randrange(4) for _ in tables]
        pt = scalar_mul(rng.randrange(1, 61), *base, p=p, a=a)
        state = pt
        try:
            for digit, table in zip(digits, tables):
                state = add_window_affine_inplace_ref(*state, digit, table)
        except AssertionError:
            excluded += 1
            continue
        group = pt
        for digit, table in zip(digits, tables):
            group = _add(group, table.points[digit], p, a)
        assert state == group
        inp = {f"j{k}": digit for k, digit in enumerate(digits)}
        seed = rng.getrandbits(64)
        res = simulate(c, dict(inp, x=pt[0], y=pt[1]), rng=random.Random(seed))
        fixed = _fixed_chain(tables, digits)
        controls = {f"c{k}": int(digit != 0) for k, digit in enumerate(digits)}
        reference = simulate(fixed, dict(controls, x=pt[0], y=pt[1]), rng=random.Random(seed))
        assert (res.outputs["x"] % p, res.outputs["y"] % p) == group
        assert all(res.outputs[k] == v for k, v in inp.items())
        assert res.measurements == reference.measurements
        assert res.phase == reference.phase
        tested += 1
    assert tested >= 40 and excluded > 0


def test_tuning_helper_resolves_fields_without_changing_inversion_config():
    for n, p, expected in [(255, 2**255-19, 45), (256, 2**256-2**32-977, 73)]:
        cfg = InversionConfig.tuned(n, p)
        kw = tuned_point_kwargs(n, p, cfg)
        assert kw["inv_kw"]["config"] is cfg
        assert kw["const_kw"]["carry_window"] == expected
        assert kw["sqr_kw"]["window"] == expected
        # Narrow rows drop an integer carry, so their ripple does not depend on F.
        assert kw["sqr_kw"]["ripple"] == 40
        legacy = tuned_point_kwargs(n, p, cfg, legacy_square_ripple=True)
        assert legacy["sqr_kw"]["ripple"] == expected
        assert legacy["const_kw"] == kw["const_kw"]


@pytest.mark.slow
@pytest.mark.parametrize("legacy", [False, True])
def test_full_width_one_bit_cost_is_existing_controlled_cost(legacy):
    case = build_window_case(1, legacy_square_ripple=legacy)
    r = count(case.circuit)
    if legacy:
        expected = (1668563, 1207) if FIELD == "secp256k1" else (1582518, 1174)
    else:
        expected = (1658267, 1207) if FIELD == "secp256k1" else (1581033, 1174)
    assert (r.toffoli, r.peak_qubits) == expected


@pytest.mark.slow
def test_full_width_window_selected_digits_and_zero():
    case = build_window_case(4)
    assert case.meta["empirical_only"] and case.meta["fail_analytic"] is None
    assert case.meta["output_moduli"] == {"x": PRIME, "y": PRIME}
    point = curve.mul(37, curve.GX, curve.GY)
    for j in (0, 1, 7, 15):
        inp = {"j": j, "x": point[0], "y": point[1]}
        expected = case.reference(inp)
        res = simulate(case.circuit, inp, rng=random.Random(0))
        assert res.outputs["j"] == j
        assert all(res.outputs[k] % PRIME == expected[k] for k in ("x", "y"))
        assert res.phase == 0


@pytest.mark.slow
@pytest.mark.skipif(FIELD != "25519", reason="fixed Wei25519 counterexamples")
def test_window_does_not_hide_existing_structured_faults():
    case = build_window_case(2)
    points = [
        (54612104369102466295186403110110132856877110769130571327805934252953404584066,
         curve.GY + 20),
        (14442679257967034402166127258426498254760573270987925111286464008990023459040,
         curve.GY + 44),
    ]
    for x, y in points:
        inp = {"j": 1, "x": x, "y": y}
        assert case.reference(inp) == dict(zip(("j", "x", "y"), (1, *curve.add(x, y, curve.GX, curve.GY))))
        with pytest.raises(DirtyAncillaError):
            simulate(case.circuit, inp, rng=random.Random(0))


def test_registered_windows_keep_legacy_ripple_and_live_windows_use_new(monkeypatch):
    from qsm.params import CARRY_WINDOW, SQUARE_RIPPLE
    from qsm.scalarmul import window as w, window_opt as wo
    seen = []

    def capture(*args, sqr_kw, **kwargs):
        seen.append(sqr_kw["ripple"])
        raise RuntimeError("captured")

    monkeypatch.setattr(w, "add_window_affine_inplace", capture)
    monkeypatch.setattr(wo, "add_window_affine_optimized_inplace", capture)
    # Module builders directly: conftest memoizes the registry entries during test runs.
    for build, ripple in [(w._build_window, CARRY_WINDOW),
                          (wo._build_optimized, CARRY_WINDOW),
                          (lambda: w.build_window_case(4), SQUARE_RIPPLE),
                          (lambda: wo.build_optimized_window_case(4), SQUARE_RIPPLE)]:
        with pytest.raises(RuntimeError, match="captured"):
            build()
        assert seen[-1] == ripple
