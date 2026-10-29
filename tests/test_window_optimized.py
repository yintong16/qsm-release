"""Small real-GCD optimized windows; full-width probes are explicitly slow.

The reused lazy arithmetic is approximate. Zero-measurement tests check values,
not coherent correctness. Only aligned arithmetic outcomes support comparison
of inherited phases; exact lookup phase proofs live in test_lookup_measured.py.
"""
import random
from dataclasses import FrozenInstanceError, replace
from inspect import unwrap

import pytest

from qsm.bench.registry import get_builder
from qsm.core.circuit import Circuit, DirtyAncillaError, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.inversion.config import InversionConfig
from qsm.params import FIELD, PRIME, U
from qsm.point import active as curve
from qsm.point.affine import add_const_affine_inplace_ref
from qsm.point.group import scalar_mul
from qsm.scalarmul import window, window_opt
from qsm.scalarmul.window import (
    add_window_affine_inplace_ref, make_window_table,
)
from qsm.scalarmul.window_opt import (
    OptimizedWindowConfig, add_window_affine_optimized_inplace,
    build_optimized_window_case,
)


def _full_config(p):
    n = p.bit_length()
    iters = 2 * n + 1
    return InversionConfig.tuned(
        n, p, iters=iters, widths=(n,) * iters, gcd_cmp_bits=None,
        replay_cmp_bits=None, replay_chunk=None, replay_chunk_cmp_bits=None,
        carry_window=None, double_window=None)


def _table(p=31, bits=2):
    a, b, base = (2, 3, (3, 6)) if p == 31 else (3, 4, (0, 2))
    return make_window_table(bits, base, p=p, a=a, b=b)


def _config(table, **kwargs):
    return OptimizedWindowConfig.tuned(
        table.p.bit_length(), table.p, table.bits,
        inversion=_full_config(table.p), **kwargs)


def _points(table):
    return [(x, y) for x in range(table.p) for y in range(table.p)
            if (y*y - x*x*x - table.a*x - table.b) % table.p == 0]


def _add(pt, qt, p, a):
    # Independent complete classical group law, not the circuit's lazy formulas.
    if pt is None:
        return qt
    if qt is None:
        return pt
    x, y = pt
    xx, yy = qt
    if x == xx and (y + yy) % p == 0:
        return None
    slope = (((3*x*x+a) * pow(2*y, -1, p)) if pt == qt else
             ((yy-y) * pow(xx-x, -1, p))) % p
    xr = (slope*slope-x-xx) % p
    return xr, (slope*(x-xr)-y) % p


def _zeros(circ):
    return [0] * sum(op.kind == "MEASURE" for op in circ.ops)


def _live_outputs(circ):
    named = {q for reg in circ.inputs.values() for q in reg}
    assert circ._live == len(named)
    assert circ._live_qids == named
    assert len(circ._free_qids) == len(set(circ._free_qids))
    assert count(circ).peak_qubits == circ.peak_qubits


def _snapshot(circ):
    return (list(circ.ops), circ._next_qid, circ._next_cbit, circ._live,
            circ._peak, set(circ._live_qids), list(circ._free_qids),
            dict(circ._cbit_const), list(circ._cond_stack),
            {name: tuple(reg) for name, reg in circ.inputs.items()})


def _build(table, config, *, legacy=False):
    circ = Circuit()
    j = circ.input("j", table.bits)
    x, y = circ.input("x", table.p.bit_length()), circ.input("y", table.p.bit_length())
    regions = []
    if legacy:
        x, y = window.add_window_affine_inplace(
            circ, x, y, j, table, config=config.inversion,
            const_kw={}, sqr_kw={}, regions=regions)
        circ.rebind("x", x)
        circ.rebind("y", y)
    else:
        x, y = add_window_affine_optimized_inplace(
            circ, x, y, j, table, config=config,
            const_kw={}, sqr_kw={}, regions=regions)
        # No caller-side rebind: this specifically exercises the new binder.
        assert circ.inputs["x"].ids == x.ids
        assert circ.inputs["y"].ids == y.ids
    _live_outputs(circ)
    return circ, regions


@pytest.mark.parametrize("strategy", ["paired", "fused"])
@pytest.mark.parametrize("chunk,block", [(None, 1), (1, 3), ("half", 7)])
def test_all_admissible_small_curve_inputs_digits_and_chunk_choices(strategy, chunk, block):
    table = _table()
    circ, _ = _build(table, _config(table, operand_strategy=strategy,
                                   adder_chunk=chunk, fused_block=block))
    measurements = _zeros(circ)
    tested = excluded = zero = 0
    for pt in _points(table):
        for digit in range(4):
            try:
                expected = add_window_affine_inplace_ref(*pt, digit, table)
            except AssertionError:
                excluded += 1
                continue
            assert expected == _add(pt, table.points[digit], table.p, table.a)
            raw = add_const_affine_inplace_ref(
                *pt, *table.operand(digit), int(digit != 0), table.p, 5)
            result = simulate(circ, {"j": digit, "x": pt[0], "y": pt[1]},
                              measurements=measurements)
            assert result.outputs == {"j": digit, "x": raw[0], "y": raw[1]}
            assert (result.outputs["x"] % 31, result.outputs["y"] % 31) == expected
            # No whole-circuit phase-clean claim: all-zero outcomes suppress
            # the inherited lazy arithmetic's measurement-conditioned phases.
            tested += 1
            zero += digit == 0
    assert tested > 20 and excluded > 0 and zero > 0


def _arithmetic_measurement_positions(circ, regions):
    indices = {i for label, start, stop in regions if label == "arithmetic"
               for i in range(start, stop)}
    return [m for m, i in enumerate(i for i, op in enumerate(circ.ops)
                                   if op.kind == "MEASURE") if i in indices]


def test_paired_inherits_reference_phase_with_aligned_arithmetic_measurements():
    table = _table()
    config = _config(table, adder_chunk=None)
    circ, regions = _build(table, config)
    reference, ref_regions = _build(table, config, legacy=True)
    positions = _arithmetic_measurement_positions(circ, regions)
    ref_positions = _arithmetic_measurement_positions(reference, ref_regions)
    assert len(positions) == len(ref_positions)
    assert len(ref_positions) == len(_zeros(reference))
    rng = random.Random(319)
    checked = phases = 0
    for pt in _points(table):
        for digit in range(4):
            try:
                add_window_affine_inplace_ref(*pt, digit, table)
            except AssertionError:
                continue
            shared = [rng.randrange(2) for _ in positions]
            measured = [rng.randrange(2) for _ in _zeros(circ)]
            old_measured = _zeros(reference)
            for i, bit in zip(positions, shared):
                measured[i] = bit
            for i, bit in zip(ref_positions, shared):
                old_measured[i] = bit
            inputs = {"j": digit, "x": pt[0], "y": pt[1]}
            got = simulate(circ, inputs, measurements=measured)
            old = simulate(reference, inputs, measurements=old_measured)
            assert got.outputs == old.outputs
            assert got.phase == old.phase
            phases += old.phase
            checked += 1
    assert checked > 20
    assert phases > 0  # Retain, rather than silently fix, the old phase contract.


@pytest.mark.parametrize("strategy", ["paired", "fused"])
@pytest.mark.parametrize("layout", ["named", "renamed", "combined"])
def test_output_binding_handles_default_names_other_names_and_sliced_input(strategy, layout):
    table = _table()
    circ = Circuit()
    j = circ.input("digit", 2)
    if layout == "combined":
        packed = circ.input("point", 11)
        x, y = packed[:5], packed[5:10]
    else:
        names = ("x", "y") if layout == "named" else ("left", "right")
        x, y = (circ.input(name, 5) for name in names)
    x, y = add_window_affine_optimized_inplace(
        circ, x, y, j, table, config=_config(table, operand_strategy=strategy),
        const_kw={}, sqr_kw={})
    if layout == "combined":
        assert circ.inputs["point"].ids == x.ids + y.ids + [packed[10]]
    else:
        assert circ.inputs[names[0]].ids == x.ids
        assert circ.inputs[names[1]].ids == y.ids
    _live_outputs(circ)
    measurements = _zeros(circ)
    for digit in (0, 1, 3):
        for pt in _points(table):
            try:
                expected = add_window_affine_inplace_ref(*pt, digit, table)
            except AssertionError:
                continue
            inputs = ({"point": pt[0] | (pt[1] << 5) | (1 << 10)} if layout == "combined"
                      else dict(zip(names, pt)))
            result = simulate(circ, dict(inputs, digit=digit), measurements=measurements)
            assert result.outputs["digit"] == digit
            if layout == "combined":
                word = result.outputs["point"]
                assert word >> 10 == 1  # Unrelated packed-input spectator survives.
                actual = (word & 31, (word >> 5) & 31)
            else:
                actual = tuple(result.outputs[name] for name in names)
            assert tuple(v % 31 for v in actual) == expected
            break


@pytest.mark.parametrize("strategy", ["paired", "fused"])
def test_three_real_windows_compose_with_group_reference(strategy):
    p, a, b, base = 127, 3, 4, (0, 2)
    tables = [make_window_table(2, scalar_mul(4**k, *base, p=p, a=a), p=p, a=a, b=b)
              for k in range(3)]
    circ = Circuit()
    digits = [circ.input(f"j{k}", 2) for k in range(3)]
    x, y = circ.input("x", 7), circ.input("y", 7)
    for address, table in zip(digits, tables):
        x, y = add_window_affine_optimized_inplace(
            circ, x, y, address, table,
            config=_config(table, operand_strategy=strategy), const_kw={}, sqr_kw={})
        assert circ.inputs["x"].ids == x.ids and circ.inputs["y"].ids == y.ids
    _live_outputs(circ)
    measurements = _zeros(circ)
    rng = random.Random(191)
    tested = excluded = 0
    while tested < 50 and tested + excluded < 150:
        js = [rng.randrange(4) for _ in tables]
        pt = scalar_mul(rng.randrange(1, 61), *base, p=p, a=a)
        state = group = pt
        try:
            for digit, table in zip(js, tables):
                state = add_window_affine_inplace_ref(*state, digit, table)
                group = _add(group, table.points[digit], p, a)
        except AssertionError:
            excluded += 1
            continue
        assert state == group
        inputs = {f"j{k}": digit for k, digit in enumerate(js)}
        result = simulate(circ, dict(inputs, x=pt[0], y=pt[1]), measurements=measurements)
        assert all(result.outputs[name] == value for name, value in inputs.items())
        assert (result.outputs["x"] % p, result.outputs["y"] % p) == group
        tested += 1
    assert tested == 50 and excluded > 0


def test_one_bit_paired_preserves_v1_gate_stream_and_resources():
    table = _table(bits=1)
    config = _config(table, adder_chunk=1)
    optimized, _ = _build(table, config)
    original, _ = _build(table, config, legacy=True)
    assert optimized.ops == original.ops
    assert count(optimized) == count(original)
    assert unwrap(get_builder("scalarmul.add_window_affine_w4_inplace.v1")) is window._build_window
    assert unwrap(get_builder("scalarmul.add_window_affine_w4_inplace.v2")) is window_opt._build_optimized


def test_configuration_is_frozen_field_resolved_and_serializable():
    inversion = _full_config(31)
    cfg = OptimizedWindowConfig.tuned(5, 31, 2, inversion=inversion)
    assert cfg.inversion is inversion
    assert (cfg.adder_chunk, cfg.fused_block, cfg.unlookup_low_bits) == (3, 3, 1)
    assert cfg.resolved(5, 31, 2) is cfg
    assert cfg.metadata() == {
        "n": 5, "p": 31, "window_bits": 2, "operand_strategy": "paired",
        "adder_chunk": 3, "fused_block": 3, "unlookup_low_bits": 1,
    }
    with pytest.raises(FrozenInstanceError):
        cfg.adder_chunk = 1
    for split in (0, 2):
        assert replace(cfg, unlookup_low_bits=split).unlookup_low_bits == split


@pytest.mark.parametrize("changes", [
    {"window_bits": 0}, {"window_bits": True}, {"operand_strategy": "other"},
    {"adder_chunk": 0}, {"adder_chunk": True}, {"adder_chunk": "other"},
    {"fused_block": 0}, {"fused_block": None}, {"fused_block": True},
    {"unlookup_low_bits": -1}, {"unlookup_low_bits": 3},
    {"unlookup_low_bits": True}, {"inversion": None},
    {"inversion": InversionConfig.tuned(7, 127)},
    {"inversion": InversionConfig.conservative(5, 31)},
])
def test_configuration_errors(changes):
    with pytest.raises(ValueError):
        replace(_config(_table()), **changes)


@pytest.mark.parametrize("case", [
    "alias", "short_y", "short_address", "narrow", "dead", "bad_id",
    "nonregister", "bad_table", "field_config", "address_config", "conditional",
])
def test_invalid_invocation_is_atomic(case):
    table = _table()
    circ = Circuit()
    j, x, y = circ.input("j", 2), circ.input("x", 5), circ.input("y", 5)
    config = _config(table)
    if case == "alias":
        y = x
    elif case == "short_y":
        y = y[:4]
    elif case == "short_address":
        j = j[:1]
    elif case == "narrow":
        x, y = x[:4], y[:4]
    elif case == "dead":
        dead = circ.alloc(1)
        circ.free(dead)
        x = Register([dead[0], *x.ids[1:]])
    elif case == "bad_id":
        x = Register([999, *x.ids[1:]])
    elif case == "nonregister":
        x = x.ids
    elif case == "bad_table":
        table = None
    elif case == "field_config":
        config = OptimizedWindowConfig.tuned(7, 127, 2)
    elif case == "address_config":
        config = replace(config, window_bits=3)
    if case == "conditional":
        cb = circ.classical_bit(1)
        with circ.condition(cb):
            before = _snapshot(circ)
            with pytest.raises(ValueError, match="condition"):
                add_window_affine_optimized_inplace(circ, x, y, j, table, config=config)
            assert _snapshot(circ) == before
    else:
        before = _snapshot(circ)
        with pytest.raises((TypeError, ValueError)):
            add_window_affine_optimized_inplace(circ, x, y, j, table, config=config)
        assert _snapshot(circ) == before


@pytest.mark.parametrize("bits,strategy", [(1, "paired"), (2, "paired"), (2, "fused")])
def test_builder_metadata_without_full_width_arithmetic(monkeypatch, bits, strategy):
    def stub(circ, x, y, address, table, *, regions, **kwargs):
        regions.append(("arithmetic", len(circ.ops), len(circ.ops)))
        return x, y

    monkeypatch.setattr(window_opt, "add_window_affine_optimized_inplace", stub)
    monkeypatch.setattr(window, "add_window_affine_inplace", stub)
    cfg = OptimizedWindowConfig.tuned(U, PRIME, bits, operand_strategy=strategy)
    new = build_optimized_window_case(bits, config=cfg)
    old = window.build_window_case(bits, config=cfg.inversion)
    for key in ("empirical_only", "fail_analytic", "input_distribution", "arithmetic_contract",
                "output_moduli", "window_bits", "table_fingerprint", "dummy_policy"):
        assert new.meta[key] == old.meta[key]
    assert new.meta["empirical_only"] is True and new.meta["fail_analytic"] is None
    assert new.meta["tuning"]["inversion"] == old.meta["tuning"]["inversion"]
    assert new.meta["tuning"]["point"] == old.meta["tuning"]["point"]
    assert new.meta["tuning"]["window"] == cfg.metadata()
    special = bits == 1 and strategy == "paired"
    assert new.meta["lookup_backend"] == ("constant-control.v1" if special else "unary-and.v1")
    assert new.meta["circuit_config"] == {
        "unlookup_backend": "constant-control.v1" if special else "measured-onehot.v1",
        "operand_strategy": strategy, "adder_chunk": cfg.adder_chunk,
        "fused_block": cfg.fused_block, "unlookup_low_bits": cfg.unlookup_low_bits,
        "one_bit_specialization": special,
    }
    assert all(op.kind == "ALLOC" for op in new.circuit.ops)
    sample = new.gen_input(random.Random(81))
    assert new.reference(sample) == old.reference(sample)


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["paired", "fused"])
@pytest.mark.parametrize("bits", [4, 16])
def test_full_width_selected_digits_zero_and_reported_peak(strategy, bits):
    case = build_optimized_window_case(bits, config=OptimizedWindowConfig.tuned(
        U, PRIME, bits, operand_strategy=strategy))
    resources = count(case.circuit)
    assert resources.peak_qubits == case.circuit.peak_qubits
    assert resources.toffoli > 0
    _live_outputs(case.circuit)
    assert case.meta["empirical_only"] and case.meta["fail_analytic"] is None
    assert case.meta["output_moduli"] == {"x": PRIME, "y": PRIME}
    measurements = _zeros(case.circuit)
    pt = curve.mul(37, curve.GX, curve.GY)
    for digit in (0, 1, 1 << (bits-1), (1 << bits)-1):
        inputs = {"j": digit, "x": pt[0], "y": pt[1]}
        expected = case.reference(inputs)
        for outcomes in (measurements, None):
            result = simulate(case.circuit, inputs, measurements=outcomes,
                              rng=random.Random(23 + digit))
            assert result.outputs["j"] == digit
            assert all(result.outputs[name] % PRIME == expected[name] for name in ("x", "y"))
            assert result.phase == 0


@pytest.mark.slow
@pytest.mark.skipif(FIELD != "25519", reason="fixed Wei25519 arithmetic counterexamples")
@pytest.mark.parametrize("strategy", ["paired", "fused"])
def test_full_width_known_arithmetic_faults_are_not_a_lookup_repair(strategy):
    case = build_optimized_window_case(2, config=OptimizedWindowConfig.tuned(
        U, PRIME, 2, operand_strategy=strategy))
    measurements = _zeros(case.circuit)
    points = [
        (54612104369102466295186403110110132856877110769130571327805934252953404584066,
         curve.GY + 20),
        (14442679257967034402166127258426498254760573270987925111286464008990023459040,
         curve.GY + 44),
    ]
    for x, y in points:
        inputs = {"j": 1, "x": x, "y": y}
        expected = curve.add(x, y, curve.GX, curve.GY)
        assert case.reference(inputs) == {"j": 1, "x": expected[0], "y": expected[1]}
        # These are inherited deterministic VALUE/dirty faults, not a claim
        # that a newly shifted random measurement stream has matching phases.
        with pytest.raises(DirtyAncillaError):
            simulate(case.circuit, inputs, measurements=measurements)
