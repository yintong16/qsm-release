"""Canonical-zero point repair, not a claim of complete tuned arithmetic."""
import itertools
import random
from functools import partial

import pytest

from qsm.bench.registry import all_versions
from qsm.core import ops
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.field.add import gidney_add
from qsm.field.add_chunked import gidney_add_chunked
from qsm.field.modarith import modsub_direct
from qsm.inversion.config import InversionConfig
from qsm.point.affine import sub_const_mod
from qsm.scalarmul import zero_point as zp
from qsm.scalarmul.odd_window import (
    build_odd_window_case, make_odd_window_table, odd_window_reference,
)
from qsm.scalarmul.window_opt import OptimizedWindowConfig


def _table(p=31, w=3):
    if p == 13:
        return make_odd_window_table(w, (0, 2), p=p, a=3, b=4, order=7)
    return make_odd_window_table(w, (1, 2), p=p, a=0, b=3,
                                 order={7: 13, 31: 43}[p])


def _kwargs(table):
    n, p = table.p.bit_length(), table.p
    iters = 2*n+1
    inv = InversionConfig.tuned(
        n, p, iters=iters, widths=(n,)*iters, gcd_cmp_bits=None,
        replay_cmp_bits=None, replay_chunk=None, replay_chunk_cmp_bits=None,
        carry_window=None, double_window=None,
    )
    cfg = OptimizedWindowConfig.tuned(n, p, table.address_bits,
                                      inversion=inv, adder_chunk=None)
    return dict(table=table, config=cfg, const_kw={}, sqr_kw={})


def _rows(case):
    table = case.analysis_context["odd_window_table"]
    rows, expected = [], []
    for x, y, word in itertools.product(range(table.p), range(table.p),
                                      range(1 << table.word_bits)):
        if (y*y-x**3-table.a*x-table.b) % table.p:
            continue
        row = dict(word=word, x=x, y=y)
        try:
            result = case.reference(row)
        except AssertionError:
            continue
        rows.append(row)
        expected.append(result)
    return rows, expected


def _audit(case, rows, expected):
    return simulate_batch(compile_symbolic(case.circuit), rows, expected,
                          output_moduli=case.meta["output_moduli"])


def _live(circ):
    assert circ._live_qids == {q for reg in circ.inputs.values() for q in reg}
    assert count(circ).peak_qubits == circ.peak_qubits
    assert len(circ._free_qids) == len(set(circ._free_qids))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_equality_phase_all_bitstrings_and_all_measurement_outcomes(n):
    c = Circuit()
    a, b = c.input("a", n), c.input("b", n)
    # A real measurement symbol, with no incoming phase from this fresh zero.
    m = c.measure(c.alloc(1)[0])
    zp._equality_phase(c, a, b, m)
    c.exact_blocks.clear()  # No summary or phase model assumed for the gadget.
    resources = count(c)
    assert resources.toffoli == n-1
    assert resources.measurements == n
    _live(c)
    for aa, bb in itertools.product(range(1 << n), repeat=2):
        row = dict(a=aa, b=bb)
        for outcomes in itertools.product((0, 1), repeat=n):
            got = simulate(c, row, measurements=outcomes)
            assert got.outputs == row
            assert got.phase == (outcomes[0] & (aa == bb))


def test_equality_full_field_resources_no_persistent_flag():
    c = Circuit()
    a, b = c.input("a", 255), c.input("b", 255)
    m = c.classical_bit(1)
    zp._equality_phase(c, a, b, m)
    resources = count(c)
    assert resources.toffoli == resources.measurements == 254
    assert resources.peak_qubits == 764
    _live(c)
    for aa, bb in [(0, 0), (0, 1), ((1 << 255)-19, (1 << 255)-19),
                   ((1 << 255)-1, (1 << 255)-2)]:
        got = simulate(c, dict(a=aa, b=bb), rng=random.Random(7))
        assert got.outputs == dict(a=aa, b=bb) and got.phase == (aa == bb)


@pytest.mark.parametrize("p", [3, 7, 13, 31])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("chunk", [None, 2])
def test_borrow_helper_is_original_primitive_gate_for_gate(p, truncated, chunk):
    n = p.bit_length()
    kw = dict(add_fn=gidney_add if chunk is None else
              partial(gidney_add_chunked, chunk=chunk, cmp_bits=None),
              carry_window=min(2, n) if truncated else None,
              cmp_bits=1 if truncated else None)
    circuits = []
    for fn in (modsub_direct, zp._modsub_direct_borrow):
        c = Circuit()
        c.classical_bit(1)  # cbit IDs intentionally differ from measurement ordinals.
        c.measure(c.alloc(1)[0])
        a, b = c.input("a", n), c.input("b", n)
        start = len(c.ops)
        m = fn(c, a, b, p, **kw)
        circuits.append(c)
    assert circuits[0].ops == circuits[1].ops
    assert circuits[0]._cbit_const == circuits[1]._cbit_const
    first = c.ops[start]
    assert first.kind == ops.ALLOC and len(first.qubits) == 1
    borrow_q = first.qubits[0]
    terminal = next(op for op in c.ops[start+1:]
                    if op.kind in (ops.MEASURE, ops.FREE) and borrow_q in op.qubits)
    assert terminal.kind == ops.MEASURE and terminal.cbit == m
    _live(c)


@pytest.mark.parametrize("p", [3, 7, 13, 31])
def test_returned_borrow_phase_coefficient_matches_direct_predicate(p):
    n, mask = p.bit_length(), (1 << p.bit_length())-1
    c = Circuit()
    c.classical_bit(0)
    c.measure(c.alloc(1)[0])
    a, b = c.input("a", n), c.input("b", n)
    m = zp._modsub_direct_borrow(c, a, b, p)
    measurements = [op for op in c.ops if op.kind == ops.MEASURE]
    index = next(i for i, op in enumerate(measurements) if op.cbit == m)
    assert index != m  # It is a cbit identifier, NOT a measurement ordinal.
    outcomes = [0]*len(measurements)
    outcomes[index] = 1
    for aa, bb in itertools.product(range(p), repeat=2):
        result = (bb-aa) % p
        got = simulate(c, dict(a=aa, b=bb), measurements=outcomes)
        assert got.outputs == dict(a=aa, b=result)
        assert got.phase == ((bb < aa) ^ ((mask ^ aa) < result))


@pytest.mark.parametrize("p", [3, 7, 13, 31])
def test_targeted_restoration_correction_zero_and_nonzero_not_literal_p(p):
    # Isolate the final direct subtraction / arithmetic negate / equality patch,
    # with the actual measured borrow. Other measurements can be anything: their
    # own clean erasures must not change the targeted coefficient.
    n = p.bit_length()
    c = Circuit()
    a, y = c.input("a", n), c.input("y", n)
    m = zp._modsub_direct_borrow(c, a, y, p)
    for q in y:
        c.x(q)
    sub_const_mod(c, y, ((1 << n)-p-1) % p, p)
    zp._equality_phase(c, y, a, m)
    measurement_ops = [op for op in c.ops if op.kind == ops.MEASURE]
    index = next(i for i, op in enumerate(measurement_ops) if op.cbit == m)
    outcomes = [0]*len(measurement_ops)
    outcomes[index] = 1
    for aa in range(1, p):
        got = simulate(c, dict(a=aa, y=0), measurements=outcomes)
        assert got.outputs == dict(a=aa, y=aa) and got.phase == 0
        # For a canonical nonzero product equality is false, so the correction
        # leaves the original direct-subtract coefficient alone.
        for product in range(1, p):
            got = simulate(c, dict(a=aa, y=product), measurements=outcomes)
            result = (product-aa) % p
            assert got.phase == ((product < aa) ^ (((1 << n)-1-aa) < result))
        # An explicit excluded representation: p maps to the same output a but
        # has no incoming borrow. Applying the zero patch then INTRODUCES m.
        got = simulate(c, dict(a=aa, y=p), measurements=outcomes)
        assert got.outputs == dict(a=aa, y=aa) and got.phase == 1
    _live(c)


@pytest.mark.parametrize("p,w", [(7, 2), (7, 3), (31, 2), (31, 3)])
def test_full_width_f1_horizontal_both_signs_old_tags_and_phase_fixed(p, w):
    table = _table(p, w)
    kw = _kwargs(table)
    old = build_odd_window_case(w, **kw)
    tags = zp.build_zero_point_case(w, phase_correction=False, **kw)
    fixed = zp.build_zero_point_case(w, **kw)
    rows, expected = _rows(fixed)
    horizontal = sum(1 << i for i, row in enumerate(rows)
                     if row["y"] == table.point_for_word(row["word"])[1])
    assert horizontal.bit_count() > 0
    hrows = [row for i, row in enumerate(rows) if horizontal >> i & 1]
    assert {table.digit(row["word"]) < 0 for row in hrows} == {False, True}
    results = [_audit(case, rows, expected) for case in (old, tags, fixed)]
    original, tagonly, corrected = results
    assert original.dirty_mask == horizontal
    assert original.phase_mask & horizontal == horizontal
    assert tagonly.dirty_count == tagonly.value_count == 0
    assert tagonly.phase_mask & horizontal == horizontal
    assert corrected.bad_mask & horizontal == 0
    assert corrected.dirty_count == corrected.value_count == corrected.summary_mismatch_count == 0
    # Preserve every previously good nonzero input and leave unrelated faults
    # visible rather than claiming universal complete/phase-clean addition.
    assert corrected.phase_mask == tagonly.phase_mask & ~horizontal
    assert corrected.phase_count > 0
    assert corrected.bad_mask & ~original.bad_mask == 0
    assert original.outputs == tagonly.outputs == corrected.outputs == expected
    for case in (old, tags, fixed):
        _live(case.circuit)
    for i, row in enumerate(rows):
        if not horizontal >> i & 1:
            continue
        for seed in (0, 1, 2):
            got = simulate(fixed.circuit, row, rng=random.Random(seed))
            assert got.outputs == expected[i] and got.phase == 0
    r0, r1, r2 = [count(case.circuit) for case in (old, tags, fixed)]
    assert r0.peak_qubits == r1.peak_qubits == r2.peak_qubits
    # Actual tiny IR counts, not a projection of a full-width point cost.
    if w == 2:
        assert (r0.toffoli, r1.toffoli, r2.toffoli) == {
            7: (2624, 2630, 2632), 31: (5074, 5088, 5092),
        }[p]


def test_reference_relaxes_only_numerator_and_retains_ideal_domain():
    table = _table(31, 3)
    for x, y, word in itertools.product(range(31), range(31), range(8)):
        if (y*y-x**3-table.a*x-table.b) % 31:
            continue
        xp, yp = table.point_for_word(word)
        try:
            new = zp.zero_point_reference(x, y, word, table)
        except AssertionError:
            with pytest.raises(AssertionError):
                odd_window_reference(x, y, word, table)
            continue
        if y == yp:
            with pytest.raises(AssertionError, match="zero numerator"):
                odd_window_reference(x, y, word, table)
        else:
            assert new == odd_window_reference(x, y, word, table)
        assert x != xp and new[0] != xp
    for x, y in [(31, 2), (-1, 2), (1, 0)]:
        with pytest.raises(ValueError):
            zp.zero_point_reference(x, y, 0, table)


def test_inherited_non_f1_failures_remain_in_sampler_support():
    table = _table(13, 2)
    case = zp.build_zero_point_case(2, **_kwargs(table))
    rows, expected = _rows(case)
    result = _audit(case, rows, expected)
    assert result.value_count > 0 and result.dirty_count > 0 and result.phase_count > 0
    bad = next(row for i, row in enumerate(rows) if result.value_mask >> i & 1)

    class FixedInput:
        def __init__(self):
            self.values = iter((bad["x"], bad["y"], bad["word"]))

        def randrange(self, stop):
            value = next(self.values)
            assert 0 <= value < stop
            return value

    assert case.gen_input(FixedInput()) == bad
    assert case.reference(bad) in expected
    assert "literal p" in case.meta["zero_contract"]
    assert "noncanonical y>p" in case.meta["sign_domain"]
    assert "not guaranteed canonical or phase-clean" in case.meta["output_representation"]


def test_same_table_three_forward_lookups_regions_live_words_and_no_registry_change():
    versions = all_versions()
    table = _table(31, 3)
    for correction in (False, True):
        case = zp.build_zero_point_case(3, phase_correction=correction, **_kwargs(table))
        c = case.circuit
        blocks = c.exact_blocks
        assert sum(b.kind == "xor" for b in blocks) == 3
        assert sum(b.kind == "unlookup" for b in blocks) == (4 if correction else 3)
        assert case.analysis_context["odd_window_table"] is table
        assert case.meta["table_fingerprint"] == table.fingerprint
        assert case.meta["original_word_restored"]
        assert set(c.inputs) == {"word", "x", "y"}
        assert case.meta["circuit_config"]["lookup_stages"] == 3
        regions = case.meta["resource_regions"]
        assert regions[0][1] == 3  # The three named input ALLOCs are not regions.
        assert regions[-1][2] == len(c.ops)
        assert all(left[2] == right[1] for left, right in zip(regions, regions[1:]))
        assert {label for label, _, _ in regions} == {
            "recoding", "sign", "lookup", "arithmetic", "unlookup",
        }
        _live(c)
        for _ in range(5):
            row = case.gen_input(random.Random(_))
            case.reference(row)
    assert all_versions() == versions


def test_argument_validation_before_arithmetic():
    table = _table(31, 3)
    with pytest.raises(ValueError, match="match"):
        zp.build_zero_point_case(2, table=table)
    with pytest.raises(ValueError, match="bool"):
        zp.build_zero_point_case(3, table=table, phase_correction=1)
    with pytest.raises(ValueError, match="paired"):
        kw = _kwargs(table)
        kw["config"] = OptimizedWindowConfig.tuned(5, 31, 2, operand_strategy="fused")
        zp.build_zero_point_case(3, **kw)
