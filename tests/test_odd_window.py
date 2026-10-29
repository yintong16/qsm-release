"""Isolated odd-window prototype: exact scaffolding, honest tuned hazards."""
import itertools
import json
import random
from dataclasses import replace

import pytest

from qsm.bench.registry import all_versions
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.inversion.config import InversionConfig
from qsm.params import PRIME
from qsm.point.group import scalar_mul
from qsm.scalarmul import odd_window as ow
from qsm.scalarmul import signed_window as sw
from qsm.scalarmul import window_opt as wo
from qsm.scalarmul.local_signed import half_shift_recode_inplace
from qsm.scalarmul.lookup import lookup_xor


def _table(w=3, p=31):
    if p == 13:
        return ow.make_odd_window_table(w, (0, 2), p=p, a=3, b=4, order=7)
    if p == 127:
        return ow.make_odd_window_table(w, (0, 2), p=p, a=3, b=4, order=61)
    return ow.make_odd_window_table(w, (1, 2), p=p, a=0, b=3,
                                    order={7: 13, 31: 43}[p])


def _config(table, **kw):
    n, p = table.p.bit_length(), table.p
    iters = 2*n+1
    inv = InversionConfig.tuned(n, p, iters=iters, widths=(n,)*iters,
                               gcd_cmp_bits=None, replay_cmp_bits=None,
                               replay_chunk=None, replay_chunk_cmp_bits=None,
                               carry_window=None, double_window=None)
    return wo.OptimizedWindowConfig.tuned(n, p, table.address_bits, inversion=inv,
                                         adder_chunk=None, **kw)


def _case(w=3, p=31):
    table = _table(w, p)
    return ow.build_odd_window_case(w, table=table, config=_config(table),
                                    const_kw={}, sqr_kw={})


def _admissible(case):
    table = case.analysis_context["odd_window_table"]
    rows, expected = [], []
    for x in range(table.p):
        for y in range(table.p):
            if (y*y-x**3-table.a*x-table.b) % table.p:
                continue
            for word in range(1 << table.word_bits):
                row = dict(x=x, y=y, word=word)
                try:
                    ref = case.reference(row)
                except AssertionError:
                    continue
                rows.append(row)
                expected.append(ref)
    return rows, expected


def _live(circ):
    named = {q for reg in circ.inputs.values() for q in reg}
    assert circ._live_qids == named
    assert circ._live == len(named)
    assert len(circ._free_qids) == len(set(circ._free_qids))
    assert count(circ).peak_qubits == circ.peak_qubits


@pytest.mark.parametrize("w", range(2, 9))
def test_exact_original_word_recoder_bijection_and_inverse(w):
    circ = Circuit()
    word = circ.input("word", w)
    half_shift_recode_inplace(circ, word)
    bound = 1 << (w-1)
    rows = [dict(word=u) for u in range(1 << w)]
    expected = []
    for u in range(1 << w):
        sign = 1-(u >> (w-1))
        j = (u % bound) ^ ((bound-1) if sign else 0)
        assert (-1 if sign else 1)*(2*j+1) == 2*u-(2**w-1)
        expected.append(dict(word=j+(sign << (w-1))))
    assert len({v["word"] for v in expected}) == 1 << w
    assert simulate_batch(compile_symbolic(circ), rows, expected).bad_count == 0
    resources = count(circ)
    assert resources.toffoli == resources.measurements == 0
    assert resources.peak_qubits == resources.clifford == w
    inverse = Circuit()
    inv_word = inverse.input("word", w)
    half_shift_recode_inplace(inverse, inv_word, inverse=True)
    assert simulate_batch(compile_symbolic(inverse), expected, rows).bad_count == 0
    half_shift_recode_inplace(circ, word, inverse=True)
    assert simulate_batch(compile_symbolic(circ), rows, rows).bad_count == 0
    _live(circ)


@pytest.mark.parametrize("w", [2, 3, 4, 5])
def test_table_is_nonzero_odd_multiples_not_odd_coordinates_or_table_halving(w):
    table = _table(w, 127)
    half = scalar_mul(pow(2, -1, table.order), *table.base, p=table.p, a=table.a)
    assert table.half_base == half
    assert scalar_mul(2, *half, p=table.p, a=table.a) == table.base
    assert table.entries == len(table.points) == 1 << (w-1)
    assert table.address_bits == table.bits == w-1
    assert any(not coordinate & 1 for point in table.points for coordinate in point)
    for j, point in enumerate(table.points):
        assert point == scalar_mul(2*j+1, *half, p=table.p, a=table.a)
        assert point is not None and point[1] != 0
    for word in range(1 << w):
        d = table.digit(word)
        assert d & 1 and d != 0
        sign = int(d < 0)
        j = (abs(d)-1)//2
        xp, yp = table.points[j]
        assert table.point_for_word(word) == (xp, -yp % table.p if sign else yp)
    signed = sw.make_signed_window_table(w-1, table.base, p=127, a=3, b=4)
    assert len(signed.points) == table.entries
    assert table.fingerprint == _table(w, 127).fingerprint
    assert len(table.fingerprint) == 64


def test_explicit_table_contract_rejects_bad_order_rows_and_widths():
    table = _table()
    for w in (True, 0, 1, 2.0):
        with pytest.raises(ValueError, match=">= 2"):
            ow.make_odd_window_table(w)
    with pytest.raises(ValueError, match="explicit"):
        ow.make_odd_window_table(3, p=31)
    for order in (2, 4, True, 1):
        with pytest.raises(ValueError, match="odd"):
            replace(table, order=order)
    with pytest.raises(ValueError, match="annihilate"):
        replace(table, order=45)
    with pytest.raises(ValueError, match="less than order"):
        replace(table, order=7)
    with pytest.raises(ValueError, match="finite"):
        replace(table, points=(None, *table.points[1:]))
    with pytest.raises(ValueError, match="canonical"):
        replace(table, points=((31, 1), *table.points[1:]))
    with pytest.raises(ValueError, match="on the declared"):
        replace(table, points=((1, 1), *table.points[1:]))
    with pytest.raises(ValueError, match="2j"):
        replace(table, points=(table.points[1], *table.points[1:]))
    with pytest.raises(ValueError, match="length"):
        replace(table, points=table.points[:-1])
    # Odd annihilator alone is insufficient: true order=13 introduces a zero row
    # at odd multiplier 13 even though declared annihilator=39 > max digit=15.
    with pytest.raises(ValueError, match="finite"):
        ow.make_odd_window_table(4, (1, 2), p=7, a=0, b=3, order=39)
    for word in (-1, 8, True, 1.0):
        with pytest.raises(ValueError, match="original word"):
            table.digit(word)
    with pytest.raises(ValueError, match="base disagrees"):
        ow.build_odd_window_case(3, table=table, base=table.points[1])
    with pytest.raises(ValueError, match="order disagrees"):
        ow.build_odd_window_case(3, table=table, order=45)


@pytest.mark.parametrize("w,p", [(2, 7), (3, 31)])
def test_new_recoding_sign_and_three_measured_lookups_phase_clean(monkeypatch, w, p):
    # Exercise the REAL odd-window scaffolding, replacing only inherited tuned
    # arithmetic by independent reversible observations/identities. All actual
    # QROM gates, p-y conjugations and measurements remain. Only the measured
    # unlookup summaries are retained (raw CPARITY is unsupported symbolically);
    # raw unlookup outcomes are exhausted independently stage-by-stage below.
    def observe(circ, operand, accumulator, p, **kw):
        for src, dst in zip(operand, accumulator):
            circ.cx(src, dst)

    monkeypatch.setattr(wo, "modsub_direct", observe)
    monkeypatch.setattr(ow, "moddiv_inplace", lambda circ, x, y, p, **kw: (x, y))
    monkeypatch.setattr(ow, "modmul_inplace", lambda circ, x, y, p, **kw: (x, y))
    monkeypatch.setattr(ow, "modsqrsub_inplace", lambda circ, y, x, p, **kw: x)
    monkeypatch.setattr(ow, "sub_const_mod", lambda *args, **kw: None)
    case = _case(w, p)
    table = case.analysis_context["odd_window_table"]
    n, mask = p.bit_length(), (1 << p.bit_length())-1
    rows, expected = [], []
    for word, y in itertools.product(range(1 << w), range(1 << n)):
        x = (3*y+word) & mask
        d = table.digit(word)
        xp, yp = table.points[(abs(d)-1)//2]
        k = ((1 << n)-p-1+3*xp) % p
        yy = (p-y) & mask if d < 0 else y
        yy = yy ^ yp ^ (-yp % p) ^ mask
        yy = (p-yy) & mask if d < 0 else yy
        rows.append(dict(word=word, x=x, y=y))
        expected.append(dict(word=word, x=x ^ xp ^ mask ^ k ^ (-xp % p), y=yy))
    case.circuit.exact_blocks[:] = [block for block in case.circuit.exact_blocks
                                    if block.kind == "unlookup"]
    result = simulate_batch(compile_symbolic(case.circuit), rows, expected)
    assert result.bad_count == result.summary_mismatch_count == 0
    assert count(case.circuit).measurements > 0
    # Literal simulation with both fixed/random outcomes independently checks
    # primitive execution; symbolic coefficients above cover every outcome.
    zeros = [0]*count(case.circuit).measurements
    for row, ref in zip(rows[:8], expected[:8]):
        for outcomes in (zeros, None):
            got = simulate(case.circuit, row, measurements=outcomes, rng=random.Random(42))
            assert got.outputs == ref and got.phase == 0
    _live(case.circuit)


def test_every_raw_measured_unlookup_outcome_for_each_positive_stage(monkeypatch):
    def observe(circ, operand, accumulator, p, **kw):
        for src, dst in zip(operand, accumulator):
            circ.cx(src, dst)
    monkeypatch.setattr(wo, "modsub_direct", observe)
    table = _table(2, 7)
    xs, ys = tuple(pt[0] for pt in table.points), tuple(pt[1] for pt in table.points)
    for columns in ((xs, ys), (tuple(3*x % 7 for x in xs),),
                    (tuple(-x % 7 for x in xs), tuple(-y % 7 for y in ys))):
        circ = Circuit()
        word = circ.input("word", 2)
        accumulators = tuple(circ.input(f"v{i}", 3) for i in range(len(columns)))
        half_shift_recode_inplace(circ, word)
        wo._packed_sub(circ, word[:-1], accumulators, columns, 7, {}, 0, [])
        half_shift_recode_inplace(circ, word, inverse=True)
        circ.exact_blocks.clear()  # Literal simulator executes CPARITY directly.
        measurements = count(circ).measurements
        assert measurements <= 6
        for u in range(4):
            j = (abs(table.digit(u))-1)//2
            row = dict(word=u, **{f"v{i}": 0 for i in range(len(columns))})
            ref = dict(word=u, **{f"v{i}": col[j] for i, col in enumerate(columns)})
            for outcomes in itertools.product((0, 1), repeat=measurements):
                got = simulate(circ, row, measurements=outcomes)
                assert got.outputs == ref and got.phase == 0
        _live(circ)


@pytest.mark.parametrize("w,p", [(2, 7), (3, 7), (2, 31), (3, 31)])
def test_actual_small_point_outputs_word_restoration_and_inherited_phases(w, p):
    case = _case(w, p)
    rows, expected = _admissible(case)
    assert len(rows) > 20
    # Admissibility is ONLY the ideal group's algebraic domain, never a successful
    # simulated trajectory. Full-width tuned phase hazards remain visible.
    result = simulate_batch(compile_symbolic(case.circuit), rows, expected,
                            output_moduli=case.meta["output_moduli"])
    assert result.value_count == result.dirty_count == result.summary_mismatch_count == 0
    assert result.phase_count > 0
    assert {row["word"] for row in rows} == set(range(1 << w))
    assert all(got["word"] == row["word"] for got, row in zip(result.outputs, rows))
    zeros = [0]*count(case.circuit).measurements
    for word in range(1 << w):
        i = next(i for i, row in enumerate(rows) if row["word"] == word)
        got = simulate(case.circuit, rows[i], measurements=zeros)
        assert dict(got.outputs, x=got.outputs["x"] % p, y=got.outputs["y"] % p) == expected[i]
    _live(case.circuit)


def test_noncanonical_interface_and_tuned_hazards_not_filtered():
    case = _case(2, 13)
    rows, expected = _admissible(case)
    result = simulate_batch(compile_symbolic(case.circuit), rows, expected,
                            output_moduli=case.meta["output_moduli"])
    assert result.bad_count > 0
    assert "noncanonical y>p" in case.meta["sign_domain"]
    assert "not guaranteed canonical or phase-clean" in case.meta["output_representation"]
    # A bad accepted input remains in the sampler's support. Neither reference
    # nor generation ever consults simulate; feed an admissible hazardous row.
    bad = next(row for row, ref, got in zip(rows, expected, result.outputs)
               if any(got[k] % 13 != ref[k] for k in ("x", "y")))
    class FixedInput:
        def __init__(self):
            self.values = iter((bad["x"], bad["y"], bad["word"]))
        def randrange(self, stop):
            value = next(self.values)
            assert 0 <= value < stop
            return value
    assert case.gen_input(FixedInput()) == bad
    assert case.reference(bad) in expected
    # Explicit exact bitspace counterexample, not a false field-negation claim.
    circ = Circuit()
    s, y = circ.input("s", 1), circ.input("y", 4)
    sw.conditional_negate_nonzero_inplace(circ, y, s[0], 13)
    ref = dict(s=1, y=15)  # 13-14 modulo 16, not -14 modulo 13 (=12)
    assert simulate_batch(compile_symbolic(circ), [dict(s=1, y=14)], [ref]).bad_count == 0
    assert ref["y"] % 13 != -14 % 13


def test_no_zero_enable_or_endpoint_and_all_folded_k_rows(monkeypatch):
    calls, controls, negations = [], [], []
    packed, square, negate = ow._packed_sub, ow.modsqrsub_inplace, ow.sub_const_mod
    def packed_spy(circ, address, accumulators, columns, *args):
        calls.append(columns)
        return packed(circ, address, accumulators, columns, *args)
    def square_spy(*args, **kw):
        controls.append(kw["ctrl"])
        return square(*args, **kw)
    def negate_spy(*args, **kw):
        negations.append(kw["ctrl"])
        return negate(*args, **kw)
    def forbidden(*args, **kw):
        raise AssertionError("odd windows must have no zero/endpoint selection machinery")
    monkeypatch.setattr(ow, "_packed_sub", packed_spy)
    monkeypatch.setattr(ow, "modsqrsub_inplace", square_spy)
    monkeypatch.setattr(ow, "sub_const_mod", negate_spy)
    monkeypatch.setattr(sw, "_signed_enabled", forbidden)
    monkeypatch.setattr(sw, "_endpoint_xor", forbidden)
    case = _case()
    table = case.analysis_context["odd_window_table"]
    xs, ys = tuple(p[0] for p in table.points), tuple(p[1] for p in table.points)
    assert calls == [(xs, ys), (tuple(3*x % 31 for x in xs),),
                     (tuple(-x % 31 for x in xs), tuple(-y % 31 for y in ys))]
    assert calls[1][0][0] != 0  # Row zero is NOT a disabled K sentinel.
    assert controls == negations == [None]
    assert "selection" not in {label for label, _, _ in case.meta["resource_regions"]}


def test_actual_loaded_words_restored_before_unlookup(monkeypatch):
    calls = []
    def checked(circ, address, target, values, **kw):
        calls.append(values)
        lookup_xor(circ, address, target, values)
        circ.free(target)  # Would diagnose a wrong restored operand as dirty.
    monkeypatch.setattr(wo, "unlookup_measure", checked)
    case = _case()
    rows, expected = _admissible(case)
    result = simulate_batch(compile_symbolic(case.circuit), rows, expected,
                            output_moduli=case.meta["output_moduli"])
    assert len(calls) == 3 and calls[1][0] != 0
    assert result.dirty_count == result.value_count == result.summary_mismatch_count == 0
    _live(case.circuit)


def test_unregistered_metadata_sampler_and_disjoint_regions():
    versions = all_versions()
    case = _case()
    assert all_versions() == versions
    assert not any("odd_window" in version for version in versions)
    meta = case.meta
    json.dumps(meta)
    assert meta["word_bits"] == meta["radix_bits"] == 3
    assert meta["address_bits"] == 2 and meta["table_entries"] == 4
    assert meta["empirical_only"] is True and meta["fail_analytic"] is None
    assert meta["circuit_config"]["lookup_stages"] == 3
    assert meta["dummy_policy"] == "none" and meta["quantum_halving"] is False
    assert "no further table halving" in meta["comparison_baseline"]
    regions = meta["resource_regions"]
    assert {label for label, _, _ in regions} == {"recoding", "sign", "lookup", "arithmetic", "unlookup"}
    indices = [i for _, start, stop in regions for i in range(start, stop)]
    assert len(indices) == len(set(indices))
    assert indices == list(range(3, len(case.circuit.ops)))
    assert sum(label == "recoding" for label, _, _ in regions) == 2
    assert sum(label == "sign" for label, _, _ in regions) == 2
    assert sum(label == "lookup" for label, _, _ in regions) == 3
    assert sum(label == "unlookup" for label, _, _ in regions) == 3
    rng = random.Random(921)
    for _ in range(30):
        row = case.gen_input(rng)
        assert case.reference(row)["word"] == row["word"]
    assert case.gen_input(random.Random(7)) == case.gen_input(random.Random(7))


def test_core_rebinds_packed_coordinate_views_and_restores_word():
    table = _table(2)
    circ = Circuit()
    point, word = circ.input("point", 11), circ.input("word", 2)
    x, y = ow.add_odd_window_inplace(circ, point[:5], point[5:10], word, table,
                                    config=_config(table), const_kw={}, sqr_kw={})
    assert circ.inputs["point"].ids == x.ids+y.ids+[point[10]]
    case = _case(2)
    rows, refs = _admissible(case)
    packed_rows = [dict(word=r["word"], point=r["x"]+(r["y"] << 5)+(1 << 10)) for r in rows]
    expected = [dict(word=r["word"], point=r["x"]+(r["y"] << 5)+(1 << 10)) for r in refs]
    result = simulate_batch(compile_symbolic(circ), packed_rows, expected)
    assert result.dirty_count == result.summary_mismatch_count == 0
    for got, ref in zip(result.outputs, expected):
        assert got["word"] == ref["word"]
        assert got["point"] >> 10 == ref["point"] >> 10 == 1
        assert (got["point"] & 31) % 31 == ref["point"] & 31
        assert ((got["point"] >> 5) & 31) % 31 == (ref["point"] >> 5) & 31
    _live(circ)


@pytest.mark.parametrize("invalid", ["table", "word_width", "alias", "fused", "config", "condition"])
def test_invalid_core_requests_rejected_before_emission(invalid):
    table = _table()
    circ = Circuit()
    word, x, y = circ.input("word", 3), circ.input("x", 5), circ.input("y", 5)
    config = _config(table)
    if invalid == "table":
        table = sw.make_signed_window_table(2, (1, 2), p=31, a=0, b=3)
    elif invalid == "word_width":
        word = word[:2]
    elif invalid == "alias":
        word = x[:3]
    elif invalid == "fused":
        config = replace(config, operand_strategy="fused")
    elif invalid == "config":
        config = replace(config, window_bits=3)
    def check():
        before = list(circ.ops)
        with pytest.raises(ValueError):
            ow.add_odd_window_inplace(circ, x, y, word, table, config=config,
                                      const_kw={}, sqr_kw={})
        assert list(circ.ops) == before
    if invalid == "condition":
        with circ.condition(circ.classical_bit(1)):
            check()
    else:
        check()


def test_n255_radix16_recoding_and_sign_resources_without_large_tables():
    p = (1 << 255)-19
    circ = Circuit()
    word, y = circ.input("word", 16), circ.input("y", 255)
    half_shift_recode_inplace(circ, word)
    sw.conditional_negate_nonzero_inplace(circ, y, word[-1], p)
    sw.conditional_negate_nonzero_inplace(circ, y, word[-1], p)
    half_shift_recode_inplace(circ, word, inverse=True)
    rows = [dict(word=u, y=v) for u in (0, 32767, 32768, 65535)
            for v in (0, 1, 2, p-1, p, p+1, (1 << 255)-1)]
    assert simulate_batch(compile_symbolic(circ), rows, rows).bad_count == 0
    resources = count(circ)
    assert resources.toffoli > 0 and resources.measurements > 0
    assert resources.peak_qubits < 16+2*255
    _live(circ)
