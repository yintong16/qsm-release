"""Signed prototype: exact new primitives, honest inherited arithmetic hazards."""
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
from qsm.point.group import scalar_mul
from qsm.scalarmul import signed_window as sw
from qsm.scalarmul.lookup import lookup_xor
from qsm.scalarmul.window import make_window_table
from qsm.scalarmul.window_opt import OptimizedWindowConfig


def test_toy_sampler_scales_coordinate_proposals_with_field_size():
    p = 8191
    case = _case(bits=1, p=p)
    point = scalar_mul(7, 0, 2, p=p, a=3)
    expected = {"j": 0, "s": 0, "x": point[0], "y": point[1]}
    case.reference(expected)

    class DelayedPoint:
        coordinate_draws = 0

        def randrange(self, stop):
            if stop != p:
                return 0
            self.coordinate_draws += 1
            if self.coordinate_draws <= 20000:
                return 0
            return point[(self.coordinate_draws - 20001) % 2]

    rng = DelayedPoint()
    assert case.gen_input(rng) == expected
    assert rng.coordinate_draws == 20002


def _table(bits=2, p=31):
    a, b, base = (2, 3, (3, 6)) if p == 31 else (3, 4, (0, 2))
    return sw.make_signed_window_table(bits, base, p=p, a=a, b=b)


def _config(table, **kw):
    n, p = table.p.bit_length(), table.p
    iters = 2*n+1
    inv = InversionConfig.tuned(n, p, iters=iters, widths=(n,)*iters,
                               gcd_cmp_bits=None, replay_cmp_bits=None,
                               replay_chunk=None, replay_chunk_cmp_bits=None,
                               carry_window=None, double_window=None)
    return OptimizedWindowConfig.tuned(n, p, table.bits, inversion=inv,
                                       adder_chunk=None, **kw)


def _case(strategy="conjugate", bits=2, p=31):
    table = _table(bits, p)
    return sw.build_signed_window_case(bits, sign_strategy=strategy, table=table,
                                       config=_config(table), const_kw={}, sqr_kw={})


def _inputs(table):
    for x in range(table.p):
        for y in range(table.p):
            if (y*y-x**3-table.a*x-table.b) % table.p:
                continue
            for s, j in itertools.product(range(2), range(1 << table.bits)):
                yield dict(x=x, y=y, s=s, j=j)


def _admissible(case):
    rows, expected = [], []
    for v in _inputs(case.analysis_context["window_table"]):
        try:
            ref = case.reference(v)
        except AssertionError:
            continue
        rows.append(v)
        expected.append(ref)
    return rows, expected


def _assert_live(circ):
    named = {q for reg in circ.inputs.values() for q in reg}
    assert circ._live_qids == named
    assert circ._live == len(named)
    assert len(circ._free_qids) == len(set(circ._free_qids))
    assert count(circ).peak_qubits == circ.peak_qubits


def _zeros(circ):
    return [0]*sum(op.kind == "MEASURE" for op in circ.ops)


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_physical_magnitudes_endpoint_and_digit_bijection(bits):
    table = _table(bits, 127)
    size = 1 << bits
    assert table.points[0] == scalar_mul(size, 0, 2, p=127, a=3)
    assert table.points[0] is not None
    assert table.points[1] == (0, 2)
    for j in range(1, size):
        assert table.points[j] == scalar_mul(j, 0, 2, p=127, a=3)
    digits = {table.digit(j, s) for s in range(2) for j in range(size)}
    assert digits == set(range(-size, size))
    assert table.digit(0, 0) == 0 and table.digit(0, 1) == -size
    assert table.operand(0, 0) == table.points[0]
    assert table.operand(0, 1) == (table.points[0][0], -table.points[0][1] % 127)
    assert len(table.fingerprint) == 64
    assert table.fingerprint == _table(bits, 127).fingerprint


def test_table_validation_does_not_misuse_unsigned_zero_contract():
    table = _table()
    with pytest.raises(ValueError, match="finite"):
        replace(table, points=(None, *table.points[1:]))
    with pytest.raises(ValueError, match="T\\[0\\]"):
        replace(table, points=(table.points[1], *table.points[1:]))
    with pytest.raises(ValueError, match="canonical"):
        replace(table, points=((table.p, 0), *table.points[1:]))
    with pytest.raises(ValueError, match="on the declared"):
        replace(table, points=((1, 1), *table.points[1:]))
    with pytest.raises(ValueError):
        sw.make_signed_window_table(0)
    with pytest.raises(ValueError, match="explicit"):
        sw.make_signed_window_table(2, p=31)
    # Order-two base: the endpoint is infinity even though T[1] is finite.
    with pytest.raises(ValueError, match="remain finite"):
        sw.make_signed_window_table(1, (30, 0), p=31, a=2, b=3)
    for j, s in ((-1, 0), (4, 0), (0, 2), (True, 0)):
        with pytest.raises(ValueError):
            table.digit(j, s)


@pytest.mark.parametrize("n,p", [(3, 7), (4, 13), (5, 29)])
def test_sign_all_words_all_controls_all_measurements_and_involution(n, p):
    circ = Circuit()
    s, y = circ.input("s", 1), circ.input("y", n)
    sw.conditional_negate_nonzero_inplace(circ, y, s[0], p)
    inputs = [dict(s=sign, y=v) for sign in range(2) for v in range(1 << n)]
    expected = [dict(s=v["s"], y=(p-v["y"]) % (1 << n) if v["s"] else v["y"])
                for v in inputs]
    hazards = simulate_batch(compile_symbolic(circ), inputs, expected)
    assert hazards.bad_count == 0
    for v, ref in zip(inputs, expected):
        if 0 < v["y"] < p:
            assert ref["y"] == (-v["y"] % p if v["s"] else v["y"])
    # Enumerate every measurement result as a second, independent tiny check.
    for outcomes in itertools.product((0, 1), repeat=len(_zeros(circ))):
        for v, ref in zip(inputs, expected):
            got = simulate(circ, v, measurements=outcomes)
            assert got.outputs == ref and got.phase == 0
    sw.conditional_negate_nonzero_inplace(circ, y, s[0], p)
    assert simulate_batch(compile_symbolic(circ), inputs, inputs).bad_count == 0
    _assert_live(circ)
    if p < (1 << n)-1:
        # p+1 -> 2**n-1, NOT p-1. The lazy-interface limitation is intentional.
        result = simulate(circ, dict(s=1, y=p+1))
        assert result.outputs["y"] == p+1  # two exact bitspace involutions
        assert (p-(p+1)) % (1 << n) % p != -(p+1) % p


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_enable_zero_endpoint_and_clean_selection(bits):
    circ = Circuit()
    j, s, out = circ.input("j", bits), circ.input("s", 1), circ.input("out", 1)
    with sw._signed_enabled(circ, j, s[0]) as flag:
        circ.cx(flag, out[0])
    rows = [dict(j=j, s=s, out=z) for j, s, z in
            itertools.product(range(1 << bits), range(2), range(2))]
    expected = [dict(v, out=v["out"] ^ int(bool(v["j"] or v["s"]))) for v in rows]
    assert simulate_batch(compile_symbolic(circ), rows, expected).bad_count == 0
    _assert_live(circ)


@pytest.mark.parametrize("mode", ["endpoint", "selected", "selected_zero"])
def test_selected_words_and_endpoint_restored_for_every_measured_outcome(monkeypatch, mode):
    # Replace only the arithmetic action with an independent XOR observation.
    # The real lookup/sign/correction/restoration/measured-unlookup code executes.
    def observe(circ, operand, accumulator, p, **kw):
        for src, dst in zip(operand, accumulator):
            circ.cx(src, dst)

    monkeypatch.setattr(sw, "modsub_direct", observe)
    circ = Circuit()
    j, s = circ.input("j", 1), circ.input("s", 1)
    x = circ.input("x", 3)
    regions = []
    if mode == "endpoint":
        columns = ((0, 2),)
        sw._signed_packed_sub(circ, j, s[0], (x,), columns, 7, {}, 0, regions,
                              endpoint=3)
        rows = [dict(j=j, s=s, x=0) for j, s in itertools.product(range(2), repeat=2)]
        expected = [dict(v, x=3 if v["s"] and not v["j"] else columns[0][v["j"]])
                    for v in rows]
    else:
        y = circ.input("y", 3)
        columns = ((1, 4), (2, 5) if mode == "selected" else (0, 5))
        sw._signed_packed_sub(circ, j, s[0], (x, y), columns, 7, {}, 0, regions,
                              sign_y=True)
        rows = [dict(j=j, s=s, x=0, y=0) for j, s in itertools.product(range(2), repeat=2)]
        expected = [dict(v, x=columns[0][v["j"]],
                         y=(-columns[1][v["j"]] if v["s"] else columns[1][v["j"]]) % 7)
                    for v in rows]
    hazards = simulate_batch(compile_symbolic(circ), rows, expected)
    assert hazards.bad_count == 0 and hazards.summary_mismatch_count == 0
    if mode != "selected_zero":
        # Canonical zero-safe fallback is larger; symbolic evaluation above
        # still verifies ALL its outcomes, not a sample.
        for outcomes in itertools.product((0, 1), repeat=len(_zeros(circ))):
            for v, expected_row in zip(rows, expected):
                result = simulate(circ, v, measurements=outcomes)
                assert result.outputs == expected_row and result.phase == 0
    _assert_live(circ)


@pytest.mark.parametrize("strategy", ["conjugate", "selected"])
@pytest.mark.parametrize("bits", [1, 2])
def test_exhaustive_toy_values_endpoint_and_symbolic_cleanup_without_phase_claim(strategy, bits):
    case = _case(strategy, bits)
    rows, expected = _admissible(case)
    hazards = simulate_batch(compile_symbolic(case.circuit), rows, expected,
                             output_moduli=case.meta["output_moduli"])
    assert hazards.samples > 50
    assert hazards.value_count == hazards.dirty_count == hazards.summary_mismatch_count == 0
    # Full-width tuning is NOT an exact coherent point addition. Preserve and
    # expose the inherited phases instead of testing only favorable outcomes.
    assert hazards.phase_count > 0
    assert len(rows) < len(list(_inputs(case.analysis_context["window_table"])))
    _assert_live(case.circuit)
    measurements = _zeros(case.circuit)
    for s, j in itertools.product(range(2), (0, 1, (1 << bits)-1)):
        i = next(i for i, v in enumerate(rows) if (v["s"], v["j"]) == (s, j))
        for outcome in (measurements, None):
            got = simulate(case.circuit, rows[i], measurements=outcome, rng=random.Random(52))
            actual = dict(got.outputs, x=got.outputs["x"] % 31, y=got.outputs["y"] % 31)
            assert actual == expected[i]
    # The same address implements different group actions at s=0 and s=1.
    paired = next(v for v in rows if v["j"] == v["s"] == 0 and dict(v, s=1) in rows)
    assert case.reference(paired) != case.reference(dict(paired, s=1))


@pytest.mark.parametrize("strategy", ["conjugate", "selected"])
def test_actual_unlookup_words_checked_by_unitary_erasure(monkeypatch, strategy):
    calls = []

    def checked_unlookup(circ, address, target, values, **kw):
        calls.append(values)
        lookup_xor(circ, address, target, values)
        circ.free(target)  # A mismatched restored word now fails, not just phase.

    monkeypatch.setattr(sw, "unlookup_measure", checked_unlookup)
    case = _case(strategy)
    assert len(calls) == 3 and calls[1][0] == 0
    rows, expected = _admissible(case)
    result = simulate_batch(compile_symbolic(case.circuit), rows, expected,
                            output_moduli=case.meta["output_moduli"])
    assert result.dirty_count == result.value_count == 0
    _assert_live(case.circuit)


@pytest.mark.parametrize("strategy", ["conjugate", "selected"])
def test_packed_sign_address_and_coordinates_rebind_for_two_small_windows(strategy):
    table = _table()
    circ = Circuit()
    # Packed coordinate spectator and packed sign/address ensure all named views
    # survive the GCD's register consumption/recycling.
    point = circ.input("point", 11)
    digits = [circ.input(f"digit{k}", 3) for k in range(2)]
    x, y = point[:5], point[5:10]
    for digit in digits:
        x, y = sw.add_signed_window_inplace(circ, x, y, digit[:2], digit[2], table,
                                           sign_strategy=strategy, config=_config(table),
                                           const_kw={}, sqr_kw={})
        assert circ.inputs["point"].ids == x.ids+y.ids+[point[10]]
    _assert_live(circ)
    rows, expected = [], []
    for first, second in itertools.product(range(8), repeat=2):
        try:
            state = sw.signed_window_reference(5, 13, first % 4, first >> 2, table)
            state = sw.signed_window_reference(*state, second % 4, second >> 2, table)
        except AssertionError:
            continue
        v = dict(point=5+(13 << 5)+(1 << 10), digit0=first, digit1=second)
        rows.append(v)
        expected.append(dict(v, point=state[0]+(state[1] << 5)+(1 << 10)))
    assert len(rows) > 10
    result = simulate_batch(compile_symbolic(circ), rows, expected)
    assert result.dirty_count == result.summary_mismatch_count == 0
    # The packed word is not a field element: reduce its coordinate slices,
    # not the whole eleven-bit word. Inherited lazy zero can be represented by p.
    for got, ref in zip(result.outputs, expected):
        assert got["digit0"] == ref["digit0"] and got["digit1"] == ref["digit1"]
        assert got["point"] >> 10 == ref["point"] >> 10 == 1
        assert (got["point"] & 31) % 31 == ref["point"] & 31
        assert ((got["point"] >> 5) & 31) % 31 == (ref["point"] >> 5) & 31


@pytest.mark.parametrize("strategy", ["conjugate", "selected"])
def test_case_metadata_sampler_registry_and_exclusive_resource_regions(strategy):
    versions = all_versions()
    case = _case(strategy)
    assert all_versions() == versions
    assert not any("signed_window" in name for name in all_versions())
    meta = case.meta
    json.dumps(meta)
    assert meta["empirical_only"] is True and meta["fail_analytic"] is None
    assert meta["address_bits"] == 2 and meta["radix_bits"] == 3
    assert meta["table_entries"] == 4
    assert meta["sign_strategy"] == strategy
    assert meta["circuit_config"]["lookup_stages"] == 3
    assert "noncanonical" in meta["sign_domain"] if strategy == "conjugate" else "restored" in meta["sign_domain"]
    regions = meta["resource_regions"]
    assert set(label for label, _, _ in regions) == {"lookup", "sign", "selection", "arithmetic", "unlookup"}
    indices = [i for _, start, stop in regions for i in range(start, stop)]
    assert len(indices) == len(set(indices))
    assert set(indices) == set(range(4, len(case.circuit.ops)))
    assert sum(label == "lookup" for label, _, _ in regions) == 3
    assert sum(label == "unlookup" for label, _, _ in regions) == 3
    rng = random.Random(812)
    for _ in range(30):
        sample = case.gen_input(rng)
        assert case.reference(sample)["s"] == sample["s"]
        assert case.reference(sample)["j"] == sample["j"]
    assert case.gen_input(random.Random(2)) == case.gen_input(random.Random(2))


@pytest.mark.parametrize("invalid", ["strategy", "sign_width", "alias", "table", "config", "fused", "condition"])
def test_invalid_build_requests_rejected_before_emission(invalid):
    table = _table()
    circ = Circuit()
    j, s = circ.input("j", 2), circ.input("s", 1)
    x, y = circ.input("x", 5), circ.input("y", 5)
    kwargs = dict(config=_config(table), const_kw={}, sqr_kw={})
    if invalid == "strategy":
        kwargs["sign_strategy"] = "wrong"
    elif invalid == "sign_width":
        s = j
    elif invalid == "alias":
        s = x[:1]
    elif invalid == "table":
        table = make_window_table(2, (3, 6), p=31, a=2, b=3)
    elif invalid == "config":
        kwargs["config"] = replace(kwargs["config"], window_bits=3)
    elif invalid == "fused":
        kwargs["config"] = replace(kwargs["config"], operand_strategy="fused")
    if invalid == "condition":
        cb = circ.classical_bit(1)
        with circ.condition(cb):
            before = list(circ.ops)
            with pytest.raises(ValueError):
                sw.add_signed_window_inplace(circ, x, y, j, s, table, **kwargs)
            assert list(circ.ops) == before
    else:
        before = list(circ.ops)
        with pytest.raises(ValueError):
            sw.add_signed_window_inplace(circ, x, y, j, s, table, **kwargs)
        assert list(circ.ops) == before
