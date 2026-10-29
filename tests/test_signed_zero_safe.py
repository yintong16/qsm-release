"""Signed canonical-zero repair; inherited lazy-arithmetic hazards stay visible."""
import itertools
import random
from dataclasses import replace

import pytest

from qsm.audit import full_scalar as fs, odd_window as raw_audit
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.scalarmul import signed_window as sw, zero_point as zp
from qsm.scalarmul.lookup import lookup_xor
from test_full_scalar import (
    SEED, _controller_only_outputs, _dense_double_inverse_qft, _group,
    _max_complex_error, _mul, _retained_circuit, _shift_scalar,
)
from test_signed_window import _admissible, _assert_live, _config, _inputs, _zeros
from test_zero_point import _audit, _table as _odd_table


def _table(p=31, bits=2):
    fixture = _odd_table(p, bits+1)
    return sw.make_signed_window_table(bits, fixture.base, p=p, a=fixture.a, b=fixture.b)


def _case(p=31, bits=2, *, zero_safe=True, chunked=False):
    table = _table(p, bits)
    config = _config(table)
    if chunked:
        config = replace(config, adder_chunk=2)
    return sw.build_signed_window_case(bits, table=table, config=config,
                                       const_kw={}, sqr_kw={}, zero_safe=zero_safe)


@pytest.mark.parametrize("p,bits", [(7, 1), (7, 2), (31, 1), (31, 2)])
@pytest.mark.parametrize("chunked", [False, True])
def test_all_f1_inputs_and_measurement_phases_include_dummy_and_endpoint(p, bits, chunked):
    case = _case(p, bits, chunked=chunked)
    old = _case(p, bits, zero_safe=False, chunked=chunked)
    table = case.analysis_context["window_table"]
    rows, expected = _admissible(case)
    horizontal = sum(1 << i for i, v in enumerate(rows)
                     if v["y"] == table.operand(v["j"], v["s"])[1])
    assert horizontal
    horizontal_rows = [v for i, v in enumerate(rows) if horizontal >> i & 1]
    assert {(v["s"], v["j"]) for v in horizontal_rows} == set(
        itertools.product(range(2), range(1 << bits)))
    before, after = [_audit(c, rows, expected) for c in (old, case)]
    assert after.outputs == expected == before.outputs
    assert after.value_count == after.dirty_count == after.noncanonical_count == 0
    assert after.summary_mismatch_count == 0
    assert not after.bad_mask & horizontal
    assert before.dirty_mask == horizontal
    assert after.phase_mask == before.phase_mask & ~horizontal
    # Pin inherited faults rather than claiming globally coherent tuned addition.
    assert (after.samples, after.phase_count) == {
        (7, 1): (37, 9), (7, 2): (73, 22),
        (31, 1): (157, 4), (31, 2): (313, 18),
    }[p, bits]
    for i, v in enumerate(rows):
        if not horizontal >> i & 1:
            assert old.reference(v) == expected[i]
            continue
        with pytest.raises(AssertionError, match="zero numerator"):
            old.reference(v)
        assert expected[i]["y"] == (v["y"] if v["j"] == v["s"] == 0 else -v["y"] % p)
        for measurements in (_zeros(case.circuit), None):
            got = simulate(case.circuit, v, measurements=measurements, rng=random.Random(52))
            assert got.outputs == expected[i] and got.phase == 0
    for c in (old, case):
        _assert_live(c.circuit)


@pytest.mark.parametrize("p", [7, 13, 31])
def test_reference_relaxes_only_zero_numerator_not_divisors_or_raw_p(p):
    table = _table(p, 1)
    seen = set()
    for row in _inputs(table):
        x, y, j, s = (row[k] for k in ("x", "y", "j", "s"))
        xp, yp = table.operand(j, s)
        result = (x, y) if table.digit(j, s) == 0 else sw._group_add((x, y), (xp, yp), p, table.a)
        admissible = x != xp and (table.digit(j, s) == 0 or
                                  result is not None and result[0] != xp)
        if not admissible:
            with pytest.raises(AssertionError):
                sw.signed_window_reference(x, y, j, s, table, zero_safe=True)
            with pytest.raises(AssertionError):
                sw.signed_window_reference(x, y, j, s, table)
            seen.add("input divisor" if x == xp else "output divisor")
            continue
        assert sw.signed_window_reference(x, y, j, s, table, zero_safe=True) == result
        if y == yp:
            seen.add("horizontal")
            with pytest.raises(AssertionError, match="zero numerator"):
                sw.signed_window_reference(x, y, j, s, table)
        else:
            assert sw.signed_window_reference(x, y, j, s, table) == result
    assert {"input divisor", "output divisor", "horizontal"} <= seen
    for x, y in ((p, 2), (1, p), (-1, 2)):
        with pytest.raises(ValueError, match="canonical"):
            sw.signed_window_reference(x, y, 0, 0, table, zero_safe=True)


def test_non_f1_inherited_hazards_and_horizontal_inputs_remain_in_sampler_support():
    case = _case(13, 1)
    rows, expected = _admissible(case)
    before = _audit(_case(13, 1, zero_safe=False), rows, expected)
    after = _audit(case, rows, expected)
    assert len(rows) == 41
    assert (after.bad_count, after.value_count, after.dirty_count,
            after.phase_count, after.noncanonical_count) == (34, 12, 19, 30, 3)
    assert (before.bad_count, before.value_count, before.dirty_count,
            before.phase_count, before.noncanonical_count) == (40, 12, 23, 36, 3)
    assert after.summary_mismatch_count == 0
    assert after.outputs == before.outputs
    assert not after.bad_mask & ~before.bad_mask
    table = case.analysis_context["window_table"]
    horizontal = next(v for v in rows if v["y"] == table.operand(v["j"], v["s"])[1])
    bad = next(v for i, v in enumerate(rows) if after.value_mask >> i & 1)

    class Fixed:
        def __init__(self, row):
            self.values = iter(row[k] for k in ("x", "y", "j", "s"))

        def randrange(self, stop):
            value = next(self.values)
            assert 0 <= value < stop
            return value

    for row in (horizontal, bad):
        assert case.gen_input(Fixed(row)) == row
        assert case.reference(row) in expected


def test_three_positive_paired_lookups_metadata_and_unitary_unlookup(monkeypatch):
    calls = []

    def checked_unlookup(circ, address, target, values, **kw):
        calls.append(tuple(values))
        lookup_xor(circ, address, target, values)
        circ.free(target)

    # The shared restoration helper resolves its own module's unlookup binding.
    monkeypatch.setattr(sw, "unlookup_measure", checked_unlookup)
    monkeypatch.setattr(zp, "unlookup_measure", checked_unlookup)
    case = _case()
    table = case.analysis_context["window_table"]
    n = table.p.bit_length()
    assert len(calls) == 3
    assert calls[0] == calls[2] == tuple(x | (y << n) for x, y in table.points)
    assert calls[1][0] == 0  # Negative endpoint correction is erased before unlookup.
    rows, expected = _admissible(case)
    got = _audit(case, rows, expected)
    assert got.dirty_count == got.value_count == got.summary_mismatch_count == 0
    assert case.meta["zero_safe"] is True
    assert case.meta["coordinate_restoration"] == "add"
    assert case.meta["restoration_exact_reduction"] is True
    assert case.meta["phase_correction"] is False
    assert case.meta["circuit_config"]["lookup_stages"] == 3
    regions = case.meta["resource_regions"]
    assert regions[0][1] == 4 and regions[-1][2] == len(case.circuit.ops)
    assert all(a[2] == b[1] for a, b in zip(regions, regions[1:]))
    assert sum(label == "unlookup" for label, _, _ in regions) == 3
    _assert_live(case.circuit)


def test_packed_views_rebind_across_two_zero_safe_windows():
    table = _table(7, 1)
    circ = Circuit()
    point = circ.input("point", 7)
    digits = [circ.input(f"digit{k}", 2) for k in range(2)]
    x, y = point[:3], point[3:6]
    for digit in digits:
        x, y = sw.add_signed_window_inplace(
            circ, x, y, digit[:1], digit[1], table, zero_safe=True,
            config=_config(table), const_kw={}, sqr_kw={})
        assert circ.inputs["point"].ids == x.ids+y.ids+[point[6]]
    rows, expected, horizontals = [], [], 0
    for v in _inputs(table):
        for second in range(4):
            try:
                middle = sw.signed_window_reference(v["x"], v["y"], v["j"], v["s"],
                                                    table, zero_safe=True)
                out = sw.signed_window_reference(*middle, second & 1, second >> 1,
                                                 table, zero_safe=True)
            except AssertionError:
                continue
            horizontals += v["y"] == table.operand(v["j"], v["s"])[1]
            row = dict(point=v["x"]+(v["y"] << 3)+64,
                       digit0=v["j"]+(v["s"] << 1), digit1=second)
            rows.append(row)
            expected.append(dict(row, point=out[0]+(out[1] << 3)+64))
    assert horizontals > 0 and len(rows) > 50
    got = simulate_batch(compile_symbolic(circ), rows, expected)
    assert got.outputs == expected
    assert got.dirty_count == got.value_count == got.summary_mismatch_count == 0
    assert got.phase_count > 0
    _assert_live(circ)


@pytest.mark.parametrize("p,bits,digest", [
    (7, 1, "a38f2c275daf30b9d16a288c0277a8fcc7ab3ec607b03a610a6812731d376d31"),
    (31, 2, "dc4d83658ca5ecc6f0ce69b31a750fad73b0c3256af461ce914a9cb621d5537d"),
])
def test_opt_out_keeps_legacy_ir_and_default_is_source_compatible(p, bits, digest):
    explicit = _case(p, bits, zero_safe=False)
    table = _table(p, bits)
    default = sw.build_signed_window_case(bits, table=table, config=_config(table),
                                          const_kw={}, sqr_kw={})
    assert compile_symbolic(explicit.circuit).ir_digest == digest
    assert compile_symbolic(default.circuit).ir_digest == digest
    rows, _ = _admissible(default)
    assert [default.reference(v) for v in rows] == [explicit.reference(v) for v in rows]


@pytest.mark.parametrize("invalid", ["selected", "bool", "alias", "width", "config", "fused", "condition"])
def test_malformed_zero_safe_requests_emit_nothing(invalid):
    table = _table()
    circ = Circuit()
    j, s, x, y = [circ.input(name, bits) for name, bits in (("j", 2), ("s", 1), ("x", 5), ("y", 5))]
    kw = dict(config=_config(table), const_kw={}, sqr_kw={}, zero_safe=True)
    if invalid == "selected":
        kw["sign_strategy"] = "selected"
    elif invalid == "bool":
        kw["zero_safe"] = 1
    elif invalid == "alias":
        s = x[:1]
    elif invalid == "width":
        s = j
    elif invalid in ("config", "fused"):
        kw["config"] = replace(kw["config"], **({"window_bits": 3} if invalid == "config" else
                                                {"operand_strategy": "fused"}))

    def reject():
        before = list(circ.ops), set(circ._live_qids), list(circ._free_qids)
        with pytest.raises(ValueError):
            sw.add_signed_window_inplace(circ, x, y, j, s, table, **kw)
        assert (list(circ.ops), set(circ._live_qids), list(circ._free_qids)) == before

    if invalid == "condition":
        with circ.condition(circ.classical_bit(1)):
            reject()
    else:
        reject()


@pytest.mark.parametrize("zero_safe", [False, True])
def test_raw_word_audit_helper_forwards_policy_and_preserves_recoding(monkeypatch, zero_safe):
    table, calls = _table(7, 1), []
    monkeypatch.setattr(raw_audit, "U", 3)
    monkeypatch.setattr(raw_audit, "PRIME", 7)
    monkeypatch.setattr(raw_audit, "make_signed_window_table", lambda bits: table)

    def toy_add(circ, x, y, address, sign, table, **kw):
        calls.append(kw["zero_safe"])
        kw.update(config=_config(table), const_kw={}, sqr_kw={})
        return sw.add_signed_window_inplace(circ, x, y, address, sign, table, **kw)

    monkeypatch.setattr(raw_audit, "add_signed_window_inplace", toy_add)
    case = raw_audit.build_centered_case(2, zero_safe=zero_safe)
    assert calls == [zero_safe]
    assert case.analysis_context["signed_window_table"] is table
    rows, expected = [], []
    for row in _inputs(table):
        raw = table.digit(row["j"], row["s"])+2
        values = dict(word=raw, x=row["x"], y=row["y"])
        try:
            out = sw.signed_window_reference(row["x"], row["y"], row["j"], row["s"],
                                             table, zero_safe=zero_safe)
        except AssertionError:
            with pytest.raises(AssertionError):
                case.reference(values)
            continue
        rows.append(values)
        expected.append(dict(word=raw, x=out[0], y=out[1]))
        assert case.reference(values) == expected[-1]
    assert len(rows) == (37 if zero_safe else 29)
    got = _audit(case, rows, expected)
    assert got.outputs == expected
    assert got.dirty_count == got.value_count == got.summary_mismatch_count == 0
    assert got.phase_count == 9
    _assert_live(case.circuit)


@pytest.mark.parametrize("kwargs", [
    {"zero_safe": True, "sign_strategy": "selected"}, {"zero_safe": 1},
])
def test_invalid_case_options_rejected_before_circuit_construction(monkeypatch, kwargs):
    table = _table()

    def forbidden():
        pytest.fail("invalid zero-safe options reached circuit construction")

    monkeypatch.setattr(sw, "Circuit", forbidden)
    with pytest.raises(ValueError):
        sw.build_signed_window_case(2, table=table, config=_config(table), **kwargs)


@pytest.mark.parametrize("u_bits,v_bits", [(2, 2), (3, 2), (4, 4)])
def test_signed_selector_initialization_tables_digits_and_global_shift_are_unchanged(u_bits, v_bits):
    group = _group()
    slots = fs.windows(u_bits, v_bits, 2)
    assert "signed_zero_safe" in fs.VARIANTS
    assert fs._initial_points(group, slots[0], "signed_zero_safe") == fs._initial_points(group, slots[0], "signed")
    shift = fs.shift_point(group, slots, "signed_zero_safe")
    assert shift == fs.shift_point(group, slots, "signed") == _mul(_shift_scalar(u_bits, v_bits, 2, "signed"))
    for slot in slots[1:]:
        new, old = [fs._table(group, slot, variant) for variant in ("signed_zero_safe", "signed")]
        assert new == old and new.fingerprint == old.fingerprint
        for raw in range(1 << slot["bits"]):
            digit = raw-(1 << (slot["bits"]-1))
            assert new.digit(abs(digit) % len(new.points), int(digit < 0)) == digit


@pytest.mark.parametrize("u_bits,v_bits", [(2, 3), (3, 3), (2, 1)])
def test_one_bit_arithmetic_rejected_before_circuit_or_tables(monkeypatch, u_bits, v_bits):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported schedule began construction")

    monkeypatch.setattr(fs, "Circuit", forbidden)
    monkeypatch.setattr(fs, "_initial_points", forbidden)
    monkeypatch.setattr(fs, "_table", forbidden)
    with pytest.raises(ValueError, match="two bits|one.bit|at least 2"):
        fs.run(group=_group(), u_bits=u_bits, v_bits=v_bits, window_bits=2,
               variant="signed_zero_safe", input_pairs=[], full_width=True)


@pytest.fixture(scope="module", params=(2, 3, 4))
def small_schedules(request):
    u_bits = request.param  # Three bits tests an allowed one-bit INITIAL lookup.
    pairs = list(itertools.product(range(1 << u_bits), range(4)))
    return {recycled: fs.run(group=_group(), u_bits=u_bits, v_bits=2, window_bits=2,
                            variant="signed_zero_safe", recycled=recycled,
                            input_pairs=pairs, seed=SEED, full_width=True,
                            track_depth=True, retain_ops=True)
            for recycled in (False, True)}


def test_retained_ir_and_recycled_diagnostics_match_without_hiding_hazards(small_schedules):
    coherent, recycled = small_schedules[False], small_schedules[True]
    circ = _retained_circuit(coherent)
    u_bits, total_bits = coherent["u_bits"], coherent["u_bits"]+2
    admitted = excluded = phases = 0
    for i, (full, streamed) in enumerate(zip(coherent["diagnostics"], recycled["diagnostics"])):
        assert full == streamed
        assert full["expected"] == _mul(full["u"]+7*full["v"]+_shift_scalar(u_bits, 2, 2, "signed"))
        if full["domain_failure"]:
            excluded += 1
            assert not full["active"] and full["output"] is full["phase"] is None
            continue
        admitted += 1
        assert full["value_failure"] is full["dirty_failure"] is full["control_failure"] is None
        got = simulate(circ, dict(u=full["u"], v=full["v"], x=0, y=0),
                       rng=random.Random(SEED+100000+i))
        assert got.outputs == dict(u=full["u"], v=full["v"], x=full["output"][0], y=full["output"][1])
        assert got.phase == full["phase"]
        assert full["observed_pass"] == (got.phase == 0 and not full["noncanonical_stages"])
        phases += got.phase
    assert (admitted, excluded, phases) == {2: (16, 0, 1), 3: (32, 0, 1), 4: (61, 3, 4)}[u_bits]
    assert coherent["resources"]["toffoli"] == count(circ).toffoli == recycled["resources"]["toffoli"]
    assert recycled["resources"]["measurements"] == coherent["resources"]["measurements"]+total_bits
    assert recycled["resources"]["live_qubits"] == coherent["resources"]["live_qubits"]-total_bits == 10
    for result in (coherent, recycled):
        assert result["zero_safe"] is True and result["phase_correction"] is False
        assert result["coordinate_restoration"] == "add" and result["restoration_exact_reduction"] is True
    # Only substitute complete group translations for the independent controller test.
    controller = dict(recycled, variant="signed")
    assert _max_complex_error(_controller_only_outputs(controller),
                              _dense_double_inverse_qft(u_bits, 2, 2, "signed")) < 2e-12
