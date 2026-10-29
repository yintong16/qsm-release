"""Independent integer/group checks for the isolated local first-lookup APIs."""
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
from qsm.params import PRIME
from qsm.point import active as curve
from qsm.point.group import scalar_mul
from qsm.scalarmul import local_signed as ls


def _table(w=3, signed=True, *, p=127):
    if p == 7:
        return ls.make_first_lookup_table(w, (1, 2), order=13, p=7, a=0, b=3,
                                          signed=signed)
    return ls.make_first_lookup_table(w, (0, 2), order=61, p=127, a=3, b=4,
                                      signed=signed)


def _live(circ):
    named = {q for reg in circ.inputs.values() for q in reg}
    assert circ._live_qids == named
    assert circ._live == len(named)
    assert len(circ._free_qids) == len(set(circ._free_qids))
    assert count(circ).peak_qubits == circ.peak_qubits


def _check(case, rows):
    expected = [case.reference(row) for row in rows]
    hazards = simulate_batch(compile_symbolic(case.circuit), rows, expected)
    assert hazards.bad_count == hazards.summary_mismatch_count == 0
    # Remove the trusted QROM summaries and independently execute its actual
    # primitive stream with symbolic coefficients for EVERY measurement outcome.
    case.circuit.exact_blocks.clear()
    raw = simulate_batch(compile_symbolic(case.circuit), rows, expected)
    assert raw.bad_count == raw.summary_mismatch_count == 0
    _live(case.circuit)
    return expected


@pytest.mark.parametrize("w", range(1, 9))
@pytest.mark.parametrize("roundtrip", [False, True])
def test_integer_bias_all_original_words_and_inverse(w, roundtrip):
    case = ls.build_bias_recode_case(w, roundtrip=roundtrip)
    rows = [{"word": u} for u in range(1 << w)]
    expected = _check(case, rows)
    assert count(case.circuit).measurements == 0
    if not roundtrip:
        bound = 1 << (w-1)
        assert len({v["word"] for v in expected}) == 1 << w
        for u, result in enumerate(expected):
            address, sign = result["word"] % bound, result["word"] // bound
            digit = (-bound if sign else 0) if not address else (-address if sign else address)
            assert digit == u-bound
        # An inverse-only circuit must undo every encoded input, not merely an
        # accidentally self-inverse round trip on a favored subset.
        circ = Circuit()
        word = circ.input("word", w)
        ls.bias_recode_inplace(circ, word, inverse=True)
        assert simulate_batch(compile_symbolic(circ), expected, rows).bad_count == 0
        _live(circ)


@pytest.mark.parametrize("w", range(1, 9))
def test_half_shift_transform_all_inputs_no_scratch(w):
    circ = Circuit()
    word = circ.input("word", w)
    ls.half_shift_recode_inplace(circ, word)
    bound = 1 << (w-1)
    rows = [{"word": u} for u in range(1 << w)]
    expected = [{"word": ((1-u//bound) << (w-1)) |
                (u % bound if u >= bound else (bound-1) ^ (u % bound))} for u in range(1 << w)]
    assert simulate_batch(compile_symbolic(circ), rows, expected).bad_count == 0
    for u, result in enumerate(expected):
        address, sign = result["word"] % bound, result["word"] // bound
        assert (-1 if sign else 1)*(2*address+1) == 2*u-2*bound+1
    metrics = count(circ)
    assert metrics.toffoli == metrics.measurements == 0
    assert metrics.peak_qubits == w and metrics.clifford == w
    ls.half_shift_recode_inplace(circ, word, inverse=True)
    assert simulate_batch(compile_symbolic(circ), rows, rows).bad_count == 0
    _live(circ)


@pytest.mark.parametrize("signed,widths", [(True, range(1, 6)), (False, range(1, 5))])
def test_exhaustive_toy_first_lookup_group_output_preservation_and_all_outcomes(signed, widths):
    for w in widths:
        table = _table(w, signed)
        case = ls.build_first_lookup_case(w, signed=signed, table=table)
        rows = [{"word": u, "x": 0, "y": 0} for u in range(1 << w)]
        expected = _check(case, rows)
        for u, point in enumerate(expected):
            k = (2*u+table.shift_numerator)*pow(2, -1, table.order) % table.order
            oracle = scalar_mul(k, *table.base, p=table.p, a=table.a)
            assert (point["x"], point["y"]) == oracle
            assert point["word"] == u
            assert 0 <= point["x"] < table.p and 0 < point["y"] < table.p
        point_ids = set(case.circuit.inputs["x"]) | set(case.circuit.inputs["y"])
        assert not any(op.kind in ("MEASURE", "FREE") and point_ids.intersection(op.qubits)
                       for op in case.circuit.ops)


@pytest.mark.parametrize("signed,w", [(True, 1), (True, 2), (True, 3), (False, 2)])
def test_tiny_raw_simulation_enumerates_every_measurement_bitstring(signed, w):
    table = _table(w, signed, p=7)
    case = ls.build_first_lookup_case(w, signed=signed, table=table)
    measurements = count(case.circuit).measurements
    assert measurements < 10
    for outcomes in itertools.product((0, 1), repeat=measurements):
        for u in range(1 << w):
            row = {"word": u, "x": 0, "y": 0}
            got = simulate(case.circuit, row, measurements=outcomes)
            assert got.outputs == case.reference(row) and got.phase == 0


@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("w", [1, 3])
def test_core_allocates_zero_point_and_retains_outputs(signed, w):
    table = _table(w, signed)
    circ = Circuit()
    word = circ.input("word", w)
    # Empty named slots merely expose the new retained coordinates to the test
    # simulator; the core itself, rather than the caller, allocates their zeros.
    circ.input("x", 0)
    circ.input("y", 0)
    before = len(circ.ops)
    x, y = ls.initialize_first_lookup(circ, word, table)
    assert circ.ops[before].kind == circ.ops[before+1].kind == "ALLOC"
    assert not circ.ops[before].is_input and not circ.ops[before+1].is_input
    circ.rebind("x", x)
    circ.rebind("y", y)
    rows = [{"word": u, "x": 0, "y": 0} for u in range(1 << w)]
    expected = [dict(row, x=table.point_for_word(row["word"])[0],
                     y=table.point_for_word(row["word"])[1]) for row in rows]
    circ.exact_blocks.clear()
    assert simulate_batch(compile_symbolic(circ), rows, expected).bad_count == 0
    _live(circ)


@pytest.mark.parametrize("signed", [True, False])
def test_public_table_metadata_shift_and_weighted_base(signed):
    versions = all_versions()
    base = scalar_mul(1 << 7, 0, 2, p=127, a=3)
    table = ls.make_first_lookup_table(3, base, order=61, p=127, a=3, b=4, signed=signed)
    assert scalar_mul(2, *table.half_base, p=127, a=3) == base
    for j, point in enumerate(table.points):
        assert point == scalar_mul(2*j+1, *table.half_base, p=127, a=3)
    assert table.entries == 1 << (3-int(signed))
    assert table.shift_numerator == (1-8 if signed else 1)
    assert table.shift_denominator == 2
    assert table.shift_point == scalar_mul(table.shift_scalar, *base, p=127, a=3)
    assert len(table.fingerprint) == 64
    assert table.fingerprint == replace(table).fingerprint
    assert table.fingerprint != _table(3, signed).fingerprint
    case = ls.build_first_lookup_case(3, table=table, signed=signed)
    json.dumps(case.meta)
    assert case.meta["table_fingerprint"] == table.fingerprint
    assert case.meta["correction_applied"] is False
    assert case.meta["measured_output_unlookup"] is False
    assert case.meta["fail_analytic"] == 0
    indices = [i for _, start, stop in case.meta["resource_regions"] for i in range(start, stop)]
    assert indices == list(range(3, len(case.circuit.ops)))
    assert all_versions() == versions
    rng = random.Random(643)
    for _ in range(20):
        row = case.gen_input(rng)
        got = simulate(case.circuit, row, rng=rng)
        assert got.outputs == case.reference(row) and got.phase == 0
    with pytest.raises(ValueError, match="zero"):
        case.reference({"word": 0, "x": 1, "y": 0})


def test_order_and_no_infinity_validation():
    with pytest.raises(ValueError, match="explicit"):
        ls.make_first_lookup_table(2, p=127)
    with pytest.raises(ValueError, match="odd"):
        ls.make_first_lookup_table(2, (3, 6), order=16, p=31, a=2, b=3)
    with pytest.raises(ValueError, match="annihilate"):
        ls.make_first_lookup_table(2, (0, 2), order=63, p=127, a=3, b=4)
    for signed, w in ((True, 6), (False, 5)):
        with pytest.raises(ValueError, match="less than order"):
            _table(w, signed)
    # An odd annihilating multiple is not an exact order certificate: explicit
    # row validation still catches the infinity at (2*j+1)=61.
    with pytest.raises(ValueError, match="finite"):
        ls.make_first_lookup_table(7, (0, 2), order=183, p=127, a=3, b=4)
    table = _table()
    for bad, pattern in ((None, "finite"), ((127, 2), "canonical"),
                         ((1, 1), "declared curve"), (table.points[1], "2j")):
        with pytest.raises(ValueError, match=pattern):
            replace(table, points=(bad, *table.points[1:]))
    with pytest.raises(ValueError, match="length"):
        replace(table, points=table.points[:-1])
    for word in (-1, 8, True):
        with pytest.raises(ValueError, match="fit"):
            table.point_for_word(word)


@pytest.mark.parametrize("fn", [ls.bias_recode_inplace, ls.half_shift_recode_inplace])
@pytest.mark.parametrize("bad", ["empty", "alias", "dead", "inverse", "condition"])
def test_invalid_recoding_rejected_before_emission(fn, bad):
    circ = Circuit()
    word = circ.input("word", 3)
    kwargs = {}
    if bad == "empty":
        word = word[:0]
    elif bad == "alias":
        word = Register([word[0], word[0]])
    elif bad == "dead":
        tmp = circ.alloc(1)
        circ.free(tmp)
        word = tmp
    elif bad == "inverse":
        kwargs["inverse"] = 1
    else:
        circ._cond_stack.append((circ.classical_bit(1), 1))
    before = list(circ.ops)
    with pytest.raises((TypeError, ValueError)):
        fn(circ, word, **kwargs)
    assert list(circ.ops) == before


def test_initializer_mismatch_rejected_before_allocating_output():
    circ = Circuit()
    word = circ.input("word", 3)
    before = list(circ.ops)
    with pytest.raises(ValueError, match="match"):
        ls.initialize_first_lookup(circ, word, _table(2))
    assert list(circ.ops) == before
    with pytest.raises(ValueError, match="match"):
        ls.build_first_lookup_case(3, table=_table(3), signed=False)


@pytest.mark.parametrize("w", [15, 16, 255, 256])
def test_fullsize_integer_primitive_costs_are_actual_emitted_gates(w):
    case = ls.build_bias_recode_case(w)
    resources = count(case.circuit)
    assert resources.toffoli == 2*(w-1)
    assert resources.clifford == 5*(w-1)+1
    assert resources.measurements == 0
    assert resources.peak_qubits == 2*w-1
    rows = [{"word": u} for u in (0, 1, (1 << (w-1))-1, 1 << (w-1), (1 << w)-1)]
    _check(case, rows)


@pytest.mark.skipif(PRIME != (1 << 255)-19, reason="Wei25519 resource calibration")
def test_wei25519_exact_negation_fullsize_cost_and_phase():
    circ = Circuit()
    s, y = circ.input("s", 1), circ.input("y", 255)
    ls.conditional_negate_nonzero_inplace(circ, y, s[0], PRIME)
    resources = count(circ)
    assert (resources.toffoli, resources.measurements, resources.peak_qubits) == (254, 252, 509)
    rows = [dict(s=s, y=y) for s in (0, 1) for y in (1, 2, PRIME//2, PRIME-1)]
    expected = [dict(row, y=(-row["y"] % PRIME if row["s"] else row["y"])) for row in rows]
    assert simulate_batch(compile_symbolic(circ), rows, expected).bad_count == 0
    _live(circ)


@pytest.mark.parametrize("signed", [True, False])
def test_full_field_small_firstlookup_no_large_table_sweep(signed):
    # Only 4/8 physical rows at full FIELD width, never a 2**15 sweep.
    table = ls.make_first_lookup_table(3, signed=signed)
    case = ls.build_first_lookup_case(3, table=table, signed=signed)
    rows = [{"word": u, "x": 0, "y": 0} for u in range(8)]
    _check(case, rows)
    metrics = count(case.circuit)
    sign = Circuit()
    s, y = sign.input("s", 1), sign.input("y", PRIME.bit_length())
    ls.conditional_negate_nonzero_inplace(sign, y, s[0], PRIME)
    assert metrics.toffoli == table.entries-2 + (count(sign).toffoli if signed else 0)
    assert metrics.measurements == table.entries-2 + (count(sign).measurements if signed else 0)
    assert table.order == curve.ORDER
