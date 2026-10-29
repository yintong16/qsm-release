"""Original guarded-prefix prototype: permutation, phase and resource tests."""
import itertools
import random

import pytest

from qsm.audit.guarded_comparator import build_compare_case, cmp_lt_guarded
from qsm.core import ops
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch


def _inputs(pairs, controlled):
    for av, bv in pairs:
        for fv in (0, 1):
            for cv in ((0, 1) if controlled else (None,)):
                row = {"a": av, "b": bv, "flag": fv}
                if controlled:
                    row["ctrl"] = cv
                yield row


def _sliding(av, bv, width, k, b):
    """Independent original one-step model, not the split comparator formula."""
    if width <= k:
        return int(bv < av)
    step = min(b, width-k)
    if av >> (width-step) == bv >> (width-step):
        mask = (1 << width) - 1
        av, bv = (av << step) & mask, (bv << step) & mask
    return int((bv >> (width-k)) < (av >> (width-k)))


def _expected(row, width, bits):
    shift = max(0, width-bits)
    predicate = int((row["b"] >> shift) < (row["a"] >> shift))
    return {**row, "flag": row["flag"] ^ (row.get("ctrl", 1) & predicate)}


def _assert_batch(case, rows, expected):
    plan = compile_symbolic(case.circuit)
    assert not plan.final_ancillas, "no persistent scratch/tape extension"
    result = simulate_batch(plan, rows, expected)
    assert result.bad_mask == 0
    assert result.dirty_mask == result.phase_mask == 0
    assert result.outputs == expected


@pytest.mark.parametrize("width", range(1, 7))
@pytest.mark.parametrize("controlled", [False, True])
def test_all_small_operands_windows_and_measurement_branches(width, controlled):
    # All a,b (including even operands), both initial flags, both controls.
    # Symbolic phase coefficients prove all measurement streams, not samples.
    rows = list(_inputs(itertools.product(range(1 << width), repeat=2), controlled))
    for k in range(1, width+2):
        for b in range(1, k+1):
            guarded = build_compare_case(width, k, b, "guarded", controlled)
            wider = build_compare_case(width, k, b, "wider", controlled)
            expected = [_expected(row, width, k+b) for row in rows]
            for row, want in zip(rows, expected):
                slide = _sliding(row["a"], row["b"], width, k, b)
                assert want["flag"] == row["flag"] ^ (row.get("ctrl", 1) & slide)
            _assert_batch(guarded, rows, expected)
            _assert_batch(wider, rows, expected)


@pytest.mark.parametrize("width,k,b", [(1, 1, 1), (2, 1, 1), (3, 2, 1), (4, 2, 2)])
@pytest.mark.parametrize("controlled", [False, True])
def test_explicit_all_tiny_measurement_streams(width, k, b, controlled):
    case = build_compare_case(width, k, b, "guarded", controlled)
    nmeas = count(case.circuit).measurements
    assert nmeas <= 5
    for row in _inputs(itertools.product(range(1 << width), repeat=2), controlled):
        expected = _expected(row, width, k+b)
        for stream in itertools.product((0, 1), repeat=nmeas):
            result = simulate(case.circuit, row, measurements=list(stream))
            assert result.outputs == expected
            assert result.phase == 0


def _wide_pairs(width, k, b):
    rng = random.Random(1701 + width + b)
    mask = (1 << width) - 1
    pairs = [(0, 0), (0, mask), (mask, 0), (mask, mask), (1, 0), (0, 1)]
    pairs += [(rng.getrandbits(width), rng.getrandbits(width)) for _ in range(8)]
    effective_b = min(b, max(0, width-k))
    split = width-effective_b
    low_mask = (1 << split) - 1
    # Force equality of the guard; random independent pairs almost never do.
    for _ in range(8):
        high = rng.getrandbits(effective_b) << split
        pairs.append((high | rng.getrandbits(split), high | rng.getrandbits(split)))
    if effective_b:
        # Opposing high/low orderings: high guard must dominate the lower k bits.
        pairs.extend([(1 << split, low_mask), (low_mask, 1 << split)])
    discarded = max(0, width-k-b)
    if discarded:
        # Entire retained prefix ties but discarded bits disagree.
        prefix = rng.getrandbits(width-discarded) << discarded
        pairs.extend([(prefix, prefix+1), (prefix+1, prefix)])
    return pairs


@pytest.mark.parametrize("width", [100, 128, 156, 200, 255])
@pytest.mark.parametrize("b", [1, 2, 4, 8, 16, 32, 64, 100])
@pytest.mark.parametrize("controlled", [False, True])
def test_fullwidth_sampled_ir_all_symbolic_phases(width, b, controlled):
    k = 100
    rows = list(_inputs(_wide_pairs(width, k, b), controlled))
    expected = [_expected(row, width, k+b) for row in rows]
    for strategy in ("guarded", "wider"):
        case = build_compare_case(width, k, b, strategy, controlled)
        assert [case.reference(row) for row in rows] == expected
        _assert_batch(case, rows, expected)
    # Independent scalar interpreter with deliberately all-zero/all-one outcomes.
    case = build_compare_case(width, k, b, "guarded", controlled)
    nmeas = count(case.circuit).measurements
    for row, want in zip(rows, expected):
        for bit in (0, 1):
            result = simulate(case.circuit, row, measurements=[bit]*nmeas)
            assert result.outputs == want
            assert result.phase == 0
        predicate = _sliding(row["a"], row["b"], width, k, b)
        assert want["flag"] == row["flag"] ^ (row.get("ctrl", 1) & predicate)


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("width,k,b", [(1, 1, 1), (6, 8, 2), (100, 100, 100)])
def test_no_guard_delegates_identical_stream(width, k, b, controlled):
    guarded = build_compare_case(width, k, b, "guarded", controlled)
    baseline = build_compare_case(width, k, b, "baseline", controlled)
    assert guarded.circuit.ops == baseline.circuit.ops
    assert count(guarded.circuit) == count(baseline.circuit)


@pytest.mark.parametrize("strategy", ["baseline", "wider", "guarded"])
@pytest.mark.parametrize("controlled", [False, True])
def test_builder_reference_generator_and_clean_lifetimes(strategy, controlled):
    case = build_compare_case(6, 3, 2, strategy, controlled)
    rng = random.Random(22)
    rows = [case.gen_input(rng) for _ in range(32)]
    bits = 3 if strategy == "baseline" else 5
    expected = [_expected(row, 6, bits) for row in rows]
    assert [case.reference(row) for row in rows] == expected
    _assert_batch(case, rows, expected)
    assert case.circuit._live == 13 + int(controlled)


@pytest.mark.parametrize("k,b,error", [
    (0, 1, ValueError), (-1, 1, ValueError), (1, 0, ValueError),
    (1, -1, ValueError), (2, 3, ValueError), (True, 1, TypeError),
    (2, False, TypeError), (2.0, 1, TypeError), (2, "1", TypeError),
])
def test_reject_invalid_windows_without_emitting(k, b, error):
    circ = Circuit()
    a, operand_b, flag = circ.input("a", 3), circ.input("b", 3), circ.input("flag", 1)
    before = list(circ.ops)
    with pytest.raises(error):
        cmp_lt_guarded(circ, a, operand_b, flag[0], k=k, b=b)
    assert circ.ops == before


def test_reject_unequal_widths_without_emitting():
    circ = Circuit()
    a, operand_b, flag = circ.input("a", 3), circ.input("b", 4), circ.input("flag", 1)
    before = list(circ.ops)
    with pytest.raises(ValueError, match="equal-width"):
        cmp_lt_guarded(circ, a, operand_b, flag[0])
    assert circ.ops == before


@pytest.mark.parametrize("args,error", [
    ((0, 2, 1, "guarded"), ValueError),
    ((True, 2, 1, "guarded"), TypeError),
    ((6, 2, 1, "unknown"), ValueError),
    ((6, 2, 1, "guarded", 1), TypeError),
])
def test_builder_invalid_arguments(args, error):
    with pytest.raises(error):
        build_compare_case(*args)


def test_regional_scratch_is_recycled_not_a_wider_carry_ladder():
    # Prototype regression, NOT optimality: count actual emitted operations.
    circ = build_compare_case(255, 100, 8, "guarded", True).circuit
    resources = count(circ)
    assert circ._live == 512
    assert resources.peak_qubits == 614  # 100 carries/top scratch + two guard bits
    assert resources.toffoli == 139  # 108 comparison + 4 controls + 26 equality + 1 AND
    # Equality uses six scratch wires, but never overlaps the 99-carry allocation.
    live = set()
    for op in circ.ops:
        if op.kind == ops.ALLOC:
            live.update(op.qubits)
            if not op.is_input and len(op.qubits) == 99:
                assert len(live) == 613
        elif op.kind == ops.FREE or op.kind == ops.MEASURE:
            live.difference_update(op.qubits)
    assert live == {q for reg in circ.inputs.values() for q in reg}
