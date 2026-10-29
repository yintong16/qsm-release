"""Streamed modular operands: exact values, inherited phase, and honest reloads."""

import itertools
from unittest.mock import patch

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.simulator import simulate
from qsm.field import modarith
from qsm.scalarmul import fused
from qsm.scalarmul.lookup import lookup_xor


def _borrow_measure_index(circ, cbit):
    # CPARITY creates classical ids without quantum measurements. In particular,
    # a cbit id is NOT an offset into simulate(measurements=...).
    cbits = [op.cbit for op in circ.ops if op.kind == ops.MEASURE]
    return cbits.index(cbit)


def _build_pair(w, n, values, p, **kwargs):
    stream = Circuit()
    address = stream.input("address", w)
    acc = stream.input("acc", n)
    direct = Circuit()
    a = direct.input("a", n)
    b = direct.input("acc", n)
    with patch.object(fused, "cmp_lt_phase", wraps=fused.cmp_lt_phase) as spy:
        fused.modsub_lookup_inplace(stream, address, acc, values, p, **kwargs)
        assert spy.call_count == 1
        borrow = spy.call_args.args[3]
        k = min(n, kwargs.get("cmp_bits") or n)
        assert len(spy.call_args.args[1]) == len(spy.call_args.args[2]) == k
        assert spy.call_args.kwargs == {"k": None}
    with patch.object(modarith, "cmp_lt_phase", wraps=modarith.cmp_lt_phase) as spy:
        modarith.modsub_direct(direct, a, b, p,
                               carry_window=kwargs.get("carry_window"),
                               cmp_bits=kwargs.get("cmp_bits"))
        direct_borrow = spy.call_args.args[3]
    assert stream._live == w + n
    assert direct._live == 2 * n
    return ((stream, _borrow_measure_index(stream, borrow)),
            (direct, _borrow_measure_index(direct, direct_borrow)))


def _outcomes(circ, borrow_index, borrow, pattern):
    size = sum(op.kind == ops.MEASURE for op in circ.ops)
    outcomes = [0 if pattern == "zero" else 1 if pattern == "one" else i % 2
                for i in range(size)]
    outcomes[borrow_index] = borrow
    return outcomes


def _compare(pair, values, p, address, acc, borrow, pattern, *, carry_window=None):
    (stream, sm), (direct, dm) = pair
    result = simulate(stream, {"address": address, "acc": acc},
                      measurements=_outcomes(stream, sm, borrow, pattern))
    baseline = simulate(direct, {"a": values[address], "acc": acc},
                        measurements=_outcomes(direct, dm, borrow, pattern))
    assert result.outputs["address"] == address
    assert baseline.outputs["a"] == values[address]
    assert result.outputs["acc"] == baseline.outputs["acc"]
    assert result.phase == baseline.phase
    if carry_window is None:
        # Do not reduce modulo p: lazy input can legitimately remain lazy.
        assert result.outputs["acc"] == modarith.modsub_ref(values[address], acc, p)
    return result


@pytest.mark.parametrize("n,p,block", [
    (1, 1, 1), (2, 3, 1), (2, 3, 2), (2, 3, 3),
    (3, 5, 1), (3, 5, 2), (3, 5, 3), (3, 5, 4), (3, 7, 2),
])
def test_every_tiny_canonical_table_and_every_lazy_accumulator(n, p, block):
    for values in itertools.product(range(p), repeat=2):
        pair = _build_pair(1, n, values, p, block=block)
        for address, acc, borrow in itertools.product(range(2), range(1 << n), range(2)):
            for pattern in ("zero", "one", "alternating"):
                _compare(pair, values, p, address, acc, borrow, pattern)


@pytest.mark.parametrize("block", [1, 2, 3, 4, 5, 6, 20])
@pytest.mark.parametrize("cmp_bits", [1, 2, 5, None])
def test_borrow_and_carry_edges_ragged_blocks_and_top_slice_alignment(block, cmp_bits):
    n, p = 5, 29
    values = (0, 1, 3, 7, 8, 15, 16, 28)
    pair = _build_pair(3, n, values, p, block=block, cmp_bits=cmp_bits)
    phases = set()
    for address, acc, borrow in itertools.product(range(8), range(1 << n), range(2)):
        # All input words cover ties and borrow/carry propagation at every cut.
        result = _compare(pair, values, p, address, acc, borrow,
                          "alternating" if acc & 1 else "one")
        phases.add(result.phase)
    assert phases == {0, 1}, "small-prime direct subtraction deliberately retains phase"


@pytest.mark.parametrize("carry_window", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("cmp_bits", [1, 2, 5, None])
def test_truncated_construction_matches_direct_not_ideal_arithmetic(carry_window, cmp_bits):
    n, p = 5, 29
    values = (0, 1, 15, 28)
    pair = _build_pair(2, n, values, p, block=2,
                       carry_window=carry_window, cmp_bits=cmp_bits)
    differences = 0
    for address, acc, borrow in itertools.product(range(4), range(1 << n), range(2)):
        result = _compare(pair, values, p, address, acc, borrow, "one",
                          carry_window=carry_window)
        differences += result.outputs["acc"] != modarith.modsub_ref(values[address], acc, p)
    assert bool(differences) == (carry_window < n)


@pytest.mark.parametrize("low_bits", [0, 1, 2, None])
def test_every_operand_is_restored_before_consuming_unlookup(low_bits):
    # Measured operands never hit FREE, so append a separate unitary equality
    # witness before EACH unlookup. A wrong restored value dirties this witness,
    # even if an all-zero measurement vector would conceal the wrong phase.
    real_unlookup = fused.unlookup_measure
    checks = []

    def checked_unlookup(circ, address, operand, words, **kwargs):
        witness = circ.alloc(len(operand))
        for source, target in zip(operand, witness):
            circ.cx(source, target)
        lookup_xor(circ, address, witness, words)
        circ.free(witness)
        checks.append((len(operand), tuple(words)))
        real_unlookup(circ, address, operand, words, **kwargs)

    values = (0, 7, 16, 28)
    with patch.object(fused, "unlookup_measure", side_effect=checked_unlookup):
        pair = _build_pair(2, 5, values, 29, block=2, cmp_bits=2, low_bits=low_bits)
    assert len(checks) == 6  # 2*ceil((5+1)/2)-1 integer pairs + phase pair
    assert checks[-1] == (2, tuple(v >> 3 for v in values))
    for address, acc, borrow in itertools.product(range(4), range(32), range(2)):
        _compare(pair, values, 29, address, acc, borrow, "zero")
        _compare(pair, values, 29, address, acc, borrow, "one")


@pytest.mark.parametrize("n,block,cmp_bits", [(5, 2, 2), (6, 3, 1), (5, 20, None)])
def test_regions_are_disjoint_complete_and_count_every_reload(n, block, cmp_bits):
    circ = Circuit()
    address = circ.input("address", 2)
    acc = circ.input("acc", n)
    start = len(circ.ops)
    regions = []
    with patch.object(fused, "lookup_xor_unary", wraps=fused.lookup_xor_unary) as load:
        fused.modsub_lookup_inplace(circ, address, acc, (0, 1, 3, 7), (1 << n) - 3,
                                   block=block, cmp_bits=cmp_bits, regions=regions)
    pairs = 2 * ((n + 1 + block - 1) // block)
    assert load.call_count == pairs
    assert sum(label == "lookup" for label, _, _ in regions) == pairs
    assert sum(label == "unlookup" for label, _, _ in regions) == pairs
    cursor = start
    for label, lo, hi in regions:
        assert label in {"lookup", "unlookup", "arithmetic"}
        assert lo == cursor and hi >= lo
        cursor = hi
    assert cursor == len(circ.ops)
    assert circ._live == n + 2


@pytest.mark.parametrize("setting", ["block", "carry_window", "cmp_bits"])
@pytest.mark.parametrize("value,error", [(0, ValueError), (-1, ValueError),
                                          (1.5, TypeError), (True, TypeError)])
def test_invalid_positive_settings_are_atomic(setting, value, error):
    _reject_atomic({setting: value}, error)


@pytest.mark.parametrize("kwargs,error", [
    ({"block": None}, TypeError),
    ({"low_bits": -1}, ValueError), ({"low_bits": 3}, ValueError),
    ({"low_bits": True}, TypeError), ({"low_bits": 1.5}, TypeError),
    ({"regions": ()}, TypeError),
    ({"p": 0}, ValueError), ({"p": -1}, ValueError), ({"p": 8}, ValueError),
    ({"p": True}, TypeError), ({"p": 5.0}, TypeError),
    ({"values": []}, ValueError), ({"values": [0, 1, 2]}, ValueError),
    ({"values": [0, 1, 2, -1]}, ValueError),
    ({"values": [0, 1, 2, 5]}, ValueError),
    ({"values": [0, 1, 2, 8]}, ValueError),
    ({"values": [0, 1, 2, True]}, TypeError),
    ({"values": [0, 1, 2, 1.5]}, TypeError),
    ({"values": iter([0, 1, 2, 3])}, TypeError),
])
def test_invalid_table_modulus_and_other_settings_are_atomic(kwargs, error):
    _reject_atomic(kwargs, error)


def _snapshot(circ):
    return (list(circ.ops), circ._next_qid, list(circ._free_qids), circ._live,
            circ._next_cbit)


def _reject_atomic(overrides, error):
    circ = Circuit()
    address = circ.input("address", 2)
    acc = circ.input("acc", 3)
    kwargs = {"values": [0, 1, 2, 3], "p": 5, "regions": []}
    kwargs.update(overrides)
    before = _snapshot(circ)
    with pytest.raises(error):
        fused.modsub_lookup_inplace(circ, address, acc, **kwargs)
    assert _snapshot(circ) == before
    if isinstance(kwargs["regions"], list):
        assert kwargs["regions"] == []


@pytest.mark.parametrize("bad_view", ["overlap", "duplicate", "freed", "unknown", "empty", "boolean"])
def test_invalid_wire_views_reject_before_emission(bad_view):
    circ = Circuit()
    address = circ.input("address", 2)
    acc = circ.input("acc", 3)
    if bad_view == "overlap":
        acc = Register([address[0], acc[1], acc[2]])
    elif bad_view == "duplicate":
        acc = Register([acc[0], acc[0], acc[2]])
    elif bad_view == "freed":
        spent = circ.alloc(1)
        circ.free(spent)
        acc = Register([spent[0], acc[1], acc[2]])
    elif bad_view == "unknown":
        acc = Register([100, acc[1], acc[2]])
    elif bad_view == "empty":
        acc = Register([])
    else:
        acc = Register([True, acc[1], acc[2]])
    before = _snapshot(circ)
    with pytest.raises((ValueError, TypeError)):
        fused.modsub_lookup_inplace(circ, address, acc, (0, 1, 2, 3), 5)
    assert _snapshot(circ) == before


def test_outer_classical_condition_is_rejected_atomically():
    circ = Circuit()
    address = circ.input("address", 2)
    acc = circ.input("acc", 3)
    m = circ.measure(circ.alloc(1)[0])
    before = _snapshot(circ)
    with circ.condition(m):
        with pytest.raises(ValueError, match="classical condition"):
            fused.modsub_lookup_inplace(circ, address, acc, (0, 1, 2, 3), 5)
    assert _snapshot(circ) == before
