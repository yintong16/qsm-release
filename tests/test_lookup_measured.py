"""Exhaustive tiny measurement branches; never build full-width arithmetic."""

import itertools
import random

import pytest

from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.scalarmul.lookup import (
    lookup_xor,
    lookup_xor_unary,
    unlookup_measure,
)


def _branches(circ):
    # Classical parity rows reserve cbits, but are NOT measurement outcomes.
    n = sum(op.kind == "MEASURE" for op in circ.ops)
    assert n <= 11, "keep exhaustive tests deliberately tiny"
    return itertools.product((0, 1), repeat=n)


def _forward(w, n, values, repetitions=1):
    circ = Circuit()
    address = circ.input("address", w)
    target = circ.input("target", n)
    for _ in range(repetitions):
        lookup_xor_unary(circ, address, target, values)
    return circ


def _assert_only_live(circ, ids):
    assert circ._live == len(ids)
    assert circ._live_qids == set(ids)
    assert len(circ._free_qids) == len(set(circ._free_qids))
    assert set(range(circ._next_qid)) - set(circ._free_qids) == set(ids)


def _snapshot(circ):
    return (
        list(circ.ops), circ._next_qid, list(circ._free_qids), circ._live,
        circ._peak, circ._next_cbit, dict(circ._cbit_const), set(circ._live_qids),
        list(circ._cond_stack),
        {name: tuple(reg) for name, reg in circ.inputs.items()},
    )


@pytest.mark.parametrize("w,n", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_unary_all_tiny_tables_addresses_targets_and_measurements(w, n):
    for values in itertools.product(range(1 << n), repeat=1 << w):
        circ = _forward(w, n, values)
        _assert_only_live(circ, range(w + n))
        for measurements in _branches(circ):
            for address, target in itertools.product(range(1 << w), range(1 << n)):
                result = simulate(circ, {"address": address, "target": target},
                                  measurements=measurements)
                assert result.outputs == {
                    "address": address, "target": target ^ values[address]
                }
                assert result.phase == 0
                assert result.measurements == list(measurements)


@pytest.mark.parametrize("values", [
    (0,) * 8,
    (7,) * 8,
    (0, 7, 1, 6, 2, 5, 3, 4),
    (7, 3, 1, 0, 0, 1, 3, 7),
])
def test_unary_three_bit_dense_constant_and_zero_all_branches(values):
    circ = _forward(3, 3, values)
    for measurements in _branches(circ):
        for address, target in itertools.product(range(8), repeat=2):
            result = simulate(circ, {"address": address, "target": target},
                              measurements=measurements)
            assert result.outputs == {
                "address": address, "target": target ^ values[address]
            }
            assert result.phase == 0


def test_compact_lookup_payloads_share_tuple_storage():
    circ = Circuit()
    address = circ.input("address", 3)
    target = circ.alloc(3)
    rows = (0, 7, 1, 6, 2, 5, 3, 4)
    lookup_xor_unary(circ, address, target, rows)
    words = [op for op in circ.ops if op.kind == "XOR_WORD"]
    assert len(words) == sum(bool(row) for row in rows)
    assert all(op.qubits is words[0].qubits for op in words)
    assert tuple(op.mask for op in words) == tuple(row for row in rows if row)
    unlookup_measure(circ, address, target, rows)
    parity, = (op for op in circ.ops if op.kind == "CPARITY")
    assert parity.values is rows
    assert len(parity.sources) == len(target)


def test_unary_is_phase_clean_involution_for_arbitrary_xor_target():
    circ = _forward(2, 2, (3, 0, 2, 1), repetitions=2)
    for measurements in _branches(circ):
        for address, target in itertools.product(range(4), repeat=2):
            inputs = {"address": address, "target": target}
            result = simulate(circ, inputs, measurements=measurements)
            assert result.outputs == inputs
            assert result.phase == 0


def test_unary_reordered_noncontiguous_views():
    circ = Circuit()
    storage = circ.input("storage", 7)
    address = Register([storage[5], storage[1]])
    target = Register([storage[6], storage[0], storage[3]])
    values = (2, 5, 3, 6)
    lookup_xor_unary(circ, address, target, values)
    for measurements in _branches(circ):
        for word in range(128):
            row = ((word >> 5) & 1) | (((word >> 1) & 1) << 1)
            expected = word
            for i, wire in enumerate(target):
                expected ^= ((values[row] >> i) & 1) << wire
            result = simulate(circ, {"storage": word}, measurements=measurements)
            assert result.outputs == {"storage": expected}
            assert result.phase == 0


@pytest.mark.parametrize("w", [1, 2, 3])
@pytest.mark.parametrize("pattern", ["zero", "constant", "dense"])
def test_unlookup_every_split_address_and_measurement_branch(w, pattern):
    n = 2
    size = 1 << w
    values = ((0,) * size if pattern == "zero" else
              (3,) * size if pattern == "constant" else
              tuple((3 * j + (j >> 1)) % 4 for j in range(size)))
    for low_bits in range(w + 1):
        circ = Circuit()
        address = circ.input("address", w)
        target = circ.alloc(n)
        # The old measurement-free circuit prepares precisely the required word;
        # all branches below therefore concern the measured unlookup itself.
        lookup_xor(circ, address, target, values)
        unlookup_measure(circ, address, target, values, low_bits=low_bits)
        _assert_only_live(circ, address.ids)
        for measurements in _branches(circ):
            for row in range(size):
                result = simulate(circ, {"address": row}, measurements=measurements)
                assert result.outputs == {"address": row}
                assert result.phase == 0
                assert result.measurements == list(measurements)


@pytest.mark.parametrize("w,low_bits", [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)])
def test_unary_forward_then_unlookup_complete_branch_composition(w, low_bits):
    n = 2
    values = tuple((j ^ (j >> 1)) % 4 for j in range(1 << w))
    # Inspect the forward output separately: the actual paired circuit cannot
    # expose target afterwards because unlookup consumes, rather than clears it.
    forward = _forward(w, n, values)
    for measurements in _branches(forward):
        for row in range(1 << w):
            result = simulate(forward, {"address": row, "target": 0},
                              measurements=measurements)
            assert result.outputs == {"address": row, "target": values[row]}
            assert result.phase == 0

    circ = Circuit()
    address = circ.input("address", w)
    target = circ.alloc(n)
    lookup_xor_unary(circ, address, target, values)
    unlookup_measure(circ, address, target, values, low_bits=low_bits)
    _assert_only_live(circ, address.ids)
    for measurements in _branches(circ):
        for row in range(1 << w):
            result = simulate(circ, {"address": row}, measurements=measurements)
            assert result.outputs == {"address": row}
            assert result.phase == 0


def test_unlookup_corrupt_output_leaves_address_dependent_phase_not_dirty_scratch():
    circ = Circuit()
    address = circ.input("address", 2)
    target = circ.alloc(2)
    values = (1, 3, 2, 0)
    lookup_xor_unary(circ, address, target, values)
    prior_measurements = sum(op.kind == "MEASURE" for op in circ.ops)
    circ.cx(address[0], target[0])  # violate the exact-restored-word precondition
    unlookup_measure(circ, address, target, values)
    measurements = [0] * sum(op.kind == "MEASURE" for op in circ.ops)
    measurements[prior_measurements] = 1  # first target measurement, NOT a CPARITY
    _assert_only_live(circ, address.ids)
    for row in range(4):
        result = simulate(circ, {"address": row}, measurements=measurements)
        assert result.outputs == {"address": row}
        assert result.phase == (row & 1)


def test_unlookup_reordered_scratch_alias_and_recycled_ids():
    circ = Circuit()
    address = circ.input("address", 3)
    storage = circ.alloc(3)
    target = Register([storage[2], storage[0], storage[1]], name="alias")
    values = (7, 0, 5, 2, 6, 1, 4, 3)
    lookup_xor_unary(circ, address, target, values)
    unlookup_measure(circ, address, target, values, low_bits=2)
    _assert_only_live(circ, address.ids)
    # Reallocate every consumed physical id and use free's zero-state check.
    recycled = circ.alloc(len(circ._free_qids))
    circ.free(recycled)
    for row in range(8):
        for seed in range(10):
            result = simulate(circ, {"address": row}, rng=random.Random(seed))
            assert result.outputs == {"address": row}
            assert result.phase == 0


@pytest.mark.parametrize("w", range(1, 7))
def test_unpruned_unary_resource_formula(w):
    values = tuple(64 | ((37 * row) & 63) for row in range(1 << w))
    circ = _forward(w, 7, values)
    resources = count(circ)
    expected = (1 << w) - 2
    assert resources.toffoli == expected
    assert resources.measurements == expected
    assert sum(op.kind == "MEASURE" for op in circ.ops) == expected
    assert resources.peak_qubits == w + 7 + w - 1
    assert resources.peak_qubits == circ.peak_qubits
    _assert_only_live(circ, range(w + 7))


@pytest.mark.parametrize("w", range(1, 7))
def test_constructed_unlookup_resource_formula_all_splits(w):
    values = tuple(64 | ((37 * row) & 63) for row in range(1 << w))
    for low_bits in range(w + 1):
        circ = Circuit()
        address = circ.input("address", w)
        target = circ.alloc(7)
        lookup_xor_unary(circ, address, target, values)
        before = count(circ)
        unlookup_measure(circ, address, target, values, low_bits=low_bits)
        after = count(circ)
        size = 1 << low_bits
        expected = size - 1 + max((1 << w) // size - 2, 0)
        assert after.toffoli - before.toffoli == expected
        assert after.measurements - before.measurements == 7 + expected
        assert sum(op.kind == "CPARITY" for op in circ.ops) == 1
        assert sum(op.kind == "MEASURE" for op in circ.ops) == after.measurements
        _assert_only_live(circ, address.ids)
        if w >= 4 and low_bits == w // 2:
            assert expected < (1 << w) - 2


@pytest.mark.parametrize("w", [1, 2, 3, 4])
def test_unlookup_default_split_is_floor_half(w):
    circuits = []
    values = tuple(row & 3 for row in range(1 << w))
    for kwargs in ({}, {"low_bits": w // 2}):
        circ = Circuit()
        address = circ.input("address", w)
        target = circ.alloc(2)
        unlookup_measure(circ, address, target, values, **kwargs)
        circuits.append(circ)
    assert circuits[0].ops == circuits[1].ops


@pytest.mark.parametrize("api", [lookup_xor_unary, unlookup_measure])
@pytest.mark.parametrize("values,error", [
    ([], ValueError), ([0], ValueError), ([0] * 5, ValueError),
    ([0, 0, 0, -1], ValueError), ([0, 0, 0, 8], ValueError),
    ([0, 0, 0, 1.0], TypeError), ([0, 0, 0, True], TypeError),
    (None, TypeError), (iter([0] * 4), TypeError),
])
def test_invalid_tables_are_atomic(api, values, error):
    circ = Circuit()
    address = circ.input("address", 2)
    target = circ.alloc(3)
    before = _snapshot(circ)
    with pytest.raises(error):
        api(circ, address, target, values)
    assert _snapshot(circ) == before


@pytest.mark.parametrize("api", [lookup_xor_unary, unlookup_measure])
@pytest.mark.parametrize("address_ids,target_ids,error", [
    ([], [2], ValueError), ([0], [], ValueError),
    ([0, 0], [2], ValueError), ([0], [2, 2], ValueError),
    ([0, 1], [1, 2], ValueError), ([-1], [2], ValueError),
    ([0], [3], ValueError), ([0.0], [2], TypeError),
    ([0], [True], TypeError),
])
def test_invalid_register_views_are_atomic(api, address_ids, target_ids, error):
    circ = Circuit()
    circ.alloc(3)
    before = _snapshot(circ)
    with pytest.raises(error):
        api(circ, Register(address_ids), Register(target_ids),
            [0] * (1 << len(address_ids)))
    assert _snapshot(circ) == before


@pytest.mark.parametrize("api", [lookup_xor_unary, unlookup_measure])
@pytest.mark.parametrize("dead_part", ["address", "target"])
@pytest.mark.parametrize("consume", ["free", "measure"])
def test_dead_wire_aliases_are_rejected_atomically(api, dead_part, consume):
    circ = Circuit()
    address, target = circ.alloc(2), circ.alloc(2)
    dead = address if dead_part == "address" else target
    if consume == "free":
        circ.free(dead)
    else:
        circ.measure(dead[0])
    before = _snapshot(circ)
    with pytest.raises(ValueError, match="live"):
        api(circ, Register(address.ids), Register(target.ids), (0, 1, 2, 3))
    assert _snapshot(circ) == before


@pytest.mark.parametrize("api", [lookup_xor_unary, unlookup_measure])
@pytest.mark.parametrize("wrong_part", ["address", "target"])
def test_nonregister_views_are_rejected_atomically(api, wrong_part):
    circ = Circuit()
    address, target = circ.alloc(2), circ.alloc(2)
    before = _snapshot(circ)
    with pytest.raises(TypeError, match="Registers"):
        api(circ, address.ids if wrong_part == "address" else address,
            target.ids if wrong_part == "target" else target, (0, 1, 2, 3))
    assert _snapshot(circ) == before


@pytest.mark.parametrize("api", [lookup_xor_unary, unlookup_measure])
@pytest.mark.parametrize("condition_value", [0, 1])
def test_outer_classical_conditions_rejected_atomically(api, condition_value):
    circ = Circuit()
    address = circ.input("address", 2)
    target = circ.alloc(2)
    cb = circ.classical_bit(condition_value)
    with circ.condition(cb):
        before = _snapshot(circ)
        with pytest.raises(ValueError, match="condition"):
            api(circ, address, target, (0, 1, 2, 3))
        assert _snapshot(circ) == before


@pytest.mark.parametrize("alias", [False, True])
def test_unlookup_named_input_target_rejected_including_aliases(alias):
    circ = Circuit()
    address = circ.input("address", 2)
    named = circ.input("target", 2)
    target = Register(reversed(named.ids), name="not-an-input") if alias else named
    before = _snapshot(circ)
    with pytest.raises(ValueError, match="scratch"):
        unlookup_measure(circ, address, target, (0, 1, 2, 3))
    assert _snapshot(circ) == before


def test_unlookup_mixed_scratch_and_named_input_target_rejected():
    circ = Circuit()
    address = circ.input("address", 2)
    named = circ.input("named", 1)
    scratch = circ.alloc(1)
    target = Register([scratch[0], named[0]])
    before = _snapshot(circ)
    with pytest.raises(ValueError, match="scratch"):
        unlookup_measure(circ, address, target, (0, 1, 2, 3))
    assert _snapshot(circ) == before


@pytest.mark.parametrize("low_bits,error", [
    (-1, ValueError), (3, ValueError), (1.0, TypeError),
    (True, TypeError), (False, TypeError), ("1", TypeError),
])
def test_invalid_low_bits_rejected_atomically(low_bits, error):
    circ = Circuit()
    address = circ.input("address", 2)
    target = circ.alloc(2)
    before = _snapshot(circ)
    with pytest.raises(error, match="low_bits"):
        unlookup_measure(circ, address, target, (0, 1, 2, 3), low_bits=low_bits)
    assert _snapshot(circ) == before
