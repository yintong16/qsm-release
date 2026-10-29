"""Exact INTEGER lookup-fed addition: correctness, cleanup and small cost probes."""

import itertools
import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import gidney_add
from qsm.field.add_chunked import gidney_add_chunked
from qsm.scalarmul.fused import add_lookup_inplace
from qsm.scalarmul.lookup import lookup_xor


def _build(w, n, values, block):
    circ = Circuit()
    address = circ.input("address", w)
    target = circ.input("target", n)
    add_lookup_inplace(circ, address, target, values, block=block)
    assert circ._live == w + n
    return circ


def _whole_word(w, n, values, block=None):
    """Load/add/unload baseline; optional chunking changes only adder carries."""
    circ = Circuit()
    address = circ.input("address", w)
    target = circ.input("target", n)
    operand = circ.alloc(n)
    lookup_xor(circ, address, operand, values)
    if block is None:
        gidney_add(circ, operand, target)
    else:
        gidney_add_chunked(circ, operand, target, chunk=block)
    lookup_xor(circ, address, operand, values)
    circ.free(operand)
    assert circ._live == w + n
    return circ


def _measurements(circ):
    m = sum(op.kind == ops.MEASURE for op in circ.ops)
    assert m <= 8, "exhaustive branch tests must stay deliberately small"
    return itertools.product(range(2), repeat=m)


def _check(circ, values, n, inputs, measurements=None, rng=None):
    result = simulate(circ, inputs, measurements=measurements, rng=rng)
    assert result.outputs == {
        "address": inputs["address"],
        "target": (inputs["target"] + values[inputs["address"]]) & ((1 << n) - 1),
    }
    assert result.phase == 0
    # simulate checks zero at EVERY free, including restored operand and carries.


@pytest.mark.parametrize("n,block", [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)])
def test_all_tiny_tables_inputs_and_measurement_branches(n, block):
    for values in itertools.product(range(1 << n), repeat=2):
        circ = _build(1, n, values, block)
        for measurements in _measurements(circ):
            for address, target in itertools.product(range(2), range(1 << n)):
                _check(circ, values, n, {"address": address, "target": target}, measurements)


@pytest.mark.parametrize("n,block", [(4, 1), (4, 2), (4, 3), (5, 2), (5, 3), (5, 4)])
def test_multiblock_exhaustive_inputs_and_measurement_branches(n, block):
    mask = (1 << n) - 1
    values = (0, mask, 1, mask - 1, 1 << (n - 1), 3, 3, 0)
    circ = _build(3, n, values, block)
    for measurements in _measurements(circ):
        for address, target in itertools.product(range(8), range(1 << n)):
            _check(circ, values, n, {"address": address, "target": target}, measurements)


@pytest.mark.parametrize("n,block", [(17, 1), (17, 3), (31, 8), (65, 8), (65, 64)])
def test_random_words_ragged_blocks_and_long_carries(n, block):
    rng = random.Random(700 + n + block)
    mask = (1 << n) - 1
    values = (0, 1, mask, mask - 1) + tuple(rng.getrandbits(n) for _ in range(12))
    circ = _build(4, n, values, block)
    baseline = _whole_word(4, n, values)
    for address in range(16):
        for target in [0, 1, mask, mask - 1] + [rng.getrandbits(n) for _ in range(4)]:
            inputs = {"address": address, "target": target}
            _check(circ, values, n, inputs, rng=rng)
            _check(baseline, values, n, inputs, rng=rng)


def test_add_and_negated_table_restore_arbitrary_target():
    n = 5
    values = (0, 1, 15, 31)
    inverse = tuple((-value) % (1 << n) for value in values)
    circ = Circuit()
    address = circ.input("address", 2)
    target = circ.input("target", n)
    add_lookup_inplace(circ, address, target, values, block=2)
    add_lookup_inplace(circ, address, target, inverse, block=3)
    rng = random.Random(87)
    for j, z in itertools.product(range(4), range(1 << n)):
        inputs = {"address": j, "target": z}
        for _ in range(4):
            result = simulate(circ, inputs, rng=rng)
            assert result.outputs == inputs
            assert result.phase == 0
    assert circ._live == 2 + n


@pytest.mark.parametrize("block", [5, 6, 100])
def test_one_block_equals_whole_word_baseline(block):
    values = (0, 31, 6, 21)
    fused = _build(2, 5, values, block)
    baseline = _whole_word(2, 5, values)
    assert fused.ops == baseline.ops
    assert count(fused) == count(baseline)


def test_small_resource_tradeoff_counts_repeated_decoding():
    rng = random.Random(112)
    w, n, block = 4, 32, 4
    values = tuple(rng.getrandbits(n) for _ in range(1 << w))
    fused = count(_build(w, n, values, block))
    baseline = count(_whole_word(w, n, values))
    chunked = count(_whole_word(w, n, values, block=block))
    # Compare against both the naive Gidney baseline AND one with matched carry
    # chunking. No point-circuit/global optimality claim follows from these.
    assert fused.peak_qubits < chunked.peak_qubits < baseline.peak_qubits
    assert fused.toffoli > chunked.toffoli > baseline.toffoli
    assert fused.measurements == chunked.measurements > baseline.measurements
    print("\nInteger-only probe: w=4, n=32, table RNG seed=112")
    print("backend block Toffoli peak measurements gates")
    for name, width, resource in [
        ("whole-gidney", n, baseline), ("whole-chunked", block, chunked)
    ] + [("lookup-fed", width, count(_build(w, n, values, width)))
         for width in [1, 2, 4, 8, 16, 32]]:
        print(name, width, resource.toffoli, resource.peak_qubits,
              resource.measurements, resource.n_gates)


@pytest.mark.parametrize("block,error", [(0, ValueError), (-1, ValueError),
                                         (1.5, TypeError), (True, TypeError)])
def test_invalid_block_is_atomic(block, error):
    circ = Circuit()
    address = circ.input("address", 1)
    target = circ.input("target", 2)
    before = list(circ.ops)
    with pytest.raises(error):
        add_lookup_inplace(circ, address, target, [0, 3], block=block)
    assert circ.ops == before


@pytest.mark.parametrize("values,error", [([], ValueError), ([0, 4], ValueError),
                                          ([0, -1], ValueError), ([0, 1.0], TypeError)])
def test_invalid_table_is_atomic(values, error):
    circ = Circuit()
    address = circ.input("address", 1)
    target = circ.input("target", 2)
    before = list(circ.ops)
    with pytest.raises(error):
        add_lookup_inplace(circ, address, target, values, block=1)
    assert circ.ops == before


def test_aliased_target_rejected_before_allocation():
    circ = Circuit()
    address = circ.input("address", 2)
    before = list(circ.ops)
    with pytest.raises(ValueError, match="disjoint"):
        add_lookup_inplace(circ, address, Register([address[0]]), [0] * 4, block=1)
    assert circ.ops == before
