"""Small exhaustive permutations plus wide-word tests; no point-circuit builds."""

import itertools
import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.scalarmul.lookup import lookup_xor, xor_nonzero


def _build(w, n, values, repetitions=1):
    circ = Circuit()
    address = circ.input("address", w)
    target = circ.input("target", n)
    for _ in range(repetitions):
        lookup_xor(circ, address, target, values)
    return circ


def _assert_unitary(circ):
    assert all(op.kind in {ops.ALLOC, ops.FREE, ops.X, ops.CX, ops.CCX}
               for op in circ.ops)
    assert all(not op.cond for op in circ.ops)
    assert circ._live == sum(len(reg) for reg in circ.inputs.values())


@pytest.mark.parametrize("w,n", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)])
def test_lookup_all_small_tables_inputs_and_targets(w, n):
    # This exhausts the TABLE too, not just a few handpicked lookup inputs.
    for values in itertools.product(range(1 << n), repeat=1 << w):
        circ = _build(w, n, values)
        _assert_unitary(circ)
        assert count(circ).peak_qubits <= w + n + w - 1
        for address in range(1 << w):
            for target in range(1 << n):
                result = simulate(circ, {"address": address, "target": target})
                assert result.outputs == {
                    "address": address, "target": target ^ values[address]
                }
                assert result.phase == 0
                assert result.measurements == []


@pytest.mark.parametrize("values", [
    (0,) * 8,
    (7,) * 8,
    (0, 0, 0, 0, 7, 7, 7, 7),
    (0, 1, 0, 1, 2, 3, 2, 3),
    (7, 3, 1, 0, 0, 1, 3, 7),
])
def test_lookup_is_an_involution_for_arbitrary_target(values):
    circ = _build(3, 3, values, repetitions=2)
    _assert_unitary(circ)
    for address, target in itertools.product(range(8), repeat=2):
        inputs = {"address": address, "target": target}
        result = simulate(circ, inputs)
        assert result.outputs == inputs
        assert result.phase == 0


def test_constant_columns_and_zero_subtrees_have_no_decoder_cost():
    zero = count(_build(4, 9, [0] * 16))
    uniform = count(_build(4, 9, [0b101010101] * 16))
    # Constant columns hoist to five X gates; an all-zero subtree emits nothing.
    assert zero.n_gates == 0
    assert uniform.n_gates == 5
    assert uniform.peak_qubits == 13
    # The low half contributes only the global column; the other two columns
    # depend on the MSB alone. No prefix ANDs should be generated.
    structured = _build(4, 3, [1] * 8 + [7] * 8)
    resource = count(structured)
    assert resource.toffoli == 0
    assert resource.n_gates == 3
    assert resource.peak_qubits == 7
    for address in range(16):
        assert simulate(structured, {"address": address, "target": 0}).outputs == {
            "address": address, "target": 1 if address < 8 else 7
        }


@pytest.mark.parametrize("w,n", [(2, 64), (4, 255), (5, 256)])
def test_lookup_wide_random_words(w, n):
    rng = random.Random(913 + n)
    values = tuple(rng.getrandbits(n) for _ in range(1 << w))
    circ = _build(w, n, values)
    _assert_unitary(circ)
    for address in range(1 << w):
        for target in [0, (1 << n) - 1, rng.getrandbits(n)]:
            result = simulate(circ, {"address": address, "target": target})
            assert result.outputs == {
                "address": address, "target": target ^ values[address]
            }
            assert result.phase == 0


def test_lookup_handles_noncontiguous_reordered_register_views():
    circ = Circuit()
    storage = circ.input("storage", 7)
    address = Register([storage[5], storage[1]])
    target = Register([storage[6], storage[0], storage[3]])
    values = (2, 5, 3, 6)
    lookup_xor(circ, address, target, values)
    for word in range(128):
        index = ((word >> 5) & 1) | (((word >> 1) & 1) << 1)
        expected = word
        for i, bit in enumerate([6, 0, 3]):
            expected ^= ((values[index] >> i) & 1) << bit
        result = simulate(circ, {"storage": word})
        assert result.outputs["storage"] == expected
        assert result.phase == 0


@pytest.mark.parametrize("w", range(1, 8))
@pytest.mark.parametrize("repetitions", [1, 2])
def test_xor_nonzero_exhaustive(w, repetitions):
    circ = Circuit()
    address = circ.input("address", w)
    target = circ.input("target", 1)
    for _ in range(repetitions):
        xor_nonzero(circ, address, target[0])
    _assert_unitary(circ)
    assert count(circ).peak_qubits <= w + 1 + max(0, w - 2)
    for j, bit in itertools.product(range(1 << w), range(2)):
        result = simulate(circ, {"address": j, "target": bit})
        assert result.outputs == {
            "address": j, "target": bit ^ (bool(j) if repetitions == 1 else 0)
        }
        assert result.phase == 0
        assert result.measurements == []


@pytest.mark.parametrize("values,error", [
    ([], ValueError), ([0], ValueError), ([0] * 5, ValueError),
    ([0, 0, 0, -1], ValueError), ([0, 0, 0, 8], ValueError),
    ([0, 0, 0, 1.0], TypeError), ([0, 0, 0, True], TypeError),
    (None, TypeError), (iter([0] * 4), TypeError),
])
def test_malformed_tables_rejected_without_emitting(values, error):
    circ = Circuit()
    address = circ.input("address", 2)
    target = circ.input("target", 3)
    before = list(circ.ops)
    with pytest.raises(error):
        lookup_xor(circ, address, target, values)
    assert circ.ops == before


@pytest.mark.parametrize("address_ids,target_ids,error", [
    ([], [2], ValueError), ([0], [], ValueError),
    ([0, 0], [2], ValueError), ([0], [2, 2], ValueError),
    ([0, 1], [1, 2], ValueError), ([-1], [2], ValueError),
    ([0], [3], ValueError), ([0.0], [2], TypeError),
    ([0], [True], TypeError),
])
def test_malformed_registers_rejected_without_emitting(address_ids, target_ids, error):
    circ = Circuit()
    circ.input("storage", 3)
    before = list(circ.ops)
    with pytest.raises(error):
        lookup_xor(circ, Register(address_ids), Register(target_ids),
                   [0] * (1 << len(address_ids)))
    assert circ.ops == before


def test_freed_wires_and_wrong_register_types_rejected():
    circ = Circuit()
    address = circ.input("address", 1)
    target = circ.alloc(1)
    circ.free(target)
    before = list(circ.ops)
    with pytest.raises(ValueError, match="live"):
        lookup_xor(circ, address, target, [0, 1])
    with pytest.raises(TypeError, match="Registers"):
        lookup_xor(circ, address.ids, target, [0, 1])
    assert circ.ops == before


@pytest.mark.parametrize("address_ids,target,error", [
    ([], 2, ValueError), ([0, 0], 2, ValueError), ([0, 1], 1, ValueError),
    ([0], -1, ValueError), ([0], 3, ValueError), ([0], 2.0, TypeError),
])
def test_xor_nonzero_malformed_inputs(address_ids, target, error):
    circ = Circuit()
    circ.input("storage", 3)
    before = list(circ.ops)
    with pytest.raises(error):
        xor_nonzero(circ, Register(address_ids), target)
    assert circ.ops == before
