"""Compact IR is exactly its physical expansion, not a unit-cost shortcut."""

from contextlib import ExitStack
from copy import copy
from dataclasses import FrozenInstanceError, fields
import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import Resources, count
from qsm.core.simulator import simulate


def _expanded(circuit):
    """Independent test oracle; production code must never build this flat list."""
    flat = copy(circuit)
    flat.ops = []
    for op in circuit.ops:
        if op.kind != ops.XOR_WORD:
            flat.ops.append(op)
            continue
        for i, target in enumerate(op.qubits):
            if op.mask & (1 << i):
                qubits = (target,) if op.ctrl is None else (op.ctrl, target)
                kind = ops.X if op.ctrl is None else ops.CX
                flat.ops.append(ops.Op(kind, qubits, cond=op.cond))
    return flat


def _snapshot(c):
    return (
        tuple(c.ops), c._next_qid, tuple(c._free_qids), frozenset(c._live_qids),
        c._live, c._peak, c._next_cbit, dict(c._cbit_const), tuple(c._cond_stack),
        tuple((name, tuple(reg)) for name, reg in c.inputs.items()),
    )


@pytest.mark.parametrize("width", range(5))
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("conditions", [(), ((0, 1),), ((0, 0),),
                                         ((0, 1), (1, 0)), ((0, 1), (0, 0))])
def test_fanout_exhaustive(width, controlled, conditions):
    for mask in range(1 << width):
        c = Circuit()
        target = c.input("target", width)
        control = c.input("control", 1)[0]
        c.classical_bit(1)
        c.classical_bit(0)
        with ExitStack() as stack:
            for cb, value in conditions:
                stack.enter_context(c.condition(cb, value))
            c.xor_word(target, mask, ctrl=control if controlled else None)
        flat = _expanded(c)
        assert count(c) == count(flat)  # all ten resource metrics, not just CCX
        fired = all((1, 0)[cb] == value for cb, value in conditions)
        for value in range(1 << width):
            for enabled in (0, 1):
                inputs = {"target": value, "control": enabled}
                result = simulate(c, inputs)
                expected = value ^ (mask if fired and (not controlled or enabled) else 0)
                assert result.outputs == {"target": expected, "control": enabled}
                assert result.phase == 0
                assert result == simulate(flat, inputs)


def test_shared_immutable_payloads_and_old_op_shape():
    c = Circuit()
    q = c.input("q", 4)
    targets = tuple(q[:3])
    for mask in range(1, 8):
        c.xor_word(targets, mask, ctrl=q[3])
    fanouts = c.ops[1:]
    assert len(fanouts) == 7
    assert all(op.qubits is targets for op in fanouts)
    assert all(isinstance(op, ops.XorWordOp) for op in fanouts)
    assert all(op.kind == ops.XOR_WORD for op in fanouts)
    with pytest.raises(FrozenInstanceError):
        fanouts[0].mask = 0
    assert [field.name for field in fields(ops.Op)] == [
        "kind", "qubits", "cbit", "cond", "name", "is_input",
    ]
    assert not hasattr(ops.Op(ops.X, (0,)), "mask")
    view = q[:2]
    c.xor_word(view, 1)
    view.ids.reverse()
    assert c.ops[-1].qubits == (q[0], q[1])  # Register snapshots do not alias lists


def test_lsb_order_and_toffoli_dependency_propagation():
    c = Circuit()
    q = c.input("q", 7)
    c.ccx(q[3], q[4], q[2])
    c.ccx(q[3], q[4], q[2])
    c.xor_word(tuple(q[:3]), 7, ctrl=q[5])
    c.ccx(q[0], q[1], q[6])
    expected = Resources(
        toffoli=3, clifford=3, measurements=0, peak_qubits=7, depth=3,
        n_gates=6, toffoli_depth=2, volume=42, toffoli_volume=21, circuit_volume=21,
    )
    assert count(c) == expected == count(_expanded(c))
    # Physical order is target-tuple order, not numeric wire-ID order.
    reverse = Circuit()
    q = reverse.input("q", 7)
    reverse.ccx(q[3], q[4], q[2])
    reverse.ccx(q[3], q[4], q[2])
    reverse.xor_word((q[2], q[1], q[0]), 7, ctrl=q[5])
    reverse.ccx(q[0], q[1], q[6])
    assert count(reverse) == count(_expanded(reverse))
    assert count(reverse).depth == 6
    assert count(reverse).toffoli_depth == 3


@pytest.mark.parametrize("seed", range(20))
def test_random_interleavings_measurement_recycling_and_live_volume(seed):
    rng = random.Random(seed)
    c = Circuit()
    q = c.input("q", 8)
    scratch = []
    cbits = [c.classical_bit(1), c.classical_bit(0)]
    for _ in range(80):
        live = list(q) + scratch
        action = rng.randrange(7)
        if action == 0:
            width = rng.randrange(1, min(6, len(live)) + 1)
            targets = tuple(rng.sample(live, width))
            outside = [wire for wire in live if wire not in targets]
            ctrl = rng.choice(outside) if rng.getrandbits(1) else None
            mask = rng.randrange(1 << width)
            with c.condition(rng.choice(cbits), rng.getrandbits(1)):
                c.xor_word(targets, mask, ctrl=ctrl)
        elif action == 1:
            c.ccx(*rng.sample(live, 3))
        elif action == 2:
            with c.condition(rng.choice(cbits), rng.getrandbits(1)):
                c.cz(*rng.sample(live, 2))
                c.z(rng.choice(live))
        elif action == 3 and len(scratch) < 12:
            scratch.extend(c.alloc(rng.randrange(1, 4)))
        elif action == 4 and scratch:
            measured = scratch.pop(rng.randrange(len(scratch)))
            cbits.append(c.measure(measured))
            replacement = c.alloc(1)
            assert replacement[0] == measured
            scratch.extend(replacement)
        elif action == 5:
            clean = c.alloc(3)
            mask = rng.randrange(8)
            c.xor_word(clean, mask, ctrl=q[0])
            c.xor_word(clean, mask, ctrl=q[0])
            c.free(clean)
        else:
            sources = tuple(rng.sample(cbits, min(4, len(cbits))))
            words = tuple(rng.randrange(1 << len(sources)) for _ in range(3))
            base = c.classical_parities(sources, words)
            cbits.extend(range(base, base + len(words)))
    for wire in scratch:
        c.measure(wire)
    assert c._live_qids == set(q)
    flat = _expanded(c)
    assert count(c) == count(flat)
    assert len(fields(count(c))) == 10
    assert count(c).peak_qubits == c.peak_qubits
    for _ in range(8):
        inputs = {"q": rng.randrange(256)}
        outcomes = [rng.getrandbits(1) for _ in range(count(c).measurements)]
        assert simulate(c, inputs, measurements=outcomes) == simulate(
            flat, inputs, measurements=outcomes,
        )


def test_zero_masks_emit_nothing_and_validate_empty_targets():
    c = Circuit()
    q = c.input("q", 3)
    before = _snapshot(c)
    for target in (q, tuple(q), (), Register(())):
        c.xor_word(target, 0)
    c.xor_word((), 0, ctrl=q[0])
    assert _snapshot(c) == before
    with pytest.raises(ValueError):
        c.xor_word((), 1)
    assert _snapshot(c) == before


@pytest.mark.parametrize("target,mask,ctrl,error", [
    ([0, 1], 1, None, TypeError),
    (None, 0, None, TypeError),
    ((0, 0), 1, None, ValueError),
    ((0, 0), 0, None, ValueError),
    ((0, 1), -1, None, ValueError),
    ((0, 1), 4, None, ValueError),
    ((0, 1), True, None, TypeError),
    ((0, 1), 1.0, None, TypeError),
    ((0, 1), "1", None, TypeError),
    ((0, 1), None, None, TypeError),
    ((0, 1), 0, 0, ValueError),
    ((0, 1), 1, 1, ValueError),  # control overlaps even an unselected target
    ((0, 1), 1, True, TypeError),
    ((0, 1), 0, False, TypeError),
    ((0, 1), 1, 1.0, TypeError),
    ((0, 1), 1, "2", TypeError),
] + [((0, wire), mask, None, error)
     for wire, error in [(-1, ValueError), (3, ValueError), (4, ValueError),
                         (5, ValueError), (True, TypeError), (1.0, TypeError)]
     for mask in (0, 1)]
  + [((0, 1), mask, ctrl, ValueError)
     for ctrl in (-1, 3, 4, 5) for mask in (0, 1)])
def test_invalid_fanouts_rejected_atomically(target, mask, ctrl, error):
    c = Circuit()
    c.input("q", 3)
    freed = c.alloc(1)
    measured = c.alloc(1)
    c.free(freed)
    cb = c.measure(measured[0])
    with c.condition(cb):
        before = _snapshot(c)
        with pytest.raises(error):
            c.xor_word(target, mask, ctrl=ctrl)
        assert _snapshot(c) == before


@pytest.mark.parametrize("width", range(5))
def test_parity_truth_tables_and_runtime_reevaluation(width):
    c = Circuit()
    c.classical_bit(1)  # offsets must not assume measurement ids start at zero
    result = c.input("result", 1 << width)
    scratch = c.alloc(width)
    sources = tuple(c.measure(wire) for wire in scratch)
    words = tuple(range(1 << width))
    base = c.classical_parities(sources, words)
    op = c.ops[-1]
    assert op.sources is sources and op.values is words
    assert op.kind == ops.CPARITY and op.base == base
    assert op.qubits == () and op.cond == ()
    assert c._next_cbit == base + len(words)
    assert c.classical_bit(0) == base + len(words)
    with pytest.raises(FrozenInstanceError):
        op.base = 100
    for j, wire in enumerate(result):
        with c.condition(base + j):
            c.x(wire)
    for packed in list(range(1 << width)) + list(reversed(range(1 << width))):
        outcomes = [(packed >> i) & 1 for i in range(width)]
        expected = sum(((packed & word).bit_count() & 1) << j
                       for j, word in enumerate(words))
        got = simulate(c, {"result": 0}, measurements=outcomes)
        assert got.outputs == {"result": expected}
        assert got.measurements == outcomes and got.phase == 0
    for seed in range(8):
        got = simulate(c, {"result": 0}, rng=random.Random(seed))
        packed = sum(bit << i for i, bit in enumerate(got.measurements))
        expected = sum(((packed & word).bit_count() & 1) << j
                       for j, word in enumerate(words))
        assert got.outputs["result"] == expected


@pytest.mark.parametrize("mask", range(8))
def test_parity_feedforward_cancels_measured_fanout_phase(mask):
    c = Circuit()
    control = c.input("control", 1)[0]
    scratch = c.alloc(3)
    c.xor_word(scratch, mask, ctrl=control)
    sources = tuple(c.measure(wire) for wire in scratch)
    base = c.classical_parities(sources, (mask,))
    with c.condition(base):
        c.z(control)
    flat = _expanded(c)
    assert count(c) == count(flat)
    for value in (0, 1):
        for packed in range(8):
            measurements = [(packed >> i) & 1 for i in range(3)]
            got = simulate(c, {"control": value}, measurements=measurements)
            assert got.outputs == {"control": value} and got.phase == 0
            assert got == simulate(flat, {"control": value}, measurements=measurements)
    if mask:
        incorrect = copy(c)
        incorrect.ops = c.ops[:-1]
        bit = (mask & -mask).bit_length() - 1
        measurements = [int(i == bit) for i in range(3)]
        assert simulate(incorrect, {"control": 1}, measurements=measurements).phase == 1


def test_classical_only_parities_constants_duplicates_chaining_and_empty_tables():
    c = Circuit()
    one, zero = c.classical_bit(1), c.classical_bit(0)
    before = count(c)
    first = c.classical_parities((one, zero, one), (0, 1, 3, 5, 7))
    second = c.classical_parities(tuple(range(first, first + 5)), (0b11111, 0b00100))
    empty_base = c.classical_parities((), ())
    assert empty_base == c._next_cbit
    zeros = c.classical_parities((), (0, 0))
    assert count(c) == before
    target = c.input("target", 4)
    for bit, cb in enumerate((second, second + 1, zeros, zeros + 1)):
        with c.condition(cb):
            c.x(target[bit])
    assert simulate(c, {"target": 0}).outputs == {"target": 2}
    assert simulate(c, {"target": 0}).measurements == []


@pytest.mark.parametrize("sources,values,error", [
    ([0], (1,), TypeError), ((0,), [1], TypeError),
    ((True,), (1,), TypeError), ((0.0,), (1,), TypeError),
    (("0",), (1,), TypeError), ((-1,), (1,), ValueError),
    ((2,), (1,), ValueError), ((99,), (), ValueError),
    ((0,), (-1,), ValueError), ((0,), (2,), ValueError),
    ((0,), (True,), TypeError), ((0,), (1.0,), TypeError),
    ((0,), ("1",), TypeError), ((0,), (0, None), TypeError),
    ((), (1,), ValueError), ((0,), (0, 1, 2), ValueError),
])
def test_invalid_parities_rejected_atomically(sources, values, error):
    c = Circuit()
    c.classical_bit(1)
    c.measure(c.alloc(1)[0])
    before = _snapshot(c)
    with pytest.raises(error):
        c.classical_parities(sources, values)
    assert _snapshot(c) == before


def test_parities_reject_outer_conditions_atomically_even_for_empty_tables():
    c = Circuit()
    cb = c.classical_bit(1)
    with c.condition(cb):
        before = _snapshot(c)
        for sources, values in [((cb,), (1,)), ((), ())]:
            with pytest.raises(ValueError, match="condition"):
                c.classical_parities(sources, values)
            assert _snapshot(c) == before


def test_existing_conditional_measurement_semantics_unchanged():
    c = Circuit()
    target = c.input("target", 1)
    scratch = c.alloc(1)
    c.x(scratch[0])
    disabled = c.classical_bit(0)
    with c.condition(disabled):
        source = c.measure(scratch[0])
    assert scratch[0] not in c._live_qids  # old unconditional allocator consumption
    replacement = c.alloc(1)
    assert replacement[0] == scratch[0]
    base = c.classical_parities((source,), (1,))
    with c.condition(base):
        c.xor_word(target, 1)
    c.free(replacement)
    got = simulate(c, {"target": 0}, measurements=[])
    assert got.outputs == {"target": 0} and got.measurements == [] and got.phase == 0
    assert count(c).measurements == 1  # old emitted, not executed, quantum counts
    assert count(c) == count(_expanded(c))


def test_physical_gate_counts_helper():
    for kind in ops.GATE_KINDS:
        assert ops.physical_gate_counts(ops.Op(kind, (0,))) == {kind: 1}
    assert ops.physical_gate_counts(ops.Op(ops.ALLOC, (0,))) == {}
    assert ops.physical_gate_counts(ops.Op(ops.FREE, (0,))) == {}
    assert ops.physical_gate_counts(ops.XorWordOp((0, 1, 2), 5)) == {ops.X: 2}
    assert ops.physical_gate_counts(ops.XorWordOp((0, 1, 2), 7, 3)) == {ops.CX: 3}
    assert ops.physical_gate_counts(ops.XorWordOp((), 0)) == {}
    assert ops.physical_gate_counts(ops.ClassicalParityOp((0,), (0, 1), 1)) == {}
