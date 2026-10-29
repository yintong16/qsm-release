"""Streaming audits must equal actual monolithic IR, not per-stage estimates."""

from copy import copy
from dataclasses import FrozenInstanceError, asdict, replace
import itertools
import json
import random
from types import SimpleNamespace

import pytest

from qsm.audit.streamed_ir import BasisAccumulator, FeedbackPhaseOp, ResourceAccumulator
from qsm.core import ops
from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import gidney_add


def _compact_circuit():
    c = Circuit()
    q = c.input("q", 4)
    one, zero = c.classical_bit(1), c.classical_bit(0)
    scratch = c.alloc(3)
    c.xor_word(scratch, 0b101, ctrl=q[0])
    measured = tuple(c.measure(wire) for wire in scratch)
    parity = c.classical_parities(measured, (0b101, 0b011, 0))
    with c.condition(parity):
        c.z(q[0])
    with c.condition(one):
        c.xor_word(q[1:], 0b101, ctrl=q[0])
    with c.condition(parity + 1):
        c.cz(q[1], q[2])
    c.ccx(q[0], q[1], q[3])
    with c.condition(zero):
        c.xor_word(q, 15)
    # Recycled IDs preserve timing dependencies, but ALLOC resets their state.
    replacement = c.alloc(3)
    c.x(replacement[0])
    with c.condition(zero):
        disabled = c.measure(replacement[0])
    recycled = c.alloc(1)
    assert recycled[0] == replacement[0]
    chained = c.classical_parities((one, zero, one, disabled), (0, 1, 3, 5, 7, 8))
    c.classical_parities(tuple(range(chained, chained + 6)), (0b111111,))
    c.classical_parities((), (0, 0))
    c.classical_parities((), ())
    c.free(recycled)
    c.free(replacement[1:])
    return c


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield iter(items[start:start + size])


@pytest.mark.parametrize("size", [1, 2, 7, 10000])
def test_compact_counts_and_basis_match_reference_at_chunk_boundaries(size):
    c = _compact_circuit()
    counter = ResourceAccumulator(digest=True)
    for chunk in _chunks(c.ops, size):
        assert counter.consume(chunk) is counter
    assert counter.resources == count(c)
    assert counter.snapshot()["op_nodes"] == len(c.ops)
    assert counter.live_qubits == c._live
    assert counter.snapshot()["phase_rotations"] == counter.snapshot()["hadamards"] == 0
    assert len(counter.digest) == 64
    assert counter.snapshot() == ResourceAccumulator(digest=True).consume(c.ops).snapshot()
    json.dumps(counter.snapshot())
    for value in (0, 1, 3, 11, 15):
        for outcomes in itertools.product((0, 1), repeat=3):
            inputs = {"q": value}
            basis = BasisAccumulator(inputs, measurements=iter(outcomes))
            for chunk in _chunks(c.ops, size):
                basis.consume(chunk, constants=c._cbit_const)
            assert basis.snapshot(c.inputs) == simulate(c, inputs, measurements=outcomes)
            assert basis.executed_measurements == 3  # disabled measurement draws nothing


def test_every_cut_and_every_prefix_keep_depth_lifetimes_and_phase():
    c = _compact_circuit()
    initial = {"q": 11}
    outcomes = [1, 0, 1]
    for cut in range(len(c.ops) + 1):
        counter = ResourceAccumulator(digest=True).consume(c.ops[:cut])
        prefix = SimpleNamespace(ops=c.ops[:cut])
        assert counter.resources == count(prefix)
        counter.consume([]).consume(c.ops[cut:])
        assert counter.resources == count(c)
        basis = BasisAccumulator(initial, measurements=outcomes, constants=c._cbit_const)
        basis.consume(c.ops[:cut]).consume([]).consume(c.ops[cut:])
        assert basis.snapshot() == simulate(c, initial, measurements=outcomes)


@pytest.mark.parametrize("seed", range(4))
def test_real_gidney_adder_rng_and_explicit_measurements(seed):
    c = Circuit()
    a, b = c.input("a", 5), c.input("b", 5)
    gidney_add(c, a, b)
    gidney_add(c, a, b)
    assert count(c).measurements > 0
    inputs = {"a": 23, "b": 19}
    expected = simulate(c, inputs, rng=random.Random(seed))
    for measurements in (None, expected.measurements):
        basis = BasisAccumulator(inputs, measurements, random.Random(seed))
        counter = ResourceAccumulator()
        for chunk in _chunks(c.ops, 11):
            chunk = tuple(chunk)
            basis.consume(chunk, c._cbit_const)
            counter.consume(chunk)
        assert counter.resources == count(c)
        assert basis.snapshot() == expected
        assert basis.outputs() == {"a": 23, "b": (19 + 46) % 32}
        assert basis.phase == 0


def test_persistent_allocator_can_flush_ops_and_sidecars_each_stage():
    c = Circuit()
    a, b = c.input("a", 3), c.input("b", 3)
    counter = ResourceAccumulator(digest=True)
    basis = BasisAccumulator({"a": 5, "b": 2}, rng=random.Random(17))
    archived = []
    for stage in range(3):
        constant = c.classical_bit(stage & 1)
        with c.condition(constant):
            c.x(b[0])
        gidney_add(c, a, b)
        archived.extend(c.ops)
        counter.consume(c.ops)
        basis.consume(c.ops, constants=c._cbit_const)
        # Real flushes, with no allocator/reset between windows.
        c.ops.clear()
        c.exact_blocks.clear()
        c._exact_block_seals.clear()
        basis.reset_classical(c._cbit_const)
        assert counter.live_qubits == c._live == 6
        assert basis.cbits == c._cbit_const
    full = copy(c)
    full.ops = archived
    assert counter.resources == count(full)
    assert counter.digest == ResourceAccumulator(digest=True).consume(archived).digest
    assert basis.snapshot(c.inputs) == simulate(full, {"a": 5, "b": 2}, rng=random.Random(17))


@pytest.mark.parametrize("size", [1, 9, 10000])
def test_optional_depth_omission_keeps_all_other_exact_counts(size):
    c = _compact_circuit()
    c.ops.extend([ops.Op("H", (0,)), FeedbackPhaseOp((0,), (0, 1), 2)])
    full = ResourceAccumulator(digest=True).consume(c.ops).snapshot()
    no_depth = ResourceAccumulator(digest=True, track_depth=False)
    for chunk in _chunks(c.ops, size):
        no_depth.consume(chunk)
    expected = dict(full, depth=None, toffoli_depth=None, depth_computed=False)
    assert no_depth.snapshot() == expected
    assert no_depth._qtime == no_depth._qtime_tof == {}
    with pytest.raises(ValueError, match="depth was not computed"):
        _ = no_depth.resources
    json.dumps(no_depth.snapshot())


def test_depth_disabled_does_not_expand_compact_targets():
    class UnindexedTargets(tuple):
        def __getitem__(self, index):
            raise AssertionError("count-only mode must never walk selected targets")

    targets = UnindexedTargets(range(2048))
    stream = [ops.Op(ops.ALLOC, targets), ops.XorWordOp(targets, (1 << 2048) - 1)]
    counts = ResourceAccumulator(track_depth=False).consume(stream).snapshot()
    assert counts["clifford"] == counts["n_gates"] == 2048
    assert counts["volume"] == 2048 * 2048
    assert counts["depth"] is None


def test_feedback_phase_and_h_are_separately_counted_unsynthesized_gates():
    c = Circuit()
    q = c.input("q", 3)
    c.ccx(*q)
    c.ops.append(ops.Op("H", (q[0],)))
    phase = FeedbackPhaseOp((q[0],), (10, 12), 3)
    c.ops.append(phase)
    c.measure(q[0])
    recycled = c.alloc(1)
    assert recycled[0] == q[0]
    c.x(recycled[0])
    counter = ResourceAccumulator(digest=True)
    for op in c.ops:
        counter.consume([op])
    assert asdict(counter.resources) == {
        "toffoli": 1, "clifford": 2, "measurements": 1,
        "peak_qubits": 3, "depth": 5, "n_gates": 5,
        "toffoli_depth": 1, "volume": 15, "toffoli_volume": 3,
        "circuit_volume": 3,
    }
    assert counter.snapshot()["phase_rotations"] == 1
    assert counter.snapshot()["hadamards"] == 1
    assert counter.live_qubits == 3
    with pytest.raises(FrozenInstanceError):
        phase.denominator_power = 5
    # The core counter already includes unknown nodes in depth/volume, but not
    # the new classification; only its Clifford count needs extending for H.
    assert counter.resources == replace(count(c), clifford=count(c).clifford + 1)


@pytest.mark.parametrize("kwargs", [
    {"qubits": (0, 1)}, {"qubits": [0]}, {"sources": [1]},
    {"sources": (-1,)}, {"denominator_power": 0},
    {"denominator_power": True}, {"denominator_power": 1.5},
])
def test_feedback_record_validates_immutable_payloads(kwargs):
    arguments = dict(qubits=(0,), sources=(1,), denominator_power=2)
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        FeedbackPhaseOp(**arguments)


@pytest.mark.parametrize("op", [
    ops.Op("H", (0,)), ops.Op("H", (0,), cond=((42, 1),)),
    FeedbackPhaseOp((0,), (1,), 2), ops.Op("UNKNOWN", (0,)),
])
def test_basis_never_claims_to_simulate_h_or_feedback_even_if_disabled(op):
    basis = BasisAccumulator().consume([ops.Op(ops.ALLOC, (0,))])
    with pytest.raises(ValueError, match="arithmetic basis simulator"):
        basis.consume([op])
    assert basis.phase == 0 and basis.state == bytearray([0])


def test_live_diagnostic_views_and_dynamic_named_inputs_and_rebind():
    c = Circuit()
    retained = c.input("retained", 2)
    basis = BasisAccumulator({"retained": 2}).consume(c.ops)
    c.ops.clear()
    control = c.alloc(3)
    basis.consume(c.ops)
    c.ops.clear()
    basis.load(control, 5)
    assert basis.read(control) == 5 and basis.read(retained) == 2
    c.cx(control[0], retained[0])
    basis.consume(c.ops)
    c.ops.clear()
    assert basis.read(retained) == 3 and basis.read(control) == 5
    # The caller skips QFT measurement simulation, retiring its arithmetic view
    # explicitly; Circuit's allocator still sees actual physical consumption.
    for q in control:
        c.measure(q)
    c.ops.clear()
    basis.discard(control)
    with pytest.raises(ValueError, match="not live"):
        basis.read(control)
    later = c.input("later", 4)
    basis.consume(c.ops, inputs={"later": 9})
    assert basis.read(later) == 9 and len(basis.state) == c._next_qid
    c.rebind("later", later[::-1])
    assert basis.outputs(c.inputs) == {"retained": 3, "later": 9}
    basis.load(later, 3)
    assert basis.outputs(c.inputs)["later"] == 12
    assert basis.outputs()["later"] == 3  # initial ALLOC view, by design
    for value in (-1, 16):
        with pytest.raises(ValueError, match="fit"):
            basis.load(later, value)
    assert basis.read(later) == 3


def test_classical_cleanup_is_explicit_and_preserves_phase_rng_and_cursor():
    basis = BasisAccumulator(measurements=[1, 0], constants={9: 1})
    basis.consume([ops.Op(ops.ALLOC, (0,)), ops.Op(ops.X, (0,)),
                   ops.Op(ops.MEASURE, (0,), cbit=10)])
    assert basis.phase == 1 and basis.cbits == {9: 1, 10: 1}
    basis.clear_classical(constants={9: 1})
    assert basis.cbits == {9: 1} and basis.phase == 1
    basis.consume_constants({11: 3})
    basis.consume([ops.Op(ops.ALLOC, (0,)), ops.Op(ops.MEASURE, (0,), cbit=12)])
    assert basis.measurements == [1, 0] and basis.phase == 1
    assert basis.cbits == {9: 1, 11: 1, 12: 0}
    basis.reset_classical()
    assert basis.cbits == {} and basis.phase == 1
    assert basis.executed_measurements == 2


def test_measurement_history_can_be_bounded_and_constants_updated_on_empty_chunk():
    basis = BasisAccumulator(measurements=[1], record_measurements=False)
    basis.consume([], constants={4: 1})
    basis.consume([ops.Op(ops.ALLOC, (0,)), ops.Op(ops.MEASURE, (0,), cbit=5)])
    assert basis.executed_measurements == 1
    assert basis.measurements == basis.snapshot().measurements == []
    assert basis.cbits == {4: 1, 5: 1}


def test_errors_expose_dirty_ancillas_missing_inputs_sources_and_outcomes():
    c = Circuit()
    q = c.alloc(1)
    c.x(q[0])
    c.free(q)
    basis = BasisAccumulator()
    basis.consume(c.ops[:-1])
    with pytest.raises(DirtyAncillaError, match="holding"):
        basis.consume(c.ops[-1:])
    assert basis.read(q) == 1
    with pytest.raises(DirtyAncillaError):
        simulate(c, {})
    with pytest.raises(KeyError):
        BasisAccumulator().consume([ops.Op(ops.ALLOC, (0,), name="missing", is_input=True)])
    with pytest.raises(KeyError):
        BasisAccumulator().consume([ops.ClassicalParityOp((999,), (1,), 1000)])
    with pytest.raises(IndexError, match="exhausted"):
        BasisAccumulator(measurements=[]).consume([
            ops.Op(ops.ALLOC, (0,)), ops.Op(ops.MEASURE, (0,), cbit=0),
        ])
    with pytest.raises(ValueError, match="unknown"):
        ResourceAccumulator().consume([ops.Op("UNKNOWN", (0,))])


def test_nonzero_phase_survives_chunk_cut_before_correction():
    c = Circuit()
    q = c.input("q", 1)
    ancilla = c.alloc(1)
    c.cx(q[0], ancilla[0])
    measured = c.measure(ancilla[0])
    split = len(c.ops)
    parity = c.classical_parities((measured,), (1,))
    with c.condition(parity):
        c.z(q[0])
    basis = BasisAccumulator({"q": 1}, measurements=[1])
    basis.consume(c.ops[:split])
    assert basis.phase == 1
    basis.consume(c.ops[split:])
    assert basis.phase == 0
    broken = copy(c)
    broken.ops = c.ops[:-1]
    assert BasisAccumulator({"q": 1}, measurements=[1]).consume(broken.ops).snapshot() == simulate(
        broken, {"q": 1}, measurements=[1],
    )
    assert simulate(broken, {"q": 1}, measurements=[1]).phase == 1


@pytest.mark.parametrize("original,changed", [
    (ops.Op(ops.ALLOC, (0,)), ops.Op(ops.ALLOC, (0,), name="named")),
    (ops.Op(ops.ALLOC, (0,)), ops.Op(ops.ALLOC, (0,), is_input=True)),
    (ops.Op(ops.X, (0,)), ops.Op(ops.X, (0,), cond=((9, 1),))),
    (ops.Op(ops.X, (0,)), ops.Op(ops.X, (1,))),
    (ops.Op(ops.MEASURE, (0,), cbit=0), ops.Op(ops.MEASURE, (0,), cbit=1)),
    (ops.XorWordOp((0, 1), 1), ops.XorWordOp((0, 1), 2)),
    (ops.XorWordOp((0, 1), 1), ops.XorWordOp((0, 1), 1, 2)),
    (ops.ClassicalParityOp((1,), (0, 1), 2), ops.ClassicalParityOp((0,), (0, 1), 2)),
    (ops.ClassicalParityOp((1,), (0, 1), 2), ops.ClassicalParityOp((1,), (1, 0), 2)),
    (ops.ClassicalParityOp((1,), (0, 1), 2), ops.ClassicalParityOp((1,), (0, 1), 3)),
    (FeedbackPhaseOp((0,), (1,), 2), FeedbackPhaseOp((0,), (1,), 3)),
    (FeedbackPhaseOp((0,), (1,), 2), FeedbackPhaseOp((0,), (2,), 2)),
])
def test_digest_covers_all_actual_fields_and_compact_payloads(original, changed):
    before = ResourceAccumulator(digest=True).consume([original]).digest
    assert before != ResourceAccumulator(digest=True).consume([changed]).digest
    assert before == ResourceAccumulator(digest=True).consume([replace(original)]).digest


def test_digest_is_order_sensitive_chunk_independent_and_cache_bounded():
    stream = [ops.ClassicalParityOp((0,), (i,), i + 1) for i in range(200)]
    expected = ResourceAccumulator(digest=True).consume(stream)
    actual = ResourceAccumulator(digest=True)
    for op in stream:
        actual.consume([op]).consume([])
    assert actual.digest == expected.digest
    assert actual.digest != ResourceAccumulator(digest=True).consume(reversed(stream)).digest
    assert len(actual._digest._tuples) <= 128
    assert ResourceAccumulator().snapshot()["digest"] is None
    assert ResourceAccumulator().consume(stream).resources == count(SimpleNamespace(ops=[]))


def test_inherited_signed_arithmetic_phase_witness_is_not_reset_or_hidden():
    from qsm.audit.signed_chain import _exact_config, group_add, group_mul
    from qsm.scalarmul.signed_window import make_signed_window_table, add_signed_window_inplace

    table = make_signed_window_table(1, group_mul(4, (0, 2), 127, 3), p=127, a=3, b=4)
    c = Circuit()
    j, s = c.input("j", 1), c.input("s", 1)
    x, y = c.input("x", 7), c.input("y", 7)
    add_signed_window_inplace(c, x, y, j, s, table, config=_exact_config(127, 1),
                              const_kw={}, sqr_kw={})
    row = {"j": 0, "s": 1, "x": 70, "y": 106}
    expected_point = group_add((70, 106), table.operand(0, 1), 127, 3)
    counter = ResourceAccumulator()
    for chunk in _chunks(c.ops, 113):
        counter.consume(chunk)
    assert counter.resources == count(c)
    observed = []
    for seed in (0, 1):
        basis = BasisAccumulator(row, rng=random.Random(seed), constants=c._cbit_const)
        for chunk in _chunks(c.ops, 113):
            basis.consume(chunk)
        assert basis.snapshot(c.inputs) == simulate(c, row, rng=random.Random(seed))
        assert basis.outputs(c.inputs) == dict(row, x=expected_point[0], y=expected_point[1])
        observed.append(basis.phase)
    assert observed == [1, 0]  # passing diagnostics must not relabel this phase-clean


def test_small_signed_lookup_executes_actual_ops_without_semantic_sidecars():
    from qsm.scalarmul.local_signed import build_first_lookup_case, make_first_lookup_table

    table = make_first_lookup_table(2, (1, 2), order=13, p=7, a=0, b=3, signed=True)
    case = build_first_lookup_case(2, signed=True, table=table)
    c = case.circuit
    counter = ResourceAccumulator()
    for chunk in _chunks(c.ops, 13):
        counter.consume(chunk)
    assert counter.resources == count(c)
    for word in range(4):
        row = {"word": word, "x": 0, "y": 0}
        for seed in range(3):
            basis = BasisAccumulator(row, rng=random.Random(seed), constants=c._cbit_const)
            for chunk in _chunks(c.ops, 13):
                basis.consume(chunk)
            assert basis.snapshot(c.inputs) == simulate(c, row, rng=random.Random(seed))
            assert basis.outputs(c.inputs) == case.reference(row)
            assert basis.phase == 0
