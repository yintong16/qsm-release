"""Tiny emitted-IR checks; no full-width builds or sampled measurement claims."""
from dataclasses import replace
from itertools import product

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch, validate_plan_binding
from qsm.scalarmul.lookup import lookup_xor_unary, unlookup_measure


def _exhaustive(circ, inputs, expected=None):
    """Compare ALL tiny basis inputs and ALL measurement branches to raw IR."""
    count = sum(op.kind == ops.MEASURE for op in circ.ops)
    assert count <= 10, "this oracle must stay tiny"
    expected_phase = 0
    raw_outputs = []
    for lane, values in enumerate(inputs):
        branches = [simulate(circ, values, measurements=list(ms))
                    for ms in product((0, 1), repeat=count)]
        assert all(result.outputs == branches[0].outputs for result in branches)
        raw_outputs.append(branches[0].outputs)
        if any(result.phase for result in branches):
            expected_phase |= 1 << lane
    result = simulate_batch(compile_symbolic(circ), inputs,
                            raw_outputs if expected is None else expected)
    assert result.outputs == raw_outputs
    assert result.phase_mask == expected_phase
    assert result.dirty_mask == result.summary_mismatch_mask == 0
    return result


@pytest.mark.parametrize("corrections", [0, 1, 2, 3])
def test_temporary_and_all_basis_and_measurement_branches(corrections):
    c = Circuit()
    a = c.input("a", 1)[0]
    b = c.input("b", 1)[0]
    t = c.alloc(1)[0]
    c.ccx(a, b, t)
    m = c.measure(t)
    # A coefficient can temporarily be nonzero or zero, then change again.
    for _ in range(corrections):
        with c.condition(m):
            c.cz(a, b)
    inputs = [{"a": a, "b": b} for a, b in product((0, 1), repeat=2)]
    result = _exhaustive(c, inputs)
    assert result.bad_mask == (0 if corrections % 2 else 8)
    assert result.bad_count == (0 if corrections % 2 else 1)
    assert result.samples == 4


@pytest.mark.parametrize("cancel_constant", [False, True])
def test_condition_zero_has_constant_and_symbol_parts(cancel_constant):
    c = Circuit()
    x = c.input("x", 1)[0]
    t = c.alloc(1)[0]
    c.cx(x, t)
    m = c.measure(t)
    with c.condition(m, 0):
        c.z(x)
    if cancel_constant:
        c.z(x)
    result = _exhaustive(c, [{"x": 0}, {"x": 1}])
    assert result.constant_phase_mask == (0 if cancel_constant else 2)
    assert result.phase_mask == result.constant_phase_mask


def test_delayed_last_use_independent_symbols_and_duplicate_conditions():
    c = Circuit()
    x = c.input("x", 2)
    t = c.alloc(1)[0]
    c.cx(x[0], t)
    m = c.measure(t)
    u = c.alloc(1)[0]  # recycled id, a distinct measurement symbol
    c.cx(x[1], u)
    n = c.measure(u)
    with c.condition(m), c.condition(m):
        c.z(x[0])
    with c.condition(n, 0):
        c.z(x[1])
    c.z(x[1])
    # This false conjunction contributes nothing, and is not a nonlinear phase.
    with c.condition(m), c.condition(n), c.condition(m, 0):
        c.cz(x[0], x[1])
    # A later nonzero term then its cancellation must be allowed to finish.
    with c.condition(m):
        c.z(x[1])
    c.x(x[0])
    with c.condition(m):
        c.z(x[1])
    result = _exhaustive(c, [{"x": x} for x in range(4)])
    assert result.bad_count == 0


def test_constant_conditions_word_xor_and_padding_lanes():
    c = Circuit()
    x = c.input("x", 9)
    ctrl = c.input("ctrl", 1)[0]
    zero, one = c.classical_bit(0), c.classical_bit(1)
    with c.condition(zero, 0), c.condition(one):
        c.xor_word(x, 0b101100101, ctrl)
    with c.condition(one, 0):
        c.x(x[0])
    c.x(x[8])
    inputs = [{"x": (i * 31) % 512, "ctrl": i % 2} for i in range(11)]
    result = _exhaustive(c, inputs)
    assert result.bad_count == 0
    assert all(mask >> 11 == 0 for mask in (result.bad_mask, result.phase_mask,
                                           result.dirty_mask, result.value_mask))


def test_dirty_free_permanent_but_other_lanes_continue_and_ids_recycle():
    c = Circuit()
    x = c.input("x", 1)[0]
    t = c.alloc(1)
    c.cx(x, t[0])
    c.free(t)
    u = c.alloc(1)
    assert u.ids == t.ids
    c.x(u[0])
    c.x(u[0])
    c.free(u)
    c.x(x)
    result = simulate_batch(compile_symbolic(c), [{"x": 0}, {"x": 1}],
                            [{"x": 1}, {"x": 0}])
    assert result.outputs == [{"x": 1}, {"x": 0}]
    assert result.dirty_mask == result.bad_mask == 2
    assert result.good_count == 1


def test_final_live_nonoutput_ancilla_and_overlapping_hazards():
    c = Circuit()
    x = c.input("x", 1)[0]
    t = c.alloc(1)[0]
    c.cx(x, t)
    c.z(x)
    result = simulate_batch(compile_symbolic(c), [{"x": 0}, {"x": 1}],
                            [{"x": 0}, {"x": 0}])
    assert result.value_mask == result.dirty_mask == result.phase_mask == 2
    assert result.constant_phase_mask == 2
    assert result.bad_count == 1  # masks overlap; this is not three bad samples
    assert result.value_count == result.dirty_count == result.phase_count == 1


def test_original_alloc_inputs_not_final_rebinding_even_empty_intermediate():
    c = Circuit()
    x = c.input("x", 3)
    target = c.alloc(3)
    for a, b in zip(x, target):
        c.cx(a, b)
        c.cx(b, a)
    c.rebind("x", Register([]))
    c.free(x)
    c.rebind("x", target)
    plan = compile_symbolic(c)
    assert dict(plan.input_registers)["x"] == tuple(x)
    assert dict(plan.output_registers)["x"] == tuple(target)
    result = _exhaustive(c, [{"x": i} for i in range(8)])
    assert result.bad_count == 0


def test_reject_overlapping_named_output_views():
    c = Circuit()
    x = c.input("x", 2)
    c.input("y", 1)
    c.rebind("y", x[:1])
    with pytest.raises(ValueError, match="disjoint"):
        compile_symbolic(c)


def test_original_input_width_may_differ_from_final_output_width():
    c = Circuit()
    x = c.input("x", 3)
    c.rebind("x", x[:2])
    result = simulate_batch(compile_symbolic(c), [{"x": 1}, {"x": 5}],
                            [{"x": 1}, {"x": 1}])
    assert result.outputs == [{"x": 1}, {"x": 1}]
    assert result.dirty_mask == 2  # excluded high bit is now a live ancilla


def test_modular_value_and_noncanonical_are_distinct_raw_outputs_preserved():
    c = Circuit()
    c.input("x", 4)
    inputs = [{"x": x} for x in (0, 4, 5, 6, 15)]
    result = simulate_batch(compile_symbolic(c), inputs,
                            [{"x": row["x"] % 5} for row in inputs], output_moduli={"x": 5})
    assert result.value_mask == 0
    assert result.noncanonical_mask == result.bad_mask == 0b11100
    assert result.noncanonical_count == 3
    assert result.outputs == inputs
    exact = simulate_batch(compile_symbolic(c), inputs, [{"x": 0}] * len(inputs))
    assert exact.value_mask == 0b11110
    assert exact.noncanonical_mask == 0


@pytest.mark.parametrize("inputs,expected,moduli", [
    ([{}], [{"x": 0}], None),
    ([{"x": 0, "extra": 1}], [{"x": 0}], None),
    ([{"x": -1}], [{"x": 0}], None),
    ([{"x": 4}], [{"x": 0}], None),
    ([{"x": True}], [{"x": 0}], None),
    ([{"x": 0}], [], None),
    ([{"x": 0}], [{}], None),
    ([{"x": 0}], [{"x": 4}], None),
    ([{"x": 0}], [{"x": 3}], {"x": 3}),
    ([{"x": 0}], [{"x": 0}], {"missing": 3}),
    ([{"x": 0}], [{"x": 0}], {"x": 0}),
    ([{"x": 0}], [{"x": 0}], {"x": 5}),
])
def test_sample_validation(inputs, expected, moduli):
    c = Circuit()
    c.input("x", 2)
    with pytest.raises(ValueError):
        simulate_batch(compile_symbolic(c), inputs, expected, output_moduli=moduli)


def test_empty_batch_and_zero_width_register():
    c = Circuit()
    c.input("x", 0)
    plan = compile_symbolic(c)
    result = simulate_batch(plan, [], [])
    assert result.samples == result.bad_count == 0
    assert result.outputs == []
    assert simulate_batch(plan, [{"x": 0}], [{"x": 0}]).bad_count == 0


@pytest.mark.parametrize("gate", ["x", "cx", "ccx", "xor_word"])
def test_reject_measurement_controlled_value_gates(gate):
    c = Circuit()
    x = c.input("x", 3)
    m = c.measure(c.alloc(1)[0])
    with c.condition(m):
        if gate == "xor_word":
            c.xor_word(x, 1)
        else:
            getattr(c, gate)(*x.ids[:{"x": 1, "cx": 2, "ccx": 3}[gate]])
    with pytest.raises(ValueError, match="measurement-controlled value"):
        compile_symbolic(c)


def test_reject_conditional_measure_even_constant_true():
    c = Circuit()
    c.input("x", 1)
    t = c.alloc(1)[0]
    with c.condition(c.classical_bit(1)):
        c.measure(t)
    with pytest.raises(ValueError):
        compile_symbolic(c)


def test_reject_nonlinear_phase_and_unknown_producer():
    c = Circuit()
    x = c.input("x", 1)[0]
    m = c.measure(c.alloc(1)[0])
    n = c.measure(c.alloc(1)[0])
    with c.condition(m), c.condition(n):
        c.z(x)
    with pytest.raises(ValueError, match="nonlinear"):
        compile_symbolic(c)
    c.ops[-1] = ops.Op(ops.Z, (x,), cond=((100, 0),))
    with pytest.raises(ValueError):
        compile_symbolic(c)


def test_reject_unannotated_classical_parity():
    c = Circuit()
    c.input("x", 1)
    c.classical_parities((c.classical_bit(0),), (0, 1))
    with pytest.raises(ValueError, match="unsupported"):
        compile_symbolic(c)


@pytest.mark.parametrize("badop", [
    ops.Op("MAGIC", (0,)),
    ops.Op(ops.CX, (0, 0)),
    ops.Op(ops.CCX, (0, 1, 1)),
    ops.Op(ops.CZ, (0, 0)),
    ops.Op(ops.X, (3,)),
    ops.Op(ops.X, (True,)),
    ops.Op(ops.X, ()),
    ops.XorWordOp((0, 1), 4),
    ops.XorWordOp((0, 0), 1),
    ops.XorWordOp((0, 1), 1, ctrl=0),
    ops.Op(ops.X, (0,), cbit=0),
])
def test_reject_malformed_gate_and_word_records(badop):
    c = Circuit()
    c.input("x", 3)
    c.ops.append(badop)
    with pytest.raises(ValueError):
        compile_symbolic(c)


@pytest.mark.parametrize("failure", ["dead_gate", "double_free", "double_alloc", "dead_output"])
def test_reject_invalid_lifetimes(failure):
    c = Circuit()
    x = c.input("x", 1)
    t = c.alloc(1)
    c.free(t)
    if failure == "dead_gate":
        c.x(t[0])
    elif failure == "double_free":
        c.free(t)
    elif failure == "double_alloc":
        c.ops.append(ops.Op(ops.ALLOC, (x[0],)))
    else:
        c.rebind("x", t)
    with pytest.raises(ValueError):
        compile_symbolic(c)


def test_exact_forward_preserves_nonzero_target_all_raw_branches():
    c = Circuit()
    a = c.input("a", 2)
    t = c.input("t", 2)
    lookup_xor_unary(c, a, t, (0, 3, 1, 2))
    inputs = [{"a": a, "t": t} for a, t in product(range(4), repeat=2)]
    expected = [{"a": row["a"], "t": row["t"] ^ (0, 3, 1, 2)[row["a"]]} for row in inputs]
    result = _exhaustive(c, inputs, expected)
    plan = compile_symbolic(c)
    assert plan.stats["exact_blocks"] == 1
    assert plan.stats["symbolic_measurements"] == 0
    assert result.bad_count == 0


@pytest.mark.parametrize("low_bits", [0, 1, 2])
def test_exact_unlookup_all_measurement_branches_and_recycled_temp_ids(low_bits):
    c = Circuit()
    a = c.input("a", 2)
    z = c.input("z", 2)
    # A previously used temporary id may occur again in a validated exact block.
    old = c.alloc(2)
    c.x(old[0])
    c.x(old[0])
    c.free(old)
    t = c.alloc(2)
    lookup_xor_unary(c, a, t, (0, 3, 1, 2))
    for q, r in zip(t, z):
        c.cx(q, r)
    unlookup_measure(c, a, t, (0, 3, 1, 2), low_bits=low_bits)
    after = c.alloc(2)
    c.free(after)
    inputs = [{"a": a, "z": z} for a, z in product(range(4), repeat=2)]
    result = _exhaustive(c, inputs)
    assert result.bad_count == 0
    assert compile_symbolic(c).stats["exact_blocks"] == 2


def test_unlookup_summary_precondition_mismatch_is_bad_not_certified():
    c = Circuit()
    a = c.input("a", 2)
    t = c.alloc(2)
    # Uninitialized scratch matches just the zero row of this table.
    unlookup_measure(c, a, t, (0, 3, 1, 2))
    inputs = [{"a": a} for a in range(4)]
    result = simulate_batch(compile_symbolic(c), inputs, inputs)
    assert result.summary_mismatch_mask == result.bad_mask == 0b1110
    assert result.summary_mismatch_count == 3
    assert result.phase_count == 0  # marginalized block: mismatch is separate


def test_exact_tamper_and_private_classical_escape_rejected_by_shared_validator():
    c = Circuit()
    a = c.input("a", 2)
    t = c.input("t", 2)
    lookup_xor_unary(c, a, t, (0, 3, 1, 2))
    block = c.exact_blocks[0]
    original = c.ops[block.start]
    c.ops[block.start] = replace(original, qubits=(t[0],))
    with pytest.raises(ValueError):
        compile_symbolic(c)
    c.ops[block.start] = original
    with c.condition(block.cbit_start):
        c.z(a[0])
    with pytest.raises(ValueError):
        compile_symbolic(c)


def test_plan_snapshot_and_digest_stable_sensitive_to_ir_constants_and_views():
    c = Circuit()
    x = c.input("x", 2)
    c.x(x[0])
    plan = compile_symbolic(c)
    assert len(plan.ir_digest) == 64
    assert plan.ir_digest == compile_symbolic(c).ir_digest
    assert plan.raw_ops[0] is c.ops[0]
    assert isinstance(plan.raw_ops, tuple)
    with pytest.raises(TypeError):
        plan.classical_last_use[0] = 99
    c.ops[-1] = ops.Op(ops.X, (x[1],))
    assert plan.ir_digest != compile_symbolic(c).ir_digest
    assert simulate_batch(plan, [{"x": 0}], [{"x": 1}]).bad_count == 0
    c.rebind("x", Register(reversed(x.ids)))
    view_digest = compile_symbolic(c).ir_digest
    assert view_digest != plan.ir_digest
    c.classical_bit(1)
    assert compile_symbolic(c).ir_digest != view_digest


@pytest.mark.parametrize("mutation", [
    "append", "delete", "equal_op_replacement", "input_alloc", "rebind",
    "inplace_view", "new_output", "constant", "new_constant", "qid_bound",
    "cbit_bound", "equal_block_replacement", "remove_block", "duplicate_block",
])
def test_plan_binding_rejects_postcompile_mutations(mutation):
    c = Circuit()
    x, y = c.input("x", 2), c.input("y", 2)
    cb = c.classical_bit(1)
    lookup_xor_unary(c, x, y, (0, 1, 2, 3))
    plan = compile_symbolic(c)
    assert validate_plan_binding(plan, c) is None
    if mutation == "append":
        c.x(x[0])
    elif mutation == "delete":
        c.ops.pop()
    elif mutation == "equal_op_replacement":
        c.ops[0] = replace(c.ops[0])
    elif mutation == "input_alloc":
        c.ops[0] = replace(c.ops[0], name="another-input")
    elif mutation == "rebind":
        c.rebind("x", Register(reversed(x.ids)))
    elif mutation == "inplace_view":
        c.inputs["x"].ids.reverse()
    elif mutation == "new_output":
        c.inputs["extra"] = Register([])
    elif mutation == "constant":
        c._cbit_const[cb] = 0
    elif mutation == "new_constant":
        c.classical_bit(0)
    elif mutation == "qid_bound":
        c._next_qid += 1
    elif mutation == "cbit_bound":
        c._next_cbit += 1
    elif mutation == "equal_block_replacement":
        c.exact_blocks[0] = replace(c.exact_blocks[0])
    elif mutation == "remove_block":
        c.exact_blocks.clear()
    else:
        c.exact_blocks.append(c.exact_blocks[0])
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_plan_binding(plan, c)


def test_plan_binding_survives_execution_and_ignores_unrelated_metadata():
    c = Circuit()
    x = c.input("x", 2)
    t = c.alloc(2)
    for a, b in zip(x, t):
        c.cx(a, b)
        c.cx(b, a)
    c.free(x)
    c.rebind("x", t)
    plan = compile_symbolic(c)
    inputs = [{"x": i} for i in range(4)]
    assert validate_plan_binding(plan, c) is None
    assert simulate_batch(plan, inputs, inputs).bad_count == 0
    for row in inputs:
        assert simulate(c, row).outputs == row
    c.unrelated_metadata = {"label": "not proof of analyzed IR identity"}
    assert validate_plan_binding(plan, c) is None


def test_exact_table_is_part_of_analyzed_ir_digest():
    def build(table):
        c = Circuit()
        a = c.input("a", 1)
        t = c.input("t", 1)
        lookup_xor_unary(c, a, t, table)
        return compile_symbolic(c)
    assert build((0, 1)).ir_digest == build((0, 1)).ir_digest
    assert build((0, 1)).ir_digest != build((1, 0)).ir_digest


@pytest.mark.parametrize("operation", ["mul", "div"])
def test_five_bit_real_gcd_composite_all_nonzero_basis_pairs(operation):
    # Actual GCD construction temporarily empties/rebinds x. No ideal arithmetic
    # evaluator is substituted: every emitted gate and measurement is analyzed.
    from qsm.inversion import schedule
    from qsm.inversion.inplace import moddiv_inplace, modmul_inplace

    p, n = 31, 5
    iterations = schedule.iterations_j2(n)
    c = Circuit()
    x, y = c.input("x", n), c.input("y", n)
    builder = modmul_inplace if operation == "mul" else moddiv_inplace
    x, y = builder(c, x, y, p, iterations, schedule.widths_j2(n, iterations))
    c.rebind("x", x)
    c.rebind("y", y)
    inputs = [{"x": x, "y": y} for x in range(1, p) for y in range(1, p)]
    expected = [{"x": row["x"], "y": row["y"] * (
        row["x"] if operation == "mul" else pow(row["x"], -1, p)) % p}
        for row in inputs]
    plan = compile_symbolic(c)
    result = simulate_batch(plan, inputs, expected, output_moduli={"x": p, "y": p})
    assert plan.stats["symbolic_measurements"] > 0
    assert result.samples == 900
    assert result.bad_count == 0
