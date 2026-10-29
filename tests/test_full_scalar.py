"""Full-schedule arithmetic integration and separately labelled controller checks.

The retained coherent IR is executed by the real basis simulator, including its
measurement phases. The dense Fourier tests instead replace ONLY arithmetic
blocks with complete subgroup translations and execute the emitted preparation
and Fourier footer ops. They validate the controller, not a phase-clean quantum
simulation of the approximate affine arithmetic or a full Shor implementation.
"""
from cmath import exp, pi
from copy import copy
from dataclasses import asdict, replace
from math import sqrt
import random

import pytest

from qsm.audit.full_scalar import Group, run, shift_point, windows
from qsm.audit.streamed_ir import BasisAccumulator, FeedbackPhaseOp, ResourceAccumulator
from qsm.core import ops
from qsm.core.resources import count
from qsm.core.simulator import simulate


P = (1, 2)
ORDER = 43
Q_LOG = 7  # A public toy fixture, not knowledge assumed by the real oracle.
SEED = 2026092905


def _add(left, right):
    """Independent complete classical law for y^2=x^3+3 over F_31."""
    if left is None:
        return right
    if right is None:
        return left
    x, y = left
    xx, yy = right
    if x == xx:
        if (y + yy) % 31 == 0:
            return None
        slope = 3*x*x * pow(2*y, -1, 31) % 31
    else:
        slope = (yy-y) * pow(xx-x, -1, 31) % 31
    xr = (slope*slope-x-xx) % 31
    return xr, (slope*(x-xr)-y) % 31


def _mul(k, point=P):
    result = None
    k %= ORDER
    while k:
        if k & 1:
            result = _add(result, point)
        point = _add(point, point)
        k >>= 1
    return result


def _group():
    return Group(31, 0, 3, ORDER, P, _mul(Q_LOG), "order43-test")


def _schedule(u_bits, v_bits, width):
    return [(axis, offset, min(width, length-offset))
            for axis, length in (("u", u_bits), ("v", v_bits))
            for offset in range(((length-1)//width)*width, -1, -width)]


def _shift_scalar(u_bits, v_bits, width, variant):
    shift = pow(2, -1, ORDER)
    if variant == "signed":
        shift -= sum((1 if axis == "u" else Q_LOG) << (offset+bits-1)
                     for axis, offset, bits in _schedule(u_bits, v_bits, width))
    return shift % ORDER


def _ideal_trace_and_first_domain_failure(u, v, variant):
    """Check every raw input, including dummy and one-bit signed exclusions.

    These are the declared algebraic exclusions, not filters based on physical
    simulation success, canonicality, or phase. We continue the complete ideal
    trajectory even after its first incomplete-arithmetic exclusion.
    """
    point = None
    first_failure = None
    trace = []
    for index, (axis, offset, bits) in enumerate(_schedule(3, 3, 2)):
        raw = ((u if axis == "u" else v) >> offset) & ((1 << bits)-1)
        digit = raw - (1 << (bits-1)) if variant == "signed" else raw
        base = P if axis == "u" else _mul(Q_LOG)
        addend = _mul(digit << offset, base)
        if index == 0:
            point = _add(addend, _mul(pow(2, -1, ORDER)))
            assert point is not None
        else:
            result = _add(point, addend)
            if variant == "unsigned":
                physical_digit = raw or 1
            elif bits == 1:
                # The constant -base remains the dummy even when raw=1 disables it.
                physical_digit = -1
            else:
                physical_digit = digit if digit else 1 << (bits-1)
            physical = _mul(physical_digit << offset, base)
            assert physical is not None
            bad = (point is None or point[0] == physical[0]
                   or point[1] == physical[1]
                   or (digit != 0 and (result is None or result[0] == physical[0])))
            if bad and first_failure is None:
                first_failure = index
            point = result
        trace.append(point)
    return trace, first_failure


def _retained_circuit(result):
    circ = copy(result["_circuit"])
    circ.ops = result["_retained_ops"]
    return circ


@pytest.fixture(scope="module", params=("unsigned", "signed"))
def complete_small_schedules(request):
    """All 64 inputs are retained, not resampled until they pass."""
    variant = request.param
    pairs = [(u, v) for u in range(8) for v in range(8)]
    return {
        recycled: run(group=_group(), u_bits=3, v_bits=3, window_bits=2,
                      variant=variant, recycled=recycled, input_pairs=pairs,
                      seed=SEED, full_width=True, track_depth=True, retain_ops=True)
        for recycled in (False, True)
    }


@pytest.mark.parametrize("length,top_bits", [(255, 15), (253, 13), (3, 1)])
def test_high_first_schedule_uses_actual_partial_top_width(length, top_bits):
    width = 2 if length == 3 else 16
    actual = windows(length, length, width)
    expected = _schedule(length, length, width)
    assert [(s["axis"], s["offset"], s["bits"]) for s in actual] == expected
    assert actual[0]["bits"] == top_bits
    assert sum(s["bits"] for s in actual) == 2*length
    for axis in ("u", "v"):
        offsets = [s["offset"] for s in actual if s["axis"] == axis]
        assert offsets == sorted(offsets, reverse=True)


@pytest.mark.parametrize("variant", ["unsigned", "signed"])
@pytest.mark.parametrize("u_bits,v_bits", [(3, 3), (2, 3), (3, 1)])
def test_fixed_global_half_shift_matches_independent_integer_formula(variant, u_bits, v_bits):
    group = _group()
    # Verify the independently constructed fixture's order without reducing k.
    point = None
    multiples = []
    for _ in range(ORDER):
        multiples.append(point)
        point = _add(point, P)
    assert point is None and len(set(multiples)) == ORDER
    assert group.Q == (25, 29)
    expected_shift = _shift_scalar(u_bits, v_bits, 2, variant)
    assert shift_point(group, windows(u_bits, v_bits, 2), variant) == _mul(expected_shift)
    for u in range(1 << u_bits):
        for v in range(1 << v_bits):
            coefficients = []
            for axis, offset, bits in _schedule(u_bits, v_bits, 2):
                raw = ((u if axis == "u" else v) >> offset) & ((1 << bits)-1)
                digit = raw-(1 << (bits-1)) if variant == "signed" else raw
                coefficients.append((digit << offset)*(1 if axis == "u" else Q_LOG))
            assert (sum(coefficients)+pow(2, -1, ORDER)) % ORDER == (
                u+Q_LOG*v+expected_shift) % ORDER


def test_coherent_retained_ir_matches_monolithic_execution_with_persistent_phase(complete_small_schedules):
    result = complete_small_schedules[False]
    circ = _retained_circuit(result)
    stages = result["stages"]
    assert circ.ops and not result["_circuit"].ops
    assert len(result["diagnostics"]) == 64
    admitted = excluded = phase_hazards = 0
    persistent_nonzero_phase = False
    for index, row in enumerate(result["diagnostics"]):
        u, v = row["u"], row["v"]
        trace, failure_stage = _ideal_trace_and_first_domain_failure(u, v, result["variant"])
        assert row["expected"] == trace[-1] == _mul(
            u+Q_LOG*v+_shift_scalar(3, 3, 2, result["variant"]))
        if failure_stage is not None:
            # Do not run undefined arithmetic and then report a filtered pass.
            excluded += 1
            assert row["domain_failure"]["stage"] == failure_stage
            assert row["domain_failure"]["reason"]
            assert not row["active"] and not row["observed_pass"]
            assert row["output"] is None and row["phase"] is None
            continue
        admitted += 1
        assert row["domain_failure"] is None and row["active"]
        assert row["value_failure"] is None
        assert row["dirty_failure"] is None
        assert row["control_failure"] is None
        inputs = {"u": u, "v": v, "x": 0, "y": 0}
        # x/y are final named views, not externally prepared input ALLOCs.
        shot_seed = SEED+100000+index
        mono = simulate(circ, inputs, rng=random.Random(shot_seed))
        basis = BasisAccumulator(inputs, rng=random.Random(shot_seed))
        basis.consume(circ.ops[:stages[0]["op_nodes_begin"]], circ._cbit_const)
        phase_stages = []
        previous_phase = 0
        for stage in stages:
            basis.consume(circ.ops[stage["op_nodes_begin"]:stage["op_nodes_end"]],
                          circ._cbit_const)
            if basis.phase:
                phase_stages.append(stage["index"])
            persistent_nonzero_phase |= bool(previous_phase and basis.phase)
            previous_phase = basis.phase
            phase, rng_state = basis.phase, basis.rng.getstate()
            basis.reset_classical(circ._cbit_const)
            assert basis.phase == phase and basis.rng.getstate() == rng_state
        assert basis.snapshot(circ.inputs) == mono
        assert mono.outputs["u"] == u and mono.outputs["v"] == v
        assert (mono.outputs["x"], mono.outputs["y"]) == tuple(row["output"])
        assert tuple(mono.outputs[k] % 31 for k in ("x", "y")) == trace[-1]
        assert mono.phase == row["phase"]
        assert phase_stages == row["phase_stages"]
        assert row["observed_pass"] == (mono.phase == 0)
        phase_hazards += mono.phase
    assert admitted+excluded == 64
    assert admitted > 0 and excluded > 0 and phase_hazards > 0
    assert persistent_nonzero_phase, "phase must survive more than one arithmetic stage"


def test_recycled_diagnostics_match_deferred_coherent_arithmetic_not_fourier_sampling(complete_small_schedules):
    coherent = complete_small_schedules[False]
    recycled = complete_small_schedules[True]
    assert len(coherent["diagnostics"]) == len(recycled["diagnostics"]) == 64
    for full, streamed in zip(coherent["diagnostics"], recycled["diagnostics"]):
        for key in ("u", "v", "expected", "domain_failure", "value_failure",
                    "dirty_failure", "control_failure", "phase_stages",
                    "noncanonical_stages", "phase", "output", "active", "observed_pass"):
            assert streamed[key] == full[key], (key, full, streamed)
    assert recycled["resources"]["live_qubits"] == 2*_group().n
    assert coherent["resources"]["live_qubits"] == 2*_group().n+6
    assert recycled["resources"]["toffoli"] == coherent["resources"]["toffoli"]
    assert recycled["resources"]["measurements"] == coherent["resources"]["measurements"]+6
    assert "not full Shor statevector" in recycled["scope"]


def test_actual_stream_counts_and_stage_ranges_match_retained_monolithic_ir(complete_small_schedules):
    for recycled, result in complete_small_schedules.items():
        circ = _retained_circuit(result)
        expected = asdict(count(circ))
        # Legacy core.count schedules H but does not classify it as Clifford.
        # Account for this explicit convention difference, not an estimated cost.
        actual_h = sum(op.kind == "H" for op in circ.ops)
        expected["clifford"] += actual_h
        for name, value in expected.items():
            assert result["resources"][name] == value, name
        counter = ResourceAccumulator(digest=True)
        beginning = result["stages"][0]["op_nodes_begin"]
        counter.consume(circ.ops[:beginning])
        previous = beginning
        for stage in result["stages"]:
            assert stage["op_nodes_begin"] == previous
            assert previous <= stage["arithmetic_nodes_begin"] < stage["arithmetic_nodes_end"]
            assert stage["arithmetic_nodes_end"] <= stage["op_nodes_end"]
            before = counter.snapshot()
            counter.consume(circ.ops[previous:stage["op_nodes_end"]])
            after = counter.snapshot()
            for key in ("toffoli", "clifford", "measurements"):
                assert stage[key] == after[key]-before[key]
            assert stage["global_peak_so_far"] == after["peak_qubits"]
            previous = stage["op_nodes_end"]
        assert previous == len(circ.ops)
        assert counter.snapshot() == result["resources"]
        assert counter.live_qubits == circ._live
        assert result["point_additions"] == len(result["stages"])-1 == 3
        assert result["resources"]["hadamards"] == (6 if recycled else 0)
        assert result["resources"]["phase_rotations"] == (4 if recycled else 0)


def _add_amplitude(state, key, amplitude):
    state[key] = state.get(key, 0j)+amplitude


def _hadamard(state, wire):
    out = {}
    bit = 1 << wire
    for (mask, group_value), amplitude in state.items():
        zero = mask & ~bit
        _add_amplitude(out, (zero, group_value), amplitude/sqrt(2))
        _add_amplitude(out, (zero | bit, group_value),
                       (-1 if mask & bit else 1)*amplitude/sqrt(2))
    return out


def _x_projection(state, wire, outcome):
    """Actual IR MEASURE is consuming X-basis measurement, not Z measurement."""
    out = {}
    bit = 1 << wire
    for (mask, group_value), amplitude in state.items():
        sign = -1 if outcome and mask & bit else 1
        _add_amplitude(out, (mask & ~bit, group_value), sign*amplitude/sqrt(2))
    return out


def _controller_only_outputs(result, retained=None, translate=None):
    """Execute retained headers/footers; substitute only ideal group arithmetic.

    The compressed group label is an exact abstract subgroup basis, including O.
    No affine arithmetic op or its measurements is treated as phase-clean here:
    that entire block is explicitly replaced for this controller-only check.
    """
    retained = result["_retained_ops"] if retained is None else retained
    branches = [({}, {(0, 0): 1+0j})]  # (Fourier cbits, {(control mask, group): amp})
    output_ids = {"u": [], "v": []}
    for stage in result["stages"]:
        axis = stage["axis"]
        wires = stage["control_qubits"]
        assert stage["fourier_output_bit_offset"] == len(output_ids[axis])
        header = retained[stage["op_nodes_begin"]:stage["arithmetic_nodes_begin"]]
        for op in header:
            assert op.kind in (ops.ALLOC, "H")
            if op.kind == ops.ALLOC:
                assert tuple(op.qubits) == tuple(wires)
                mask = sum(1 << q for q in wires)
                assert all(not (key[0] & mask) for _, state in branches for key in state)
            else:
                branches = [(cbits, _hadamard(state, op.qubits[0])) for cbits, state in branches]

        translated = []
        for cbits, state in branches:
            out = {}
            for (mask, group_value), amplitude in state.items():
                raw = sum(((mask >> q) & 1) << j for j, q in enumerate(wires))
                if translate is not None:
                    value = translate(group_value, raw, stage)
                else:
                    digit = raw-(1 << (stage["bits"]-1)) if result["variant"] == "signed" else raw
                    amount = (digit << stage["offset"])*(1 if axis == "u" else Q_LOG)
                    if stage["index"] == 0:
                        amount += pow(2, -1, ORDER)
                    value = (group_value+amount) % ORDER
                _add_amplitude(out, (mask, value), amplitude)
            translated.append((cbits, out))
        branches = translated

        measured = []
        footer = retained[stage["arithmetic_nodes_end"]:stage["op_nodes_end"]]
        for op in footer:
            if isinstance(op, FeedbackPhaseOp):
                next_branches = []
                for cbits, state in branches:
                    # Missing/wrong-axis dependencies must fail, not become zero.
                    assert all(source in cbits for source in op.sources)
                    assert tuple(op.sources) == tuple(output_ids[axis]+measured)
                    history = sum(cbits[source] << j for j, source in enumerate(op.sources))
                    phase = exp(-2j*pi*history/(1 << op.denominator_power))
                    bit = 1 << op.qubits[0]
                    out = {key: amplitude*(phase if key[0] & bit else 1)
                           for key, amplitude in state.items()}
                    next_branches.append((cbits, out))
                branches = next_branches
            else:
                assert op.kind == ops.MEASURE and not op.cond
                assert op.qubits == (wires[-1-len(measured)],)
                branches = [(cbits | {op.cbit: outcome},
                             _x_projection(state, op.qubits[0], outcome))
                            for cbits, state in branches for outcome in (0, 1)]
                measured.append(op.cbit)
        assert measured == stage["fourier_output_cbits"]
        output_ids[axis].extend(measured)
    assert {axis: len(ids) for axis, ids in output_ids.items()} == {
        "u": result["u_bits"], "v": result["v_bits"]}
    outputs = {}
    for cbits, state in branches:
        key = tuple(sum(cbits[cb] << j for j, cb in enumerate(output_ids[axis]))
                    for axis in ("u", "v"))
        assert key not in outputs
        assert all(mask == 0 for mask, _ in state)
        outputs[key] = {g: amplitude for (_, g), amplitude in state.items()}
    return outputs


def _dense_double_inverse_qft(u_bits, v_bits, width, variant):
    """Independent direct DFT of the ideal +uP+vQ shifted oracle; no IR reuse."""
    nu, nv = 1 << u_bits, 1 << v_bits
    shift = _shift_scalar(u_bits, v_bits, width, variant)
    outputs = {}
    for yu in range(nu):
        for yv in range(nv):
            target = {}
            for u in range(nu):
                for v in range(nv):
                    g = (u+Q_LOG*v+shift) % ORDER
                    amplitude = exp(-2j*pi*(u*yu/nu+v*yv/nv))/(nu*nv)
                    _add_amplitude(target, g, amplitude)
            outputs[yu, yv] = target
    return outputs


def _max_complex_error(actual, expected):
    assert actual.keys() == expected.keys()
    return max(abs(actual[key].get(g, 0j)-expected[key].get(g, 0j))
               for key in expected for g in range(ORDER))


def test_emitted_recycled_fourier_controller_matches_dense_inverse_qft(complete_small_schedules):
    result = complete_small_schedules[True]
    actual = _controller_only_outputs(result)
    expected = _dense_double_inverse_qft(3, 3, 2, result["variant"])
    assert _max_complex_error(actual, expected) < 2e-12
    assert sum(abs(a)**2 for vec in actual.values() for a in vec.values()) == pytest.approx(1)
    assert result["scalar_qft_signs"] == {"u": -1, "v": -1}
    # This controller-only success coexists with real arithmetic exclusions/errors.
    assert any(row["domain_failure"] for row in result["diagnostics"])
    assert any(row["phase"] == 1 for row in result["diagnostics"])


def test_controller_check_detects_wrong_emitted_feedback_denominator(complete_small_schedules):
    result = complete_small_schedules[True]
    retained = list(result["_retained_ops"])
    index = next(i for i, op in enumerate(retained) if isinstance(op, FeedbackPhaseOp))
    retained[index] = replace(retained[index], denominator_power=retained[index].denominator_power+1)
    corrupted = _controller_only_outputs(result, retained)
    expected = _dense_double_inverse_qft(3, 3, 2, result["variant"])
    assert _max_complex_error(corrupted, expected) > 1e-4


@pytest.mark.parametrize("variant", ["unsigned", "signed"])
@pytest.mark.parametrize("u_bits,v_bits", [(2, 3), (3, 1)])
def test_controller_only_unequal_registers_and_one_bit_signed_stages(variant, u_bits, v_bits):
    result = run(group=_group(), u_bits=u_bits, v_bits=v_bits, window_bits=2,
                 variant=variant, recycled=True, input_pairs=[], full_width=True,
                 track_depth=True, retain_ops=True)
    assert result["diagnostics"] == []  # No claim of physical-arithmetic smoke passes.
    assert [(s["axis"], s["offset"], s["bits"]) for s in result["stages"]] == _schedule(u_bits, v_bits, 2)
    assert any(s["axis"] == "v" and s["bits"] == 1 for s in result["stages"])
    assert _max_complex_error(
        _controller_only_outputs(result),
        _dense_double_inverse_qft(u_bits, v_bits, 2, variant),
    ) < 2e-12
