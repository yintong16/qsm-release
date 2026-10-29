"""Coset domain, emitted arithmetic and Fourier controller are separate checks."""
from cmath import exp, pi
from dataclasses import replace

import pytest

import coset_fixtures as fixture
from qsm.audit import full_scalar as audit
from qsm.audit.streamed_ir import FeedbackPhaseOp
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.point.group import scalar_mul
from qsm.scalarmul.lookup import lookup_xor_unary
from qsm.scalarmul.zero_point import add_zero_point_inplace, build_zero_point_case
from test_full_scalar import _controller_only_outputs


def group():
    return audit.Group(**fixture.group_kwargs(), torsion=fixture.T)


def mul(k, point):
    return scalar_mul(k, *point, p=fixture.MODULUS, a=fixture.A)


def expected(u, v):
    g = group()
    half = pow(2, -1, g.order)
    return g.add(fixture.T, g.add(g.mul((2*u-7)*half, g.P), g.mul((2*v-15)*half, g.Q)))


def monolithic(*, symbolic_workspace=False):
    g = group()
    c = Circuit()
    registers = {"u": c.input("u", 3), "v": c.input("v", 4)}
    for index, slot in enumerate(audit.windows(3, 4, 2)):
        word = registers[slot["axis"]][slot["offset"]:slot["offset"]+slot["bits"]]
        if index == 0:
            points = audit._initial_points(g, slot, "odd_torsion")
            if symbolic_workspace:
                # The batch compiler requires named input ALLOCs for named outputs.
                x, y = c.input("x", g.n), c.input("y", g.n)
                packed = Register(x.ids+y.ids)
            else:
                packed = c.alloc(2*g.n)
                x, y = packed[:g.n], packed[g.n:]
            lookup_xor_unary(c, word, packed, tuple(xx | (yy << g.n) for xx, yy in points))
            c.inputs["x"], c.inputs["y"] = x, y
        else:
            table = audit._table(g, slot, "odd_torsion")
            x, y = add_zero_point_inplace(c, x, y, word, table,
                config=audit._config(g, slot["bits"]-1, True),
                const_kw={}, sqr_kw={}, phase_correction=True)
            c.rebind("x", x)
            c.rebind("y", y)
    return c


@pytest.fixture(scope="module")
def small_results():
    return {recycled: audit.run(group=group(), u_bits=3, v_bits=4, window_bits=2,
                variant="odd_torsion", recycled=recycled, input_pairs=fixture.INPUT_PAIRS,
                full_width=True, track_depth=True, retain_ops=True)
            for recycled in (False, True)}


def test_fixture_is_complete_cofactor_eight_curve_not_selected_on_circuit_success():
    result = fixture.validate_fixture()
    assert result["horizontal_events"] == 41
    assert result["input_pairs"] == 128 and result["exceptional_events"] == 0


def test_initializer_and_shift_keep_one_fixed_torsion_point():
    g = group()
    schedule = audit.windows(3, 4, 2)
    shift = audit.shift_point(g, schedule, "odd_torsion")
    assert shift == expected(0, 0)
    points = audit._initial_points(g, schedule[0], "odd_torsion")
    assert len(points) == 2
    assert all(mul(g.order, point) == mul(g.order, g.torsion) for point in points)
    for u, v in fixture.INPUT_PAIRS:
        assert g.add(g.add(g.mul(u, g.P), g.mul(v, g.Q)), shift) == expected(u, v)
        assert points[u >> 2] == fixture.odd_trace(u, v)[0]["after"]


def test_all_positional_tables_use_their_own_half_base():
    g = group()
    for slot in audit.windows(3, 4, 2)[1:]:
        table = audit._table(g, slot, "odd_torsion")
        base = g.P if slot["axis"] == "u" else g.Q
        for raw in range(1 << slot["bits"]):
            digit = 2*raw-((1 << slot["bits"])-1)
            assert table.point_for_word(raw) == g.mul(digit*(1 << slot["offset"])*pow(2,-1,g.order), base)


def test_all_inputs_have_canonical_values_clean_scratch_and_no_group_exceptions(small_results):
    for recycled, result in small_results.items():
        assert result["point_additions"] == 3
        assert result["resources"]["toffoli"] == 15276
        assert result["resources"]["live_qubits"] == (10 if recycled else 17)
        assert result["resources"]["peak_qubits"] == (51 if recycled else 56)
        assert len(result["diagnostics"]) == 128
        assert sum(len(d["horizontal_stages"]) for d in result["diagnostics"]) == 41
        for d in result["diagnostics"]:
            assert d["expected"] == d["output"] == expected(d["u"], d["v"])
            assert not any(d[k] for k in ("domain_failure", "value_failure", "dirty_failure", "control_failure", "noncanonical_stages"))
        assert sum(not d["observed_pass"] for d in result["diagnostics"]) == 9
        assert all(d["phase"] == 1 for d in result["diagnostics"] if not d["observed_pass"])
    assert small_results[False]["diagnostics"] == small_results[True]["diagnostics"]


def test_coherent_stream_is_gate_identical_to_monolithic_builder(small_results):
    c = monolithic()
    result = small_results[False]
    assert list(c.ops) == result["_retained_ops"]
    resources = count(c)
    for key in ("toffoli", "clifford", "measurements", "peak_qubits", "toffoli_depth", "depth"):
        assert getattr(resources, key) == result["resources"][key]


def test_all_outcome_chain_phase_census_retains_every_input():
    c = monolithic(symbolic_workspace=True)
    original = monolithic()
    assert [op for op in c.ops if op.kind != "ALLOC"] == [op for op in original.ops if op.kind != "ALLOC"]
    inputs = [{"u": u, "v": v, "x": 0, "y": 0} for u, v in fixture.INPUT_PAIRS]
    outputs = [{"u": u, "v": v, "x": expected(u,v)[0], "y": expected(u,v)[1]} for u, v in fixture.INPUT_PAIRS]
    h = simulate_batch(compile_symbolic(c), inputs, outputs, output_moduli={"x":31,"y":31})
    assert h.samples == 128
    assert h.value_count == h.dirty_count == h.noncanonical_count == h.summary_mismatch_count == 0
    assert h.bad_mask == h.phase_mask and h.phase_count == 19
    assert h.outputs == outputs


def test_all_horizontal_stages_are_clean_and_other_phase_faults_are_inherited():
    g = group()
    phase_union = 0
    for index, slot in enumerate(audit.windows(3,4,2)[1:], 1):
        table = audit._table(g,slot,"odd_torsion")
        case = build_zero_point_case(2,table=table,config=audit._config(g,1,True),const_kw={},sqr_kw={})
        traces = [fixture.odd_trace(u,v)[index] for u,v in fixture.INPUT_PAIRS]
        inputs = [{"word": t["raw"], "x":t["before"][0], "y":t["before"][1]} for t in traces]
        outputs = [{"word": t["raw"], "x":t["after"][0], "y":t["after"][1]} for t in traces]
        h = simulate_batch(compile_symbolic(case.circuit),inputs,outputs,output_moduli={"x":31,"y":31})
        horizontal = sum(1 << lane for lane,t in enumerate(traces) if t["before"][1] == t["operand"][1])
        assert horizontal.bit_count() == {1:16,2:12,3:13}[index]
        assert not h.bad_mask & horizontal
        assert h.value_count == h.dirty_count == h.noncanonical_count == h.summary_mismatch_count == 0
        assert h.phase_count == {1:0,2:8,3:13}[index]
        phase_union |= h.phase_mask
    assert phase_union.bit_count() == 19
    c = monolithic(symbolic_workspace=True)
    inputs = [{"u":u,"v":v,"x":0,"y":0} for u,v in fixture.INPUT_PAIRS]
    outputs = [{"u":u,"v":v,"x":expected(u,v)[0],"y":expected(u,v)[1]} for u,v in fixture.INPUT_PAIRS]
    assert simulate_batch(compile_symbolic(c),inputs,outputs,output_moduli={"x":31,"y":31}).phase_mask == phase_union


def test_inherited_nonhorizontal_square_phase_has_deterministic_witness():
    g = group()
    table = audit._table(g,audit.windows(3,4,2)[-1],"odd_torsion")
    case = build_zero_point_case(2,table=table,config=audit._config(g,1,True),const_kw={},sqr_kw={})
    inputs = {"word":3,"x":18,"y":12}
    outcomes = [0]*count(case.circuit).measurements
    assert simulate(case.circuit,inputs,measurements=outcomes).phase == 0
    outcomes[1429] = 1
    result = simulate(case.circuit,inputs,measurements=outcomes)
    assert result.outputs == {"word":3,"x":4,"y":2}
    assert result.phase == 1


def translate(point, raw, stage):
    g = group()
    base = g.P if stage["axis"] == "u" else g.Q
    digit = 2*raw-((1 << stage["bits"])-1)
    addend = g.mul(digit*(1 << stage["offset"])*pow(2,-1,g.order), base)
    return g.add(g.torsion if stage["index"] == 0 else point, addend)


def dense_fourier():
    outputs = {}
    for j in range(8):
        for k in range(16):
            vector = {}
            for u, v in fixture.INPUT_PAIRS:
                point = expected(u, v)
                vector[point] = vector.get(point, 0j)+exp(-2j*pi*(u*j/8+v*k/16))/128
            outputs[j,k] = vector
    return outputs


def max_error(actual, ideal):
    assert actual.keys() == ideal.keys()
    return max(abs(actual[key].get(point,0j)-ideal[key].get(point,0j))
               for key in ideal for point in actual[key].keys() | ideal[key].keys())


def test_emitted_controller_matches_dense_qft_with_coset_point_labels(small_results):
    actual = _controller_only_outputs(small_results[True], translate=translate)
    assert max_error(actual, dense_fourier()) < 2e-12
    assert sum(abs(a)**2 for vec in actual.values() for a in vec.values()) == pytest.approx(1)
    assert small_results[True]["resources"]["phase_rotations"] == 5
    assert small_results[True]["resources"]["hadamards"] == 7


def test_controller_negative_control_detects_wrong_phase(small_results):
    result = small_results[True]
    retained = list(result["_retained_ops"])
    index = next(i for i, op in enumerate(retained) if isinstance(op, FeedbackPhaseOp))
    retained[index] = replace(retained[index], denominator_power=retained[index].denominator_power+1)
    assert max_error(_controller_only_outputs(result, retained, translate=translate), dense_fourier()) > 1e-4


@pytest.mark.parametrize("point", [None, fixture.P, (1,0), (0,0), (31,1), (1,31), (3,11)])
def test_invalid_torsion_rejected_before_circuit(point):
    with pytest.raises(ValueError):
        audit.run(group=replace(group(),torsion=point),u_bits=3,v_bits=4,window_bits=2,variant="odd_torsion",input_pairs=[])


def test_one_bit_arithmetic_rejected_but_one_bit_initializer_supported():
    with pytest.raises(ValueError, match="arithmetic window"):
        audit.run(group=group(),u_bits=3,v_bits=3,window_bits=2,variant="odd_torsion",input_pairs=[])
    assert audit.windows(255,208,16)[0]["bits"] == 15
    assert len(audit.windows(255,208,16))-1 == 28
    assert all(s["bits"] == 16 for s in audit.windows(255,208,16)[1:])


def test_active_order_eight_offset_is_valid_only_for_wei25519(monkeypatch):
    from qsm.params import FIELD
    if FIELD == "25519":
        g = audit.active_group(torsion=True)
        g.validate()
        assert g.torsion == audit.WEI25519_TORSION
        assert scalar_mul(8,*g.torsion,p=g.p,a=g.a) is None
        assert scalar_mul(4,*g.torsion,p=g.p,a=g.a) is not None
    monkeypatch.setattr(audit,"FIELD","secp256k1")
    with pytest.raises(ValueError,match="order-eight"):
        audit.active_group(torsion=True)
