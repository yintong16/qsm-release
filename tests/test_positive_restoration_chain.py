"""Complete coset inputs and inherited hazards for the additive restoration chain."""
from dataclasses import replace
import json

import pytest

import coset_fixtures as fixture
from qsm.audit import full_scalar as audit
from qsm.audit.zero_point import _diagnostics
from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.point.group import scalar_mul
from qsm.scalarmul.lookup import lookup_xor_unary
from qsm.scalarmul.odd_window import make_odd_window_table
from qsm.scalarmul.zero_point import add_zero_point_inplace, build_zero_point_case
from test_full_scalar import _controller_only_outputs
from test_full_scalar_odd import dense_fourier, max_error, translate
from test_zero_point import _kwargs


def group(p=31):
    if p == 31:
        return audit.Group(**fixture.group_kwargs(), torsion=fixture.T)
    return audit.Group(29, 8, 2, 5, (17, 11), (18, 27), "order5-coset8-f3", (21, 8))


def monolithic(g, positive):
    variant = "odd_torsion_add" if positive else "odd_torsion"
    c = Circuit()
    registers = {axis: c.input(axis, width) for axis, width in (("u", 3), ("v", 4))}
    for index, slot in enumerate(audit.windows(3, 4, 2)):
        word = registers[slot["axis"]][slot["offset"]:slot["offset"]+slot["bits"]]
        if index == 0:
            points = audit._initial_points(g, slot, variant)
            x, y = c.input("x", g.n), c.input("y", g.n)
            lookup_xor_unary(c, word, Register(x.ids+y.ids),
                             tuple(xx | (yy << g.n) for xx, yy in points))
        else:
            table = audit._table(g, slot, variant)
            x, y = add_zero_point_inplace(c, x, y, word, table,
                config=audit._config(g, slot["bits"]-1, True), const_kw={}, sqr_kw={},
                restoration="add" if positive else "subtract", phase_correction=not positive)
            c.rebind("x", x)
            c.rebind("y", y)
    return c


def expected(g, u, v):
    half = pow(2, -1, g.order)
    return g.add(g.torsion, g.add(g.mul((2*u-7)*half, g.P), g.mul((2*v-15)*half, g.Q)))


@pytest.fixture(scope="module")
def chains():
    results = {}
    for p in (31, 29):
        g = group(p)
        inputs = [{"u": u, "v": v, "x": 0, "y": 0} for u, v in fixture.INPUT_PAIRS]
        outputs = [{"u": row["u"], "v": row["v"],
                    "x": expected(g, row["u"], row["v"])[0],
                    "y": expected(g, row["u"], row["v"])[1]} for row in inputs]
        results[p] = {}
        for positive in (False, True):
            c = monolithic(g, positive)
            h = simulate_batch(compile_symbolic(c), inputs, outputs, output_moduli={"x": p, "y": p})
            results[p][positive] = (c, h)
    return results


@pytest.mark.parametrize("p", [31, 29])
def test_same_initialization_tables_and_fixed_coset_translation(p):
    g = group(p)
    g.validate()
    schedule = audit.windows(3, 4, 2)
    assert audit._initial_points(g, schedule[0], "odd_torsion") == audit._initial_points(g, schedule[0], "odd_torsion_add")
    assert audit.shift_point(g, schedule, "odd_torsion") == audit.shift_point(g, schedule, "odd_torsion_add")
    projection = scalar_mul(g.order, *g.torsion, p=g.p, a=g.a)
    for u, v in fixture.INPUT_PAIRS:
        point = expected(g, u, v)
        assert point is not None and point[1] != 0
        assert scalar_mul(g.order, *point, p=g.p, a=g.a) == projection
        assert g.add(g.add(g.mul(u, g.P), g.mul(v, g.Q)), audit.shift_point(g, schedule, "odd_torsion_add")) == point
    for slot in schedule[1:]:
        assert audit._table(g, slot, "odd_torsion") == audit._table(g, slot, "odd_torsion_add")


def test_f1_chain_preserves_all_outcome_phase_faults_and_all_128_inputs(chains):
    old, a = chains[31][False]
    new, b = chains[31][True]
    assert a.samples == b.samples == 128
    assert b.outputs == a.outputs
    assert b.bad_mask == a.bad_mask == b.phase_mask
    assert b.phase_count == 19
    assert b.value_count == b.dirty_count == b.noncanonical_count == b.summary_mismatch_count == 0
    assert count(old).toffoli == 15276
    assert count(new).toffoli == 15330
    assert count(old).peak_qubits == count(new).peak_qubits == 56


def test_non_f1_torsion_fixture_is_chosen_and_validated_algebraically():
    g = group(29)
    points = {(x, y) for x in range(29) for y in range(29) if (y*y-x*x*x-8*x-2) % 29 == 0}
    assert len(points) == 39
    assert len({g.add(g.mul(i, g.P), scalar_mul(j, *g.torsion, p=29, a=8))
                for i in range(5) for j in range(8)}) == 40
    assert g.mul(2, g.P) == g.Q
    g.validate()


def test_non_f1_chain_retains_failures_and_changed_failure_classes(chains):
    old, a = chains[29][False]
    new, b = chains[29][True]
    assert a.samples == b.samples == 128
    assert (a.bad_count, a.value_count, a.dirty_count, a.phase_count) == (128, 98, 85, 120)
    assert (b.bad_count, b.value_count, b.dirty_count, b.phase_count) == (128, 98, 86, 111)
    assert a.noncanonical_count == b.noncanonical_count == 0
    assert a.summary_mismatch_count == b.summary_mismatch_count == 0
    assert count(old).toffoli == 15300 and count(new).toffoli == 15354
    assert not b.bad_mask & ~a.bad_mask


@pytest.mark.parametrize("multiple,good,noncanonical", [(1, 1, 1), (2, 1, 2), (3, 0, 0), (4, 0, 1)])
def test_non_f1_coset_point_cases_keep_all_inputs_and_prefix_encoding_hazards(multiple, good, noncanonical):
    g = group(29)
    table = make_odd_window_table(2, g.mul(multiple, g.P), order=5, p=29, a=8, b=2)
    fixtures = []
    for k in range(5):
        x, y = g.add(g.torsion, g.mul(k, g.P))
        fixtures.extend({"kind": "coset", "input": {"word": w, "x": x, "y": y}} for w in range(4))
    before, after = [
        _diagnostics(build_zero_point_case(2, **_kwargs(table), restoration=mode,
                                          phase_correction=mode == "subtract"), table, fixtures)[0]
        for mode in ("subtract", "add")]
    assert len(before) == len(after) == 20
    assert sum(d["all_outcome_clean"] for d in before) == sum(d["all_outcome_clean"] for d in after) == good
    assert not any(a["all_outcome_clean"] and not b["all_outcome_clean"] for a, b in zip(before, after))
    assert sum(not d["restoration_input"]["canonical_payloads"] for d in after) == noncanonical


@pytest.fixture(scope="module")
def streamed():
    return {recycled: audit.run(group=group(), u_bits=3, v_bits=4, window_bits=2,
                variant="odd_torsion_add", recycled=recycled, input_pairs=fixture.INPUT_PAIRS,
                full_width=True, track_depth=True, retain_ops=True)
            for recycled in (False, True)}


def test_integrated_arithmetic_control_restoration_and_horizontal_chords(streamed, chains):
    for recycled, result in streamed.items():
        assert result["variant"] == "odd_torsion_add"
        assert result["version"] == "full-scalar-odd-torsion-positive-add-v1"
        assert result["coordinate_restoration"] == "add" and result["phase_correction"] is False
        assert result["restoration_unlookup"] == "paired"
        assert result["resources"]["toffoli"] == 15330
        assert result["resources"]["peak_qubits"] == (51 if recycled else 56)
        assert result["point_additions"] == 3 and len(result["diagnostics"]) == 128
        assert sum(len(d["horizontal_stages"]) for d in result["diagnostics"]) == 41
        for i, d in enumerate(result["diagnostics"]):
            assert d["output"] == d["expected"] == expected(group(), d["u"], d["v"])
            assert not any(d[k] for k in ("control_failure", "domain_failure", "dirty_failure", "value_failure", "noncanonical_stages"))
            if not d["observed_pass"]:
                assert chains[31][True][1].phase_mask >> i & 1
    assert streamed[False]["diagnostics"] == streamed[True]["diagnostics"]
    assert [op for op in streamed[False]["_retained_ops"] if op.kind != ops.ALLOC] == [
        op for op in chains[31][True][0].ops if op.kind != ops.ALLOC]


def test_recycled_controller_remains_the_same_ideal_fourier_experiment(streamed):
    result = streamed[True]
    actual = _controller_only_outputs(result, translate=translate)
    assert max_error(actual, dense_fourier()) < 2e-12
    assert result["resources"]["phase_rotations"] == 5
    assert result["resources"]["hadamards"] == 7


def test_scalar_selector_calls_positive_restoration_explicitly(monkeypatch):
    calls = []
    def point(c, x, y, *args, **kwargs):
        calls.append((kwargs["restoration"], kwargs["phase_correction"]))
        return x, y
    monkeypatch.setattr(audit, "add_zero_point_inplace", point)
    for variant, expected_policy in (("odd_torsion", ("subtract", True)), ("odd_torsion_add", ("add", False))):
        calls.clear()
        result = audit.run(group=group(), u_bits=3, v_bits=4, window_bits=2, variant=variant, input_pairs=[])
        assert calls == [expected_policy]*3
        assert result["coordinate_restoration"] == expected_policy[0]


def test_cli_selects_new_variant_and_keeps_explicit_native_widths(monkeypatch, tmp_path):
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {"variant": kwargs["variant"], "diagnostics": [{"observed_pass": True}]}
    monkeypatch.setattr(audit, "run", run)
    path = tmp_path/"new.json"
    assert audit.main(["--variants", "odd_torsion_add", "--u-bits", "253", "--v-bits", "208", "--json", str(path)]) == 0
    assert calls[0]["variant"] == "odd_torsion_add"
    assert (calls[0]["u_bits"], calls[0]["v_bits"]) == (253, 208)
    assert json.loads(path.read_text())["rows"][0]["variant"] == "odd_torsion_add"


@pytest.mark.parametrize("variant", ["odd_torsion", "odd_torsion_add"])
def test_both_torsion_variants_reject_missing_torsion_and_one_bit_arithmetic(variant):
    with pytest.raises(ValueError, match="order-eight"):
        audit.run(group=replace(group(), torsion=None), u_bits=3, v_bits=4, window_bits=2,
                  variant=variant, input_pairs=[])
    with pytest.raises(ValueError, match="arithmetic window"):
        audit.run(group=group(), u_bits=3, v_bits=3, window_bits=2, variant=variant, input_pairs=[])
