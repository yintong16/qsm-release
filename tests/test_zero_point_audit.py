import json
from types import SimpleNamespace

import pytest

from qsm.audit import zero_point as audit
from qsm.bench.registry import Case
from qsm.core.circuit import Circuit
from qsm.params import PRIME, U
from qsm.point.group import scalar_mul
from qsm.scalarmul.odd_window import make_odd_window_table
from qsm.scalarmul.zero_point import zero_point_reference


@pytest.fixture
def fake_cases(monkeypatch):
    calls = []
    table = SimpleNamespace(p=PRIME, entries=4, fingerprint="table")
    def build(bits, variant, table):
        calls.append((bits, variant))
        c = Circuit()
        c.input("word", bits)
        c.input("x", U)
        c.input("y", U)
        start = len(c.ops)
        x = c.inputs["x"][0]
        c.x(x)
        c.x(x)
        return Case(c, lambda v: dict(v), lambda rng: {}, meta={
            "resource_regions": [("arithmetic", start, len(c.ops))]})
    monkeypatch.setattr(audit, "_build", build)
    monkeypatch.setattr(audit, "make_odd_window_table", lambda bits: table)
    monkeypatch.setattr(audit, "_inputs", lambda *args: [{"kind": "edge", "input": {"word": 0, "x": 1, "y": 2}}])
    monkeypatch.setattr(audit, "horizontal_inputs", lambda table: [{"kind": "horizontal", "input": {"word": 7, "x": 3, "y": 4}}])
    monkeypatch.setattr(audit, "zero_point_reference", lambda x, y, word, table: (x, y))
    monkeypatch.setattr(audit, "source_digest", lambda: "fixed")
    return calls, build


def test_matched_inputs_and_all_outcome_diagnostics(fake_cases):
    calls, _ = fake_cases
    result = audit.run(word_bits=3, random_cases=0)
    assert calls == [(3, v) for v in audit.VARIANTS]
    assert result["added_toffoli"] == 0
    assert result["source_digest"] == "fixed"
    assert result["production_changed"] is False
    for row in result["rows"]:
        assert row["all_outcome_clean"]
        assert row["hazard_counts"] == {"ordinary": 0, "horizontal": 0}
        assert row["diagnostics"] == result["rows"][0]["diagnostics"]
        assert row["analysis"]["ir_digest"]


@pytest.mark.parametrize("hazard", ["phase", "dirty", "value", "noncanonical", "summary_mismatch"])
def test_failure_categories_are_retained(fake_cases, monkeypatch, hazard):
    def simulate(plan, inputs, expected, **kwargs):
        flags = {name+"_mask": int(name == hazard) for name in audit.MASKS}
        return SimpleNamespace(outputs=expected, bad_mask=1, **flags)
    monkeypatch.setattr(audit, "simulate_batch", simulate)
    result = audit.run(word_bits=3)
    for row in result["rows"]:
        assert not row["all_outcome_clean"]
        assert row["hazard_counts"]["ordinary"] == 1
        assert row["diagnostics"][0]["hazards"][hazard]


def test_symbolic_phase_fault_is_detected_with_correct_values(fake_cases, monkeypatch):
    _, build = fake_cases
    def faulty(bits, variant, table):
        case = build(bits, variant, table)
        start = len(case.circuit.ops)
        flag = case.circuit.alloc(1)
        case.circuit.cx(case.circuit.inputs["x"][0], flag[0])
        case.circuit.measure(flag[0])
        case.meta["resource_regions"].append(("phase", start, len(case.circuit.ops)))
        return case
    monkeypatch.setattr(audit, "_build", faulty)
    result = audit.run(word_bits=3)
    for row in result["rows"]:
        assert row["hazard_counts"] == {"ordinary": 1, "horizontal": 1}
        assert all(d["hazards"]["phase"] and not d["hazards"]["value"] for d in row["diagnostics"])


def test_domain_failures_are_not_resampled(fake_cases, monkeypatch):
    def reference(x, y, word, table):
        assert word != 0, "domain witness"
        return x, y
    monkeypatch.setattr(audit, "zero_point_reference", reference)
    result = audit.run(word_bits=3)
    for row in result["rows"]:
        assert row["hazard_counts"]["ordinary"] == 1
        assert row["diagnostics"][0]["domain_failure"].startswith("domain witness")
        assert len(row["diagnostics"]) == 2


def test_source_change_rejected(fake_cases, monkeypatch):
    digests = iter(("before", "after"))
    monkeypatch.setattr(audit, "source_digest", lambda: next(digests))
    with pytest.raises(RuntimeError, match="sources changed"):
        audit.run(word_bits=3)


@pytest.mark.parametrize("kwargs", [{"word_bits": 1}, {"word_bits": 18}, {"word_bits": True},
                                    {"random_cases": -1}, {"random_cases": True}])
def test_invalid_request(kwargs):
    with pytest.raises(ValueError):
        audit.run(**kwargs)


def test_cli_refuses_overwrite(fake_cases, tmp_path):
    path = tmp_path / "result.json"
    args = ["--word-bits", "3", "--json", str(path)]
    assert audit.main(args) == 0
    assert json.loads(path.read_text())["added_toffoli"] == 0
    with pytest.raises(SystemExit):
        audit.main(args)


def test_cli_archives_failure_and_exits_nonzero(fake_cases, monkeypatch, tmp_path):
    def simulate(plan, inputs, expected, **kwargs):
        return SimpleNamespace(outputs=expected, bad_mask=1,
                               **{name+"_mask": int(name == "phase") for name in audit.MASKS})
    monkeypatch.setattr(audit, "simulate_batch", simulate)
    path = tmp_path / "failed.json"
    assert audit.main(["--word-bits", "3", "--json", str(path)]) == 1
    assert not json.loads(path.read_text())["rows"][-1]["all_outcome_clean"]


def test_real_horizontal_fixtures_have_both_roots_and_signs():
    table = make_odd_window_table(5)
    fixtures = audit.horizontal_inputs(table)
    assert len(fixtures) == 4
    assert {d["negative_digit"] for d in fixtures} == {False, True}
    assert len({d["input"]["x"] for d in fixtures}) == 2
    for item in fixtures:
        v = item["input"]
        xp, yp = table.point_for_word(v["word"])
        assert v["y"] == yp
        assert (v["y"]**2-v["x"]**3-table.a*v["x"]-table.b) % table.p == 0
        out = zero_point_reference(v["x"], v["y"], v["word"], table)
        assert v["x"] != xp and out[0] != xp
        assert out[1] == -yp % table.p
        projection = scalar_mul(table.order, v["x"], v["y"], p=table.p, a=table.a)
        assert (None if projection is None else tuple(item["torsion_projection"])) == projection
        if item["torsion_order"] is not None and projection is not None:
            assert scalar_mul(item["torsion_order"], *projection, p=table.p, a=table.a) is None


def test_explicit_variant_selection_and_comparison_reference(fake_cases):
    calls, _ = fake_cases
    result = audit.run(word_bits=3, variants=("zero_phase", "positive_add"))
    assert calls == [(3, "zero_phase"), (3, "positive_add")]
    assert result["added_toffoli"] is None
    assert result["deltas_from_zero_phase"]["positive_add"]["toffoli"] == 0
    assert result["diagnostic_differences_from_zero_phase"]["positive_add"] == {
        "newly_bad_indices": [], "newly_clean_indices": []}


@pytest.mark.parametrize("variants", [(), ("missing",), ("zero_phase", "zero_phase")])
def test_invalid_variant_selections_are_rejected(fake_cases, variants):
    with pytest.raises(ValueError, match="variants"):
        audit.run(word_bits=3, variants=variants)


def test_coset_fixtures_are_algebraic_translations_with_same_words():
    from qsm.params import FIELD
    if FIELD != "25519":
        pytest.skip("fixed order-eight input coset is Wei25519-specific")
    from qsm.audit.full_scalar import active_group
    table = make_odd_window_table(3)
    ordinary = audit._inputs(3, 2, 2026101501)
    shifted = audit.coset_inputs(table, ordinary)
    group = active_group(torsion=True)
    assert len(ordinary) == len(shifted) == 6
    projection = scalar_mul(group.order, *group.torsion, p=group.p, a=group.a)
    for before, after in zip(ordinary, shifted):
        assert before["input"]["word"] == after["input"]["word"]
        expected = group.add(group.torsion, (before["input"]["x"], before["input"]["y"]))
        point = after["input"]["x"], after["input"]["y"]
        assert point == expected and after["kind"].startswith("coset_")
        assert scalar_mul(group.order, *point, p=group.p, a=group.a) == projection
