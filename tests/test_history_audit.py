from __future__ import annotations

import json
import pytest

from qsm.audit.history import (
    _odd_scale_model, _residue_expected, analyze, format_report, main, width_schedule,
)
from qsm.audit.history_math import trace_history


@pytest.mark.parametrize("p", [7, 13, 31, 61, 127])
@pytest.mark.parametrize("kind", ["tuned", "safe", "full"])
def test_census_conditional_counts_and_history_reweighting(p, kind):
    n = p.bit_length()
    result = analyze(p, list(range(1, p)), compare_bits=[1, max(1, n - 2), n - 1],
                     schedule_kind=kind, exhaustive=True, checkpoints=[1, 2])
    for row in result["comparison"].values():
        assert row["conditional_bad_slots"]["mean"] == pytest.approx(
            row["direct_bad_slots"]["mean"], abs=1e-12)
        assert row["residue_upper_slots"]["mean"] + 1e-12 >= row["conditional_bad_slots"]["mean"]
        assert row["conditional_bad_slots"]["standard_error"] == 0
        assert row["observed_any_comparison_hazard"] <= row["direct_bad_slots"]["mean"] + 1e-12
    for slot, summary in result["checkpoints"].items():
        prefixes = {}
        for x in range(1, p):
            for rec in trace_history(p, x):
                if rec.i == slot and rec.active:
                    prefixes[rec.history_key] = rec.prefix
        assert summary["observed_distinct_histories"] == len(prefixes)
        assert summary["estimated_distinct_histories"] == pytest.approx(len(prefixes))
        class_counts = {}
        for prefix in prefixes.values():
            j = (prefix.a + prefix.b).bit_length() - 1
            class_counts[j] = class_counts.get(j, 0) + 1
        for j, row in summary["m_log2_classes"].items():
            assert row["estimated_distinct_histories"] == pytest.approx(class_counts[j])


@pytest.mark.parametrize("m", range(1, 130))
def test_uniform_residue_model_accounts_for_discrete_endpoints(m):
    expected = [0] * 10
    for q in range(1, m + 1):
        expected[min(9, 10 * q // m)] += 1
    assert _residue_expected(m) == pytest.approx([v / m for v in expected])
    assert sum(_residue_expected(m)) == pytest.approx(1)


@pytest.mark.parametrize("bits", range(1, 8))
@pytest.mark.parametrize("discarded", range(0, 9))
def test_odd_scale_model_matches_its_declared_null_distribution(bits, discarded):
    pairs = [(u, v) for u in range(1, 1 << bits, 2)
             for v in range(1, 1 << bits, 2) if max(u, v).bit_length() == bits]
    bad = sum(u > v and u >> discarded == v >> discarded for u, v in pairs)
    assert _odd_scale_model(bits, discarded) == pytest.approx(bad / len(pairs))


def test_small_divisor_witness_is_not_hidden_by_small_operand_filter():
    p = (1 << 255) - 19
    result = analyze(p, [3], compare_bits=[100], checkpoints=[129])
    assert result["width_clip_inputs"] == 0
    assert result["comparison"][100]["bad_inputs"] == 1
    assert result["slots"][129]["k"][100]["direct_bad"] == 1
    assert result["comparison"][100]["conditional_bad_slots"]["standard_error"] is None
    assert result["mode"] == "diagnostic"
    assert result["checkpoints"][129]["estimated_distinct_histories"] is None


def test_exact_comparisons_have_zero_counts_and_models():
    result = analyze(251, list(range(1, 251)), compare_bits=[8, 9], exhaustive=True)
    for row in result["comparison"].values():
        assert row["bad_inputs"] == 0
        for name in ("direct_bad_slots", "conditional_bad_slots", "residue_upper_slots",
                     "uniform_residue_model_slots", "naive_independent_model_slots"):
            assert row[name]["mean"] == 0


def test_same_inputs_reproduce_all_non_timing_fields():
    kwargs = dict(compare_bits=[3, 4, 5], checkpoints=[1, 3], seed=9)
    first = analyze(61, [1, 2, 3, 10, 20, 30], **kwargs)
    second = analyze(61, [1, 2, 3, 10, 20, 30], **kwargs)
    first.pop("seconds")
    second.pop("seconds")
    assert first == second
    assert "NOT certified" in format_report(first)


@pytest.mark.parametrize("inputs", [[1], [1, 1, 2, 3, 4, 5]])
def test_false_census_claim_rejected(inputs):
    with pytest.raises(ValueError, match="exhaustive"):
        analyze(7, inputs, compare_bits=[2], exhaustive=True)


@pytest.mark.parametrize("ks", [[], [0], [-1], [True], [1, True], [1, 1.0]])
def test_invalid_comparison_width_rejected(ks):
    with pytest.raises(ValueError, match="comparison"):
        analyze(7, [1], compare_bits=ks)


@pytest.mark.parametrize("indices", [[1.5], [True]])
def test_invalid_checkpoint_type_rejected(indices):
    with pytest.raises(ValueError, match="checkpoints"):
        analyze(31, [1], compare_bits=[2], checkpoints=indices)


def test_empty_checkpoint_has_no_residue_agreement_claim():
    result = analyze(7, [1], compare_bits=[2], checkpoints=[4])
    row = result["checkpoints"][4]
    assert row["active_inputs"] == 0
    assert row["input_residue_decile_tv"] is None
    assert row["history_weighted_residue_decile_tv"] is None


def test_odd_scale_prediction_obeys_declared_current_fit_event(monkeypatch):
    from types import SimpleNamespace
    import qsm.audit.history as module

    prefix = SimpleNamespace(count=1, bad_count=lambda *a, **kw: 0,
                             residue_bound=lambda *a: 0)
    record = SimpleNamespace(i=1, active=True, entry_bits=5, u=17, v=9,
                             prefix=prefix, m=3)
    monkeypatch.setattr(module, "trace_history", lambda p, x: iter([record]))
    monkeypatch.setattr(module, "width_schedule", lambda n, kind: [5, 3])
    result = analyze(31, [1], compare_bits=[1], checkpoints=[])
    assert result["width_clip_inputs"] == 1
    assert result["comparison"][1]["independent_odd_scale_model_slots"]["mean"] == 0


def test_selected_inputs_do_not_receive_population_inference():
    result = analyze(31, [1, 3, 5, 7], compare_bits=[2], checkpoints=[1])
    assert result["mode"] == "diagnostic"
    assert result["checkpoints"][1]["estimated_distinct_histories"] is None
    assert result["comparison"][2]["conditional_bad_slots"]["standard_error"] is None
    sampled = analyze(31, [1, 3, 5, 7], compare_bits=[2], checkpoints=[1], uniform_samples=True)
    assert sampled["mode"] == "sampled"
    assert sampled["checkpoints"][1]["estimated_distinct_histories"] is not None


def test_safe_width_schedule_has_proved_budget():
    for n in (8, 255, 256):
        widths = width_schedule(n, "safe")
        assert len(widths) == 2 * n - 1
        assert widths[0] == n
        assert all(a >= b >= 1 for a, b in zip(widths, widths[1:]))


def test_cli_runs_fresh_census_and_writes_provenance(tmp_path, capsys):
    destination = tmp_path / "history.json"
    assert main(["--prime", "31", "--exhaustive", "--compare-bits", "2", "4",
                 "--json", str(destination)]) == 0
    result = json.loads(destination.read_text())
    assert result["mode"] == "exhaustive"
    assert result["samples"] == 30
    assert result["seed"] is None
    assert len(result["source_digest"]) == 64
    assert "exact-prefix slot mean" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--prime", "31", "--exhaustive", "--json", str(destination)])
    assert exc.value.code == 2
    assert json.loads(destination.read_text()) == result


@pytest.mark.parametrize("args", [
    ["--prime", "15"], ["--prime", "2"], ["--prime", "1000005"],
    ["--exhaustive"], ["--samples", "0"], ["--prime", "31", "--checkpoints", "0"],
])
def test_cli_rejects_invalid_experiments(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
