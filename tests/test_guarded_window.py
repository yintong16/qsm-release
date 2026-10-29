import json
import random

import pytest

from qsm.audit.guarded_comparator import build_compare_case
from qsm.audit.guarded_window import (
    _pairs, guarded_order, main, measure_comparators, prefix_order, sample_trajectories,
)
from qsm.audit.history_math import trace_history
from qsm.core.simulator import simulate
from qsm.inversion.schedule import widths_j2


@pytest.mark.parametrize("w,k", [(3, 1), (4, 2), (5, 3), (6, 4), (7, 4), (8, 4)])
def test_guarded_rule_is_wider_comparison_and_never_worsens(w, k):
    for b in range(1, k + 1):
        for u in range(1 << w):
            for v in range(1 << w):
                result, _ = guarded_order(u, v, w, k, b)
                assert result == prefix_order(u, v, w, k + b)
                if prefix_order(u, v, w, k) == (v < u):
                    assert result == (v < u)


@pytest.mark.parametrize("p", [31, 61, 251])
def test_exact_census_guard_improvements_and_full_width_endpoint(p):
    n = p.bit_length()
    k = n - 2
    result = sample_trajectories(p, list(range(1, p)), base_bits=[k], guard_bits=[1, 2], exhaustive=True)
    rows = result["rows"]
    assert [r["guard_bits"] for r in rows] == [0, 1, 2]
    assert rows[0]["bad_inputs"] >= rows[1]["bad_inputs"] >= rows[2]["bad_inputs"]
    assert rows[2]["bad_inputs"] == 0
    assert rows[2]["rescued_baseline_bad_inputs"] == rows[0]["bad_inputs"]
    assert rows[2]["removed_bad_slots"] == rows[0]["bad_slots"]


def test_known_small_divisor_needs_large_enough_extension():
    p = (1 << 255) - 19
    result = sample_trajectories(p, [3], base_bits=[100], guard_bits=[8, 16, 32, 54, 55, 64, 100])
    rows = {r["guard_bits"]: r for r in result["rows"]}
    assert rows[0]["bad_inputs"] == 1
    assert rows[8]["bad_inputs"] == 1
    assert rows[16]["bad_inputs"] == 1
    assert rows[54]["bad_inputs"] == 1
    assert rows[55]["bad_inputs"] == 0
    assert rows[64]["bad_inputs"] == 0
    assert rows[100]["bad_inputs"] == 0


@pytest.mark.parametrize("p,x,i,w,u,v", [
    ((1 << 255) - 19,
     43631305065782444984070340065901002510198174364762144979655210659194163183820,
     198, 82, 16713975, 16594657),
    ((1 << 256) - (1 << 32) - 977,
     77262945111446695464812132347834798548160176174972158362598024440070563336267,
     204, 78, 840269, 838373),
])
def test_late_singleton_64_bit_hazards_in_emitted_comparators(p, x, i, w, u, v):
    rec = next(rec for rec in trace_history(p, x) if rec.i == i)
    assert rec.active and rec.prefix.count == 1
    assert (rec.u, rec.v) == (u, v)
    assert widths_j2(p.bit_length())[i] == w
    assert v < u and not prefix_order(u, v, w, 64)
    inputs = {"a": u, "b": v, "flag": 0, "ctrl": 1}
    for strategy in ("baseline", "wider", "guarded"):
        case = build_compare_case(w, 64, 8, strategy)
        result = simulate(case.circuit, inputs, rng=random.Random(41))
        assert result.outputs == {**inputs, "flag": int(strategy != "baseline")}
        assert result.phase == 0
    diagnostic = sample_trajectories(p, [x], base_bits=[64], guard_bits=[8])
    assert diagnostic["width_clip_inputs"] == diagnostic["iteration_overrun_inputs"] == 0
    assert [row["bad_inputs"] for row in diagnostic["rows"]] == [1, 0]


def test_resource_measurements_are_actual_ir_and_schedule_totals():
    result = measure_comparators(8, base_bits=[4], guard_bits=[1, 2, 4])
    for row in result["rows"]:
        resource = row["at_full_width"]
        for r in resource.values():
            assert r["scratch_peak"] == r["peak_qubits"] - 18
            assert r["toffoli"] > 0
        assert row["forward_comparison_regions"]["guarded"]["toffoli"] >= resource["guarded"]["toffoli"]
    assert "NOT a rebuilt" in result["scope"]


@pytest.mark.parametrize("ks,bs", [([], [1]), ([0], [1]), ([4], []), ([4], [True]), ([1, True], [1])])
def test_invalid_grid_rejected(ks, bs):
    with pytest.raises(ValueError):
        _pairs(ks, bs)


def test_cli_records_scope_and_refuses_overwrite(tmp_path, capsys):
    path = tmp_path / "study.json"
    args = ["--prime", "31", "--exhaustive", "--base-bits", "3", "--guard-bits", "1", "2", "--json", str(path)]
    assert main(args) == 0
    result = json.loads(path.read_text())
    assert result["trajectories"]["samples"] == 30
    assert result["production_gcd_changed"] is False
    assert "Whole-window peak is not inferred" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(args)


@pytest.mark.parametrize("args", [["--samples", "0"], ["--prime", "15"], ["--exhaustive"],
                                  ["--prime", "31", "--base-bits", "0"]])
def test_cli_invalid_input(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
