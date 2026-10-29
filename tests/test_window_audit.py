"""Lightweight sweep tests: never construct a full-width arithmetic circuit."""
from copy import deepcopy
from dataclasses import asdict
import io
import json
from types import SimpleNamespace
import weakref

import pytest

from qsm import params
from qsm.audit import window
from qsm.bench import report
from qsm.bench.harness import Result
from qsm.bench.registry import Case
from qsm.core.circuit import Circuit


def tiny_case(bits=1, *, config=None):
    c = Circuit()
    x = c.input("x", 3)
    c.ccx(x[0], x[1], x[2])
    c.ccx(x[0], x[1], x[2])
    c.cx(x[0], x[1])
    c.cx(x[0], x[1])
    c.x(x[2])
    c.x(x[2])
    c.z(x[0])  # Count phase-kind operations by their string kind, too.
    c.measure(c.alloc(1)[0])
    return Case(c, dict, lambda rng: {"x": 0}, {
        "empirical_only": True, "fail_analytic": None,
        "window_bits": bits, "lookup_backend": "test-prefix.v1",
        "table_fingerprint": "a" * 64, "dummy_policy": "T[1]",
        "tuning": {"inversion": {"iters": 5, "widths": [3, 3, 2, 2, 1]}},
        "resource_regions": [("lookup", 1, 2), ("unlookup", 2, 3),
                             ("selection", 3, 5), ("arithmetic", 5, 10)],
    })


def fake_result(version, case, **kwargs):
    return Result(version, None, kwargs["n_trials"], 2, 5, 1, 4, 7, 8,
                  empirical_only=True, observed_pass=True,
                  p_correct=None, fail_analytic=None, batch_fail_analytic=None,
                  field=params.FIELD, field_u=params.U, field_prime=params.PRIME,
                  circuit_config=deepcopy(case.meta["circuit_config"]))


def test_region_counts_use_one_unsliced_scan():
    class Stream(list):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

        def __getitem__(self, key):
            assert not isinstance(key, slice), "large op streams must not be sliced"
            return super().__getitem__(key)

    case = tiny_case()
    case.circuit.ops = Stream(case.circuit.ops)
    counts = window.count_window_regions(case)
    assert case.circuit.ops.scans == 1
    assert counts == {
        "ccx": 2, "cx": 2, "measurements": 1, "n_gates": 8, "clifford": 5,
        "gate_kinds": {"CCX": 2, "CX": 2, "X": 2, "Z": 1, "MEASURE": 1},
        "toffoli_by_region": {"lookup": 1, "unlookup": 1, "selection": 0, "arithmetic": 0},
        "peak_qubits": 4,
        "peak_qubits_by_region": {"lookup": 3, "unlookup": 3, "selection": 3, "arithmetic": 4},
        "lookup_calls": 1, "unlookup_calls": 1, "ir_ops": 10,
        "classical_parity_rows": 0, "classical_parity_bit_terms": 0, "classical_parity_sources": 0,
    }


def test_region_counts_repeated_empty_and_unsorted_regions():
    case = tiny_case()
    case.meta["resource_regions"] = [
        ("arithmetic", 5, 10), ("lookup", 1, 2), ("selection", 3, 3),
        ("lookup", 2, 3), ("selection", 3, 5),
    ]
    assert window.count_window_regions(case)["toffoli_by_region"] == {
        "lookup": 2, "unlookup": 0, "selection": 0, "arithmetic": 0,
    }


@pytest.mark.parametrize("regions", [
    [("arithmetic", 1, 11)], [("arithmetic", -1, 10)],
    [("arithmetic", 1, 5), ("lookup", 4, 10)],
    [("mystery", 1, 10)], [("arithmetic", 3, 10)],
    [("arithmetic", 1, 2)], [("arithmetic", True, 10)], [],
])
def test_invalid_or_incomplete_regions_rejected(regions):
    case = tiny_case()
    case.meta["resource_regions"] = regions
    with pytest.raises(ValueError):
        window.count_window_regions(case)


def test_version_stable_json_order_and_distinct_operating_points(monkeypatch):
    case = tiny_case(4)
    baseline = window.window_case_version(case)
    case.meta = dict(reversed(list(case.meta.items())))
    assert window.window_case_version(case) == baseline
    for key, value in [
        ("window_bits", 8), ("lookup_backend", "other.v2"),
        ("table_fingerprint", "b" * 64), ("dummy_policy", "T[2]"),
        ("tuning", {"inversion": {"iters": 5, "widths": [3, 3, 3, 2, 1]}}),
        ("circuit_config", {"future_knob": 12}),
    ]:
        changed = tiny_case(4)
        changed.meta[key] = value
        assert window.window_case_version(changed) != baseline
    for key, value in [("FIELD", "other"), ("U", params.U + 1), ("PRIME", params.PRIME + 2)]:
        with monkeypatch.context() as patch:
            patch.setattr(params, key, value)
            assert window.window_case_version(case) != baseline
    assert "w4.test-prefix.v1" in baseline
    assert len(baseline.rsplit(".", 2)[1]) == 64


def test_sweep_releases_cases_and_circuits_before_next_build(monkeypatch):
    refs = []
    events = []
    output = io.StringIO()

    def build(bits, *, config):
        assert all(ref() is None for ref in refs)
        if events:
            assert "local CCX/window bit=" in output.getvalue()
        events.append(("build", bits))
        case = tiny_case(bits)
        refs.extend((weakref.ref(case), weakref.ref(case.circuit)))
        return case

    def benchmark(version, case, **kwargs):
        events.append(("benchmark", case.meta["window_bits"]))
        assert kwargs == dict(n_trials=3, meas_reps=2, seed=17, persist=False)
        return fake_result(version, case, **kwargs)

    monkeypatch.setattr(window, "build_window_case", build)
    monkeypatch.setattr(window, "benchmark_case", benchmark)
    rows = window.sweep([1, 2, 1, 4], n_trials=3, meas_reps=2, seed=17,
                        persist=False, stream=output)
    assert events == [(stage, w) for w in [1, 2, 4] for stage in ["build", "benchmark"]]
    assert all(ref() is None for ref in refs)
    assert len({r.version for r in rows}) == 3
    assert all(isinstance(r, Result) for r in rows)


@pytest.mark.parametrize("kwargs", [
    {"bits": []}, {"bits": [0]}, {"bits": [-1]}, {"bits": [True]},
    {"bits": [1.5]}, {"bits": [1, 13]}, {"bits": [10**100]},
    {"n_trials": 0}, {"n_trials": True}, {"meas_reps": 0},
    {"config": SimpleNamespace(arithmetic_family="canonical")},
])
def test_bad_arguments_rejected_before_any_build(monkeypatch, kwargs):
    monkeypatch.setattr(window, "build_window_case", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(ValueError):
        window.sweep(**kwargs)


def test_large_width_requires_explicit_override(monkeypatch):
    monkeypatch.setattr(window, "build_window_case", tiny_case)
    monkeypatch.setattr(window, "benchmark_case", fake_result)
    rows = window.sweep([13], allow_large=True, persist=False, stream=io.StringIO())
    assert len(rows) == 1 and ".w13." in rows[0].version


def test_real_harness_serialization_and_atomic_history(monkeypatch, tmp_path):
    path = tmp_path / "results.json"
    monkeypatch.setattr(report, "RESULTS_PATH", path)
    monkeypatch.setattr(window, "build_window_case", tiny_case)
    output = io.StringIO()
    rows = window.sweep([1, 2], n_trials=3, meas_reps=2, seed=42, stream=output)
    stored = report.load_history()
    assert stored == json.loads(json.dumps([asdict(r) for r in rows], allow_nan=False))
    assert set(report.latest_by_version()) == {r.version for r in rows}
    for r in rows:
        assert r.correct is None and r.observed_pass is True and r.empirical_only
        assert r.p_correct is r.fail_analytic is r.batch_fail_analytic is None
        assert r.regression_pvalue is None and r.log2_fail is None
        assert r.trials == 6 and r.batches_completed == 3 and r.meas_reps == 2
        assert r.seed == 42
        assert r.fail_seen == r.dirty_seen == r.phase_seen == r.output_seen == 0
        counts = r.circuit_config["window_resources"]
        assert counts["ccx"] == r.toffoli == 2
        assert r.circuit_config["table_fingerprint"] == "a" * 64
        assert r.circuit_config["tuning"]["inversion"]["widths"] == [3, 3, 2, 2, 1]
    rendered = output.getvalue()
    for text in ("EMPIRICAL ONLY", "analytic correctness=unknown", "95% CP upper=",
                 "CCX=2; CX=2; measurements=1; total_gates=8; peak_qubits=4",
                 "lookup=1; unlookup=1; selection=0; arithmetic=0",
                 "not a full-scalar extrapolation", "not quantum norm"):
        assert text in rendered
    assert "exact" not in rendered.lower()
    assert "exact" not in report.compare(columns=["version", "correct", "log2_fail"]).lower()


def test_real_empirical_failures_print_independent_counts(monkeypatch):
    def failing_case(*args, **kwargs):
        case = tiny_case(*args, **kwargs)
        case.gen_input = lambda rng: {"x": 1}  # Z leaves a phase error.
        case.reference = lambda v: {"x": 0}  # Independent output mismatch.
        return case

    monkeypatch.setattr(window, "build_window_case", failing_case)
    output = io.StringIO()
    result, = window.sweep([1], n_trials=2, meas_reps=2, persist=False, stream=output)
    assert result.correct is None and result.observed_pass is False
    assert result.trials == result.fail_seen == result.output_seen == result.phase_seen == 4
    assert result.dirty_seen == 0 and result.batches_failed == 2
    assert "failed_union=4; output=4; dirty=0; phase=4" in output.getvalue()
    assert "observed_pass=False" in output.getvalue()


def test_no_persist_skips_history(monkeypatch, tmp_path):
    monkeypatch.setattr(window, "build_window_case", tiny_case)
    monkeypatch.setattr(report, "RESULTS_PATH", tmp_path / "absent.json")
    window.sweep([1], n_trials=1, persist=False, stream=io.StringIO())
    assert not report.RESULTS_PATH.exists()


def test_count_mismatch_never_persists(monkeypatch):
    monkeypatch.setattr(window, "build_window_case", tiny_case)

    def mismatch(*args, **kwargs):
        r = fake_result(*args, **kwargs)
        r.toffoli += 1
        return r

    monkeypatch.setattr(window, "benchmark_case", mismatch)
    monkeypatch.setattr(report, "append_result", lambda row: pytest.fail("persisted bad counts"))
    with pytest.raises(ValueError, match="disagree"):
        window.sweep([1], stream=io.StringIO())


def test_non_empirical_case_rejected(monkeypatch):
    def build(*args, **kwargs):
        case = tiny_case(*args, **kwargs)
        case.meta["empirical_only"] = False
        return case

    monkeypatch.setattr(window, "build_window_case", build)
    monkeypatch.setattr(window, "benchmark_case", lambda *a, **k: pytest.fail("benchmarked"))
    with pytest.raises(ValueError, match="empirical_only"):
        window.sweep([1], persist=False, stream=io.StringIO())


@pytest.mark.parametrize("observed, status", [(True, 0), (False, 1), (None, 1)])
def test_cli_args_and_empirical_exit_status(monkeypatch, observed, status):
    def sweep(bits, **kwargs):
        assert bits == [1, 2, 4, 8]
        assert kwargs == dict(n_trials=4, meas_reps=1, seed=0, persist=False, allow_large=True,
                              backend="v1", operand_strategy="paired", adder_chunk="half",
                              fused_block="half", low_bits=None)
        return [SimpleNamespace(observed_pass=observed)]

    monkeypatch.setattr(window, "sweep", sweep)
    assert window.main(["--bits", "1", "2", "4", "8", "--trials", "4",
                        "--meas-reps", "1", "--seed", "0", "--no-persist",
                        "--allow-large"]) == status


def test_cli_defaults(monkeypatch):
    def sweep(bits, **kwargs):
        assert bits == [1, 2, 4, 8]
        assert kwargs == dict(n_trials=64, meas_reps=4, seed=0, persist=True, allow_large=False,
                              backend="v1", operand_strategy="paired", adder_chunk="half",
                              fused_block="half", low_bits=None)
        return [SimpleNamespace(observed_pass=True)]

    monkeypatch.setattr(window, "sweep", sweep)
    assert window.main([]) == 0


@pytest.mark.parametrize("args", [["--trials", "0"], ["--bits", "1", "13"],
                                 ["--bits", "0"], ["--meas-reps", "-1"]])
def test_cli_validation_reports_usage_error_before_build(monkeypatch, capsys, args):
    monkeypatch.setattr(window, "build_window_case", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(SystemExit) as exc:
        window.main(args)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
