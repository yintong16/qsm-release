"""Calibration CLI, fresh-row rendering, and sweep lifecycle without large builds."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import io
import json

import pytest

from qsm import params
from qsm.audit import window
from qsm.bench import __main__ as cli
from qsm.bench import calibration, harness, registry, report
from qsm.bench.harness import Result
from qsm.bench.registry import Case
from qsm.core import symbolic_batch
from qsm.core.circuit import Circuit


def unexpected(*args, **kwargs):
    pytest.fail("unexpected production circuit build or history lookup")


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    monkeypatch.setattr(report, "RESULTS_PATH", path)
    monkeypatch.setattr(window, "build_window_case", unexpected)
    monkeypatch.setattr(window, "build_optimized_window_case", unexpected)
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(report, "compare", unexpected)
    monkeypatch.setattr(report, "latest_by_version", unexpected)
    monkeypatch.setattr(calibration, "source_digest", lambda: "a" * 64)
    return path


def fresh_result(*, calibrated=True):
    result = Result(
        "scalarmul.window_sweep.w16.test." + "b" * 64 + ".v1",
        None, 4, 123, 456, 3, 89, 20, 579,
        empirical_only=True, observed_pass=True,
        p_correct=None, fail_analytic=None, batch_fail_analytic=None,
        field=params.FIELD, field_u=params.U, field_prime=params.PRIME,
        trials=12, meas_reps=3,
        circuit_config={"window_bits": 16, "operand_strategy": "paired"},
    )
    if calibrated:
        result.success_samples = 100
        result.bad_inputs = 1
        result.success_lower = .937123456789
        result.success_confidence = .995
        result.clean_input_fraction = .99
    return result


def table_rows(output):
    lines = [line for line in output.splitlines() if line.startswith("|")]
    assert len(lines) >= 2, output
    cells = lambda line: [cell.strip() for cell in line.strip("|").split("|")]
    headers = cells(lines[0])
    return headers, [dict(zip(headers, cells(line), strict=True)) for line in lines[2:]]


@pytest.mark.parametrize("alias", ["--bits", "--window-bits"])
def test_cli_forwards_calibration_separately_from_smoke_shots(
        monkeypatch, capsys, isolated_history, alias):
    calls = []
    result = fresh_result()
    before = asdict(result)
    isolated_history.write_text(json.dumps([dict(before, toffoli=987654321)]))
    history_before = isolated_history.read_bytes()
    monkeypatch.setattr(report, "load_history", unexpected)

    def sweep(bits, **kwargs):
        calls.append((bits, kwargs))
        return [result]

    monkeypatch.setattr(cli, "window_sweep", sweep)
    assert cli.main([alias, "16", "--success-samples", "100", "--confidence", ".995",
                     "--success-batch-size", "1024", "--calibration-seed", "23",
                     "--trials", "4", "--meas-reps", "3", "--seed", "17"]) == 0
    assert calls == [([16], dict(
        n_trials=4, meas_reps=3, seed=17, verbose=False,
        backend="optimized", operand_strategy="paired", success_samples=100,
        confidence=.995, success_batch_size=1024, calibration_seed=23))]
    output = capsys.readouterr().out
    headers, rows = table_rows(output)
    assert headers == cli.SUCCESS_COLUMNS
    row, = rows
    assert row["success_samples"] == "100" and row["bad_inputs"] == "1"
    assert row["success_confidence"] == "0.995"
    assert Decimal(row["success_lower"]) <= Decimal(str(result.success_lower))
    assert row["toffoli"] == "123" and row["peak_qubits"] == "89"
    assert row["version"] == "window[16] optimized/paired #bbbbbbbb [EMPIRICAL]"
    assert "987654321" not in output
    assert "statistical lower bound" in output.lower()
    assert "confidence is not success probability" in output.lower()
    assert "not a full-shor recovery guarantee" in output.lower()
    assert "exact" not in output.lower()
    assert asdict(result) == before
    assert isolated_history.read_bytes() == history_before


def test_cli_defaults_not_invented_or_forwarded_without_request(monkeypatch, capsys):
    calls = []

    def sweep(bits, **kwargs):
        calls.append(kwargs)
        return [fresh_result(calibrated="success_samples" in kwargs)]

    monkeypatch.setattr(cli, "window_sweep", sweep)
    assert cli.main(["--bits", "16", "--success-samples", "100"]) == 0
    assert calls[-1] == dict(n_trials=64, meas_reps=4, seed=0, verbose=False,
                             backend="optimized", operand_strategy="paired", success_samples=100)
    assert table_rows(capsys.readouterr().out)[0] == cli.SUCCESS_COLUMNS
    assert cli.main(["--bits", "16"]) == 0
    assert calls[-1] == dict(n_trials=64, meas_reps=4, seed=0, verbose=False,
                             backend="optimized", operand_strategy="paired")
    output = capsys.readouterr().out
    assert table_rows(output)[0] == cli.WINDOW_COLUMNS
    assert "no analytic correctness guarantee" in output
    assert "Statistical lower bound" not in output


@pytest.mark.parametrize("options", [
    ["--success-samples", "1"],
    ["--confidence", ".99"], ["--success-batch-size", "2"], ["--calibration-seed", "3"],
    ["--bits", "1", "--confidence", ".99"],
    ["--bits", "1", "--success-batch-size", "2"],
    ["--bits", "1", "--calibration-seed", "3"],
])
def test_calibration_options_require_samples_and_window_before_sweep(
        monkeypatch, capsys, options):
    monkeypatch.setattr(cli, "window_sweep", unexpected)
    monkeypatch.setattr(report, "append_result", unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.main(options)
    assert exc.value.code == 2
    assert "require" in capsys.readouterr().err


@pytest.mark.parametrize("options", [
    ["--success-samples", "0"], ["--success-samples", "-1"],
    ["--success-samples", "nope"],
    ["--success-samples", "1", "--confidence", "0"],
    ["--success-samples", "1", "--confidence", "1"],
    ["--success-samples", "1", "--confidence", "nan"],
    ["--success-samples", "1", "--confidence", "inf"],
    ["--success-samples", "1", "--success-batch-size", "0"],
    ["--success-samples", "1", "--success-batch-size", "-1"],
    ["--success-samples", "1", "--calibration-seed", "1.5"],
])
def test_invalid_calibration_never_builds_or_persists(monkeypatch, capsys, options):
    monkeypatch.setattr(window, "benchmark_case", unexpected)
    monkeypatch.setattr(calibration, "calibrate_window", unexpected)
    monkeypatch.setattr(report, "append_result", unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--bits", "16", *options])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("options,columns", [
    (["--metrics", "success_lower,success_confidence,correct,fail_analytic,p_correct,trials"],
     ["version", "success_lower", "success_confidence", "correct", "fail_analytic", "p_correct", "trials"]),
    (["--hide", "field,peak_qubits"],
     [column for column in cli.SUCCESS_COLUMNS if column not in ("field", "peak_qubits")]),
])
def test_calibration_metrics_preserve_unknown_analytic_claims(
        monkeypatch, capsys, options, columns):
    monkeypatch.setattr(cli, "window_sweep", lambda *a, **kw: [fresh_result()])
    assert cli.main(["--bits", "16", "--success-samples", "100", *options]) == 0
    headers, rows = table_rows(capsys.readouterr().out)
    assert headers == columns
    for name in ("correct", "fail_analytic", "p_correct"):
        if name in rows[0]:
            assert rows[0][name] == "unknown"
    if "trials" in rows[0]:
        assert rows[0]["trials"] == "12"


@pytest.mark.parametrize("value", [.99999999999999, .123456789999, 0.0, 1.0])
def test_report_never_rounds_statistical_lower_bound_up(monkeypatch, value):
    monkeypatch.setattr(report, "load_history", unexpected)
    row = asdict(fresh_result())
    row["success_lower"] = value
    before = deepcopy(row)
    _, rows = table_rows(report.format_rows([row], ["version", "success_lower"]))
    rendered = Decimal(rows[0]["success_lower"])
    assert rendered <= Decimal(str(value))
    assert Decimal(str(value)) - rendered < Decimal("0.0000000001")
    assert row == before


def test_legacy_result_has_unknown_calibration_not_an_invented_bound(monkeypatch):
    monkeypatch.setattr(report, "load_history", unexpected)
    row = asdict(fresh_result(calibrated=False))
    for key in ("success_lower", "success_confidence", "success_samples", "bad_inputs"):
        row.pop(key)
    _, rows = table_rows(report.format_rows([row], cli.SUCCESS_COLUMNS))
    assert all(rows[0][key] == "—" for key in
               ("success_lower", "success_confidence", "success_samples", "bad_inputs"))


def tiny_smoke_case(bits):
    circ = Circuit()
    circ.input("x", 1)
    circ.measure(circ.alloc(1)[0])
    return Case(circ, dict, lambda rng: {"x": 0}, {
        "empirical_only": True, "fail_analytic": None,
        "window_bits": bits, "lookup_backend": "mock.v1",
        "table_fingerprint": "c" * 64, "dummy_policy": "T[1]", "tuning": {},
        "resource_regions": [("arithmetic", 1, len(circ.ops))],
    })


@pytest.mark.parametrize("persist", [True, False])
@pytest.mark.parametrize("explicit_seed", [None, 31])
def test_sweep_calibrates_same_case_after_smoke_before_persistence(
        monkeypatch, isolated_history, persist, explicit_seed):
    events, cases, results, plans = [], [], [], []
    output = io.StringIO()
    bound = object()
    compile_symbolic = symbolic_batch.compile_symbolic

    def compile_once(circ):
        events.append("compile")
        assert circ is cases[-1].circuit
        plan = compile_symbolic(circ)
        plans.append(plan)
        return plan

    def build(bits, *, config):
        events.append("build")
        case = tiny_smoke_case(bits)
        cases.append(case)
        return case

    def benchmark(version, case, **kwargs):
        events.append("smoke")
        assert case is cases[-1]
        assert kwargs == dict(n_trials=4, meas_reps=3, seed=17, persist=False,
                              analysis_plan=plans[-1])
        result = harness.benchmark_case(version, case, **kwargs)
        assert result.analysis_ir_digest == plans[-1].ir_digest
        assert result.trials == 12
        assert result.correct is result.fail_analytic is None
        assert result.success_samples is None
        results.append(result)
        return result

    def calibrate(case, **kwargs):
        events.append("calibrate")
        assert case is cases[-1]
        assert kwargs == dict(samples=19, confidence=.995, batch_size=7,
                             seed=18 if explicit_seed is None else explicit_seed,
                             stream=output, plan=plans[-1])
        return bound

    def attach(result, got_bound):
        events.append("attach")
        assert result is results[-1] and got_bound is bound
        result.success_samples = 19
        result.bad_inputs = 0
        result.success_lower = .005 ** (1 / 19)
        result.success_confidence = .995
        result.clean_input_fraction = 1
        result.circuit_config["success_calibration"] = {"samples": 19, "source_digest": "a" * 64}

    append = report.append_result

    def persist_row(row):
        events.append("persist")
        assert row["trials"] == 12 and row["success_samples"] == 19
        assert row["correct"] is row["fail_analytic"] is None
        append(row)

    monkeypatch.setattr(window, "build_window_case", build)
    monkeypatch.setattr(symbolic_batch, "compile_symbolic", compile_once)
    monkeypatch.setattr(window, "benchmark_case", benchmark)
    monkeypatch.setattr(calibration, "calibrate_window", calibrate)
    monkeypatch.setattr(calibration, "attach_calibration", attach)
    monkeypatch.setattr(report, "append_result", persist_row)
    returned = window.sweep([1], n_trials=4, meas_reps=3, seed=17, persist=persist,
                            stream=output, verbose=False, success_samples=19,
                            confidence=.995, success_batch_size=7, calibration_seed=explicit_seed)
    assert returned == results
    assert events == ["build", "compile", "smoke", "calibrate", "attach"] + (["persist"] if persist else [])
    assert len(plans) == 1
    if persist:
        assert json.loads(isolated_history.read_text()) == [asdict(results[0])]
    else:
        assert not isolated_history.exists()


@pytest.mark.parametrize("stage", ["calibrate", "attach"])
def test_failed_calibration_cannot_persist_partial_result(monkeypatch, isolated_history, stage):
    monkeypatch.setattr(window, "build_window_case", lambda bits, **kw: tiny_smoke_case(bits))
    monkeypatch.setattr(report, "append_result", unexpected)

    def fail(*args, **kwargs):
        raise ValueError("source changed; uncertified")

    monkeypatch.setattr(calibration, "calibrate_window", fail if stage == "calibrate" else lambda *a, **kw: object())
    monkeypatch.setattr(calibration, "attach_calibration", fail if stage == "attach" else unexpected)
    with pytest.raises(ValueError, match="uncertified"):
        window.sweep([1], n_trials=1, success_samples=3, stream=io.StringIO(), verbose=False)
    assert not isolated_history.exists()


def test_ordinary_sweep_never_calibrates_or_changes_analytic_state(monkeypatch):
    monkeypatch.setattr(window, "build_window_case", lambda bits, **kw: tiny_smoke_case(bits))
    monkeypatch.setattr(calibration, "calibrate_window", unexpected)
    monkeypatch.setattr(calibration, "attach_calibration", unexpected)
    monkeypatch.setattr(symbolic_batch, "compile_symbolic", unexpected)
    result, = window.sweep([1], n_trials=2, meas_reps=3, persist=False,
                           stream=io.StringIO(), verbose=False)
    assert result.trials == 6 and result.success_samples is None
    assert result.analysis_ir_digest is None
    assert result.success_lower is result.success_confidence is result.bad_inputs is None
    assert "success_calibration" not in result.circuit_config
    assert result.correct is result.fail_analytic is None


@pytest.mark.parametrize("kwargs", [
    {"success_samples": 0}, {"success_samples": True},
    {"confidence": 0}, {"confidence": 1}, {"confidence": float("nan")},
    {"success_batch_size": 0}, {"success_batch_size": True},
    {"calibration_seed": True}, {"calibration_seed": 2.5},
])
def test_audit_validates_calibration_before_build(monkeypatch, kwargs):
    monkeypatch.setattr(window, "benchmark_case", unexpected)
    monkeypatch.setattr(calibration, "calibrate_window", unexpected)
    with pytest.raises(ValueError):
        window.sweep([1], stream=io.StringIO(), **({"success_samples": 3} | kwargs))
