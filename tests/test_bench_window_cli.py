"""Live bench CLI regressions; only tiny circuits and temporary history are used."""
from copy import deepcopy
from dataclasses import asdict
import json
import runpy
import sys

import pytest

from qsm import params
from qsm.audit import window
from qsm.bench import __main__ as cli
from qsm.bench import harness, registry, report
from test_window_audit import fake_result, tiny_case


def unexpected(*args, **kwargs):
    pytest.fail("unexpected builder, benchmark, or history operation")


@pytest.fixture(autouse=True)
def history_path(tmp_path, monkeypatch):
    path = tmp_path / "benchmarks" / "results.json"
    monkeypatch.setattr(report, "RESULTS_PATH", path)
    # An accidental fallback must never build a production arithmetic circuit.
    monkeypatch.setattr(window, "build_window_case", unexpected)
    monkeypatch.setattr(window, "build_optimized_window_case", unexpected)
    monkeypatch.setattr(registry, "_REGISTRY", {})
    return path


@pytest.fixture
def no_history_comparison(monkeypatch):
    monkeypatch.setattr(report, "compare", unexpected)
    monkeypatch.setattr(report, "latest_by_version", unexpected)


def tiny_window(bits, *, config=None):
    case = tiny_case(bits)
    if config is not None:
        case.meta["circuit_config"] = config.metadata()
    return case


@pytest.fixture
def live_window(monkeypatch):
    events, configs, results = [], [], []

    def build(bits, *, config):
        events.append(("build", bits))
        configs.append(config)
        return tiny_window(bits, config=config)

    def benchmark(version, case, **kwargs):
        events.append(("benchmark", case.meta["window_bits"]))
        result = harness.benchmark_case(version, case, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(window, "build_window_case", build)
    monkeypatch.setattr(window, "build_optimized_window_case", build)
    monkeypatch.setattr(window, "benchmark_case", benchmark)
    return events, configs, results


def result_for(bits=2, *, observed=True):
    case = tiny_case(bits)
    case.meta["circuit_config"] = {"window_bits": bits, "operand_strategy": "paired"}
    result = fake_result(window.window_case_version(case), case, n_trials=3)
    result.observed_pass = observed
    return result


def table_rows(output):
    """Read the simple scalar-column GitHub table, ignoring progress and footer."""
    lines = [line for line in output.splitlines() if line.startswith("|")]
    assert len(lines) >= 2, output
    cells = lambda line: [value.strip() for value in line.strip("|").split("|")]
    headers = cells(lines[0])
    return headers, [dict(zip(headers, cells(line), strict=True)) for line in lines[2:]]


@pytest.mark.parametrize("alias", ["--bits", "--window-bits"])
@pytest.mark.parametrize("backend", ["optimized", "v1"])
def test_live_builds_benchmarks_and_persists_fresh_results(
        history_path, live_window, no_history_comparison, capsys, alias, backend):
    config = (window.OptimizedWindowConfig.tuned(params.U, params.PRIME, 2)
              if backend == "optimized" else None)
    version = window.window_case_version(tiny_window(2, config=config))
    stale = asdict(result_for())
    stale.update(version=version, toffoli=987654321, observed_pass=False, note="STALE ROW")
    report.append_result(stale)
    events, configs, results = live_window
    # Omit --backend for optimized to test the bench CLI's default, not audit's v1.
    options = [] if backend == "optimized" else ["--backend", backend]
    assert cli.main([alias, "2", "1", "2", "--trials", "3", "--meas-reps", "2",
                     "--seed", "17", *options]) == 0
    output = capsys.readouterr().out
    assert events == [(stage, bits) for bits in (2, 1) for stage in ("build", "benchmark")]
    assert len(configs) == 2
    if backend == "optimized":
        assert all(config.operand_strategy == "paired" for config in configs)
        assert all(config.adder_chunk == (params.U + 2) // 2 for config in configs)
    else:
        assert configs == [None, None]
    stored = json.loads(history_path.read_text())
    assert stored == [stale] + [asdict(result) for result in results]
    assert results[0].version == version
    headers, rows = table_rows(output)
    assert headers == ["version", "field", "observed_pass", "trials", "toffoli",
                       "peak_qubits", "measurements"]
    assert len(rows) == 2
    for bits, row, result in zip((2, 1), rows, results, strict=True):
        digest = result.version.rsplit(".", 2)[1]
        strategy = "/paired" if backend == "optimized" else ""
        assert row["version"] == f"window[{bits}] {backend}{strategy} #{digest[:8]} [EMPIRICAL]"
        assert len(digest) == 64 and result.version not in output
        assert result.circuit_config["window_bits"] == bits
        assert row["field"] == params.FIELD and row["observed_pass"] == "True"
        assert row["trials"] == "6" and result.trials == 6
        assert row["toffoli"] == "2" and row["peak_qubits"] == "4"
        assert row["measurements"] == "1"
        assert result.n_trials == result.batches_completed == 3
        assert result.meas_reps == 2 and result.seed == 17
        assert result.correct is result.p_correct is result.fail_analytic is None
    assert "987654321" not in output and "STALE ROW" not in output
    assert "region CCX" not in output and "exact" not in output.lower()
    assert "no analytic correctness guarantee" in output
    assert output.index("Building w=2") < output.index("Building w=1") < output.index("| version")


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_controls_actual_diagnostics(live_window, capsys, verbose):
    args = ["--bits", "2", "--trials", "1", "--meas-reps", "1"]
    assert cli.main(args + (["--verbose"] if verbose else [])) == 0
    output = capsys.readouterr().out
    result, = live_window[2]
    assert ("region CCX" in output) is verbose
    assert (result.version in output) is verbose
    assert ("analytic correctness=unknown" in output) is verbose
    assert ("95% CP upper=" in output) is verbose
    assert ("not quantum norm" in output) is verbose
    assert "Building w=2" in output
    if verbose:
        assert output.index("region CCX") < output.index("| version")
    _, rows = table_rows(output)
    assert len(rows) == 1 and rows[0]["observed_pass"] == "True"


@pytest.mark.parametrize("options, expected", [
    ([], {}),
    (["--backend", "v1"], {"backend": "v1"}),
    (["--operand", "fused", "--adder-chunk", "full", "--fused-block", "128",
      "--low-bits", "2", "--allow-large", "--verbose"],
     {"operand_strategy": "fused", "adder_chunk": None, "fused_block": 128,
      "low_bits": 2, "allow_large": True, "verbose": True}),
    (["--adder-chunk", "7", "--fused-block", "half", "--low-bits", "0"],
     {"adder_chunk": 7, "fused_block": "half", "low_bits": 0}),
    (["--adder-chunk", "half", "--fused-block", "3"],
     {"adder_chunk": "half", "fused_block": 3}),
])
def test_window_sweep_bound_name_receives_only_explicit_options(monkeypatch, options, expected):
    calls = []

    def sweep(bits, **kwargs):
        calls.append((bits, kwargs))
        return [result_for(4)]

    monkeypatch.setattr(cli, "window_sweep", sweep)
    assert cli.main(["--window-bits", "4", *options]) == 0
    assert calls == [([4], dict(n_trials=64, meas_reps=4, seed=0, verbose=False,
                                backend="optimized", operand_strategy="paired") | expected)]
    # Persistence is the sweep's default, not a new --no-persist CLI contract.
    assert "persist" not in calls[0][1]


@pytest.mark.parametrize("chunk, resolved", [("full", None), ("7", 7), ("half", "half")])
def test_fused_options_reach_builder_config(live_window, chunk, resolved):
    assert cli.main(["--bits", "17", "--allow-large", "--backend", "optimized",
                     "--operand", "fused", "--adder-chunk", chunk, "--fused-block", "9",
                     "--low-bits", "3", "--trials", "1", "--meas-reps", "1"]) == 0
    config, = live_window[1]
    assert config.window_bits == 17 and config.operand_strategy == "fused"
    assert config.adder_chunk == ((params.U + 2) // 2 if resolved == "half" else resolved)
    assert config.fused_block == 9 and config.unlookup_low_bits == 3
    # Even this guarded address width is represented by the same four-qubit mock.
    result, = live_window[2]
    assert result.peak_qubits == 4


@pytest.mark.parametrize("options, expected_columns", [
    (["--metrics", "correct,p_correct,log2_fail,fail_analytic,toffoli,peak_qubits",
      "--hide", "toffoli"],
     ["version", "correct", "p_correct", "log2_fail", "fail_analytic", "peak_qubits"]),
    (["--hide", "measurements,field"],
     ["version", "observed_pass", "trials", "toffoli", "peak_qubits"]),
    (["--metrics", "all", "--hide", "circuit_config"],
     [column for column in report.ALL_COLUMNS if column != "circuit_config"]),
])
def test_metrics_hide_and_unknown_analytic_values(live_window, capsys, options, expected_columns):
    assert cli.main(["--bits", "1", "--trials", "1", "--meas-reps", "1", *options]) == 0
    output = capsys.readouterr().out
    headers, rows = table_rows(output)
    assert headers == expected_columns
    row, = rows
    assert "[EMPIRICAL]" in row["version"]
    for column in ("correct", "p_correct", "log2_fail", "fail_analytic"):
        if column in row:
            assert row[column] == "unknown"
    assert "exact" not in output.lower()


@pytest.mark.parametrize("args", [
    ["test.*", "--bits", "1"],
    ["--bits", "1", "--history"],
    ["--bits", "1", "--all"],
    ["--bits", "1", "--exclude", "test.*"],
    ["--bits", "1", "--include-legacy"],
    ["--bits", "1", "--history", "--include-legacy"],
    ["--backend", "optimized"], ["--operand", "paired"],
    ["--adder-chunk", "half"], ["--fused-block", "half"],
    ["--low-bits", "0"], ["--allow-large"], ["--verbose"],
    ["--bits"], ["--bits", "nope"], ["--bits", "0"], ["--bits", "-1"],
    ["--bits", "1", "17"], ["--bits", "1", "13", "--backend", "v1"],
    ["--bits", "1", "--backend", "unknown"],
    ["--bits", "1", "--operand", "unknown"],
    ["--bits", "1", "--backend", "v1", "--operand", "fused"],
    ["--bits", "1", "--backend", "v1", "--adder-chunk", "full"],
    ["--bits", "1", "--backend", "v1", "--low-bits", "0"],
    ["--bits", "1", "--adder-chunk", "0"],
    ["--bits", "1", "--adder-chunk", "nope"],
    ["--bits", "1", "--fused-block", "-1"],
    ["--bits", "1", "--fused-block", "full"],
    ["--bits", "1", "--low-bits", "-1"],
    ["--bits", "4", "1", "--low-bits", "2"],
    ["--bits", "1", "--trials", "0"],
    ["--bits", "1", "--meas-reps", "0"],
    ["--bits", "1", "--metrics", "typo"],
    ["--bits", "1", "--hide", "typo"],
    ["--bits", "1", "--no-persist"],
])
def test_invalid_options_never_build_benchmark_or_write(history_path, monkeypatch, capsys, args):
    report.append_result({"version": "untouched historical row"})
    before = history_path.read_bytes()
    monkeypatch.setattr(window, "benchmark_case", unexpected)
    monkeypatch.setattr(harness, "benchmark", unexpected)
    monkeypatch.setattr(report, "append_result", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert history_path.read_bytes() == before


@pytest.mark.parametrize("observed, status", [
    ([True, True], 0), ([True, False], 1), ([False, True], 1),
    ([True, None], 1), ([None, True], 1),
])
def test_all_results_must_explicitly_pass(monkeypatch, observed, status):
    results = [result_for(bits, observed=value) for bits, value in enumerate(observed, 1)]
    monkeypatch.setattr(cli, "window_sweep", lambda *args, **kwargs: results)
    assert cli.main(["--bits", "1", "2"]) == status


def test_real_empirical_failure_exits_one_and_persists(history_path, monkeypatch, capsys):
    def build(bits, *, config):
        case = tiny_window(bits, config=config)
        case.gen_input = lambda rng: {"x": 1}  # The tiny circuit's Z leaks a phase.
        case.reference = lambda values: {"x": 0}  # Also an independent output mismatch.
        return case

    monkeypatch.setattr(window, "build_optimized_window_case", build)
    assert cli.main(["--bits", "1", "--trials", "2", "--meas-reps", "2"]) == 1
    result, = json.loads(history_path.read_text())
    assert result["observed_pass"] is False and result["correct"] is None
    assert result["trials"] == result["fail_seen"] == result["phase_seen"] == result["output_seen"] == 4
    _, rows = table_rows(capsys.readouterr().out)
    assert rows[0]["observed_pass"] == "False"


@pytest.mark.parametrize("observed, status", [(True, 0), (False, 1), (None, 1)])
def test_module_entrypoint_propagates_window_exit(monkeypatch, observed, status):
    monkeypatch.setattr(window, "sweep", lambda *args, **kwargs: [result_for(observed=observed)])
    monkeypatch.setattr(sys, "argv", ["qsm.bench", "--bits", "2"])
    # runpy gets a fresh CLI namespace; patch the audit binding it imports.
    monkeypatch.delitem(sys.modules, cli.__name__)
    with pytest.raises(SystemExit) as error:
        runpy.run_module("qsm.bench", run_name="__main__")
    assert error.value.code == status


@pytest.mark.parametrize("strategy", [None, "paired", "fused"])
def test_short_labels_are_detached_display_copies(strategy):
    result = result_for(4)
    result.version = f"scalarmul.window_sweep.w4.test.{'0123abcd' * 8}.v1"
    if strategy is None:
        result.circuit_config.pop("operand_strategy")
    else:
        result.circuit_config["operand_strategy"] = strategy
    result.circuit_config["nested"] = {"values": [1, 2]}
    before = asdict(result)
    row, = cli._window_rows([result], "optimized")
    suffix = "" if strategy is None else f"/{strategy}"
    assert row["version"] == f"window[4] optimized{suffix} #0123abcd"
    assert {**row, "version": result.version} == before
    row["circuit_config"]["nested"]["values"].append(3)
    assert asdict(result) == before


@pytest.mark.parametrize("version, has_bits", [
    ("registered.demo.v1", True),
    (f"window.{'a' * 63}.v1", True),
    (f"window.{'a' * 65}.v1", True),
    (f"window.{'z' * 64}.v1", True),
    (f"window.{'a' * 64}.v1", False),
])
def test_ineligible_results_keep_full_display_identifier(version, has_bits):
    result = result_for()
    result.version = version
    if not has_bits:
        result.circuit_config.pop("window_bits")
    assert cli._window_rows([result], "v1") == [asdict(result)]


def test_format_rows_never_loads_history_and_preserves_input(monkeypatch, no_history_comparison):
    monkeypatch.setattr(report, "load_history", unexpected)
    rows = [asdict(result_for())]
    before = deepcopy(rows)
    output = report.format_rows(rows, columns=["version", "correct", "log2_fail", "toffoli"])
    headers, rendered = table_rows(output)
    assert headers == ["version", "correct", "log2_fail", "toffoli"]
    assert rendered[0]["correct"] == rendered[0]["log2_fail"] == "unknown"
    assert rendered[0]["toffoli"] == "2" and "[EMPIRICAL]" in output
    assert "exact" not in output.lower() and rows == before
    assert table_rows(report.format_rows(rows))[0] == report.DEFAULT_COLUMNS


def test_registered_run_renders_returned_result_not_stale_history(
        history_path, monkeypatch, no_history_comparison, capsys):
    version = "test.cli.live.v1"
    stale = asdict(result_for())
    stale.update(version=version, toffoli=987654321, observed_pass=False)
    report.append_result(stale)
    before = history_path.read_bytes()
    monkeypatch.setitem(registry._REGISTRY, version, tiny_case)
    calls = []

    def benchmark(name, **kwargs):
        calls.append((name, kwargs))
        # Deliberately leave history stale, isolating returned-row rendering.
        return harness.benchmark_case(name, registry.get_builder(name)(), persist=False, **kwargs)

    monkeypatch.setattr(harness, "benchmark", benchmark)
    monkeypatch.setattr(cli, "window_sweep", unexpected)
    assert cli.main([version, "--trials", "3", "--meas-reps", "2", "--seed", "19",
                     "--metrics", "observed_pass,trials,toffoli"]) is None
    assert calls == [(version, dict(n_trials=3, meas_reps=2, seed=19))]
    output = capsys.readouterr().out
    _, rows = table_rows(output)
    assert rows == [{"version": f"{version} [EMPIRICAL]", "observed_pass": "True",
                     "trials": "6", "toffoli": "2"}]
    assert "987654321" not in output and "exact" not in output.lower()
    assert history_path.read_bytes() == before


def test_history_mode_stays_read_only(history_path, monkeypatch, capsys):
    row = asdict(result_for())
    row.update(version="saved.window.v1", toffoli=7654321)
    report.append_result(row)
    before = history_path.read_bytes()
    monkeypatch.setattr(cli, "window_sweep", unexpected)
    monkeypatch.setattr(harness, "benchmark", unexpected)
    monkeypatch.setattr(report, "append_result", unexpected)
    assert cli.main(["--history", "saved.*", "--metrics", "observed_pass,toffoli"]) is None
    _, rows = table_rows(capsys.readouterr().out)
    assert rows == [{"version": "saved.window.v1 [EMPIRICAL]", "observed_pass": "True",
                     "toffoli": "7654321"}]
    assert history_path.read_bytes() == before
