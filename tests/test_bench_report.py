import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from qsm import params
from qsm.bench import harness, registry, report
from qsm.core.circuit import Circuit


@pytest.fixture
def history_path(tmp_path, monkeypatch):
    path = tmp_path / "benchmarks" / "results.json"
    monkeypatch.setattr(report, "RESULTS_PATH", path)
    return path


def _row(version="test.report.v1", *, field=None, correct=True, toffoli=101):
    field = report.active_field() if field is None else field
    return {"version": version, "field": field[0], "field_u": field[1],
            "field_prime": field[2], "correct": correct, "toffoli": toffoli,
            "p_correct": 1.0, "log2_fail": -100.0, "note": "diagnostic message"}


def test_latest_is_field_aware_and_keeps_latest_failure(history_path):
    active = report.active_field()
    other = ("other", 256, (1 << 256) - 189)
    report.append_result(_row())
    failure = _row(correct=False, toffoli=123)
    report.append_result(failure)
    foreign = _row(field=other, toffoli=987)
    report.append_result(foreign)
    assert report.latest_by_version() == {failure["version"]: failure}
    assert report.latest_by_version(field=other) == {foreign["version"]: foreign}
    assert report.latest_by_version(field=active) == report.latest_by_version()
    table = report.compare()
    assert "False" in table and "123" in table and "987" not in table
    assert active[0] in table
    assert "correct" in table and "field" in table


@pytest.mark.parametrize("changed", ["field", "field_u", "field_prime"])
def test_every_identity_component_is_required(history_path, changed):
    row = _row()
    row[changed] = "not-active" if changed == "field" else row[changed] + 1
    report.append_result(row)
    assert report.latest_by_version() == {}
    assert "101" not in report.compare([row["version"]])


def test_no_cross_field_fallback(history_path):
    row = _row(field=("foreign", 5, 29))
    report.append_result(row)
    assert report.latest_by_version() == {}
    table = report.compare([row["version"]])
    assert row["version"] in table
    assert "foreign" not in table and "101" not in table


def test_legacy_rows_unchanged_warned_excluded_and_inspectable(history_path):
    legacy = {"version": "legacy.only.v1", "correct": True, "toffoli": 777}
    partial = {**legacy, "version": "legacy.partial.v1", "field": params.FIELD}
    # No field inference, even when a legacy width/count looks familiar.
    report.append_result(legacy)
    report.append_result(partial)
    report.append_result(_row())
    assert report.load_history()[:2] == [legacy, partial]
    with pytest.warns(report.UnknownProvenanceWarning, match="Excluded 2"):
        assert list(report.latest_by_version()) == ["test.report.v1"]
    assert report.latest_by_version(legacy=True) == {
        legacy["version"]: legacy, partial["version"]: partial}
    with pytest.warns(report.UnknownProvenanceWarning):
        ordinary = report.compare()
    assert "777" not in ordinary
    with pytest.warns(report.UnknownProvenanceWarning):
        explicit = report.compare(include_legacy=True)
    assert "777" in explicit and "unknown" in explicit
    assert report.load_history()[:2] == [legacy, partial]
    with pytest.raises(ValueError, match="legacy"):
        report.latest_by_version(field=report.active_field(), legacy=True)


def test_legacy_cannot_overwrite_current_same_version(history_path):
    current = _row(correct=False)
    report.append_result(current)
    report.append_result({"version": current["version"], "correct": True, "toffoli": 999})
    with pytest.warns(report.UnknownProvenanceWarning):
        assert report.latest_by_version()[current["version"]] == current
    with pytest.warns(report.UnknownProvenanceWarning):
        table = report.compare(include_legacy=True)
    assert "False" in table and "unknown" in table
    assert table.count(current["version"]) == 2


def test_columns_notes_and_version_filter_preserved(history_path):
    report.append_result(_row("test.keep.v1", correct=False))
    report.append_result(_row("test.drop.v1", toffoli=999))
    columns = report.resolve_columns(["note", "correct", "field"])
    assert columns == ["version", "note", "correct", "field"]
    table = report.compare(["test.keep.v1"], columns)
    assert "diagnostic message" in table and "False" in table
    assert "test.drop.v1" not in table and "999" not in table
    assert "note" in report.resolve_columns(["all"])
    assert "p_correct" not in report.resolve_columns(hide=["p_correct"])
    with pytest.raises(ValueError, match="unknown metric"):
        report.resolve_columns(["not-a-column"])
    assert "test.keep.v1" in report.compare(columns=["correct", "version"])


def test_atomic_append_and_legacy_preservation(history_path, monkeypatch):
    original = [{"version": "old", "extra": {"retained": True}}]
    history_path.parent.mkdir()
    history_path.write_text(json.dumps(original))
    previous_bytes = history_path.read_bytes()

    def failed_replace(source, destination):
        assert destination == history_path
        assert json.loads(source.read_text()) == original + [_row()]
        assert history_path.read_bytes() == previous_bytes
        raise OSError("injected replacement failure")

    with monkeypatch.context() as patch:
        patch.setattr(report.os, "replace", failed_replace)
        with pytest.raises(OSError, match="injected"):
            report.append_result(_row())
    assert history_path.read_bytes() == previous_bytes
    assert list(history_path.parent.iterdir()) == [history_path]
    report.append_result(_row())
    assert report.load_history() == original + [_row()]
    assert list(history_path.parent.iterdir()) == [history_path]


def test_concurrent_appends_do_not_lose_rows(history_path):
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda i: report.append_result(_row(f"test.concurrent.{i}")), range(30)))
    rows = report.load_history()
    assert len(rows) == 30
    assert {r["version"] for r in rows} == {f"test.concurrent.{i}" for i in range(30)}


def test_bad_json_or_unserializable_result_does_not_destroy_history(history_path):
    report.append_result(_row())
    original = history_path.read_bytes()
    with pytest.raises(TypeError):
        report.append_result({"version": "bad", "value": object()})
    assert history_path.read_bytes() == original
    history_path.write_text("truncated history")
    with pytest.raises(json.JSONDecodeError):
        report.append_result(_row())
    assert history_path.read_text() == "truncated history"


def test_explicit_path_api(tmp_path):
    path = tmp_path / "other.json"
    assert report.load_history(path) == []
    report.append_result(_row(), path)
    assert report.load_history(path) == [_row()]


def test_harness_result_serializes_contract_and_field(history_path, monkeypatch):
    c = Circuit()
    c.input("x", 1)
    meta = {"input_distribution": "uniform bit", "arithmetic_contract": {"exact": True}}
    case = registry.Case(c, dict, lambda rng: {"x": rng.getrandbits(1)}, meta)
    monkeypatch.setitem(registry._REGISTRY, "test.report.v1", lambda: case)
    # Tests never let benchmark() append to the developer's local cache.
    result = harness.benchmark("test.report.v1", n_trials=3, persist=False)
    report.append_result(asdict(result))
    row = report.load_history()[0]
    assert report.field_identity(row) == (params.FIELD, params.U, params.PRIME)
    assert row["input_distribution"] == meta["input_distribution"]
    assert row["arithmetic_contract"] == meta["arithmetic_contract"]
    assert row["n_trials"] == row["inputs_executed"] == row["trials"] == 3


def test_real_profiles_share_history_without_cross_field_fallback(history_path):
    from test_field_profile import in_field

    for field, width, deficit in [("25519", 255, 19), ("secp256k1", 256, (1 << 32) + 977)]:
        row = in_field(field, f'''
import json
from dataclasses import asdict
from pathlib import Path
from qsm.bench import demos, harness, report
report.RESULTS_PATH = Path({str(history_path)!r})
result = harness.benchmark("demo.and.v1", n_trials=2, persist=False)
report.append_result(asdict(result))
print("JSON" + json.dumps(report.latest_by_version()["demo.and.v1"]))
''')
        assert (row["field"], row["field_u"], row["field_prime"]) == (
            field, width, (1 << width) - deficit)
        assert row["correct"]
    assert len(report.load_history()) == 2
    assert report.latest_by_version()["demo.and.v1"]["field"] == params.FIELD


def test_old_result_dictionary_still_loads_with_unknown_provenance():
    row = {"version": "old", "correct": True, "n_trials": 1, "toffoli": 0,
           "clifford": 0, "measurements": 0, "peak_qubits": 0, "depth": 0, "n_gates": 0}
    result = harness.Result(**row)
    assert result.field is None and result.field_u is None and result.field_prime is None
    assert result.fail_bound95 == 1


def test_cli_history_is_read_only_and_preserves_globs(history_path, monkeypatch, capsys):
    from qsm.bench import __main__ as cli

    report.append_result(_row("test.keep.v1", correct=False))
    report.append_result(_row("test.drop.v1"))
    report.append_result({"version": "test.legacy.v1", "correct": True, "toffoli": 999})
    before = history_path.read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("history inspection must not benchmark")
    monkeypatch.setattr(harness, "benchmark", unexpected)
    monkeypatch.setattr("sys.argv", ["qsm.bench", "--history", "--include-legacy",
                                     "test.*", "--exclude", "test.drop.*"])
    with pytest.warns(report.UnknownProvenanceWarning):
        cli.main()
    table = capsys.readouterr().out
    assert "test.keep.v1" in table and "False" in table
    assert "test.legacy.v1" in table and "unknown" in table
    assert "test.drop.v1" not in table
    assert history_path.read_bytes() == before


@pytest.mark.parametrize("args", [["--trials", "0"], ["--meas-reps", "-1"],
                                  ["--include-legacy"]])
def test_cli_rejects_invalid_options(history_path, monkeypatch, args):
    from qsm.bench import __main__ as cli

    monkeypatch.setattr("sys.argv", ["qsm.bench", *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not history_path.exists()
