"""Descriptive benchmark files keep precise provenance without overwriting runs."""
from pathlib import Path

import pytest

from test_paper_tables import paper

names = paper.benchmark_files


@pytest.mark.parametrize("table,stem", [
    ("table1", "square-breakdown"),
    ("table2", "field-primitives"),
    ("table3", "constant-point-addition"),
    ("table4", "scalar-schedules"),
    ("table5", "odd-encoding-savings"),
    ("table6", "resource-comparison"),
    ("signed", "signed-scalar-comparison"),
])
def test_result_names_are_descriptive(table, stem):
    paths = names.paths(Path("run"), table)
    assert {key: path.name for key, path in paths.items()} == {
        ext: f"{stem}.{ext}" for ext in ("json", "tex", "md")}


def test_default_directory_is_readable_and_same_second_runs_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setattr(names, "timestamp", lambda: "2026-10-16_12-34-56Z")
    first = names.create_directory(tmp_path, "table1")
    assert first == tmp_path/"runs/benchmarks/square-breakdown/2026-10-16_12-34-56Z"
    sentinel = first/"square-breakdown.json"
    sentinel.write_text("first measurement")
    second = names.create_directory(tmp_path, "table1")
    assert second == first.with_name(first.name+"-2")
    assert sentinel.read_text() == "first measurement"
    assert list(second.iterdir()) == []


def test_explicit_output_directory_is_respected_and_never_overwritten(tmp_path):
    requested = tmp_path/"review/squares"
    assert names.create_directory(tmp_path, "table1", requested) == requested
    with pytest.raises(FileExistsError):
        names.create_directory(tmp_path, "table1", requested)


def test_older_run_directories_are_untouched(tmp_path):
    old = tmp_path/"runs/paper/table1-20261015T162822212668Z"
    old.mkdir(parents=True)
    (old/"table.tex").write_text("archived table")
    new = names.create_directory(tmp_path, "table1")
    assert new != old and (old/"table.tex").read_text() == "archived table"


def test_file_at_output_parent_is_an_error_not_a_collision_loop(tmp_path):
    parent = tmp_path/"runs/benchmarks"
    parent.mkdir(parents=True)
    (parent/"square-breakdown").write_text("keep")
    with pytest.raises(FileExistsError):
        names.create_directory(tmp_path, "table1")
