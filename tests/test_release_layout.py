"""Public commands resolve this standalone installation without research archives."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import qsm

ROOT = Path(__file__).resolve().parents[1]


def isolated_environment(field="25519"):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["QSM_FIELD"] = field
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def test_package_and_runtime_assets_belong_to_this_distribution():
    assert Path(qsm.__file__).resolve().is_relative_to(ROOT/"qsm")
    for name in ("paper", "setup.sh", "pyproject.toml", "README.md",
                 "scripts/paper_tables.py", "scripts/benchmark_files.py",
                 "scripts/signed_paper_bench.py", "scripts/odd_savings_bench.py",
                 "scripts/bench_zero_safe_signed.py"):
        assert (ROOT/name).is_file()
    assert os.access(ROOT/"paper", os.X_OK) and os.access(ROOT/"setup.sh", os.X_OK)


@pytest.mark.parametrize("field,bits", [("25519", 255), ("secp256k1", 256)])
def test_cli_resolves_release_from_an_unrelated_working_directory(tmp_path, field, bits):
    result = subprocess.run([sys.executable, "-I", "-B", "-m", "qsm", "--field", field, "info"],
                            cwd=tmp_path, env=isolated_environment(),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    info = json.loads(result.stdout)
    assert Path(info["source_directory"]).resolve() == ROOT/"qsm"
    assert info["field"] == field and info["field_bits"] == bits
    assert info["virtual_environment"] is True
    assert not info["dependency_errors"] and len(info["source_digest"]) == 64
    assert not list(tmp_path.iterdir())


def test_all_six_table_commands_are_available_without_building(tmp_path):
    result = subprocess.run([str(ROOT/"paper"), "--list"], cwd=tmp_path,
                            env=isolated_environment(), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    for i in range(1, 7):
        assert f"table{i} " in result.stdout
    assert "signed " in result.stdout
    assert not list(tmp_path.iterdir())
