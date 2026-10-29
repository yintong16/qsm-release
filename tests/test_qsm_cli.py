"""Launcher tests must not build full-width circuits or modify saved history."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qsm import cli

ROOT = Path(__file__).resolve().parents[1]


def invoke(*arguments, env_field="25519"):
    env = dict(os.environ, QSM_FIELD=env_field, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "qsm", *arguments], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=30)


@pytest.fixture
def launched(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", run)
    return calls


def test_import_and_global_help_do_not_import_arithmetic():
    code = ("import sys; import qsm.cli as c; "
            "assert c.main([])==0; "
            "assert 'qsm.params' not in sys.modules; "
            "assert 'qsm.bench.__main__' not in sys.modules; "
            "assert 'qsm.point' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command,module,args", [
    ("bench", "qsm.bench", ["field.*", "--all", "--trials", "4"]),
    ("test", "pytest", ["tests/test_field_canonical.py", "-q"]),
    ("list", "qsm.cli", ["list", "field.*", "--json"]),
    ("info", "qsm.cli", ["info"]),
])
def test_arguments_pass_through_fresh_process(launched, command, module, args):
    tail = args[1:] if command in ("info", "list") else args
    assert cli.main(["--field", "secp256k1", command, *tail]) == 0
    call, options = launched[0]
    assert call == [sys.executable, "-m", module, *args]
    assert options["env"]["QSM_FIELD"] == "secp256k1"
    assert options["check"] is False
    assert "shell" not in options and "cwd" not in options


def test_environment_is_not_mutated(launched, monkeypatch):
    monkeypatch.setenv("QSM_FIELD", "25519")
    cli.main(["--field", "secp256k1", "info"])
    assert os.environ["QSM_FIELD"] == "25519"


@pytest.mark.parametrize("command", ["bench", "test"])
def test_no_argument_heavy_commands_only_show_help(launched, command):
    cli.main([command])
    assert launched[0][0][-1] == "--help"


@pytest.mark.parametrize("topic", ["history", "guarded-window"])
def test_field_argument_modules_receive_selected_field(launched, topic):
    cli.main(["--field", "secp256k1", "audit", topic, "--samples", "2"])
    assert launched[0][0][-4:] == ["--field", "secp256k1", "--samples", "2"]
    assert launched[0][1]["env"]["QSM_FIELD"] == "secp256k1"


def test_local_field_and_environment_are_kept_consistent(launched, monkeypatch):
    monkeypatch.setenv("QSM_FIELD", "25519")
    cli.main(["audit", "history", "--field=secp256k1", "--samples", "1"])
    assert launched[0][1]["env"]["QSM_FIELD"] == "secp256k1"
    assert launched[0][0].count("--field=secp256k1") == 1


@pytest.mark.parametrize("topic", ["history", "guarded-window"])
def test_custom_prime_does_not_get_conflicting_field_argument(launched, topic):
    cli.main(["audit", topic, "--prime", "31", "--exhaustive"])
    assert "--field" not in launched[0][0]


@pytest.mark.parametrize("arguments", [
    ["--field", "p256", "info"],
    ["--field", "ed25519", "info"],
    ["unknown"],
    ["audit", "unknown"],
    ["--field", "25519", "audit", "history", "--field", "secp256k1"],
    ["audit", "history", "--field", "25519", "--field", "secp256k1"],
    ["audit", "history", "--field"],
    ["audit", "port", "--samples", "1"],
])
def test_invalid_routes_do_not_launch(launched, arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2
    assert launched == []


@pytest.mark.parametrize("topic", [*cli.AUDITS, *cli.LEGACY_AUDITS])
def test_every_audit_route_and_help_is_safe(launched, topic):
    assert cli.main(["audit", topic, "--help"]) == 0
    if topic in cli.LEGACY_AUDITS:
        assert not launched
    else:
        assert launched[0][0][2] == cli.AUDITS[topic][0]
        assert launched[0][0][-1] == "--help"


def test_legacy_aggregate_is_explicit(launched):
    cli.main(["audit", "all"])
    assert launched[0][0] == [sys.executable, "-m", "qsm.audit", "all"]


def test_underscore_alias_preserves_module_options(launched):
    cli.main(["audit", "full_scalar", "--variants", "odd_torsion", "--json", "new.json"])
    assert launched[0][0][2:] == ["qsm.audit.full_scalar", "--variants", "odd_torsion", "--json", "new.json"]


@pytest.mark.parametrize("status,expected", [(0,0), (1,1), (2,2), (-15,143)])
def test_child_exit_status_is_preserved(monkeypatch, status, expected):
    monkeypatch.setattr(cli.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=status))
    assert cli.main(["info"]) == expected


def test_interrupt_returns_shell_interrupt_status(monkeypatch):
    def interrupted(*args, **kwargs): raise KeyboardInterrupt
    monkeypatch.setattr(cli.subprocess,"run",interrupted)
    assert cli.main(["info"]) == 130


def test_info_selects_field_before_child_imports():
    result = invoke("--field", "secp256k1", "info", env_field="25519")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["field"] == "secp256k1" and data["field_bits"] == 256
    assert data["prime"] == (1 << 256)-(1 << 32)-977
    assert data["dependency_errors"] == {}
    assert len(data["source_digest"]) == 64


def test_inherited_field_and_explicit_override():
    result = invoke("info", env_field="secp256k1")
    assert json.loads(result.stdout)["field"] == "secp256k1"
    result = invoke("--field", "25519", "info", env_field="secp256k1")
    assert json.loads(result.stdout)["field_bits"] == 255


def test_registered_listing_and_help_leave_history_untouched():
    history = ROOT/"benchmarks/results.json"
    before = history.read_bytes() if history.exists() else None
    result = invoke("list", "demo.*", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["versions"] == ["demo.and.v1", "demo.xor.v1"]
    for arguments in (("--help",), ("audit", "--help"), ("bench", "--help"),
                      ("audit", "full-scalar", "--help"), ("audit", "port", "--help")):
        result = invoke(*arguments)
        assert result.returncode == 0, (arguments, result.stderr)
    after = history.read_bytes() if history.exists() else None
    assert before == after


def test_actual_history_profile_propagation_and_overwrite_guard(tmp_path):
    output = tmp_path/"history.json"
    result = invoke("--field", "secp256k1", "audit", "history", "--samples", "1", "--json", str(output))
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["p"] == (1 << 256)-(1 << 32)-977
    before = output.read_bytes()
    result = invoke("audit", "history", "--prime", "31", "--samples", "1", "--json", str(output))
    assert result.returncode != 0
    assert output.read_bytes() == before


def test_actual_small_prime_study_does_not_inject_field(tmp_path):
    output = tmp_path/"toy.json"
    result = invoke("audit", "history", "--prime", "31", "--samples", "1", "--json", str(output))
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["p"] == 31


@pytest.mark.skipif(os.name == "nt", reason="POSIX setup helper")
def test_setup_rejects_non_environment_without_overwriting(tmp_path):
    script = tmp_path/"setup.sh"
    script.write_bytes((ROOT/"setup.sh").read_bytes())
    existing = tmp_path/".venv"
    existing.mkdir()
    marker = existing/"user-file"
    marker.write_text("keep")
    result = subprocess.run(["bash",str(script)], env=dict(os.environ, VENV_DIR=".venv", PYTHON=sys.executable), capture_output=True,text=True,timeout=10)
    assert result.returncode == 1 and "Refusing to overwrite" in result.stderr
    assert marker.read_text() == "keep"


@pytest.mark.skipif(os.name == "nt", reason="POSIX setup helper")
def test_setup_reuses_environment_and_does_not_upgrade_tools(tmp_path):
    script = tmp_path/"setup.sh"
    script.write_bytes((ROOT/"setup.sh").read_bytes())
    existing = tmp_path/".venv"
    (existing/"bin").mkdir(parents=True)
    config = existing/"pyvenv.cfg"
    config.write_text("test environment")
    interpreter = existing/"bin/python"
    interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    interpreter.chmod(0o755)
    log = tmp_path/"calls"
    result = subprocess.run(["bash",str(script)],env=dict(os.environ,CALL_LOG=str(log),VENV_DIR=".venv",PYTHON=sys.executable),capture_output=True,text=True,timeout=10)
    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "-m pip install -e .[dev]" in calls
    assert "-m venv" not in calls and "--upgrade" not in calls
    assert config.read_text() == "test environment"
