#!/usr/bin/env bash
# QSM has its own environment; reference-project environments are not dependencies.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ -e "$VENV_DIR" ]]; then
    if [[ ! -f "$VENV_DIR/pyvenv.cfg" || ! -x "$VENV_DIR/bin/python" ]]; then
        printf 'Refusing to overwrite %s: it is not a usable POSIX virtual environment.\n' "$VENV_DIR" >&2
        exit 1
    fi
    printf 'Reusing virtual environment: %s\n' "$VENV_DIR"
else
    "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "QSM requires Python 3.12 or newer")'
    printf 'Creating virtual environment (%s) in %s ...\n' "$PYTHON" "$VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

ENV_PYTHON="$VENV_DIR/bin/python"
"$ENV_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Existing environment is too old; choose a new VENV_DIR with Python >=3.12")'
"$ENV_PYTHON" -m pip install -e '.[dev]'

printf '\nDone. Activate with: source "%s/bin/activate"\n' "$VENV_DIR"
printf 'Inspect the environment: python -m qsm info\n'
printf 'List implementations:   python -m qsm list\n'
printf 'Command guide:          python -m qsm --help\n'
