"""Append-ordered local history and field-aware comparison tables.

Rows are not keyed by git commit or timestamp: last append wins for a (field name,
width, modulus, version). Legacy rows remain unchanged and have unknown provenance.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import warnings
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

from tabulate import tabulate

from .. import params

FieldIdentity = tuple[str, int, int]


class UnknownProvenanceWarning(UserWarning):
    """History contains rows that cannot enter an active-field comparison."""


def active_field() -> FieldIdentity:
    return params.FIELD, params.U, params.PRIME


def field_identity(row: dict) -> FieldIdentity | None:
    """Use only explicit, complete provenance; never infer an old row's field."""
    name, width, prime = (row.get(k) for k in ("field", "field_u", "field_prime"))
    if (not isinstance(name, str) or not name or
            type(width) is not int or width <= 0 or
            type(prime) is not int or prime <= 0):
        return None
    return name, width, prime

RESULTS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "results.json"

# Every metric the harness persists. `compare()` shows DEFAULT_COLUMNS unless told
# otherwise, so a metric we don't currently care about can be switched off without
# losing it from the stored history (results.json always keeps everything).
ALL_COLUMNS = [
    "version", "field", "field_u", "field_prime",
    "input_distribution", "arithmetic_contract", "seed", "circuit_config",
    "empirical_only", "observed_pass", "output_seen",
    "success_lower", "success_confidence", "success_samples", "bad_inputs", "clean_input_fraction",
    "analysis_ir_digest",
    # p_correct is the ANALYTIC chance of a right answer
    # on one call; log2_fail is the same thing in the readable 2^-k form.
    "p_correct", "log2_fail", "fail_analytic",
    # what the sampling alone established (much weaker than log2_fail -- see harness)
    "fail_seen", "dirty_seen", "phase_seen", "trials", "fail_bound95",
    # regression flag: False = misbehaved far more than designed, or leaked phase
    "correct", "note", "n_trials", "inputs_executed", "batches_completed",
    "batches_failed", "meas_reps", "aborted", "batch_fail_analytic",
    "regression_pvalue", "regression_alpha",
    # primitive counts
    "toffoli", "peak_qubits", "clifford", "measurements", "n_gates",
    # runtime axes
    "depth", "toffoli_depth",
    # cost models
    "circuit_volume", "toffoli_volume", "volume",
]

# Curated default: correctness as a probability (log2_fail is the legible form), the two
# primitives everything is judged on, plus the conventional 2D-surface-code cost
# (circuit_volume, == the ecdsa.fail score) and its transient-aware refinement.
# Edit this list, or pass columns=/--metrics, to change it.
DEFAULT_COLUMNS = [
    "version", "field", "correct", "p_correct", "log2_fail",
    "toffoli", "peak_qubits", "circuit_volume", "toffoli_volume",
]

_ALWAYS_FIRST = "version"


def resolve_columns(
    metrics: list[str] | None = None,
    hide: list[str] | None = None,
) -> list[str]:
    """Pick display columns. `metrics` selects (or "all"); `hide` drops from the result.

    Unknown names raise, so a typo in --metrics fails loudly instead of silently
    dropping a column.
    """
    if metrics is None:
        cols = list(DEFAULT_COLUMNS)
    elif len(metrics) == 1 and metrics[0] == "all":
        cols = list(ALL_COLUMNS)
    else:
        cols = list(metrics)

    for name in cols + list(hide or []):
        if name not in ALL_COLUMNS:
            raise ValueError(
                f"unknown metric {name!r}; choose from {', '.join(ALL_COLUMNS)} (or 'all')"
            )

    if hide:
        cols = [c for c in cols if c not in hide]
    # keep `version` present and leftmost so rows stay identifiable
    cols = [c for c in cols if c != _ALWAYS_FIRST]
    return [_ALWAYS_FIRST] + cols


def load_history(path: Path | None = None) -> list[dict]:
    """Read every row, including unchanged legacy records."""
    path = RESULTS_PATH if path is None else Path(path)
    try:
        with path.open() as f:
            history = json.load(f)
    except FileNotFoundError:
        return []
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError("benchmark history must be a list of result objects")
    return history


def append_result(result: dict, path: Path | None = None) -> None:
    """Atomically replace history; serialize concurrent cooperating POSIX writers."""
    path = RESULTS_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock the stable directory inode, not the file that replace() swaps out.
    # This also avoids leaving an unignored lock file beside the ignored history.
    lock = os.open(path.parent, os.O_RDONLY)
    temporary = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        history = load_history(path)
        history.append(result)
        payload = json.dumps(history, indent=2, allow_nan=False)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        os.close(lock)


def latest_by_version(
    *, field: FieldIdentity | None = None, legacy: bool = False,
) -> dict[str, dict]:
    """Latest row per version for one field (active by default), even if failed.

    `legacy=True` inspects ONLY unknown-provenance rows, never a fallback field.
    `load_history()` remains the lossless API for all records and all fields.
    """
    if legacy and field is not None:
        raise ValueError("legacy inspection cannot select a known field")
    selected = None if legacy else (active_field() if field is None else field)
    latest: dict[str, dict] = {}
    unknown = 0
    for row in load_history():
        identity = field_identity(row)
        unknown += int(identity is None)
        if identity == selected:
            latest[row["version"]] = row
    if unknown and not legacy:
        warnings.warn(
            f"Excluded {unknown} unknown-provenance benchmark row(s) from the field "
            "comparison; inspect with --history --include-legacy or "
            "latest_by_version(legacy=True).",
            UnknownProvenanceWarning, stacklevel=2,
        )
    return latest


# Probabilities near 1 are unreadable at default precision (1 - 5.4e-13 rounds to "1"),
# so each is formatted explicitly rather than left to tabulate.
_FMT = {
    "p_correct": lambda v: f"{v:.15g}",
    "log2_fail": lambda v: f"2^{v:.1f}",
    "fail_analytic": lambda v: f"{v:.3e}",
    "fail_bound95": lambda v: f"{v:.2e}",
    "success_lower": lambda v: format(Decimal(str(v)).quantize(
        Decimal("0.0000000001"), rounding=ROUND_FLOOR), "f"),
    "success_confidence": lambda v: f"{v:.6g}",
    "clean_input_fraction": lambda v: f"{v:.10g}",
}


def _fmt(col: str, val, *, empirical_only: bool = False) -> str:
    if empirical_only and col == "version":
        return f"{val} [EMPIRICAL]"
    if val is None:
        if empirical_only and col in ("log2_fail", "fail_analytic", "p_correct",
                                      "batch_fail_analytic", "regression_pvalue", "correct"):
            return "unknown"
        return "exact" if col in ("log2_fail",) else "—"
    f = _FMT.get(col)
    if f is None or not isinstance(val, (int, float)) or isinstance(val, bool):
        return val
    try:
        return f(val)
    except (TypeError, ValueError):  # pragma: no cover
        return val


def format_rows(rows, columns: list[str] | None = None) -> str:
    """Render supplied result dictionaries without consulting saved history."""
    cols = list(columns) if columns is not None else resolve_columns()
    cells = [[_fmt(c, row.get(c, "—"),
                   empirical_only=row.get("empirical_only", False)) for c in cols]
             for row in rows]
    return tabulate(cells, headers=cols, tablefmt="github", disable_numparse=True)


def compare(
    versions: list[str] | None = None,
    columns: list[str] | None = None,
    *,
    field: FieldIdentity | None = None,
    include_legacy: bool = False,
) -> str:
    """Render latest rows for one field; optional legacy rows stay separately labelled."""
    cols = list(columns) if columns is not None else resolve_columns()
    latest = latest_by_version(field=field)
    legacy = latest_by_version(legacy=True) if include_legacy else {}
    keys = versions if versions is not None else sorted(latest.keys() | legacy.keys())
    rows = []
    for v in keys:
        candidates = []
        if v in latest:
            candidates.append(latest[v])
        if v in legacy:
            candidates.append({**legacy[v], "field": "unknown", "field_u": None,
                               "field_prime": None})
        if not candidates:
            candidates.append({"version": v})
        rows.extend(candidates)
    return format_rows(rows, columns=cols)
