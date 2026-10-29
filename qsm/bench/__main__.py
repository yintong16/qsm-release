"""Live circuit benchmarks and saved-history inspection.

`--bits` builds parameterized windows and prints the fresh results table; `--history`
only reads saved rows. Otherwise the settled list is hidden, `--all` includes it,
and naming a registered version always runs it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib

from . import demos  # noqa: F401  — importing registers the demo circuits
from . import harness, registry, report, settled

# Importing each layer package runs its circuit modules' @circuit_version decorators,
# so every registered version becomes visible to the CLI. Each layer's __init__.py is
# responsible for importing its own circuit submodules. Missing/empty layers are skipped.
_LAYERS = ("qsm.field", "qsm.mul", "qsm.inversion", "qsm.point", "qsm.scalarmul")
for _layer in _LAYERS:
    try:
        importlib.import_module(_layer)
    except ImportError:
        pass


from ..audit.window import _block_choice, sweep as window_sweep


WINDOW_COLUMNS = ["version", "field", "observed_pass", "trials", "toffoli",
                  "peak_qubits", "measurements"]
SUCCESS_COLUMNS = ["version", "field", "success_samples", "bad_inputs", "success_lower",
                   "success_confidence", "toffoli", "peak_qubits"]


def _window_rows(results, backend):
    rows = []
    for result in results:
        row = asdict(result)
        cfg = result.circuit_config
        parts = result.version.rsplit(".", 2)
        digest = parts[-2] if len(parts) > 1 else ""
        if ("window_bits" in cfg and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest)):
            label = f"window[{cfg['window_bits']}] {backend}"
            if "operand_strategy" in cfg:
                label += f"/{cfg['operand_strategy']}"
            row["version"] = f"{label} #{digest[:8]}"
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int | None:
    ap = argparse.ArgumentParser(
        prog="qsm.bench", description="Build and benchmark circuit versions or live window configurations."
    )
    ap.add_argument(
        "versions",
        nargs="*",
        help=("version keys or globs (e.g. 'field.modadd.*'). Naming a version runs it "
              "even if it is settled. Default: all registered except the settled list."),
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="include settled versions too (see qsm/bench/settled.py)",
    )
    ap.add_argument(
        "--exclude",
        help="comma-separated version keys or globs to drop (applied last, always wins)",
    )
    ap.add_argument("--trials", type=int, default=64, help="random inputs per version")
    ap.add_argument("--meas-reps", type=int, default=4, help="measurement shots per input")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history", action="store_true", help="inspect history without running circuits")
    ap.add_argument("--include-legacy", action="store_true",
                    help="with --history, also show unknown-provenance rows separately")
    ap.add_argument(
        "--metrics",
        help=("comma-separated columns to display, or 'all'. "
              f"Available: {', '.join(report.ALL_COLUMNS)}. "
              f"Default: {','.join(report.DEFAULT_COLUMNS)}. "
              "Live windows default to observed checks and resource counts."),
    )
    ap.add_argument("--hide", help="comma-separated columns to drop from the table")
    window = ap.add_argument_group("live window benchmarks")
    window.add_argument("--bits", "--window-bits", dest="window_bits", nargs="+", type=int,
                        help="build and test these window widths, then print a fresh results table")
    window.add_argument("--backend", choices=("v1", "optimized"), default=argparse.SUPPRESS,
                        help="window backend (default optimized)")
    window.add_argument("--operand", dest="operand_strategy", choices=("paired", "fused"),
                        default=argparse.SUPPRESS, help="window operands (default paired)")
    window.add_argument("--adder-chunk", type=_block_choice, default=argparse.SUPPRESS,
                        help="half (default), full, or a positive chunk size")
    window.add_argument("--fused-block", type=_block_choice, default=argparse.SUPPRESS,
                        help="half (default), or a positive block size")
    window.add_argument("--low-bits", type=int, default=argparse.SUPPRESS,
                        help="measured-unlookup split (default half the address width)")
    window.add_argument("--allow-large", action="store_true", default=argparse.SUPPRESS,
                        help="override the backend's window-size memory guard")
    window.add_argument("--verbose", action="store_true",
                        help="also show window diagnostics and full configuration identifiers")
    window.add_argument("--success-samples", type=int, default=argparse.SUPPRESS,
                        help="calibrate an all-measurement-clean input success bound with this fixed sample count")
    window.add_argument("--confidence", type=float, default=argparse.SUPPRESS,
                        help="confidence of the statistical success bound (default 0.99)")
    window.add_argument("--success-batch-size", type=int, default=argparse.SUPPRESS,
                        help="number of concrete input lanes evaluated together (default 1024)")
    window.add_argument("--calibration-seed", type=int, default=argparse.SUPPRESS,
                        help="independent calibration RNG seed (default smoke seed + 1)")
    args = ap.parse_args(argv)
    if args.trials < 1 or args.meas_reps < 1:
        ap.error("--trials and --meas-reps must be positive")
    if args.include_legacy and not args.history:
        ap.error("--include-legacy requires --history")

    window_options = {key: getattr(args, key) for key in (
        "backend", "operand_strategy", "adder_chunk", "fused_block", "low_bits", "allow_large",
        "success_samples", "confidence", "success_batch_size", "calibration_seed",
    ) if hasattr(args, key)}
    if (any(key in window_options for key in ("confidence", "success_batch_size", "calibration_seed"))
            and "success_samples" not in window_options):
        ap.error("calibration settings require --success-samples")
    if args.window_bits is None and (window_options or args.verbose):
        ap.error("window settings and --verbose require --bits/--window-bits")
    if args.window_bits is not None and (
            args.history or args.include_legacy or args.versions or args.all or args.exclude):
        ap.error("--bits runs live windows; it cannot be combined with --history or version filters")

    def _split(s: str | None) -> list[str] | None:
        return [x.strip() for x in s.split(",") if x.strip()] if s else None

    try:
        metrics = _split(args.metrics)
        if metrics is None and args.window_bits is not None:
            metrics = SUCCESS_COLUMNS if "success_samples" in window_options else WINDOW_COLUMNS
        columns = report.resolve_columns(metrics, _split(args.hide))
    except ValueError as e:
        ap.error(str(e))

    if args.window_bits is not None:
        window_options.setdefault("backend", "optimized")
        window_options.setdefault("operand_strategy", "paired")
        try:
            results = window_sweep(
                args.window_bits, n_trials=args.trials, meas_reps=args.meas_reps,
                seed=args.seed, verbose=args.verbose, **window_options)
        except ValueError as e:
            ap.error(str(e))
        print()
        print(report.format_rows(_window_rows(results, window_options["backend"]), columns))
        if "success_samples" in window_options:
            print("Statistical lower bound for all-outcome-clean canonical window execution on "
                  "the declared conditional input law; confidence is not success probability. "
                  "This is not a full-Shor recovery guarantee.")
        else:
            print("Observed checks only; no analytic correctness guarantee. "
                  "Full configuration identifiers are saved in history (--verbose shows them).")
        return 0 if all(r.observed_pass is True for r in results) else 1

    if args.history:
        registered = sorted({row["version"] for row in report.load_history()
                             if report.field_identity(row) == report.active_field() or
                             (args.include_legacy and report.field_identity(row) is None)})
    else:
        registered = registry.all_versions()
    versions = settled.select(
        registered,
        requested=args.versions,
        exclude=_split(args.exclude),
        include_settled=args.all,
    )

    if args.history:
        print(report.compare(versions, columns=columns, include_legacy=args.include_legacy))
        return

    if not versions:
        if not registered:
            print("No circuit versions registered.")
        else:
            print("No versions selected. Registered: " + ", ".join(registered))
            print("(the settled list is hidden by default — use --all to include it)")
        return

    hidden = len(registered) - len(versions)
    if hidden > 0 and not args.versions:
        print(f"({hidden} settled/excluded version(s) hidden — --all to show)\n")

    live_rows = []
    for v in versions:
        r = harness.benchmark(v, n_trials=args.trials, seed=args.seed,
                              meas_reps=args.meas_reps)
        live_rows.append(asdict(r))
        if r.empirical_only:
            status = "EMPIRICAL"
            acc = (f"analytic P(fail)=unknown; observed_pass={r.observed_pass}; "
                   f"failed batches={r.batches_failed}/{r.batches_completed}; "
                   f"failed calls={r.fail_seen}/{r.trials} "
                   f"(output={r.output_seen}, phase={r.phase_seen}, dirty={r.dirty_seen}); "
                   f"batch diagnostic CP95 upper={r.fail_bound95:.3e} "
                   "(not a quantum-norm bound)")
        else:
            status = "OK  " if r.correct else "FAIL"
            acc = ("declared exact" if r.log2_fail is None else
                   f"analytic P(fail)=2^{r.log2_fail:.1f}")
        line = (f"[{status}] {v} field={r.field}: {acc} toffoli={r.toffoli} "
                f"peak_qubits={r.peak_qubits} depth={r.depth}")
        if r.correct is False:
            line += f"  <- {r.note}"
        print(line)

    print()
    print(report.format_rows(live_rows, columns=columns))


if __name__ == "__main__":
    raise SystemExit(main())
