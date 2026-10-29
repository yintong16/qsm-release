"""Unified launcher; arithmetic commands run in a fresh, field-selected process."""
from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

FIELDS = ("25519", "secp256k1")
LEGACY_AUDITS = {
    "rates": "Approximate-arithmetic rate diagnostics (fixed study).",
    "refs": "Classical reference checks (fixed study).",
    "baselines": "Arithmetic baseline comparisons (fixed study).",
    "curves": "Cross-field primitive comparisons (fixed study).",
    "preconditions": "Arithmetic precondition checks (fixed study).",
    "port": "Cross-field point/inversion comparison; expensive subprocess builds.",
    "all": "Legacy aggregate studies; excludes the expensive port study.",
}
AUDITS = {
    "window": ("qsm.audit.window", "Unsigned point windows; legacy v1 is this runner's default."),
    "signed-window": ("qsm.audit.signed_window", "Signed/unsigned lookup-window comparison."),
    "signed-recoding": ("qsm.audit.signed_recoding", "Classical recoding schedules and optional recoder circuits."),
    "signed-schedule": ("qsm.audit.signed_schedule", "Terminal circuits or composition of saved window results."),
    "odd-window": ("qsm.audit.odd_window", "Centered versus odd-only single-addition comparison."),
    "zero-point": ("qsm.audit.zero_point", "Odd-window subtraction/positive-addition restoration and zero-slope audit."),
    "full-scalar": ("qsm.audit.full_scalar", "Actually emitted double-scalar schedules; long builds."),
    "history": ("qsm.audit.history", "Classical GCD histories and comparison-error counts."),
    "guarded-window": ("qsm.audit.guarded_window", "Comparison guards on classical histories and small comparators."),
    "guarded-full-window": ("qsm.audit.guarded_full_window", "Complete windows with direct and guarded comparisons."),
}
FIELD_ARGUMENT_AUDITS = {"history", "guarded-window"}


def _parser():
    parser = argparse.ArgumentParser(prog="qsm", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
               "  info   Show interpreter, dependencies, field and source identity.\n"
               "  list   List registered versions without building circuits.\n"
               "  bench  Registered circuits or live unsigned-window benchmarks.\n"
               "  audit  Named research drivers; use 'qsm audit --help'.\n"
               "  test   Forward explicit arguments to pytest.\n\n"
               "Put the global --field option before the command.\n"
               "No-argument bench/test commands show help, not an expensive run.")
    parser.add_argument("--field", choices=FIELDS,
                        help="override QSM_FIELD (default: 25519 when unset)")
    parser.add_argument("command", nargs="?", choices=("info", "list", "bench", "audit", "test"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _audit_help():
    print("usage: qsm [--field FIELD] audit NAME [options]\n")
    for name, (_, description) in AUDITS.items():
        print(f"  {name:22} {description}")
    print("\nFixed legacy studies (no additional options):")
    for name, description in LEGACY_AUDITS.items():
        print(f"  {name:22} {description}")
    print("\nNo study runs unless named. Port and curves compare both fields.")


def _option_values(arguments, option, parser):
    values = []
    for index, token in enumerate(arguments):
        if token == "--":
            break
        if token == option:
            if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
                parser.error(f"{option} requires a value")
            values.append(arguments[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def _field_and_arguments(args, topic, arguments, parser):
    local = _option_values(arguments, "--field", parser) if topic in FIELD_ARGUMENT_AUDITS else []
    if any(value not in FIELDS for value in local):
        parser.error(f"--field must be one of {', '.join(FIELDS)}")
    if len(set(local)) > 1 or (args.field is not None and any(v != args.field for v in local)):
        parser.error("conflicting global and audit --field selections")
    field = args.field or (local[-1] if local else os.environ.get("QSM_FIELD", "25519"))
    if field not in FIELDS:
        parser.error(f"QSM_FIELD={field!r}: choose {', '.join(FIELDS)}")
    prime = _option_values(arguments, "--prime", parser) if topic in FIELD_ARGUMENT_AUDITS else []
    if topic in FIELD_ARGUMENT_AUDITS and not local and not prime:
        arguments = ["--field", field, *arguments]
    return field, arguments


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    arguments = list(args.arguments)
    topic = None
    if args.command == "audit":
        if not arguments or arguments[0] in ("--help", "-h"):
            _audit_help()
            return 0
        topic = arguments.pop(0).replace("_", "-")
        if topic in LEGACY_AUDITS:
            if arguments in (["--help"], ["-h"]):
                print(f"usage: qsm [--field FIELD] audit {topic}\n\n{LEGACY_AUDITS[topic]}\nNo options are accepted by this fixed study.")
                return 0
            if arguments:
                parser.error(f"audit {topic} accepts no options; use its --help for scope")
            module, arguments = "qsm.audit", [topic]
        elif topic in AUDITS:
            module = AUDITS[topic][0]
        else:
            parser.error(f"unknown audit {topic!r}; use 'qsm audit --help'")
    elif args.command in ("info", "list"):
        module, arguments = "qsm.cli", [args.command, *arguments]
    else:
        module = "qsm.bench" if args.command == "bench" else "pytest"
        if not arguments:
            arguments = ["--help"]
    field, arguments = _field_and_arguments(args, topic, arguments, parser)
    env = dict(os.environ, QSM_FIELD=field)
    # Child imports must see the field before any arithmetic module binds constants.
    try:
        status = subprocess.run([sys.executable, "-m", module, *arguments], env=env, check=False).returncode
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"qsm: could not launch {sys.executable}: {exc}", file=sys.stderr)
        return 1
    return status if status >= 0 else 128-status


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _info():
    from .params import FIELD, U, PRIME
    dependencies = {name: _version(name) for name in ("numpy", "tabulate", "pytest")}
    errors = {}
    for name in ("numpy", "tabulate"):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            errors[name] = str(exc)
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "virtual_environment": sys.prefix != sys.base_prefix,
              "qsm_version": _version("qsm"), "source_directory": str(Path(__file__).resolve().parent),
              "field": FIELD, "field_bits": U, "prime": PRIME,
              "dependencies": dependencies, "dependency_errors": errors}
    if not errors:
        from .bench.calibration import source_digest
        result["source_digest"] = source_digest()
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        print('Install dependencies with: python -m pip install -e ".[dev]"', file=sys.stderr)
    return int(bool(errors))


def _list_versions(patterns, json_output):
    from .params import FIELD
    importlib.import_module("qsm.bench.demos")
    for layer in ("field", "mul", "inversion", "point", "scalarmul"):
        importlib.import_module("qsm." + layer)
    from .bench.registry import all_versions
    versions = [v for v in all_versions() if any(fnmatch.fnmatchcase(v, p) for p in patterns or ["*"])]
    if json_output:
        print(json.dumps({"field": FIELD, "versions": versions}, indent=2))
    else:
        print(f"Registered circuits for {FIELD} (including settled versions; no circuits built):")
        print("\n".join(versions) if versions else "No matching versions.")
    return 0


def _support_main(argv=None):
    parser = argparse.ArgumentParser(prog="qsm")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Show interpreter, dependencies and selected field.")
    listing = sub.add_parser("list", help="List registered versions without building them.")
    listing.add_argument("patterns", nargs="*")
    listing.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return _info() if args.command == "info" else _list_versions(args.patterns, args.json)


if __name__ == "__main__":
    raise SystemExit(_support_main())
