"""Opt-in whole-window measurements of isolated comparator prototypes.

Construction temporarily substitutes the GCD/undo comparison emitters in this
single-threaded process, then restores them before simulation. No production
source, registry version or saved benchmark history is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time
from typing import Sequence

from ..bench.calibration import source_digest
from ..bench.harness import benchmark_case
from ..inversion import gcd, undo
from ..params import FIELD, PRIME, U
from ..scalarmul.window_opt import OptimizedWindowConfig, build_optimized_window_case
from .guarded_comparator import cmp_lt_guarded


@contextmanager
def guarded_gcd_emission(b: int):
    if type(b) is not int or b < 1:
        raise ValueError("guard width must be a positive integer")
    original_gcd, original_undo = gcd.cmp_lt, undo.cmp_lt

    def emitter(original):
        def emit(circ, a, B, flag, k=None, ctrl=None):
            if k is None:
                return original(circ, a, B, flag, k=k, ctrl=ctrl)
            return cmp_lt_guarded(circ, a, B, flag, k=k, b=min(b, k), ctrl=ctrl)
        return emit

    gcd.cmp_lt = emitter(original_gcd)
    undo.cmp_lt = emitter(original_undo)
    try:
        yield
    finally:
        gcd.cmp_lt, undo.cmp_lt = original_gcd, original_undo


def run(*, window_bits: int = 16, guard_bits: Sequence[int] = (8, 16),
        trials: int = 2, meas_reps: int = 1, seed: int = 2026092803) -> dict:
    if not guard_bits or any(type(b) is not int or not 1 <= b <= 100 for b in guard_bits):
        raise ValueError("guard widths must be integers from 1 through 100")
    if type(window_bits) is not int or not 1 <= window_bits <= 16:
        raise ValueError("window size must be between 1 and 16")
    if type(trials) is not int or trials < 1 or type(meas_reps) is not int or meas_reps < 1:
        raise ValueError("trials and measurement repetitions must be positive integers")
    digest = source_digest()
    original_gcd, original_undo = gcd.cmp_lt, undo.cmp_lt
    base = OptimizedWindowConfig.tuned(U, PRIME, window_bits)
    specs = [("baseline", 0)] + [(strategy, b) for b in sorted(set(guard_bits))
                                 for strategy in ("wider", "guarded")]
    results = []
    for strategy, b in specs:
        print(f"Building {FIELD} window[{window_bits}] {strategy} b={b}", flush=True)
        started = time.monotonic()
        config = base
        if strategy == "wider":
            config = replace(base, inversion=replace(base.inversion,
                             gcd_cmp_bits=base.inversion.gcd_cmp_bits + b))
        if strategy == "guarded":
            with guarded_gcd_emission(b):
                case = build_optimized_window_case(window_bits, config=config)
        else:
            case = build_optimized_window_case(window_bits, config=config)
        if (gcd.cmp_lt, undo.cmp_lt) != (original_gcd, original_undo):
            raise AssertionError("experimental emitters were not restored")
        case.meta["circuit_config"]["experimental_gcd_comparison"] = {"strategy": strategy, "b": b}
        result = benchmark_case(f"experiment.guarded_window.{strategy}.b{b}", case,
                                n_trials=trials, seed=seed, meas_reps=meas_reps, persist=False)
        row = {"strategy": strategy, "guard_bits": b, "seconds": time.monotonic() - started,
               "result": asdict(result)}
        results.append(row)
        print(f"{strategy} b={b}: CCX={result.toffoli}, peak={result.peak_qubits}, "
              f"smoke_pass={result.observed_pass}, shots={result.trials}", flush=True)
        del case, result
        gc.collect()
    if source_digest() != digest:
        raise RuntimeError("QSM sources changed during the experiment")
    return {"version": "guarded-full-window-v1", "field": FIELD, "window_bits": window_bits,
            "seed": seed, "source_digest": digest, "rows": results,
            "scope": "actually built and counted complete tuned windows; basis/phase smoke diagnostics only; "
                     "no analytic success bound, no production configuration change"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--guard-bits", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--meas-reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026092803)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.json is not None and args.json.exists():
        parser.error("JSON target already exists; choose a new path")
    try:
        result = run(window_bits=args.bits, guard_bits=args.guard_bits, trials=args.trials,
                     meas_reps=args.meas_reps, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
    return int(any(not row["result"]["observed_pass"] for row in result["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
