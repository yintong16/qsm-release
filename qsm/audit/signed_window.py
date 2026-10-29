"""Opt-in complete-window measurements; no production defaults or history changes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
from typing import Sequence

from ..bench.calibration import source_digest
from ..bench.harness import benchmark_case
from ..core import ops
from ..params import FIELD
from ..scalarmul.window_opt import build_optimized_window_case


STRATEGIES = ("unsigned", "conjugate", "selected")


def _build(bits: int, strategy: str):
    if strategy == "unsigned":
        return build_optimized_window_case(bits)
    from ..scalarmul.signed_window import build_signed_window_case
    return build_signed_window_case(bits, sign_strategy=strategy)


def region_resources(circuit, regions: Sequence[tuple[str, int, int]]) -> dict:
    """Exclusive region gate counts and peak total live qubits, not local scratch."""
    spans = sorted(regions, key=lambda r: (r[1], r[2]))
    previous = 0
    for name, start, stop in spans:
        if not isinstance(name, str) or not 0 <= start <= stop <= len(circuit.ops):
            raise ValueError("invalid resource region")
        if start < previous:
            raise ValueError("resource regions must not overlap")
        previous = stop
    output = {}
    cursor = live = 0
    for index, op in enumerate(circuit.ops):
        while cursor < len(spans) and index >= spans[cursor][2]:
            cursor += 1
        label = spans[cursor][0] if cursor < len(spans) and index >= spans[cursor][1] else "unattributed"
        row = output.setdefault(label, {"toffoli": 0, "clifford": 0,
                                       "measurements": 0, "peak_live_qubits": 0})
        if op.kind == ops.ALLOC:
            live += len(op.qubits)
        row["peak_live_qubits"] = max(row["peak_live_qubits"], live)
        physical = ops.physical_gate_counts(op)
        row["toffoli"] += physical.get(ops.CCX, 0)
        row["clifford"] += sum(physical.get(k, 0) for k in (ops.X, ops.CX, ops.CZ, ops.Z))
        row["measurements"] += physical.get(ops.MEASURE, 0)
        if op.kind == ops.FREE:
            live -= len(op.qubits)
        elif op.kind == ops.MEASURE:
            live -= 1
        if live < 0:
            raise ValueError("negative live-qubit count")
    return output


def run(*, address_bits: Sequence[int] = (15, 16, 17),
        strategies: Sequence[str] = STRATEGIES, trials: int = 4,
        meas_reps: int = 1, seed: int = 2026092901) -> dict:
    if not address_bits or any(type(a) is not int or not 1 <= a <= 17 for a in address_bits):
        raise ValueError("address widths must be integers from 1 through 17")
    if not strategies or any(s not in STRATEGIES for s in strategies):
        raise ValueError("unknown or empty strategy list")
    if type(trials) is not int or trials < 1 or type(meas_reps) is not int or meas_reps < 1:
        raise ValueError("trials and measurement repetitions must be positive integers")
    digest = source_digest()
    rows = []
    for a in sorted(set(address_bits)):
        for strategy in dict.fromkeys(strategies):
            print(f"Building {FIELD}: address={a}, strategy={strategy}", flush=True)
            started = time.monotonic()
            case = _build(a, strategy)
            regions = region_resources(case.circuit, case.meta.get("resource_regions", ()))
            input_qubits = sum(len(op.qubits) for op in case.circuit.ops
                               if op.kind == ops.ALLOC and op.is_input)
            result = benchmark_case(f"experiment.signed_window.a{a}.{strategy}", case,
                                    n_trials=trials, seed=seed, meas_reps=meas_reps, persist=False)
            for metric in ("toffoli", "clifford", "measurements"):
                if sum(r[metric] for r in regions.values()) != getattr(result, metric):
                    raise AssertionError(f"region totals disagree for {metric}")
            if max(r["peak_live_qubits"] for r in regions.values()) != result.peak_qubits:
                raise AssertionError("region/global peak mismatch")
            row = {"address_bits": a, "radix_bits": a + (strategy != "unsigned"),
                   "magnitude_entries": 1 << a, "strategy": strategy,
                   "input_qubits": input_qubits, "regions": regions,
                   "seconds": time.monotonic() - started, "result": asdict(result)}
            rows.append(row)
            print(f"{strategy} a={a}: CCX={result.toffoli}, peak={result.peak_qubits}, "
                  f"smoke_pass={result.observed_pass}, shots={result.trials}", flush=True)
            del case, result
            gc.collect()
            if source_digest() != digest:
                raise RuntimeError("QSM source changed during the experiment")
    return {"version": "signed-window-v1", "field": FIELD, "seed": seed,
            "source_digest": digest, "rows": rows, "production_changed": False,
            "scope": "actual complete windows with sign and endpoint controls; "
                     "recoding and scalar-chain preparation are NOT included; "
                     "regional peaks include all live qubits, not just scratch; "
                     "unpaired basis/phase smoke diagnostics, not a probability bound"}


def format_report(result: dict) -> str:
    from tabulate import tabulate
    rows = [[r["address_bits"], r["radix_bits"], r["strategy"], r["result"]["toffoli"],
             r["result"]["peak_qubits"], r["result"]["toffoli_depth"],
             r["result"]["observed_pass"]] for r in result["rows"]]
    return "\n".join([f"Signed-window experiment: {result['field']}",
        tabulate(rows, headers=["address bits", "radix bits", "strategy", "CCX", "peak",
                               "CCX depth", "smoke pass"], tablefmt="github"), result["scope"]])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address-bits", type=int, nargs="+", default=[15, 16, 17])
    parser.add_argument("--strategies", choices=STRATEGIES, nargs="+", default=list(STRATEGIES))
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--meas-reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026092901)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.json is not None and args.json.exists():
        parser.error("JSON target already exists; choose a new path")
    try:
        result = run(address_bits=args.address_bits, strategies=args.strategies,
                     trials=args.trials, meas_reps=args.meas_reps, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print(format_report(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x") as f:
            json.dump(result, f, sort_keys=True, indent=2)
            f.write("\n")
    return int(any(not row["result"]["observed_pass"] for row in result["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
