"""Isolated b-bit guarded-comparison experiments; production GCD is unchanged."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Sequence, TextIO

from ..bench.calibration import source_digest
from ..core.resources import count
from ..inversion.schedule import widths_j2
from ..params import FIELDS
from .guarded_comparator import build_compare_case
from .history_math import trace_history


def prefix_order(u: int, v: int, width: int, bits: int) -> bool:
    return (v >> max(0, width - bits)) < (u >> max(0, width - bits))


def guarded_order(u: int, v: int, width: int, k: int, b: int) -> tuple[bool, bool]:
    """Return (v<u predicate, lower-window selected) for in-width operands."""
    if not 0 <= u < 1 << width or not 0 <= v < 1 << width:
        raise ValueError("operands must fit the scheduled width")
    if not 1 <= b <= k or k < 1:
        raise ValueError("need 1 <= b <= k")
    shift = min(b, max(0, width - k))
    if not shift:
        return v < u, False
    same = u >> (width - shift) == v >> (width - shift)
    if not same:
        return prefix_order(u, v, width, k), False
    lo = width - shift - k
    mask = (1 << k) - 1
    return ((v >> lo) & mask) < ((u >> lo) & mask), True


def _pairs(base_bits: Sequence[int], guard_bits: Sequence[int]) -> list[tuple[int, int]]:
    if not base_bits or not guard_bits or any(
        type(value) is not int or value < 1 for value in (*base_bits, *guard_bits)
    ):
        raise ValueError("comparison and guard widths must be positive integers")
    return [(k, b) for k in sorted(set(base_bits))
            for b in [0] + sorted({b for b in guard_bits if b <= k})]


def sample_trajectories(
    p: int, inputs: Sequence[int], *, base_bits: Sequence[int], guard_bits: Sequence[int],
    exhaustive: bool = False, stream: TextIO | None = None,
) -> dict:
    if not inputs:
        raise ValueError("at least one input is required")
    if exhaustive and (len(inputs) != p - 1 or set(inputs) != set(range(1, p))):
        raise ValueError("a census requires every canonical nonzero divisor exactly once")
    widths = widths_j2(p.bit_length())
    pairs = _pairs(base_bits, guard_bits)
    rows = {pair: {"base_bits": pair[0], "guard_bits": pair[1], "bad_inputs": 0,
                   "bad_slots": 0, "extension_slots": 0, "lower_window_selected": 0,
                   "rescued_baseline_bad_inputs": 0, "removed_bad_slots": 0}
            for pair in pairs}
    clips = overruns = active_slots = 0
    for index, x in enumerate(inputs, 1):
        bad = dict.fromkeys(pairs, False)
        clipped = False
        total_iterations = 0
        for rec in trace_history(p, x):
            total_iterations = rec.i + 1
            if rec.i >= len(widths):
                continue
            w = widths[rec.i]
            clipped |= rec.entry_bits > w
            if not rec.active or max(rec.u, rec.v) >= 1 << w:
                continue
            active_slots += 1
            exact = rec.v < rec.u
            baseline = {k: prefix_order(rec.u, rec.v, w, k) for k in set(base_bits)}
            for pair, row in rows.items():
                k, b = pair
                if b:
                    predicted, selected = guarded_order(rec.u, rec.v, w, k, b)
                    widened = prefix_order(rec.u, rec.v, w, k + b)
                    if predicted != widened:
                        raise AssertionError("guarded and widened predicates disagree")
                    if predicted != exact and baseline[k] == exact:
                        raise AssertionError("guard introduced a comparison hazard")
                    row["extension_slots"] += w > k
                    row["lower_window_selected"] += selected
                else:
                    predicted = baseline[k]
                error = predicted != exact
                row["bad_slots"] += error
                row["removed_bad_slots"] += baseline[k] != exact and not error
                bad[pair] |= error
        clips += clipped
        overruns += total_iterations > len(widths)
        for pair, row in rows.items():
            row["bad_inputs"] += bad[pair]
            row["rescued_baseline_bad_inputs"] += bad[(pair[0], 0)] and not bad[pair]
        if stream is not None and index % max(1, len(inputs) // 10) == 0:
            print(f"Guard inputs {index}/{len(inputs)}", file=stream, flush=True)
    for row in rows.values():
        row["observed_any_hazard"] = row["bad_inputs"] / len(inputs)
        row["mean_bad_slots"] = row["bad_slots"] / len(inputs)
        row["lower_window_fraction"] = row["lower_window_selected"] / row["extension_slots"] \
            if row["extension_slots"] else None
    return {"p": p, "n": p.bit_length(), "samples": len(inputs), "exhaustive": exhaustive,
            "widths": widths, "width_clip_inputs": clips, "iteration_overrun_inputs": overruns,
            "active_in_width_comparisons": active_slots, "rows": list(rows.values()),
            "event": "enabled, currently in-width comparison disagreement along the exact integer trajectory; "
                     "not conditioned on earlier faults, not full-circuit success"}


def measure_comparators(n: int, *, base_bits: Sequence[int], guard_bits: Sequence[int]) -> dict:
    widths = widths_j2(n)
    cache: dict[tuple[int, int, int, str], dict] = {}

    def resource(w: int, k: int, b: int, strategy: str) -> dict:
        key = (w, k, b if strategy != "baseline" else 1, strategy)
        if key not in cache:
            case = build_compare_case(w, k, max(1, b), strategy, controlled=True)
            result = asdict(count(case.circuit))
            result["scratch_peak"] = result["peak_qubits"] - (2 * w + 2)
            cache[key] = result
        return cache[key]

    output = []
    for k, b in _pairs(base_bits, guard_bits):
        if not b:
            continue
        row = {"base_bits": k, "guard_bits": b, "at_full_width": {}, "forward_comparison_regions": {}}
        for strategy in ("baseline", "wider", "guarded"):
            row["at_full_width"][strategy] = resource(n, k, b, strategy)
            metrics = [resource(w, k, b, strategy) for w in widths[1:]]
            row["forward_comparison_regions"][strategy] = {
                "toffoli": sum(m["toffoli"] for m in metrics),
                "max_scratch_peak": max(m["scratch_peak"] for m in metrics),
            }
        row["guarded_minus_wider_forward_toffoli"] = (
            row["forward_comparison_regions"]["guarded"]["toffoli"] -
            row["forward_comparison_regions"]["wider"]["toffoli"])
        output.append(row)
    return {"n": n, "rows": output,
            "scope": "physical emitted standalone comparator IR; forward totals sum scheduled comparison "
                     "regions excluding boundary; NOT a rebuilt GCD/window peak or full circuit benchmark"}


def format_report(result: dict) -> str:
    from tabulate import tabulate
    rows = [[r["base_bits"], r["guard_bits"], r["bad_inputs"], r["bad_slots"],
             r["rescued_baseline_bad_inputs"], r["lower_window_fraction"]]
            for r in result["trajectories"]["rows"]]
    lines = [f"Guarded comparison study: {result['field']}, seed={result['seed']}",
        tabulate(rows, headers=["k", "b", "bad inputs", "bad slots", "rescued inputs", "guard selected"],
                 tablefmt="github", floatfmt=".6g")]
    rows = []
    for r in result["resources"]["rows"]:
        if r["base_bits"] != max(result["base_bits"]):
            continue
        z = r["at_full_width"]
        rows.append([r["guard_bits"], z["baseline"]["toffoli"], z["wider"]["toffoli"],
                     z["guarded"]["toffoli"], z["baseline"]["scratch_peak"],
                     z["wider"]["scratch_peak"], z["guarded"]["scratch_peak"]])
    lines.extend(["", "Full-width standalone comparator costs (largest base k):",
        tabulate(rows, headers=["b", "baseline CCX", "wider CCX", "guarded CCX", "base scratch",
                               "wider scratch", "guarded scratch"], tablefmt="github"),
        "Guarded and widened predicates agree; guards do not make emitted conditional gates free.",
        "Observed rates are diagnostics, not analytic success bounds. Whole-window peak is not inferred."])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    domain = parser.add_mutually_exclusive_group()
    domain.add_argument("--field", choices=sorted(FIELDS), default="25519")
    domain.add_argument("--prime", type=int)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026092801)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--base-bits", type=int, nargs="+")
    parser.add_argument("--guard-bits", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 100])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.prime is not None:
        p = args.prime
        if not 3 <= p <= 65537 or any(p % d == 0 for d in range(2, math.isqrt(p) + 1)):
            parser.error("--prime must be prime and at most 65537")
    else:
        profile = FIELDS[args.field]
        p = (1 << profile["u"]) - profile["f"]
    if args.exhaustive and p > 65537:
        parser.error("censuses are limited to small primes")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.json is not None and args.json.exists():
        parser.error("JSON target already exists; use a new path")
    n = p.bit_length()
    ks = args.base_bits or ([48, 56, 100] if n > 100 else [max(1, n - 4)])
    try:
        _pairs(ks, args.guard_bits)
    except ValueError as exc:
        parser.error(str(exc))
    rng = random.Random(args.seed)
    xs = list(range(1, p)) if args.exhaustive else [rng.randrange(1, p) for _ in range(args.samples)]
    digest = source_digest()
    started = time.monotonic()
    trajectories = sample_trajectories(p, xs, base_bits=ks, guard_bits=args.guard_bits,
                                      exhaustive=args.exhaustive, stream=sys.stderr)
    resources = measure_comparators(n, base_bits=ks, guard_bits=args.guard_bits)
    if digest != source_digest():
        raise RuntimeError("QSM sources changed during the study")
    result = {"version": "guarded-window-v1", "field": args.field if args.prime is None else f"p={p}",
              "seed": None if args.exhaustive else args.seed, "base_bits": ks,
              "guard_bits": args.guard_bits, "source_digest": digest,
              "seconds": time.monotonic() - started, "trajectories": trajectories,
              "resources": resources, "production_gcd_changed": False}
    print(format_report(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n")
        print(f"Saved {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
