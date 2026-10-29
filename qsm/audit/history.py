"""Exact-prefix accounting and heuristic history-distribution experiments.

Run with ``python -m qsm.audit.history --help``. This analyzes ideal integer GCD
trajectories, not a quantum circuit or a subgroup-point input distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Sequence, TextIO

from ..inversion import schedule
from ..params import FIELDS
from .history_math import trace_history

VERSION = "jump2-history-v1"


@dataclass
class _Moments:
    total: float = 0.0
    squares: float = 0.0

    def add(self, value: float) -> None:
        self.total += value
        self.squares += value * value

    def finish(self, draws: int, exhaustive: bool, iid: bool = True) -> dict:
        mean = self.total / draws
        variance = max(0.0, self.squares - self.total * mean)
        return {"mean": mean, "standard_error": (
            0.0 if exhaustive else math.sqrt(variance / (draws * (draws - 1)))
            if iid and draws > 1 else None
        )}


def _residue_expected(m: int) -> list[float]:
    out = []
    for j in range(10):
        lo = max(1, (j * m + 9) // 10)
        hi = m if j == 9 else ((j + 1) * m + 9) // 10 - 1
        out.append(max(0, hi - lo + 1) / m)
    return out


def _odd_scale_model(bits: int, discarded: int) -> float:
    """Strict-order ties for independent odd operands conditioned on max bitlength."""
    if bits <= 1 or discarded <= 1:
        return 0.0
    equality_half = math.ldexp(1.0 / 3, 1 - bits)
    if discarded >= bits:
        return 0.5 - equality_half
    return math.ldexp(1.0 / 3, discarded - bits) - equality_half


def _normalized(values: list[float]) -> list[float]:
    total = sum(values)
    return [v / total for v in values] if total else [0.0] * len(values)


@dataclass
class _Snapshot:
    active: int = 0
    seen: set[int] = field(default_factory=set)
    inverse_population: _Moments = field(default_factory=_Moments)
    population_log2: float = 0.0
    singleton: int = 0
    residue_observed: list[float] = field(default_factory=lambda: [0.0] * 10)
    residue_expected: list[float] = field(default_factory=lambda: [0.0] * 10)
    history_residue_observed: list[float] = field(default_factory=lambda: [0.0] * 10)
    history_residue_expected: list[float] = field(default_factory=lambda: [0.0] * 10)
    classes: dict[int, dict] = field(default_factory=dict)

    def add(self, rec) -> None:
        prefix = rec.prefix
        population = prefix.count
        weight = 1.0 / population
        self.active += 1
        self.seen.add(rec.history_key)
        self.inverse_population.add(weight)
        self.population_log2 += math.log2(population)
        self.singleton += population == 1
        bucket = min(9, (5 * prefix.residue_first_positive) // rec.m)
        expected = _residue_expected(rec.m)
        self.residue_observed[bucket] += 1
        self.history_residue_observed[bucket] += weight
        for j, probability in enumerate(expected):
            self.residue_expected[j] += probability
            self.history_residue_expected[j] += weight * probability
        j = rec.m.bit_length() - 1
        row = self.classes.setdefault(j, {
            "observations": 0, "inverse_population_sum": 0.0,
            "population_log2_sum": 0.0, "singletons": 0,
        })
        row["observations"] += 1
        row["inverse_population_sum"] += weight
        row["population_log2_sum"] += math.log2(population)
        row["singletons"] += population == 1

    def finish(self, p: int, draws: int, exhaustive: bool, uniform_samples: bool) -> dict:
        infer = exhaustive or uniform_samples
        correction = self.inverse_population.finish(draws, exhaustive, uniform_samples)
        observed = _normalized(self.residue_observed)
        expected = _normalized(self.residue_expected)
        h_observed = _normalized(self.history_residue_observed)
        h_expected = _normalized(self.history_residue_expected)
        for row in self.classes.values():
            row["estimated_distinct_histories"] = (
                (p - 1) * row["inverse_population_sum"] / draws if infer else None)
        total = self.inverse_population.total
        squares = self.inverse_population.squares
        return {
            "active_inputs": self.active,
            "active_input_fraction": self.active / draws,
            "observed_distinct_histories": len(self.seen),
            "estimated_distinct_histories": (p - 1) * correction["mean"] if infer else None,
            "history_count_standard_error": (
                None if correction["standard_error"] is None else
                (p - 1) * correction["standard_error"]),
            "inverse_population_weight_ess": total * total / squares if squares else 0,
            "mean_log2_population_given_active": self.population_log2 / self.active
            if self.active else None,
            "singleton_fraction_given_active": self.singleton / self.active
            if self.active else None,
            "input_residue_deciles": observed,
            "input_uniform_residue_model_deciles": expected,
            "history_weighted_residue_deciles": h_observed,
            "history_weighted_uniform_residue_model_deciles": h_expected,
            "input_residue_decile_tv": sum(abs(a - b) for a, b in
                                            zip(observed, expected)) / 2 if self.active else None,
            "history_weighted_residue_decile_tv": sum(abs(a - b) for a, b in
                zip(h_observed, h_expected)) / 2 if self.active else None,
            "m_log2_classes": self.classes,
        }


def width_schedule(n: int, kind: str) -> list[int]:
    if kind == "tuned":
        return schedule.widths_j2(n)
    if kind == "safe":
        return [n] + [max(1, min(n, 2 * n - i - 1)) for i in range(1, 2 * n - 1)]
    if kind == "full":
        return [n] * (2 * n - 1)
    raise ValueError("schedule must be tuned, safe or full")


def default_checkpoints(n: int, widths: Sequence[int]) -> list[int]:
    candidates = [1] + [round(n * f) for f in (.125, .25, .375, .5, .625, .70)]
    return sorted({i for i in candidates if 0 < i < len(widths)})


def source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = ["audit/history.py", "audit/history_math.py", "inversion/schedule.py", "params.py"]
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def analyze(
    p: int, inputs: Sequence[int], *, compare_bits: Sequence[int],
    schedule_kind: str = "tuned", checkpoints: Sequence[int] | None = None,
    exhaustive: bool = False, uniform_samples: bool = False,
    seed: int | None = None, stream: TextIO | None = None,
) -> dict:
    if p < 3 or p % 2 == 0 or not inputs:
        raise ValueError("need an odd modulus and at least one input")
    if exhaustive and (len(inputs) != p - 1 or set(inputs) != set(range(1, p))):
        raise ValueError("exhaustive mode requires every nonzero canonical divisor exactly once")
    if not compare_bits or any(not isinstance(k, int) or isinstance(k, bool) or k < 1
                               for k in compare_bits):
        raise ValueError("comparison widths must be positive integers")
    ks = sorted(set(compare_bits))
    n, draws = p.bit_length(), len(inputs)
    widths = width_schedule(n, schedule_kind)
    if checkpoints is None:
        checkpoints = default_checkpoints(n, widths)
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 1 or i >= len(widths)
           for i in checkpoints):
        raise ValueError("checkpoints must be positive indices inside the chosen schedule")
    snapshots = {i: _Snapshot() for i in sorted(set(checkpoints))}
    keys = ("direct_bad_slots", "conditional_bad_slots", "residue_upper_slots",
            "uniform_residue_model_slots", "naive_independent_model_slots",
            "independent_odd_scale_model_slots")
    moments = {k: {name: _Moments() for name in keys} for k in ks}
    bad_inputs = {k: 0 for k in ks}
    slot_totals: dict[int, dict] = {}
    clip_inputs = overruns = 0
    first_clip_slots: dict[int, int] = {}
    iterations: dict[int, int] = {}
    started = time.monotonic()
    digest = source_digest()
    for sample_index, x in enumerate(inputs, 1):
        totals = {k: dict.fromkeys(keys, 0.0) for k in ks}
        first_clip = None
        total_iters = 0
        for rec in trace_history(p, x):
            total_iters = rec.i + 1
            if rec.i >= len(widths):
                continue
            w = widths[rec.i]
            if first_clip is None and rec.entry_bits > w:
                first_clip = rec.i
            if not rec.active:
                continue
            if rec.i in snapshots:
                snapshots[rec.i].add(rec)
            slot = slot_totals.setdefault(rec.i, {"active_inputs": 0, "k": {
                k: {"direct_bad": 0, "conditional_sum": 0.0,
                    "uniform_residue_model_sum": 0.0} for k in ks}})
            slot["active_inputs"] += 1
            for k in ks:
                if k >= w - 1:
                    continue
                B = 1 << (w - k)
                current_fit = max(rec.u, rec.v) < (1 << w)
                direct = int(current_fit and rec.u > rec.v and rec.u // B == rec.v // B)
                count = rec.prefix.bad_count(k, w, fit_only=True)
                conditional = count / rec.prefix.count
                residue_upper = rec.prefix.residue_bound(k, w) / rec.prefix.count
                residue_model = min(1.0, ((B - 1) // 2) / rec.m / rec.prefix.count)
                naive = math.ldexp(0.5, -k)
                scale_model = _odd_scale_model(max(rec.u, rec.v).bit_length(), w - k) if current_fit else 0.0
                for name, value in zip(keys, (direct, conditional, residue_upper,
                                              residue_model, naive, scale_model)):
                    totals[k][name] += value
                row = slot["k"][k]
                row["direct_bad"] += direct
                row["conditional_sum"] += conditional
                row["uniform_residue_model_sum"] += residue_model
        iterations[total_iters] = iterations.get(total_iters, 0) + 1
        overruns += total_iters > len(widths)
        if first_clip is not None:
            clip_inputs += 1
            first_clip_slots[first_clip] = first_clip_slots.get(first_clip, 0) + 1
        for k in ks:
            bad_inputs[k] += totals[k]["direct_bad_slots"] > 0
            for name in keys:
                moments[k][name].add(totals[k][name])
        if stream is not None and (sample_index % max(1, draws // 10) == 0 or sample_index == draws):
            print(f"History inputs {sample_index}/{draws}", file=stream, flush=True)
    comparison = {}
    for k in ks:
        row = {name: moments[k][name].finish(draws, exhaustive, uniform_samples) for name in keys}
        row["bad_inputs"] = bad_inputs[k]
        row["observed_any_comparison_hazard"] = bad_inputs[k] / draws
        row["conditional_union_estimate_capped"] = min(1.0, row["conditional_bad_slots"]["mean"])
        row["uniform_residue_model_union_capped"] = min(1.0, row["uniform_residue_model_slots"]["mean"])
        comparison[k] = row
    if source_digest() != digest:
        raise RuntimeError("analysis sources changed during the experiment")
    return {
        "version": VERSION, "p": p, "n": n, "samples": draws, "seed": seed,
        "mode": "exhaustive" if exhaustive else "sampled" if uniform_samples else "diagnostic",
        "input_law": "all canonical nonzero divisors" if exhaustive else
        "declared IID-uniform nonzero field divisors (seeded PRNG in CLI)" if uniform_samples else
        "selected diagnostic inputs; no population inference or sampling standard errors",
        "schedule": schedule_kind, "widths": widths, "compare_bits": ks,
        "source_digest": digest, "seconds": time.monotonic() - started,
        "event": "enabled ideal-trajectory strict-order disagreement with tied high prefixes, "
        "restricted to current operands fitting the scheduled width; earlier faults are not conditioned away",
        "interpretation": "exhaustive counts or sampled diagnostics, not a certified full-size error bound; "
        "uniform-residue candidate counts are a heuristic, and empirical standard errors do not certify rare tails",
        "width_clip_inputs": clip_inputs, "first_width_clip_slots": first_clip_slots,
        "iteration_overrun_inputs": overruns, "iteration_histogram": iterations,
        "comparison": comparison,
        "checkpoints": {i: row.finish(p, draws, exhaustive, uniform_samples) for i, row in snapshots.items()},
        "slots": slot_totals,
    }


def format_report(result: dict) -> str:
    from tabulate import tabulate
    rows = []
    for k, row in result["comparison"].items():
        rows.append([k, row["bad_inputs"], row["observed_any_comparison_hazard"],
                     row["conditional_bad_slots"]["mean"],
                     row["uniform_residue_model_slots"]["mean"],
                     row["independent_odd_scale_model_slots"]["mean"],
                     row["residue_upper_slots"]["mean"]])
    lines = [f"Jump-2 history study: {result['n']} bits; {result['schedule']}; "
             f"{result['mode']}; {result['samples']} inputs",
             tabulate(rows, headers=["k", "bad inputs", "any hazard", "exact-prefix slot mean",
                                     "residue-model slot mean", "odd-scale model", "residue-upper slot mean"],
                      tablefmt="github", floatfmt=".6g"), ""]
    rows = []
    for i, row in result["checkpoints"].items():
        rows.append([i, row["active_inputs"], row["observed_distinct_histories"],
                     row["estimated_distinct_histories"], row["mean_log2_population_given_active"],
                     row["singleton_fraction_given_active"], row["history_weighted_residue_decile_tv"]])
    lines.extend([tabulate(rows, headers=["slot", "active", "seen histories", "estimated histories",
        "mean log2(pop.)", "singleton fraction", "weighted residue TV"],
        tablefmt="github", floatfmt=".6g"),
        f"Width clips: {result['width_clip_inputs']}; iteration overruns: {result['iteration_overrun_inputs']}",
        "Slot means are sums of local hazards, not probabilities of independent events.",
        "Sampled estimates and the residue model are NOT certified bounds or full-window success rates."])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    domain = parser.add_mutually_exclusive_group()
    domain.add_argument("--field", choices=sorted(FIELDS), default="25519")
    domain.add_argument("--prime", type=int, help="small prime, at most 1,000,003")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--compare-bits", type=int, nargs="+")
    parser.add_argument("--schedule", choices=("tuned", "safe", "full"), default="tuned")
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--json", type=Path, help="create a new JSON result file (never overwrite)")
    args = parser.parse_args(argv)
    if args.prime is None:
        profile = FIELDS[args.field]
        p = (1 << profile["u"]) - profile["f"]
    else:
        p = args.prime
        if not 3 <= p <= 1000003 or any(p % d == 0 for d in range(2, math.isqrt(p) + 1)):
            parser.error("--prime must be a prime between 3 and 1,000,003")
    if args.exhaustive and p > 1000003:
        parser.error("exhaustive enumeration is limited to small primes")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.json is not None and args.json.exists():
        parser.error("--json target already exists; use a new filename")
    n = p.bit_length()
    ks = args.compare_bits or ([48, 56, 64, 80, 100] if n > 100 else
                               sorted({max(1, n - 4), max(1, n - 2), n - 1}))
    rng = random.Random(args.seed)
    inputs = list(range(1, p)) if args.exhaustive else [rng.randrange(1, p) for _ in range(args.samples)]
    try:
        result = analyze(p, inputs, compare_bits=ks, schedule_kind=args.schedule,
                         checkpoints=args.checkpoints, exhaustive=args.exhaustive,
                         uniform_samples=not args.exhaustive,
                         seed=None if args.exhaustive else args.seed, stream=sys.stderr)
    except ValueError as exc:
        parser.error(str(exc))
    print(format_report(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        print(f"Saved {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
