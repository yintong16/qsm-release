"""Matched old/new odd-window restoration audit with raw-prefix diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

from ..bench.calibration import source_digest
from ..core.resources import count
from ..core.symbolic_batch import compile_symbolic, simulate_batch
from ..params import FIELD
from ..point.group import scalar_mul
from ..point.sampling import sqrt_mod
from ..scalarmul.odd_window import build_odd_window_case, make_odd_window_table
from ..scalarmul.zero_point import build_zero_point_case, zero_point_reference
from .odd_window import _inputs
from .signed_window import region_resources


VARIANTS = ("original", "zero_tag", "zero_phase", "positive_add", "positive_add_split")
MASKS = ("value", "noncanonical", "dirty", "phase", "constant_phase", "summary_mismatch")


def horizontal_inputs(table):
    """Construct both roots and signs of one horizontal chord, without circuit filtering."""
    p, a = table.p, table.a
    inverse_two = pow(2, -1, p)
    for j, (xp, yp) in enumerate(table.points):
        root = sqrt_mod((-3*xp*xp-4*a) % p, p)
        if root is None or root == 0:
            continue
        xs = sorted({(-xp+root)*inverse_two % p, (-xp-root)*inverse_two % p})
        if xp in xs:
            continue
        result = []
        for negative in (False, True):
            word = (1 << (table.word_bits-1))+j
            if negative:
                word = (1 << table.word_bits)-1-word
            y = (-yp) % p if negative else yp
            for x in xs:
                values = {"word": word, "x": x, "y": y}
                output = zero_point_reference(x, y, word, table)
                if output[1] != -y % p:
                    raise AssertionError("horizontal chord did not negate y")
                projection = scalar_mul(table.order, x, y, p=p, a=a)
                torsion_order = next((k for k in (1, 2, 4, 8)
                                      if projection is None or
                                      scalar_mul(k, *projection, p=p, a=a) is None), None)
                result.append({"kind": "horizontal", "input": values,
                               "negative_digit": negative, "table_index": j,
                               "torsion_projection": projection, "torsion_order": torsion_order})
        return result
    raise ValueError("table contains no nondegenerate horizontal-chord fixture")


def _build(word_bits, variant, table):
    if variant not in VARIANTS:
        raise ValueError("unknown zero-point variant")
    if variant == "original":
        return build_odd_window_case(word_bits, table=table)
    if variant.startswith("positive_add"):
        return build_zero_point_case(word_bits, table=table, phase_correction=False,
            restoration="add", addition_unlookup="split" if variant.endswith("split") else "paired")
    return build_zero_point_case(word_bits, table=table, phase_correction=variant == "zero_phase")


def _restoration_prefix(case):
    """Compile the actual prefix without forging or copying exact-block seals.

    Coordinate handles are unchanged after the third lookup boundary. Temporarily
    truncate this same owner circuit, then restore every modified attribute;
    the compiled plan retains an immutable operation snapshot.
    """
    c = case.circuit
    lookups = [b for b in c.exact_blocks if b.kind == "xor"]
    if len(lookups) != 3:
        return None
    starts = [start for label, start, stop in case.meta["resource_regions"] if label == "lookup"]
    if len(starts) != 3:
        raise ValueError("expected three lookup regions")
    cut, limit = starts[2], lookups[2].cbit_start
    saved = c.ops, c.exact_blocks, c._next_cbit, c._cbit_const
    try:
        c.ops = c.ops[:cut]
        c.exact_blocks = [b for b in c.exact_blocks if b.stop <= cut]
        c._next_cbit = limit
        c._cbit_const = {k: v for k, v in c._cbit_const.items() if k < limit}
        return compile_symbolic(c)
    finally:
        c.ops, c.exact_blocks, c._next_cbit, c._cbit_const = saved


def _prefix_reference(table, values, expected):
    digit = table.digit(values["word"])
    negative = digit < 0
    index = (abs(digit)-1)//2
    ax, ay = table.points[index]
    positive_y = -expected["y"] % table.p if negative else expected["y"]
    return {"word": index | (int(negative) << table.address_bits),
            "x": (expected["x"]-ax) % table.p, "y": (-positive_y-ay) % table.p}


def coset_inputs(table, ordinary):
    """Translate an unchanged fixture list algebraically, never by circuit success."""
    from .full_scalar import active_group
    group = active_group(torsion=True)
    if (table.p, table.a, table.b) != (group.p, group.a, group.b):
        raise ValueError("coset fixtures require the active Wei25519 curve")
    result = []
    for item in ordinary:
        values = item["input"]
        point = group.add(group.torsion, (values["x"], values["y"]))
        if point is None:
            raise AssertionError("torsion coset unexpectedly contains infinity")
        result.append({**item, "kind": "coset_"+item["kind"],
                       "input": {"word": values["word"], "x": point[0], "y": point[1]}})
    return result


def _diagnostics(case, table, fixtures):
    diagnostics, inputs, expected, indices = [], [], [], []
    for item in fixtures:
        values = dict(item["input"])
        row = {**item, "input": values, "domain_failure": None, "all_outcome_clean": False}
        diagnostics.append(row)
        try:
            x, y = zero_point_reference(values["x"], values["y"], values["word"], table)
        except AssertionError as exc:
            row["domain_failure"] = str(exc)
            continue
        row["expected"] = {"word": values["word"], "x": x, "y": y}
        inputs.append(values)
        expected.append(row["expected"])
        indices.append(len(diagnostics)-1)
    plan = compile_symbolic(case.circuit)
    if inputs:
        hazards = simulate_batch(plan, inputs, expected, output_moduli={"x": table.p, "y": table.p})
        for lane, index in enumerate(indices):
            row = diagnostics[index]
            row["output"] = hazards.outputs[lane]
            row["hazards"] = {name: bool(getattr(hazards, name+"_mask") & (1 << lane)) for name in MASKS}
            row["all_outcome_clean"] = not bool(hazards.bad_mask & (1 << lane))
    analysis = {"ir_digest": plan.ir_digest, "stats": dict(plan.stats)}
    prefix = _restoration_prefix(case)
    if prefix is not None:
        analysis["restoration_prefix"] = {"ir_digest": prefix.ir_digest, "stats": dict(prefix.stats)}
        if inputs:
            prefix_expected = [_prefix_reference(table, v, e) for v, e in zip(inputs, expected)]
            before = simulate_batch(prefix, inputs, prefix_expected,
                                    output_moduli={"x": table.p, "y": table.p})
            for lane, index in enumerate(indices):
                output = before.outputs[lane]
                diagnostics[index]["restoration_input"] = {
                    "output": output, "expected": prefix_expected[lane],
                    "canonical_payloads": all(0 <= output[k] < table.p for k in ("x", "y")),
                    "hazards": {name: bool(getattr(before, name+"_mask") & (1 << lane)) for name in MASKS},
                }
    return diagnostics, analysis


def run(*, word_bits=16, random_cases=4, seed=2026100103, variants=None,
        torsion_coset=False):
    if type(word_bits) is not int or not 2 <= word_bits <= 17:
        raise ValueError("word width must be between two and seventeen")
    if type(random_cases) is not int or random_cases < 0:
        raise ValueError("random case count must be a nonnegative integer")
    selected = VARIANTS if variants is None else tuple(variants)
    if not selected or any(v not in VARIANTS for v in selected) or len(set(selected)) != len(selected):
        raise ValueError("variants must be a nonempty unique selection")
    if type(torsion_coset) is not bool:
        raise ValueError("torsion_coset must be a bool")
    digest = source_digest()
    table = make_odd_window_table(word_bits)
    ordinary = _inputs(word_bits, random_cases, seed)
    fixtures = ordinary+horizontal_inputs(table)
    if torsion_coset:
        fixtures += coset_inputs(table, ordinary)
    rows = []
    for variant in selected:
        print(f"Building {FIELD} odd point window: {variant}", flush=True)
        started = time.monotonic()
        case = _build(word_bits, variant, table)
        resources = count(case.circuit)
        regions = region_resources(case.circuit, case.meta["resource_regions"])
        for metric in ("toffoli", "clifford", "measurements"):
            if sum(r[metric] for r in regions.values()) != getattr(resources, metric):
                raise AssertionError(f"regional {metric} count mismatch")
        if max(r["peak_live_qubits"] for r in regions.values()) != resources.peak_qubits:
            raise AssertionError("regional/global peak mismatch")
        lookup_starts = [start for label, start, stop in case.meta["resource_regions"] if label == "lookup"]
        restoration_regions = None
        if lookup_starts:
            restore_start = lookup_starts[-1]
            restore_stop = next(start for label, start, stop in case.meta["resource_regions"]
                                if label == "sign" and start > restore_start)
            restoration_regions = region_resources(case.circuit, [("restoration", restore_start, restore_stop)])["restoration"]
        diagnostics, analysis = _diagnostics(case, table, fixtures)
        counts = {kind: sum(not d["all_outcome_clean"] for d in diagnostics
                            if (d["kind"] == "horizontal") == (kind == "horizontal"))
                  for kind in ("ordinary", "horizontal")}
        rows.append({"variant": variant, "resources": asdict(resources), "regions": regions,
                     "metadata": case.meta, "diagnostics": diagnostics, "analysis": analysis,
                     "restoration_region": restoration_regions,
                     "hazard_counts": counts, "all_outcome_clean": all(d["all_outcome_clean"] for d in diagnostics),
                     "seconds": time.monotonic()-started})
        print(f"{variant}: CCX={resources.toffoli}, peak={resources.peak_qubits}, "
              f"ordinary hazards={counts['ordinary']}, horizontal hazards={counts['horizontal']}", flush=True)
        del case
        gc.collect()
        if source_digest() != digest:
            raise RuntimeError("QSM sources changed during the zero-point experiment")
    by_name = {row["variant"]: row for row in rows}
    baseline = by_name.get("zero_phase")
    deltas = ({name: {metric: row["resources"][metric]-baseline["resources"][metric]
                      for metric in ("toffoli", "peak_qubits", "measurements", "clifford")}
               for name, row in by_name.items()} if baseline is not None else {})
    legacy_added = (baseline["resources"]["toffoli"]-by_name["original"]["resources"]["toffoli"]
                    if baseline is not None and "original" in by_name else None)
    differences = {}
    if baseline is not None:
        for name, row in by_name.items():
            differences[name] = {
                "newly_bad_indices": [i for i, (old, new) in enumerate(zip(baseline["diagnostics"], row["diagnostics"]))
                                      if old["all_outcome_clean"] and not new["all_outcome_clean"]],
                "newly_clean_indices": [i for i, (old, new) in enumerate(zip(baseline["diagnostics"], row["diagnostics"]))
                                        if not old["all_outcome_clean"] and new["all_outcome_clean"]],
            }
    return {"version": "odd-zero-point-v2", "field": FIELD, "prime": table.p,
            "word_bits": word_bits, "table_entries": table.entries,
            "table_fingerprint": table.fingerprint, "source_digest": digest, "seed": seed,
            "rows": rows, "production_changed": False, "torsion_coset_fixtures": torsion_coset,
            "added_toffoli": legacy_added, "deltas_from_zero_phase": deltas,
            "diagnostic_differences_from_zero_phase": differences,
            "scope": "Matched actual odd additions with identical table, tuning and input lanes. "
                     "Common ideal reference permits horizontal chords even for the original circuit. "
                     "All-measurement-outcome symbolic diagnostics of emitted arithmetic, with validated "
                     "exact lookup summaries; deterministic edge/horizontal fixtures and seeded ordinary probes, "
                     "not a failure-rate estimate or universal completeness claim. "
                     "No torsion-coset initializer or full scalar integration."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-bits", type=int, default=16)
    parser.add_argument("--random-cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026100103)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--torsion-coset", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.json.exists():
        parser.error("JSON target already exists")
    try:
        result = run(word_bits=args.word_bits, random_cases=args.random_cases, seed=args.seed,
                     variants=args.variants, torsion_coset=args.torsion_coset)
    except ValueError as exc:
        parser.error(str(exc))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved {args.json}; legacy phase-repair delta CCX: {result['added_toffoli']}")
    return int(not result["rows"][-1]["all_outcome_clean"])


if __name__ == "__main__":
    raise SystemExit(main())
