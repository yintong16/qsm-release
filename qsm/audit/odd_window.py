"""Matched single-addition odd-only audit; no scalar schedule/default changes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time

from ..bench.calibration import source_digest
from ..bench.registry import Case
from ..core.circuit import Circuit, DirtyAncillaError
from ..core.resources import count
from ..core.simulator import simulate
from ..params import FIELD, PRIME, U
from ..point import active as curve
from ..scalarmul.local_signed import bias_recode_inplace
from ..scalarmul.signed_window import add_signed_window_inplace, make_signed_window_table, signed_window_reference
from ..scalarmul.window_opt import OptimizedWindowConfig
from .signed_chain import group_add
from .signed_window import region_resources


def build_centered_case(word_bits=16, *, zero_safe=False) -> Case:
    if type(zero_safe) is not bool:
        raise ValueError("zero_safe must be a bool")
    if type(word_bits) is not int or not 2 <= word_bits <= 17:
        raise ValueError("raw word width must be between two and seventeen")
    address_bits = word_bits-1
    table = make_signed_window_table(address_bits)
    config = OptimizedWindowConfig.tuned(U, PRIME, address_bits).resolved(U, PRIME, address_bits)
    circ = Circuit()
    word, x, y = circ.input("word", word_bits), circ.input("x", U), circ.input("y", U)
    regions = []
    start = len(circ.ops)
    bias_recode_inplace(circ, word)
    regions.append(("recoding", start, len(circ.ops)))
    x, y = add_signed_window_inplace(circ, x, y, word[:-1], word[-1], table,
                                    sign_strategy="conjugate", config=config, regions=regions,
                                    zero_safe=zero_safe)
    start = len(circ.ops)
    bias_recode_inplace(circ, word, inverse=True)
    regions.append(("recoding", start, len(circ.ops)))
    circ.rebind("x", x)
    circ.rebind("y", y)

    def reference(values):
        raw = values["word"]
        if type(raw) is not int or not 0 <= raw < 1 << word_bits:
            raise ValueError("word outside declared radix")
        digit = raw-(1 << address_bits)
        xr, yr = signed_window_reference(values["x"], values["y"],
                                         abs(digit) % (1 << address_bits), int(digit < 0), table,
                                         zero_safe=zero_safe)
        return {"word": raw, "x": xr, "y": yr}

    def sample(rng):
        while True:
            point = curve.mul(rng.randrange(1, curve.ORDER), curve.GX, curve.GY)
            if point is None:
                continue
            values = {"word": rng.randrange(1 << word_bits), "x": point[0], "y": point[1]}
            try:
                reference(values)
            except AssertionError:
                continue
            return values

    metadata = {
        "empirical_only": True, "fail_analytic": None,
        "input_distribution": "original uniform word and nonidentity subgroup point, ideal-domain conditioned only",
        "arithmetic_contract": "current centered signed window with local recoder and inverse; tuned lazy/incomplete arithmetic",
        "output_moduli": {"x": PRIME, "y": PRIME},
        "resource_regions": regions,
        "circuit_config": {"word_bits": word_bits, "address_bits": address_bits,
                           "strategy": "centered", "table_fingerprint": table.fingerprint,
                           "window": config.metadata(), "inversion": asdict(config.inversion)},
    }
    if zero_safe:
        metadata.update({
            "zero_safe": True, "coordinate_restoration": "add",
            "restoration_adder": "modadd.borrowed_pseudo_mersenne",
            "restoration_unlookup": "paired", "restoration_exact_reduction": True,
            "restoration_carry_window": None, "restoration_comparator_bits": None,
            "phase_correction": False,
            "zero_contract": "literal canonical zero; lazy p-as-zero is not supported",
            "arithmetic_contract": "zero-safe centered signed window with canonical positive restoration; inherited tuned hazards remain",
        })
    return Case(circ, reference, sample, meta=metadata,
                analysis_context={"window_table": table, "signed_window_table": table})


def _build(word_bits, strategy):
    if strategy == "centered":
        return build_centered_case(word_bits)
    from ..scalarmul.odd_window import build_odd_window_case
    return build_odd_window_case(word_bits)


def _inputs(word_bits, random_cases, seed):
    rng = random.Random(seed)
    midpoint = 1 << (word_bits-1)
    words = [0, midpoint-1, midpoint, (1 << word_bits)-1]
    words += [rng.randrange(1 << word_bits) for _ in range(random_cases)]
    result = []
    for index, word in enumerate(words):
        point = curve.mul(rng.randrange(1, curve.ORDER), curve.GX, curve.GY)
        if point is None:
            raise AssertionError("nonzero subgroup scalar yielded identity")
        result.append({"kind": "edge" if index < 4 else "seeded",
                       "input": {"word": word, "x": point[0], "y": point[1]}})
    return result


def run(*, word_bits=16, random_cases=4, seed=2026100101) -> dict:
    if type(word_bits) is not int or not 2 <= word_bits <= 17:
        raise ValueError("word width must be between two and seventeen")
    if type(random_cases) is not int or random_cases < 0:
        raise ValueError("random case count must be a nonnegative integer")
    digest = source_digest()
    inputs = _inputs(word_bits, random_cases, seed)
    rows = []
    for si, strategy in enumerate(("centered", "odd")):
        print(f"Building {FIELD}, raw word {word_bits}, {strategy}", flush=True)
        started = time.monotonic()
        case = _build(word_bits, strategy)
        if set(case.circuit.inputs) != {"word", "x", "y"}:
            raise ValueError("single-addition case must expose word,x,y")
        resources = count(case.circuit)
        regions = region_resources(case.circuit, case.meta.get("resource_regions", ()))
        for metric in ("toffoli", "clifford", "measurements"):
            if sum(r[metric] for r in regions.values()) != getattr(resources, metric):
                raise AssertionError(f"regional {metric} count mismatch")
        if max(r["peak_live_qubits"] for r in regions.values()) != resources.peak_qubits:
            raise AssertionError("regional/global peak mismatch")
        diagnostics = []
        for index, item in enumerate(inputs):
            values = dict(item["input"])
            measurement_seed = seed+1000000*(si+1)+index
            row = {"input": values, "kind": item["kind"], "measurement_seed": measurement_seed,
                   "domain_failure": None, "dirty_failure": None, "value_failure": None,
                   "phase": None, "canonical_output": None, "observed_pass": False}
            try:
                expected = case.reference(values)
                row["expected"] = expected
            except AssertionError as exc:
                row["domain_failure"] = str(exc)
                diagnostics.append(row)
                continue
            try:
                result = simulate(case.circuit, values, rng=random.Random(measurement_seed))
            except DirtyAncillaError as exc:
                row["dirty_failure"] = str(exc)
                diagnostics.append(row)
                continue
            row["output"] = result.outputs
            row["phase"] = result.phase
            row["measurements"] = len(result.measurements)
            row["measurement_digest"] = hashlib.sha256(bytes(result.measurements)).hexdigest()
            normalized = {k: v % PRIME if k in ("x", "y") else v for k, v in result.outputs.items()}
            row["value_failure"] = normalized != expected
            row["canonical_output"] = all(0 <= result.outputs[k] < PRIME for k in ("x", "y"))
            row["observed_pass"] = not row["value_failure"] and result.phase == 0
            diagnostics.append(row)
        rows.append({"strategy": strategy, "word_bits": word_bits, "address_bits": word_bits-1,
                     "table_entries": 1 << (word_bits-1), "resources": asdict(resources),
                     "regions": regions, "metadata": case.meta, "diagnostics": diagnostics,
                     "seconds": time.monotonic()-started,
                     "observed_pass": all(d["observed_pass"] for d in diagnostics)})
        print(f"{strategy}: CCX={resources.toffoli}, peak={resources.peak_qubits}, "
              f"pass={rows[-1]['observed_pass']}, cases={len(diagnostics)}", flush=True)
        del case
        gc.collect()
        if source_digest() != digest:
            raise RuntimeError("QSM sources changed during the single-addition experiment")
    half = curve.mul(pow(2, -1, curve.ORDER), curve.GX, curve.GY)
    relation_checks = 0
    for left, right in zip(rows[0]["diagnostics"], rows[1]["diagnostics"]):
        if left["domain_failure"] is None and right["domain_failure"] is None:
            a, b = left["expected"], right["expected"]
            if group_add((a["x"], a["y"]), half, PRIME, curve.CURVE.A) != (b["x"], b["y"]):
                raise AssertionError("ideal odd/centered outputs are not related by the declared half-point shift")
            relation_checks += 1
    return {"version": "single-odd-window-v1", "field": FIELD, "prime": PRIME,
            "curve": {"a": curve.CURVE.A, "b": curve.CURVE.B,
                      "order": curve.ORDER, "base": [curve.GX, curve.GY]},
            "word_bits": word_bits, "seed": seed, "source_digest": digest,
            "rows": rows, "half_point": half, "ideal_shift_checks": relation_checks,
            "saved_toffoli": rows[0]["resources"]["toffoli"]-rows[1]["resources"]["toffoli"],
            "production_changed": False,
            "scope": "actual single additions with recoding+inverse, fixed tuning and equal magnitude-table size; "
                     "same original words and starting points, different ideal outputs related by fixed half-point; "
                     "edge/seeded diagnostics with independent measurement streams, not failure-rate comparison; "
                     "no initialization/full scalar schedule change, no new full-schedule domain certificate"}


def main(argv=None):
    from tabulate import tabulate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-bits", type=int, default=16)
    parser.add_argument("--random-cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026100101)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.json.exists():
        parser.error("JSON target already exists")
    try:
        result = run(word_bits=args.word_bits, random_cases=args.random_cases, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print(tabulate([[r["strategy"], r["table_entries"], r["resources"]["toffoli"],
                     r["resources"]["peak_qubits"], r["resources"]["toffoli_depth"], r["observed_pass"]]
                    for r in result["rows"]], headers=["variant", "entries", "CCX", "peak", "CCX depth", "pass"], tablefmt="github"))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved {args.json}; actual CCX saved: {result['saved_toffoli']}")
    return int(any(not row["observed_pass"] for row in result["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
