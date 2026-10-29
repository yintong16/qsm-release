"""Fresh raw-word signed-addition resources and all-outcome zero-slope diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def horizontal_fixtures(table, word_bits):
    from qsm.point.sampling import sqrt_mod

    p, midpoint = table.p, 1 << (word_bits-1)
    fixtures, selected = [], []
    for j, (ax, ay) in enumerate(table.points):
        root = sqrt_mod((-3*ax*ax-4*table.a) % p, p)
        if root is None or root == 0:
            continue
        xs = sorted({(-ax+root)*pow(2, -1, p) % p, (-ax-root)*pow(2, -1, p) % p})
        if ax in xs:
            continue
        selected.append(j)
        words = [(midpoint, ay, "dummy_horizontal"), (0, -ay % p, "endpoint_horizontal")] if j == 0 else [
            (midpoint+j, ay, "positive_horizontal"), (midpoint-j, -ay % p, "negative_horizontal")]
        for word, y, kind in words:
            for x in xs:
                fixtures.append({"kind": kind, "table_index": j,
                                 "input": {"word": word, "x": x, "y": y}})
        if j > 0:
            break
    if not any(j > 0 for j in selected):
        raise ValueError("no nondegenerate nonzero-digit horizontal fixture found")
    return fixtures, {"indices": selected, "dummy_horizontal_exists": 0 in selected,
                      "selection": "first algebraic horizontal chord, plus row-zero chords when they exist; no circuit-outcome filtering"}


def run(field, output, *, word_bits=16, random_cases=4, seed=2026101501):
    if output.exists():
        raise FileExistsError("JSON target exists; choose a fresh path")
    if not 2 <= word_bits <= 17 or random_cases < 0:
        raise ValueError("word_bits must be 2..17 and random_cases must be nonnegative")
    os.environ["QSM_FIELD"] = field
    sys.path.insert(0, str(ROOT))
    from qsm.audit.odd_window import build_centered_case, _inputs
    from qsm.audit.signed_window import region_resources
    from qsm.audit.zero_point import MASKS, coset_inputs
    from qsm.bench.calibration import source_digest
    from qsm.core.resources import count
    from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
    from qsm import params

    if params.FIELD != field:
        raise ValueError("field-dependent modules were imported before selecting the field")
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    source = source_digest()
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"Building fresh {field} zero-safe signed point, raw word bits={word_bits}", flush=True)
    case = build_centered_case(word_bits, zero_safe=True)
    table = case.analysis_context["signed_window_table"]
    resources = count(case.circuit)
    regions = region_resources(case.circuit, case.meta["resource_regions"])
    for metric in ("toffoli", "clifford", "measurements"):
        if sum(row[metric] for row in regions.values()) != getattr(resources, metric):
            raise ValueError("regional/full-circuit count mismatch")
    if max(row["peak_live_qubits"] for row in regions.values()) != resources.peak_qubits:
        raise ValueError("regional/global peak mismatch")
    print(f"Measured CCX={resources.toffoli:,}; peak={resources.peak_qubits:,}; checking all measurement outcomes", flush=True)
    ordinary = _inputs(word_bits, random_cases, seed)
    horizontal, selection = horizontal_fixtures(table, word_bits)
    fixtures = ordinary+horizontal
    if field == "25519":
        fixtures += coset_inputs(table, ordinary)
    diagnostics, inputs, expected, indices = [], [], [], []
    for item in fixtures:
        row = {**item, "domain_failure": None, "all_outcome_clean": False}
        diagnostics.append(row)
        try:
            want = case.reference(item["input"])
        except AssertionError as exc:
            row["domain_failure"] = str(exc)
            continue
        row["expected"] = want
        indices.append(len(diagnostics)-1)
        inputs.append(item["input"])
        expected.append(want)
    plan = compile_symbolic(case.circuit)
    hazards = simulate_batch(plan, inputs, expected, output_moduli={"x": table.p, "y": table.p})
    for lane, index in enumerate(indices):
        diagnostics[index].update(output=hazards.outputs[lane],
            hazards={name: bool(getattr(hazards, name+"_mask") & (1 << lane)) for name in MASKS},
            all_outcome_clean=not bool(hazards.bad_mask & (1 << lane)))
    if source_digest() != source or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runner_sha:
        raise RuntimeError("source or runner changed during signed point measurement")
    data = {"version": "zero-safe-signed-point-v1", "field": field, "variant": "signed_zero_safe",
            "started": started, "finished": datetime.now(timezone.utc).isoformat(),
            "seconds": time.perf_counter()-before, "source_digest": source, "runner_sha256": runner_sha,
            "word_bits": word_bits, "seed": seed, "resources": asdict(resources), "regions": regions,
            "metadata": case.meta, "table_fingerprint": table.fingerprint,
            "ir_digest": plan.ir_digest, "ir_digest_scheme": "qsm-symbolic-ir-v1",
            "operation_nodes": len(case.circuit.ops), "horizontal_selection": selection,
            "diagnostics": diagnostics, "all_outcome_clean": all(d["all_outcome_clean"] for d in diagnostics),
            "scope": "Fresh raw-word point circuit including forward/inverse recoding; fixed subgroup, coset and horizontal fixtures with all-measurement-outcome diagnostics. Not an IID calibration or universal exactness claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    clean = sum(d["all_outcome_clean"] for d in diagnostics)
    print(f"{field}: {clean}/{len(diagnostics)} strict-clean fixtures; saved {output}", flush=True)
    return int(not data["all_outcome_clean"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("25519", "secp256k1"), required=True)
    parser.add_argument("--word-bits", type=int, default=16)
    parser.add_argument("--random-cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026101501)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.field, args.json, word_bits=args.word_bits, random_cases=args.random_cases, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"signed benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
