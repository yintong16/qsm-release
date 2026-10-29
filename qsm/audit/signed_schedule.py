"""Measured terminal-carry operations and explicitly compositional scalar ledgers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

from ..bench.calibration import source_digest
from ..bench.harness import benchmark_case
from ..bench.registry import Case
from ..core.circuit import Circuit
from ..params import FIELD, PRIME, U
from ..point import active as curve
from ..point.affine import add_const_affine_inplace, tuned_point_kwargs
from .signed_recoding import recoder_resource_report, schedule_counts


def terminal_case(exponent: int) -> Case:
    if type(exponent) is not int or exponent < 0:
        raise ValueError("terminal exponent must be a nonnegative integer")
    point = curve.mul(1 << exponent, curve.GX, curve.GY)
    if point is None:
        raise ValueError("terminal point is the identity")
    if not curve.is_on_curve(*point):
        raise ValueError("terminal point is not on the active curve")
    circ = Circuit()
    ctrl = circ.input("c", 1)
    x, y = circ.input("x", U), circ.input("y", U)
    kwargs = tuned_point_kwargs(U, PRIME)
    x, y = add_const_affine_inplace(circ, x, y, *point, ctrl=ctrl[0], p=PRIME, **kwargs)
    circ.rebind("x", x)
    circ.rebind("y", y)

    def reference(v):
        if (v["x"] - point[0]) % PRIME == 0 or (v["y"] - point[1]) % PRIME == 0:
            raise AssertionError("invalid initial coordinate difference")
        out = curve.add(v["x"], v["y"], *point) if v["c"] else (v["x"], v["y"])
        if out is None or (v["c"] and out[0] == point[0]):
            raise AssertionError("invalid output coordinate")
        return {"c": v["c"], "x": out[0], "y": out[1]}

    def sample(rng):
        while True:
            r = curve.mul(rng.randrange(1, curve.ORDER), curve.GX, curve.GY)
            if r is None:
                continue
            v = {"c": rng.randrange(2), "x": r[0], "y": r[1]}
            try:
                reference(v)
            except AssertionError:
                continue
            return v

    return Case(circ, reference, sample, meta={
        "empirical_only": True, "fail_analytic": None,
        "input_distribution": "uniform nonidentity subgroup point and control, conditioned on ideal algebraic domain",
        "arithmetic_contract": "tuned incomplete affine terminal-carry addition; no certified error bound",
        "output_moduli": {"x": PRIME, "y": PRIME},
        "circuit_config": {"terminal_exponent": exponent, "point": point,
                           "curve": {"p": PRIME, "a": curve.CURVE.A, "b": curve.CURVE.B,
                                     "base": [curve.GX, curve.GY], "order": curve.ORDER}},
    })


def measure_terminals(exponents=(252, 255, 256), *, trials=4, seed=2026092902) -> dict:
    if not exponents or any(type(e) is not int or e < 0 for e in exponents):
        raise ValueError("nonnegative terminal exponents required")
    if type(trials) is not int or trials < 1:
        raise ValueError("positive trial count required")
    digest = source_digest()
    rows = []
    for exponent in sorted(set(exponents)):
        print(f"Building {FIELD} terminal carry at exponent {exponent}", flush=True)
        started = time.monotonic()
        case = terminal_case(exponent)
        result = benchmark_case(f"experiment.signed_terminal.{exponent}", case,
                                n_trials=trials, seed=seed, meas_reps=1, persist=False)
        rows.append({"exponent": exponent, "result": asdict(result),
                     "seconds": time.monotonic() - started})
        print(f"terminal {exponent}: CCX={result.toffoli}, peak={result.peak_qubits}, "
              f"smoke_pass={result.observed_pass}", flush=True)
        del case, result
        gc.collect()
    if source_digest() != digest:
        raise RuntimeError("QSM sources changed during terminal measurements")
    return {"version": "signed-terminals-v1", "field": FIELD, "source_digest": digest,
            "seed": seed, "rows": rows,
            "scope": "actually built controlled additions of [2^exponent]G; unpaired smoke tests only"}


def compose(windows: dict, terminals: dict, *, scalar_bits=(252, 253, 255, 256),
            strategy="conjugate") -> dict:
    if strategy not in ("conjugate", "selected"):
        raise ValueError("a signed strategy is required")
    if not scalar_bits or any(type(v) is not int or v < 1 for v in scalar_bits):
        raise ValueError("positive scalar widths required")
    if windows.get("version") != "signed-window-v1" or terminals.get("version") != "signed-terminals-v1":
        raise ValueError("incorrect input archive type")
    if windows["field"] != terminals["field"] or windows["source_digest"] != terminals["source_digest"]:
        raise ValueError("measurement field/source mismatch")
    if source_digest() != windows["source_digest"]:
        raise ValueError("current source differs from component measurements")
    wrows = {(r["address_bits"], r["strategy"]): r for r in windows["rows"]}
    trows = {r["exponent"]: r for r in terminals["rows"]}
    addresses = sorted({a for a, s in wrows if s == strategy})
    if not addresses:
        raise ValueError("no measured signed windows for requested strategy")
    output = []
    for length in sorted(set(scalar_bits)):
        for a in addresses:
            if (a, "unsigned") not in wrows:
                raise ValueError("missing matched unsigned measurement")
            selected = [wrows[a, s]["result"] for s in ("unsigned", strategy)]
            if any(not r["observed_pass"] for r in selected):
                raise ValueError("component smoke failures cannot be silently used as passing estimates")
            slots = schedule_counts(length, a)
            unsigned_ccx, signed_ccx = (r["toffoli"] for r in selected)
            rows = {"unsigned": {"radix_slots": slots["unsigned"]["radix_slots"],
                                  "ccx_model": slots["unsigned"]["radix_slots"] * unsigned_ccx,
                                  "recoder_ccx": 0, "terminal_ccx": 0}}
            for variant in ("carry", "bias"):
                q = slots[variant]["radix_slots"]
                recoder = recoder_resource_report(length, a, variant, validate=False)
                terminal_ccx = 0
                exponent = q * (a + 1)
                if slots[variant]["terminal_constant_addition_slots"]:
                    if exponent not in trows or not trows[exponent]["result"]["observed_pass"]:
                        raise ValueError(f"missing passing terminal measurement for exponent {exponent}")
                    terminal_ccx = trows[exponent]["result"]["toffoli"]
                recoder_ccx = recoder["compute_uncompute"]["toffoli"]
                rows[variant] = {"radix_slots": q, "ccx_model": q * signed_ccx + terminal_ccx + recoder_ccx,
                                 "recoder_ccx": recoder_ccx, "terminal_ccx": terminal_ccx,
                                 "recoder_resources": recoder}
            output.append({"scalar_bits": length, "address_bits": a, "signed_strategy": strategy,
                           "schedule": slots, "variants": rows})
    return {"version": "signed-schedule-model-v1", "field": windows["field"],
            "source_digest": windows["source_digest"], "rows": output,
            "scope": "per-scalar compositional CCX model, NOT an emitted full scalar circuit; "
                     "fixed known-start policy with bias correction folded into the prepared start; "
                     "no first-addition lookup credit; generic full-width window charged even at partial top; "
                     "no three-addition postprocessing credit; Fourier/preparation Clifford costs omitted; "
                     "no scalar-chain peak estimate, streaming claim, or success probability; "
                     "G-base terminal constants, not measured P/Q double-scalar oracle"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("terminals")
    measure.add_argument("--exponents", nargs="+", type=int, default=[252, 255, 256])
    measure.add_argument("--trials", type=int, default=4)
    measure.add_argument("--seed", type=int, default=2026092902)
    estimate = sub.add_parser("compose")
    estimate.add_argument("--windows", type=Path, required=True)
    estimate.add_argument("--terminals", type=Path, required=True)
    estimate.add_argument("--scalar-bits", nargs="+", type=int, default=[252, 253, 255, 256])
    estimate.add_argument("--strategy", choices=["conjugate", "selected"], default="conjugate")
    for p in (measure, estimate):
        p.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.json.exists():
        parser.error("JSON target already exists")
    try:
        if args.command == "terminals":
            result = measure_terminals(args.exponents, trials=args.trials, seed=args.seed)
        else:
            result = compose(json.loads(args.windows.read_text()), json.loads(args.terminals.read_text()),
                             scalar_bits=args.scalar_bits, strategy=args.strategy)
    except ValueError as exc:
        parser.error(str(exc))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved {args.json}")
    if args.command == "terminals":
        return int(any(not r["result"]["observed_pass"] for r in result["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
