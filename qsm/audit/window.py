"""Opt-in, sequential diagnostics for individual tuned affine windows.

Run ``python -m qsm.audit.window --bits 1 2 4 8 --trials 4 --meas-reps 1``.
QSM_FIELD selects the process's active field. These are emitted whole-window
resources and empirical input-batch diagnostics, never full-scalar estimates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import sys
from typing import TextIO

from .. import params
from ..bench import report
from ..bench.harness import Result, benchmark_case
from ..bench.registry import Case
from ..bench.statistics import probability, trial_count
from ..core import ops
from ..inversion.config import InversionConfig
from ..scalarmul.window import build_window_case
from ..scalarmul.window_opt import OptimizedWindowConfig, build_optimized_window_case

DEFAULT_BITS = (1, 2, 4, 8)
MAX_DEFAULT_BITS = 12
REGION_LABELS = ("lookup", "unlookup", "selection", "arithmetic")


def window_case_version(case: Case) -> str:
    """Stable identity of the active field, complete table hash and resolved tuning.

    Resource markers are construction bookkeeping, not an operating point. Every
    other builder metadata item enters the hash, including any future config keys.
    Sampling seed/trial count deliberately do not create new circuit versions.
    """
    metadata = {k: v for k, v in case.meta.items() if k != "resource_regions"}
    payload = {"field": [params.FIELD, params.U, params.PRIME], "meta": metadata}
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return (f"scalarmul.window_sweep.w{case.meta['window_bits']}."
            f"{case.meta['lookup_backend']}.{params.FIELD}.{digest}.v1")


def count_window_regions(case: Case) -> dict:
    """One op-index scan; no circuit copies, slices or depth/resource recount.

    All gate operations must belong to disjoint half-open named regions. Unlabelled
    ALLOC/FREE markers are allowed (notably the input register allocations).
    String kinds also handle phase gates without inspecting gate object classes.
    """
    regions = sorted(case.meta["resource_regions"], key=lambda r: (r[1], r[2]))
    end = 0
    for label, start, stop in regions:
        if (label not in REGION_LABELS or type(start) is not int or
                type(stop) is not int or not end <= start <= stop <= len(case.circuit.ops)):
            raise ValueError("resource_regions must be disjoint valid named op intervals")
        end = stop
    per_region = dict.fromkeys(REGION_LABELS, 0)
    region_peaks = dict.fromkeys(REGION_LABELS, 0)
    kinds: dict[str, int] = {}
    region_index = 0
    live = peak = parity_rows = parity_terms = parity_sources = 0
    for index, op in enumerate(case.circuit.ops):
        kind = op.kind
        while region_index < len(regions) and index >= regions[region_index][2]:
            region_index += 1
        label = (regions[region_index][0] if region_index < len(regions) and
                 index >= regions[region_index][1] else None)
        before = live
        if kind == ops.ALLOC:
            live += len(op.qubits)
            peak = max(peak, live)
        elif kind == ops.FREE:
            live -= len(op.qubits)
        elif kind == ops.CPARITY:
            parity_rows += len(op.values)
            parity_terms += sum(word.bit_count() for word in op.values)
            parity_sources += len(op.sources)
        else:
            if label is None:
                raise ValueError(f"unlabelled gate at op index {index}")
            if kind == ops.XOR_WORD:
                physical = ops.X if op.ctrl is None else ops.CX
                kinds[physical] = kinds.get(physical, 0) + op.mask.bit_count()
            elif kind in ops.GATE_KINDS:
                kinds[kind] = kinds.get(kind, 0) + 1
            else:
                raise ValueError(f"unknown op kind {kind}")
            if kind == ops.CCX:
                per_region[label] += 1
            if kind == ops.MEASURE:
                live -= 1
        if label is not None:
            region_peaks[label] = max(region_peaks[label], before, live)
    return {
        "ccx": kinds.get("CCX", 0),
        "cx": kinds.get("CX", 0),
        "measurements": kinds.get("MEASURE", 0),
        "n_gates": sum(kinds.values()),
        "clifford": sum(kinds.get(k, 0) for k in ("X", "CX", "CZ", "Z")),
        "gate_kinds": kinds,
        "toffoli_by_region": per_region,
        "peak_qubits": peak,
        "peak_qubits_by_region": region_peaks,
        "lookup_calls": sum(label == "lookup" for label, _, _ in regions),
        "unlookup_calls": sum(label == "unlookup" for label, _, _ in regions),
        "ir_ops": len(case.circuit.ops),
        "classical_parity_rows": parity_rows,
        "classical_parity_bit_terms": parity_terms,
        "classical_parity_sources": parity_sources,
    }


def _validate_counts(counts: dict, result: Result) -> None:
    for local, total in (("ccx", "toffoli"), ("clifford", "clifford"),
                         ("measurements", "measurements"), ("n_gates", "n_gates"),
                         ("peak_qubits", "peak_qubits")):
        if counts[local] != getattr(result, total):
            raise ValueError(f"window op counts disagree with benchmark {total}")
    if sum(counts["toffoli_by_region"].values()) != result.toffoli:
        raise ValueError("window region Toffoli counts do not sum to whole circuit")


def _print_result(result: Result, bits: int, counts: dict, stream: TextIO) -> None:
    print(f"\n{result.version}", file=stream)
    print(f"  EMPIRICAL ONLY; observed_pass={result.observed_pass}; "
          "analytic correctness=unknown (not regression acceptance)", file=stream)
    print(f"  calls={result.trials}; failed_union={result.fail_seen}; "
          f"output={result.output_seen}; dirty={result.dirty_seen}; phase={result.phase_seen}",
          file=stream)
    print(f"  failed input batches={result.batches_failed}/{result.batches_completed}; "
          f"95% CP upper={result.fail_bound95:.6g} "
          "(sampled batch diagnostics, not quantum norm)", file=stream)
    print(f"  emitted CCX={result.toffoli}; CX={counts['cx']}; "
          f"measurements={result.measurements}; total_gates={result.n_gates}; "
          f"peak_qubits={result.peak_qubits}", file=stream)
    print("  region CCX: " + "; ".join(
        f"{label}={counts['toffoli_by_region'][label]}" for label in REGION_LABELS),
        file=stream)
    print("  region live peaks (including inputs): " + "; ".join(
        f"{label}={counts['peak_qubits_by_region'][label]}" for label in REGION_LABELS),
        file=stream)
    print(f"  lookup calls={counts['lookup_calls']}; unlookup calls={counts['unlookup_calls']}; "
          f"stored IR ops={counts['ir_ops']}", file=stream)
    print(f"  classical parity rows={counts['classical_parity_rows']}; "
          f"bit terms={counts['classical_parity_bit_terms']}; "
          "classical latency excluded from quantum resources", file=stream)
    print(f"  local CCX/window bit={result.toffoli / bits:.6g}; "
          "not a full-scalar extrapolation", file=stream, flush=True)
    if result.success_samples is not None:
        print(f"  statistical all-outcome success lower={result.success_lower!r}; "
              f"confidence={result.success_confidence}; "
              f"hazardous inputs={result.bad_inputs}/{result.success_samples}; "
              "conditional input law, not Shor recovery", file=stream, flush=True)


def sweep(
    bits=DEFAULT_BITS, *, n_trials: int = 64, meas_reps: int = 4,
    seed: int = 0, persist: bool = True, allow_large: bool = False,
    config: InversionConfig | None = None, stream: TextIO | None = None,
    backend: str = "v1", operand_strategy: str = "paired",
    adder_chunk: int | str | None = "half", fused_block: int | str = "half",
    low_bits: int | None = None, verbose: bool = True,
    success_samples: int | None = None, confidence: float = 0.99,
    success_batch_size: int = 1024, calibration_seed: int | None = None,
) -> list[Result]:
    """Build, diagnose, print and release each window before building the next.

    The returned Results contain only small JSON-safe summaries, not Cases or
    builders. Widths >12 on v1 and >16 on the compact optimized backend require
    an explicit override. Larger permitted widths can still exhaust memory.
    """
    trial_count(n_trials, "n_trials", minimum=1)
    trial_count(meas_reps, "meas_reps", minimum=1)
    if success_samples is not None:
        trial_count(success_samples, "success samples", minimum=1)
        trial_count(success_batch_size, "success batch size", minimum=1)
        confidence = probability(confidence, "confidence")
        if not 0 < confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one")
        if calibration_seed is None:
            if type(seed) is not int:
                raise ValueError("calibration requires an integer seed")
            calibration_seed = seed + 1
        if type(calibration_seed) is not int:
            raise ValueError("calibration seed must be an integer")
    if backend not in ("v1", "optimized"):
        raise ValueError("backend must be v1 or optimized")
    if backend == "v1" and (operand_strategy != "paired" or adder_chunk != "half" or
                            fused_block != "half" or low_bits is not None):
        raise ValueError("operand/split settings require the optimized backend")
    limit = MAX_DEFAULT_BITS if backend == "v1" else 16
    widths = []
    for width in bits:
        trial_count(width, "window bits", minimum=1)
        if width > limit and not allow_large:
            raise ValueError(f"window bits >{limit} require --allow-large; "
                             "table and circuit sizes grow exponentially")
        if width not in widths:
            widths.append(width)
    if not widths:
        raise ValueError("at least one window width is required")
    if config is not None and config.arithmetic_family != "tuned":
        raise ValueError("window sweep supports tuned arithmetic only, not conservative")
    configs = {}
    if backend == "optimized":
        configs = {width: OptimizedWindowConfig.tuned(
            params.U, params.PRIME, width, operand_strategy=operand_strategy,
            adder_chunk=adder_chunk, fused_block=fused_block,
            unlookup_low_bits=low_bits, inversion=config) for width in widths}
    stream = sys.stdout if stream is None else stream
    if verbose:
        print(f"Individual-window sweep: field={params.FIELD}, u={params.U}, "
              f"p={params.PRIME}; backend={backend}; empirical diagnostics only", file=stream, flush=True)
    else:
        print(f"Live window benchmark: field={params.FIELD}; backend={backend}",
              file=stream, flush=True)
    results = []
    for width in widths:
        print(f"Building w={width} (one circuit at a time)", file=stream, flush=True)
        if backend == "optimized" and verbose:
            cfg = configs[width]
            print(f"  operand={cfg.operand_strategy}; adder_chunk={cfg.adder_chunk}; "
                  f"fused_block={cfg.fused_block}; low_bits={cfg.unlookup_low_bits}",
                  file=stream, flush=True)
        case = (build_window_case(width, config=config) if backend == "v1" else
                build_optimized_window_case(width, config=configs[width]))
        analysis_plan = None
        benchmark_kwargs = {}
        try:
            if case.meta.get("empirical_only") is not True or case.meta.get("fail_analytic") is not None:
                raise ValueError("window sweep requires empirical_only and fail_analytic=None")
            version = window_case_version(case)
            counts = count_window_regions(case)
            case.meta["circuit_config"] = {
                **case.meta.get("circuit_config", {}),
                "window_resources": counts,
            }
            if success_samples is not None:
                from ..core.symbolic_batch import compile_symbolic
                print("Compiling all-outcome checker and binding emitted IR...", file=stream, flush=True)
                analysis_plan = compile_symbolic(case.circuit)
                benchmark_kwargs["analysis_plan"] = analysis_plan
            # Persist only after consistency checks, using the existing atomic writer.
            result = benchmark_case(version, case, n_trials=n_trials, seed=seed,
                                    meas_reps=meas_reps, persist=False, **benchmark_kwargs)
            _validate_counts(counts, result)
            if result.correct is not None or not result.empirical_only:
                raise ValueError("benchmark returned non-empirical window diagnostics")
            if success_samples is not None:
                from ..bench.calibration import attach_calibration, calibrate_window
                calibration = calibrate_window(
                    case, samples=success_samples, confidence=confidence,
                    seed=calibration_seed, batch_size=success_batch_size, stream=stream,
                    plan=analysis_plan)
                attach_calibration(result, calibration)
            if persist:
                report.append_result(asdict(result))
            if verbose:
                _print_result(result, width, counts, stream)
            results.append(result)
        finally:
            analysis_plan = None
            benchmark_kwargs.clear()
            del case
            gc.collect()
    return results


def _block_choice(value: str):
    if value == "half":
        return value
    if value == "full":
        return None
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("choose half, full, or a positive block size") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("block size must be positive")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", nargs="+", type=int, default=list(DEFAULT_BITS))
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--meas-reps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-persist", action="store_true", help="do not append benchmark history")
    parser.add_argument("--backend", choices=("v1", "optimized"), default="v1")
    parser.add_argument("--operand", dest="operand_strategy", choices=("paired", "fused"), default="paired")
    parser.add_argument("--adder-chunk", type=_block_choice, default="half",
                        help="optimized: half of n+1 (default), full, or a positive size")
    parser.add_argument("--fused-block", type=_block_choice, default="half",
                        help="fused: half of n+1 (default), or a positive size")
    parser.add_argument("--low-bits", type=int, help="measured-unlookup low-address split (default w//2)")
    parser.add_argument("--allow-large", action="store_true",
                        help="permit widths >12 (v1) or >16 (optimized); memory grows exponentially")
    args = parser.parse_args(argv)
    try:
        results = sweep(args.bits, n_trials=args.trials, meas_reps=args.meas_reps,
                        seed=args.seed, persist=not args.no_persist,
                        allow_large=args.allow_large, backend=args.backend,
                        operand_strategy=args.operand_strategy, adder_chunk=args.adder_chunk,
                        fused_block=args.fused_block, low_bits=args.low_bits)
    except ValueError as exc:
        parser.error(str(exc))
    # Zero means no sampled failure, not an exactness or model-acceptance claim.
    return 0 if all(r.observed_pass is True for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
