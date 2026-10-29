"""Measure the exclusive gate savings of matched signed and odd-only additions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_files
from paper_tables import measure, primitive_ccx

ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    ("recoding", "Centered recoding and its inverse"),
    ("enable", "Zero-digit enable logic"),
    ("endpoint", "Negative-endpoint correction"),
    ("square_copy", "Controlled square-source copy"),
)
KINDS = ("signed", "odd")


def now():
    return datetime.now(timezone.utc).isoformat()


def tool_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(benchmark_files.__file__), ROOT/"scripts/paper_tables.py")}


def function_config(fn):
    base = getattr(fn, "func", fn)
    return dict(function=base.__name__, module=base.__module__,
                keywords=dict(getattr(fn, "keywords", {}) or {}))


def traced_addition(build):
    from qsm.scalarmul import signed_window as signed, zero_point as odd
    from qsm.mul import accumulate

    spans, signatures, calls = [], {}, {}
    active_square = None
    original_enable = signed._signed_enabled
    original_endpoint = signed._endpoint_xor
    original_and, original_unand = accumulate._and, accumulate._unand

    def record(component, operation, start, circ):
        spans.append(dict(component=component, operation=operation, start=start, stop=len(circ.ops)))

    @contextmanager
    def enabled(circ, *args, **kwargs):
        calls["enable"] = calls.get("enable", 0)+1
        start = len(circ.ops)
        with original_enable(circ, *args, **kwargs) as flag:
            record("enable", "prepare", start, circ)
            yield flag
            start = len(circ.ops)
        record("enable", "erase", start, circ)

    @wraps(original_endpoint)
    def endpoint(circ, *args, **kwargs):
        calls["endpoint"] = calls.get("endpoint", 0)+1
        start = len(circ.ops)
        result = original_endpoint(circ, *args, **kwargs)
        record("endpoint", "adjust", start, circ)
        return result

    def copy_gate(original, operation):
        @wraps(original)
        def wrapped(circ, a, b, target):
            if active_square is None or a != active_square["ctrl"] or b not in active_square["source"]:
                raise ValueError("unexpected AND helper use outside the square-source copy")
            start = len(circ.ops)
            result = original(circ, a, b, target)
            record("square_copy", operation, start, circ)
            calls[operation] = calls.get(operation, 0)+1
            return result
        return wrapped

    def square(original):
        @wraps(original)
        def wrapped(*args, **kwargs):
            nonlocal active_square
            if active_square is not None or "square" in signatures:
                raise ValueError("expected one non-nested fused square")
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            v = bound.arguments
            signatures["square"] = {k: v[k] for k in ("p", "carry_window", "cmp_bits", "window", "ripple")}
            signatures["square"].update(width=len(v["x"]), controlled=v["ctrl"] is not None,
                                         adder=function_config(v["add_fn"]),
                                         diagonal_adder=function_config(v["diag_add_fn"] or v["add_fn"]))
            active_square = dict(ctrl=v["ctrl"], source=set(v["x"].ids))
            try:
                return original(*args, **kwargs)
            finally:
                active_square = None
        return wrapped

    def inversion(original, operation):
        @wraps(original)
        def wrapped(circ, x, y, p, *, config):
            if operation in signatures:
                raise ValueError("duplicate zero-safe inversion stage")
            signatures[operation] = asdict(config.resolved(len(x), p))
            return original(circ, x, y, p, config=config)
        return wrapped

    original_restore = odd._restore_positive
    @wraps(original_restore)
    def restore(*args, **kwargs):
        if "restoration" in signatures:
            raise ValueError("duplicate positive-restoration stage")
        bound = inspect.signature(original_restore).bind(*args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        signatures["restoration"] = dict(p=v["p"], width=len(v["x"]), unlookup=v["unlookup"],
                                           low_bits=v["low_bits"], adder=function_config(v["add_fn"]))
        return original_restore(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(signed, "_signed_enabled", enabled))
        stack.enter_context(patch.object(signed, "_endpoint_xor", endpoint))
        stack.enter_context(patch.object(accumulate, "_and", copy_gate(original_and, "copy_compute")))
        stack.enter_context(patch.object(accumulate, "_unand", copy_gate(original_unand, "copy_erase")))
        stack.enter_context(patch.object(odd, "_restore_positive", restore))
        for module in (signed, odd):
            stack.enter_context(patch.object(module, "modsqrsub_inplace", square(module.modsqrsub_inplace)))
            for name, operation in (("moddiv_zero_safe_inplace", "division"), ("modmul_zero_safe_inplace", "multiplication")):
                stack.enter_context(patch.object(module, name, inversion(getattr(module, name), operation)))
        case = build()
    for label, start, stop in case.meta["resource_regions"]:
        if label == "recoding":
            spans.append(dict(component="recoding", operation="recode", start=start, stop=stop))
    previous = 0
    totals = {key: 0 for key, _ in LABELS}
    for span in sorted(spans, key=lambda s: s["start"]):
        if not previous <= span["start"] <= span["stop"] <= len(case.circuit.ops):
            raise ValueError("overlapping or invalid savings spans")
        span["toffoli"] = primitive_ccx(case.circuit.ops[span["start"]:span["stop"]])
        totals[span["component"]] += span["toffoli"]
        previous = span["stop"]
    return case, dict(components=totals, component_spans=spans, signatures=signatures, calls=calls)


def worker(kind, output):
    if output.exists():
        raise FileExistsError("worker output exists")
    os.environ["QSM_FIELD"] = "25519"
    sys.path.insert(0, str(ROOT))
    from qsm import params
    from qsm.audit.odd_window import build_centered_case
    from qsm.audit.signed_window import region_resources
    from qsm.bench.calibration import source_digest
    from qsm.scalarmul.zero_point import build_zero_point_case

    if (params.FIELD, params.U, params.F, params.CARRY_WINDOW, params.CMP_BITS, params.SQUARE_RIPPLE) != ("25519", 255, 19, 45, 40, 40):
        raise ValueError("Table 5 requires the selected Wei25519 arithmetic settings")
    source, hashes, started = source_digest(), tool_hashes(), now()
    print(f"Building fresh {kind} 16-bit addition with matched zero-safe positive restoration", flush=True)
    build = (lambda: build_centered_case(16, zero_safe=True)) if kind == "signed" else (
        lambda: build_zero_point_case(16, restoration="add", phase_correction=False, addition_unlookup="paired"))
    case, trace = traced_addition(build)
    record = measure(case)
    record.update(trace)
    regions = region_resources(case.circuit, case.meta["resource_regions"])
    for metric in ("toffoli", "clifford", "measurements"):
        if sum(r[metric] for r in regions.values()) != record["resources"][metric]:
            raise ValueError("regional/full-circuit resource mismatch")
    def region(name):
        return regions.get(name, {}).get("toffoli", 0)
    components = record["components"]
    if region("unattributed") or region("recoding") != components["recoding"]:
        raise ValueError("unclassified CCX or inconsistent recoding spans")
    if region("selection") != components["enable"]+components["endpoint"]:
        raise ValueError("selection contains CCX outside the measured enable/endpoint spans")
    for span in record["component_spans"]:
        required = "arithmetic" if span["component"] == "square_copy" else (
            "recoding" if span["component"] == "recoding" else "selection")
        if not any(label == required and start <= span["start"] <= span["stop"] <= stop
                   for label, start, stop in case.meta["resource_regions"]):
            raise ValueError("component span escaped its declared resource region")
    shared = {name: value["toffoli"] for name, value in regions.items()
              if name not in ("recoding", "selection", "unattributed")}
    shared["arithmetic"] -= components["square_copy"]
    if any(v < 0 for v in shared.values()) or sum(shared.values())+sum(components.values()) != record["resources"]["toffoli"]:
        raise ValueError("exclusive component/shared partition mismatch")
    table = case.analysis_context["window_table"]
    base = table.points[1] if kind == "signed" else table.base
    record.update(kind=kind, field=params.FIELD, word_bits=16, public_base=list(base),
                  curve=dict(p=table.p, a=table.a, b=table.b), table_entries=len(table.points),
                  table_fingerprint=table.fingerprint, regions=regions, shared_regions=shared,
                  source_digest=source, tools=hashes, started=started, finished=now())
    validate_record(record, kind)
    if source_digest() != source or tool_hashes() != hashes:
        raise RuntimeError("source or measurement tools changed during the build")
    with output.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Measured {kind}: {record['resources']['toffoli']:,} CCX; components={components}", flush=True)


def validate_record(record, kind):
    if (record["kind"], record["field"], record["word_bits"], record["table_entries"]) != (kind, "25519", 16, 1 << 15):
        raise ValueError("point measurement identity mismatch")
    meta = record["case_metadata"]
    if (meta.get("coordinate_restoration"), meta.get("restoration_exact_reduction"),
        meta.get("phase_correction"), meta.get("restoration_unlookup")) != ("add", True, False, "paired"):
        raise ValueError("both additions must use the selected zero-safe positive-restoration policy")
    if kind == "signed" and meta.get("zero_safe") is not True:
        raise ValueError("signed addition is not zero-safe")
    if record["input_widths"] != {"word": 16, "x": 255, "y": 255} or record["curve"]["p"] != (1 << 255)-19:
        raise ValueError("point operands do not match the declared Wei25519/raw-word widths")
    signatures = record["signatures"]
    if set(signatures) != {"square", "division", "multiplication", "restoration"}:
        raise ValueError("missing observed arithmetic stage")
    if signatures["division"] != signatures["multiplication"]:
        raise ValueError("division/multiplication settings differ")
    if signatures["square"]["controlled"] != (kind == "signed"):
        raise ValueError("unexpected square-control policy")
    expected_calls = dict(enable=2, endpoint=2, copy_compute=255, copy_erase=255) if kind == "signed" else {}
    if record["calls"] != expected_calls:
        raise ValueError("digit handling or square-copy structure changed; review component definitions")
    if signatures["restoration"]["unlookup"] != "paired":
        raise ValueError("restoration cleanup no longer matches")


def savings(signed, odd):
    for record, kind in ((signed, "signed"), (odd, "odd")):
        validate_record(record, kind)
    for key in ("source_digest", "tools", "word_bits", "curve", "public_base", "table_entries", "shared_regions"):
        if signed[key] != odd[key]:
            raise ValueError(f"unmatched additions: {key}")
    for key in ("division", "multiplication", "restoration"):
        if signed["signatures"][key] != odd["signatures"][key]:
            raise ValueError(f"unmatched arithmetic support: {key}")
    a, b = (dict(record["signatures"]["square"]) for record in (signed, odd))
    a.pop("controlled"); b.pop("controlled")
    if a != b:
        raise ValueError("unmatched square settings")
    rows = [[label, signed["components"][key]-odd["components"][key]] for key, label in LABELS]
    total = signed["resources"]["toffoli"]-odd["resources"]["toffoli"]
    if sum(value for _, value in rows) != total:
        raise ValueError("component savings do not equal the independently counted whole-circuit difference")
    return rows+[["Total", total]]


def tables(rows):
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{CCX savings from odd-only encoding per 16-bit addition compared to signed half-tables with the same arithmetic support.}",
           r"\label{tab:odd}", r"\begin{tabularx}{\linewidth}{Xr}", r"\toprule",
           r"Work saved & CCX saved\\", r"\midrule"]
    md = ["| Work saved | CCX saved |", "|---|---:|"]
    for label, value in rows:
        if label == "Total":
            tex.append(r"\midrule")
        tex.append(f"{label} & {value:,}"+r"\\")
        md.append(f"| {label} | {value:,} |")
    tex += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(tex)+"\n", "\n".join(md)+"\n"


def stream(process, handle, kind):
    for line in process.stdout:
        handle.write(line); handle.flush()
        print(f"[{kind}] {line.rstrip()}", flush=True)


def run(output=None):
    sys.path.insert(0, str(ROOT))
    from qsm.bench.calibration import source_digest
    directory = benchmark_files.create_directory(ROOT, "table5", output)
    outputs = benchmark_files.paths(directory, "table5")
    source, hashes = source_digest(), tool_hashes()
    manifest = dict(schema="fresh-odd-savings-v1", table="table5", complete=False, started=now(),
                    source_digest=source, tools=hashes, circuits={}, jobs=[],
                    scope="Static resource accounting of fresh matched point circuits, not a new correctness or success-rate calibration.")
    running = []
    try:
        for kind in KINDS:
            target = directory/("signed-half-tables.json" if kind == "signed" else "odd-only.json")
            command = [sys.executable, "-I", "-B", "-u", str(Path(__file__).resolve()),
                       "--worker", kind, "--json", str(target)]
            job = dict(kind=kind, command=command, output=target.name)
            manifest["jobs"].append(job)
            handle = (directory/(kind+".log")).open("x")
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, QSM_FIELD="25519"),
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            except BaseException:
                handle.close()
                raise
            thread = threading.Thread(target=stream, args=(process, handle, kind), daemon=True)
            running.append((process, handle, thread, job, target))
            thread.start()
        for process, _, thread, job, target in running:
            job["exit_code"] = process.wait()
            thread.join()
            if job["exit_code"] != 0 or not target.exists():
                raise RuntimeError(f"{job['kind']} build failed; inspect its log")
            raw = target.read_bytes()
            record = json.loads(raw)
            if record["source_digest"] != source or record["tools"] != hashes:
                raise ValueError("worker source/tools do not match the parent")
            job["sha256"] = hashlib.sha256(raw).hexdigest()
            manifest["circuits"][job["kind"]] = record
        rows = savings(manifest["circuits"]["signed"], manifest["circuits"]["odd"])
        if source_digest() != source or tool_hashes() != hashes:
            raise RuntimeError("source or measurement tools changed while measuring")
        tex, md = tables(rows)
        outputs["tex"].write_text(tex)
        outputs["md"].write_text(md)
        manifest.update(complete=True, rows=rows)
        print(md, flush=True)
        print(f"Completed odd-encoding savings: {directory}", flush=True)
    except BaseException as exc:
        manifest["error"] = str(exc)
        raise
    finally:
        for process, handle, thread, _, _ in running:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            thread.join(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            handle.close()
        manifest["finished"] = now()
        outputs["json"].write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", choices=KINDS, help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if args.json is None:
                parser.error("worker output is required")
            worker(args.worker, args.json)
        else:
            run(args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"odd savings: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
