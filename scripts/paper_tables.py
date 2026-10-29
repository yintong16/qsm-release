"""Fresh circuit measurements and publication tables; no saved benchmark counts.

Each field is built in a separate process before any field-dependent import.
Progress is reported while workers run. The complete table is printed only after
both workers finish, their identities/partitions pass validation, and outputs are saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_files

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("25519", "secp256k1")
ROWS = {
    "table1": (
        ("halvings", "Initial modular halvings"),
        ("doublings", "Modular doublings"),
        ("rows", "Triangular-row additions"),
        ("diagonals", "Initial subtraction and diagonal additions"),
        ("total", "Total"),
    ),
    "table2": (
        ("addition", "Modular addition"),
        ("doubling", "Modular doubling"),
        ("multiply_f", "Multiplication by F"),
        ("square", "Square subtraction"),
    ),
    "table3": (
        ("default", "261 / 264"),
        ("matched", "261 / 261"),
    ),
}
POINT_VERSION = "point.add_const_affine_inplace.v2"
POINT_ITERATIONS = {
    "25519": {"default": 261, "matched": 261},
    "secp256k1": {"default": 264, "matched": 261},
}
TABLE_NOTES = {
    "table1": "Current square ripple=40 in both fields; modular-reduction windows=45/73.",
    "table2": "Current square ripple=40 in both fields; modular addition is tuned/lazy.",
    "table3": "Constant-point addition, ripple=40; CCX / peak qubits; saving is percent CCX relative to secp256k1. Matched iterations do not imply matched error rates.",
}
SQUARE_VERSION = "mul.modsqrsub_inplace_trunc_flat.v2"
PRIMITIVES = (
    ("addition", "field.modadd_lazy.v4"),
    ("doubling", "field.moddouble_trunc.v1"),
    ("multiply_f", "mul.modmul_const_f.v1"),
    ("square", SQUARE_VERSION),
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primitive_ccx(operations):
    from qsm.core import ops
    return sum(ops.physical_gate_counts(op).get(ops.CCX, 0) for op in operations)


def traced_square(build, module):
    """Observe exclusive spans of the real builder without changing any gates."""
    spans, calls, invocation = [], Counter(), {}
    active = False
    originals = {name: getattr(module, name) for name in (
        "modhalve_inplace", "moddouble", "_signed_row", "_narrow_row",
        "modsub_direct", "modadd_lazy", "modsqrsub_inplace")}
    categories = {"modhalve_inplace": "halvings", "moddouble": "doublings",
                  "_signed_row": "rows", "_narrow_row": "rows",
                  "modsub_direct": "diagonals", "modadd_lazy": "diagonals"}

    def trace(name):
        original = originals[name]
        @wraps(original)
        def wrapped(circ, *args, **kwargs):
            nonlocal active
            # A full triangular row owns its internal modular addition.
            if active:
                return original(circ, *args, **kwargs)
            start = len(circ.ops)
            active = True
            try:
                result = original(circ, *args, **kwargs)
            finally:
                active = False
            spans.append({"component": categories[name], "operation": name,
                          "start": start, "stop": len(circ.ops)})
            calls[name] += 1
            return result
        return wrapped

    @wraps(originals["modsqrsub_inplace"])
    def capture(*args, **kwargs):
        bound = inspect.signature(originals["modsqrsub_inplace"]).bind(*args, **kwargs)
        bound.apply_defaults()
        invocation.update({k: bound.arguments[k] for k in (
            "p", "carry_window", "cmp_bits", "window", "ripple", "ctrl")})
        invocation["width"] = len(bound.arguments["x"])
        diag = bound.arguments["diag_add_fn"]
        invocation["diagonal_adder"] = {
            "function": getattr(getattr(diag, "func", diag), "__name__", None),
            "keywords": dict(getattr(diag, "keywords", {}) or {}),
        }
        return originals["modsqrsub_inplace"](*args, **kwargs)

    with ExitStack() as stack:
        for name in categories:
            stack.enter_context(patch.object(module, name, trace(name)))
        stack.enter_context(patch.object(module, "modsqrsub_inplace", capture))
        case = build()
    totals = {key: 0 for key in ("halvings", "doublings", "rows", "diagonals")}
    previous, uncovered = 0, 0
    for span in sorted(spans, key=lambda s: s["start"]):
        if span["start"] < previous:
            raise ValueError("overlapping square component spans")
        uncovered += primitive_ccx(case.circuit.ops[previous:span["start"]])
        value = primitive_ccx(case.circuit.ops[span["start"]:span["stop"]])
        span["toffoli"] = value
        totals[span["component"]] += value
        previous = span["stop"]
    uncovered += primitive_ccx(case.circuit.ops[previous:])
    if uncovered:
        raise ValueError(f"{uncovered} unclassified CCX: update the component adapter rather than hiding gates in a remainder")
    return case, totals, dict(calls), spans, invocation


def measure(case):
    from qsm.core.resources import count
    from qsm.core.symbolic_batch import compile_symbolic
    resources = count(case.circuit)
    plan = compile_symbolic(case.circuit)
    return {"resources": asdict(resources), "operation_nodes": len(case.circuit.ops),
            "input_widths": {k: len(v) for k, v in case.circuit.inputs.items()},
            "ir_digest": plan.ir_digest, "ir_digest_scheme": "qsm-symbolic-ir-v1",
            "case_metadata": case.meta}


def traced_point(build, module):
    """Capture the configuration actually passed to the constant-point emitter."""
    original = module.add_const_affine_inplace
    invocations = []

    @wraps(original)
    def capture(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        n = len(v["x"])
        config = v["inv_kw"]["config"].resolved(n, v["p"])
        invocations.append({
            "width": n, "y_width": len(v["y"]), "p": v["p"], "ctrl": v["ctrl"],
            "constant_point": {"x": v["xp"], "y": v["yp"]},
            "inversion_config": asdict(config),
            "square": {k: v["sqr_kw"][k] for k in ("carry_window", "cmp_bits", "window", "ripple")},
            "constant_subtraction": {k: v["const_kw"][k] for k in ("carry_window", "cmp_bits")},
        })
        return original(*args, **kwargs)

    with patch.object(module, "add_const_affine_inplace", capture):
        case = build()
    if len(invocations) != 1:
        raise ValueError("expected exactly one constant-point addition")
    return case, invocations[0]


def measure_points(field, params, send):
    from qsm.bench.registry import get_builder
    from qsm.inversion.config import InversionConfig
    from qsm.point import affine

    default = InversionConfig.tuned(params.U, params.PRIME)
    if default.iters != POINT_ITERATIONS[field]["default"]:
        raise ValueError("default GCD iteration budget changed; review Table 3's row labels")
    records, values = {}, {}
    for row, _ in ROWS["table3"]:
        iters = POINT_ITERATIONS[field][row]
        key = f"iters_{iters}"
        if key not in records:
            config = InversionConfig.tuned(params.U, params.PRIME, iters=iters)
            send("building", version=POINT_VERSION, iterations=iters)
            before = time.perf_counter()
            build = (get_builder(POINT_VERSION) if row == "default" else
                     lambda: affine._affine_case(False, config, legacy_square_ripple=False))
            case, actual = traced_point(build, affine)
            if (actual["width"], actual["y_width"], actual["p"], actual["ctrl"]) != (params.U, params.U, params.PRIME, None):
                raise ValueError("Table 3 requires an uncontrolled full-width constant-point addition")
            if actual["constant_point"] != {"x": affine.curve.GX, "y": affine.curve.GY}:
                raise ValueError("constant-point builder no longer uses the active curve's generator")
            if actual["inversion_config"] != asdict(config):
                raise ValueError("emitted point addition used a different inversion configuration")
            if actual["square"] != dict(carry_window=params.CARRY_WINDOW, cmp_bits=40,
                                         window=params.CARRY_WINDOW, ripple=40):
                raise ValueError("point square tuning no longer matches the paper definition")
            if actual["constant_subtraction"] != dict(carry_window=params.CARRY_WINDOW, cmp_bits=40):
                raise ValueError("point constant-subtraction tuning changed")
            record = measure(case)
            record.update(version=POINT_VERSION, actual_parameters=actual,
                          builder="registered" if row == "default" else "qsm.point.affine._affine_case",
                          overrides={} if row == "default" else {"iters": iters},
                          elapsed_seconds=time.perf_counter()-before)
            records[key] = record
            del case
            gc.collect()
        resource = records[key]["resources"]
        value = dict(ccx=resource["toffoli"], peak_qubits=resource["peak_qubits"],
                     iterations=iters, circuit_key=key)
        values[row] = value
        send("value", row=row, **value)
    return records, values


def worker(table, field):
    os.environ["QSM_FIELD"] = field
    sys.path.insert(0, str(ROOT))
    import qsm.field
    import qsm.mul
    from qsm import params as p
    from qsm.bench.calibration import source_digest
    from qsm.bench.registry import get_builder
    from qsm.mul import accumulate

    def send(kind, **values):
        print(json.dumps({"kind": kind, "field": field, **values}), flush=True)

    started = utc()
    source = source_digest()
    runner = file_sha(Path(__file__))
    if p.FIELD != field or (p.ERROR_BUDGET, p.CMP_BITS, p.SQUARE_RIPPLE) != (40, 40, 40):
        raise ValueError("paper definitions require error_budget=40, comparator=40 and current square ripple=40; do not silently relabel changed settings")
    width, deficit = (255, 19) if field == "25519" else (256, (1 << 32)+977)
    if (p.U, p.F, p.PRIME) != (width, deficit, (1 << width)-deficit):
        raise ValueError("live field parameters do not match the paper's named field")
    if p.CARRY_WINDOW != 40+(deficit-1).bit_length():
        raise ValueError("reduction window no longer matches the paper's 40-bit budget")
    profile = {"field": field, "field_bits": p.U, "prime": p.PRIME, "deficit": p.F,
               "error_budget": p.ERROR_BUDGET, "carry_window": p.CARRY_WINDOW,
               "comparator_bits": p.CMP_BITS, "current_square_ripple": p.SQUARE_RIPPLE,
               "diagonal_chunk": accumulate.FLAT_DIAG_CHUNK}
    send("start", source_digest=source, runner_sha256=runner, started=started, profile=profile)
    records, values = {}, {}
    if table == "table1":
        send("building", version=SQUARE_VERSION)
        before = time.perf_counter()
        case, totals, calls, spans, actual = traced_square(get_builder(SQUARE_VERSION), accumulate)
        record = measure(case)
        record.update(version=SQUARE_VERSION, components=totals, calls=calls,
                      component_spans=spans, actual_parameters=actual,
                      elapsed_seconds=time.perf_counter()-before)
        if actual["ctrl"] is not None or actual["ripple"] != 40:
            raise ValueError("table1 did not select the uncontrolled common-40 square")
        if (actual["width"], actual["p"], actual["carry_window"], actual["cmp_bits"], actual["window"]) != (p.U, p.PRIME, p.CARRY_WINDOW, 40, p.CARRY_WINDOW):
            raise ValueError("square construction/configuration mismatch")
        if calls.get("modhalve_inplace") != p.U or calls.get("moddouble") != p.U:
            raise ValueError("square scaling schedule changed; review the paper component definition")
        if calls.get("_signed_row", 0)+calls.get("_narrow_row", 0) != p.U-1:
            raise ValueError("triangular row schedule changed")
        if calls.get("modsub_direct") != 1 or calls.get("modadd_lazy") != 2:
            raise ValueError("initial/diagonal operation partition changed")
        total = record["resources"]["toffoli"]
        if sum(totals.values()) != total:
            raise ValueError("component sum disagrees with the independently counted full circuit")
        values = {**totals, "total": total}
        records["square"] = record
        for key, _ in ROWS[table]:
            send("value", row=key, ccx=values[key])
    elif table == "table3":
        records, values = measure_points(field, p, send)
    else:
        for key, version in PRIMITIVES:
            send("building", version=version)
            before = time.perf_counter()
            if key == "square":
                case, totals, calls, spans, actual = traced_square(get_builder(version), accumulate)
                if actual["ripple"] != 40 or actual["ctrl"] is not None:
                    raise ValueError("table2 square did not select the common 40-bit margin")
                record = measure(case)
                if sum(totals.values()) != record["resources"]["toffoli"]:
                    raise ValueError("square component/total mismatch")
                record.update(components=totals, calls=calls, component_spans=spans,
                              actual_parameters=actual)
            else:
                case = get_builder(version)()
                record = measure(case)
            record.update(version=version, elapsed_seconds=time.perf_counter()-before)
            records[key] = record
            values[key] = record["resources"]["toffoli"]
            send("value", row=key, ccx=values[key])
            del case
    if source_digest() != source or file_sha(Path(__file__)) != runner:
        raise RuntimeError("source or table runner changed while measuring; no completed table is valid")
    send("complete", source_digest=source, runner_sha256=runner, started=started, finished=utc(),
         profile=profile, values=values, circuits=records)


def measured_value(table, field, event):
    key, ccx = event["row"], event["ccx"]
    if key not in dict(ROWS[table]) or type(ccx) is not int or ccx < 0:
        raise ValueError("invalid measured row")
    if table != "table3":
        return ccx
    peak, iters = event.get("peak_qubits"), event.get("iterations")
    if ccx == 0 or type(peak) is not int or peak < 1:
        raise ValueError("Table 3 requires positive measured CCX and peak qubits")
    if type(iters) is not int or iters != POINT_ITERATIONS[field][key]:
        raise ValueError("point measurement does not match its GCD-iteration row")
    if event.get("circuit_key") != f"iters_{iters}":
        raise ValueError("point measurement is missing its circuit identity")
    return dict(ccx=ccx, peak_qubits=peak, iterations=iters, circuit_key=event["circuit_key"])


def table_rows(table, values):
    rows = []
    for key, label in ROWS[table]:
        a, b = (values[field][key] for field in FIELDS)
        saving = 100*(b["ccx"]-a["ccx"])/b["ccx"] if table == "table3" else b-a
        rows.append([label, a, b, saving])
    return rows


def headings(table):
    first = {"table1": "Component", "table2": "Operation", "table3": "GCD iterations"}[table]
    return first, "Curve25519" if table == "table1" else "Wei25519"


def formatted_row(table, row):
    label, a, b, saving = row
    if table == "table3":
        return [label, f"{a['ccx']:,} / {a['peak_qubits']:,}",
                f"{b['ccx']:,} / {b['peak_qubits']:,}", f"{saving:.2f}%"]
    return [label, f"{a:,}", f"{b:,}", f"{saving:,}"]


def console_table(table, rows):
    heading, curve = headings(table)
    width = max(len(heading), *(len(row[0]) for row in rows))
    resource_width = 21 if table == "table3" else 12
    lines = [f"{heading:<{width}}  {curve:>{resource_width}}  {'secp256k1':>{resource_width}}  {'Saving':>12}",
             "-"*(width+2*resource_width+18)]
    for row in rows:
        label, left, right, delta = formatted_row(table, row)
        lines.append(f"{label:<{width}}  {left:>{resource_width}}  {right:>{resource_width}}  {delta:>12}")
    return "\n".join(lines)


def markdown(table, rows):
    first, curve = headings(table)
    return "\n".join([f"| {first} | {curve} | secp256k1 | Saving |",
                       "|---|---:|---:|---:|",
                       *["| "+" | ".join(formatted_row(table, row))+" |" for row in rows]])+"\n"


def latex(table, rows):
    if table == "table1":
        caption = "Square-subtraction costs with a common 40-bit narrow-row ripple margin and modular-reduction windows of 45 bits for Curve25519 and 73 bits for secp256k1. All entries are CCX counts."
        label, begin, end = "tab:square-breakdown", r"\begin{tabular}{lrrr}", r"\end{tabular}"
        header = r"Component & Curve25519 & \texttt{secp256k1} & Saving\\"
    elif table == "table2":
        caption = "Field-operation costs with the same 40-bit narrow-row carry margin in both square circuits. All entries are CCX counts."
        label, begin, end = "tab:field-primitives", r"\begin{tabularx}{\linewidth}{Xrrr}", r"\end{tabularx}"
        header = r"Operation & Wei25519 & \texttt{secp256k1} & Saving\\"
    else:
        caption = r"Constant-point addition costs. Each resource pair is CCX / peak qubits. Percentages are CCX savings relative to \texttt{secp256k1}."
        label, begin, end = "tab:point-fields", r"\begin{tabularx}{\linewidth}{Xrrr}", r"\end{tabularx}"
        header = r"GCD iterations & Wei25519 & \texttt{secp256k1} & Saving\\"
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{"+caption+"}", r"\label{"+label+"}",
             begin, r"\toprule", header, r"\midrule"]
    for row in rows:
        text, a, b, saving = formatted_row(table, row)
        if table == "table1" and text == "Total":
            lines.append(r"\midrule")
        if text == "Multiplication by F": text = r"Multiplication by $F$"
        lines.append(" & ".join((text, a, b, saving.replace("%", r"\%")))+r"\\")
    lines += [r"\bottomrule", end, r"\end{table}"]
    if table == "table2":
        lines.insert(0, "% Modular addition is the tuned/lazy field.modadd_lazy.v4, not the canonical positive-restoration adder.")
    return "\n".join(lines)+"\n"


def stream_worker(process, field, events):
    try:
        for line in process.stdout:
            events.put((field, line))
    finally:
        events.put((field, None))


def run(table, output=None):
    started = utc()
    directory = benchmark_files.create_directory(ROOT, table, output)
    outputs = benchmark_files.paths(directory, table)
    naming_sha = benchmark_files.digest()
    events, processes, handles, threads = queue.Queue(), {}, [], []
    values = {field: {} for field in FIELDS}
    completed, ended = {}, set()
    result = {"schema": "fresh-paper-table-v1", "table": table, "started": started,
              "complete": False, "measurement_mode": "fresh emitted circuit IR; no stored counts read",
              "runner_sha256": file_sha(Path(__file__)), "python": sys.version,
              "output_naming_sha256": naming_sha,
              "wrapper_sha256": file_sha(ROOT/"paper") if (ROOT/"paper").exists() else None,
              "fields": {}, "rows": []}
    print(f"\n{table}: fresh circuit builds started {started}", flush=True)
    print(TABLE_NOTES[table], flush=True)
    try:
        for field in FIELDS:
            stderr = (directory/(field+".stderr.log")).open("x")
            raw = (directory/(field+".jsonl")).open("x")
            handles.extend((stderr, raw))
            process = subprocess.Popen([sys.executable, "-I", "-B", "-u", str(Path(__file__).resolve()),
                                        "--worker", table, "--worker-field", field],
                cwd=ROOT, env=dict(os.environ, QSM_FIELD=field, PYTHONDONTWRITEBYTECODE="1"),
                stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
            processes[field] = (process, raw)
            thread = threading.Thread(target=stream_worker, args=(process, field, events), daemon=True)
            thread.start(); threads.append(thread)
        while len(ended) < len(FIELDS):
            field, line = events.get()
            process, raw = processes[field]
            if line is None:
                ended.add(field)
                code = process.wait()
                if code != 0 or field not in completed:
                    raise RuntimeError(f"{field} worker failed (exit {code}); see {directory/(field+'.stderr.log')}")
                continue
            raw.write(line); raw.flush()
            event = json.loads(line)
            if event.get("field") != field:
                raise ValueError("worker field mismatch")
            if event["kind"] == "start":
                print(f"[{field}] source {event['source_digest'][:16]}...; field bits={event['profile']['field_bits']}", file=sys.stderr, flush=True)
            elif event["kind"] == "building":
                suffix = f"; GCD iterations={event['iterations']}" if "iterations" in event else ""
                print(f"[{field}] building and counting {event['version']}{suffix}", file=sys.stderr, flush=True)
            elif event["kind"] == "value":
                key = event["row"]
                if key in values[field]:
                    raise ValueError("duplicate measured row")
                values[field][key] = measured_value(table, field, event)
            elif event["kind"] == "complete":
                if field in completed: raise ValueError("duplicate worker completion")
                completed[field] = event
            else:
                raise ValueError("unknown worker event")
        expected_rows = {key for key, _ in ROWS[table]}
        if set(completed) != set(FIELDS) or any(set(values[field]) != expected_rows for field in FIELDS):
            raise ValueError("incomplete measurements")
        if len({v["source_digest"] for v in completed.values()}) != 1:
            raise ValueError("fields were measured from different source versions")
        for field in FIELDS:
            if completed[field]["runner_sha256"] != result["runner_sha256"]:
                raise ValueError("workers used a different table runner")
            if completed[field]["values"] != values[field]:
                raise ValueError("live values disagree with final worker result")
            if table == "table3":
                for value in values[field].values():
                    record = completed[field]["circuits"].get(value["circuit_key"])
                    if record is None or (record["resources"]["toffoli"], record["resources"]["peak_qubits"],
                                          record["actual_parameters"]["inversion_config"]["iters"]) != (
                                          value["ccx"], value["peak_qubits"], value["iterations"]):
                        raise ValueError("point row disagrees with its measured circuit record")
        if benchmark_files.digest() != naming_sha:
            raise RuntimeError("output naming code changed while measuring")
        rows = table_rows(table, values)
        result.update(complete=True, finished=utc(), fields=completed, rows=rows,
                      source_digest=completed["25519"]["source_digest"],
                      scope="Static resources from fresh circuit construction; no hardware timing or correctness/success guarantee.")
        outputs["tex"].write_text(latex(table, rows))
        outputs["md"].write_text(markdown(table, rows))
    except BaseException as exc:
        result.update(complete=False, error=str(exc), finished=utc(), fields=completed, partial_values=values)
        raise
    finally:
        for process, _ in processes.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for thread in threads: thread.join(timeout=5)
        for process, _ in processes.values():
            if process.stdout is not None:
                process.stdout.close()
        for handle in handles: handle.close()
        outputs["json"].write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print("\n"+console_table(table, rows), flush=True)
    print(f"\nCompleted fresh measurements. JSON: {outputs['json']}", flush=True)
    print(f"LaTeX: {outputs['tex']}\nMarkdown: {outputs['md']}", flush=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="?", choices=(*ROWS, "table4", "table5", "table6", "signed"), help="paper table or zero-safe signed schedules to rebuild")
    parser.add_argument("--out", type=Path, help="fresh output directory; existing paths are refused")
    parser.add_argument("--field", choices=FIELDS, help="signed only: build one field instead of both")
    parser.add_argument("--list", action="store_true", help="list supported paper tables without building")
    parser.add_argument("--worker", choices=tuple(ROWS), help=argparse.SUPPRESS)
    parser.add_argument("--worker-field", choices=FIELDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.worker_field: parser.error("worker field is required")
        worker(args.worker, args.worker_field)
        return 0
    if args.list:
        print("table1  Square-subtraction breakdown: current v2, ripple40/40")
        print("table2  Field primitives: current v2 square, ripple40/40")
        print("table3  Constant-point addition: current v2, GCD 261/264 and 261/261")
        print("table4  Wei25519 full schedules: unsigned, zero-safe signed half-tables, complete odd-only")
        print("table5  Odd-only savings: fresh matched 16-bit additions and component accounting")
        print("table6  Published secp256k1 references plus a fresh full Wei25519 odd-only circuit")
        print("signed  Zero-safe centered-signed full schedules only: both fields in parallel")
        return 0
    if args.table == "table5":
        if args.field is not None:
            parser.error("Table 5 is fixed to Wei25519; --field applies only to signed")
        command = [sys.executable, "-I", "-B", "-u", str(ROOT/"scripts/odd_savings_bench.py")]
        if args.out is not None:
            command.extend(("--out", str(args.out.resolve())))
        return subprocess.call(command)
    if args.table in ("signed", "table4", "table6"):
        if args.table != "signed" and args.field is not None:
            parser.error("--field applies only to the signed schedules command")
        command = [sys.executable, "-I", "-B", "-u", str(ROOT/"scripts/signed_paper_bench.py")]
        if args.table in ("table4", "table6"):
            command.append("--"+args.table)
        if args.out is not None:
            command.extend(("--out", str(args.out.resolve())))
        if args.field is not None:
            command.extend(("--field", args.field))
        return subprocess.call(command)
    if args.field is not None:
        parser.error("--field applies only to the signed schedules command")
    if args.table is None:
        parser.print_help()
        return 0
    try:
        run(args.table, args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"paper: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("paper: interrupted; no completed table claimed", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
