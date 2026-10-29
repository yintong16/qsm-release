"""Fresh scalar benchmarks: signed comparisons, Wei25519 schedules or Table 6."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_files

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("25519", "secp256k1")
SEED = 2026092905
SCHEDULES = (
    ("unsigned", "Unsigned tables"),
    ("signed_zero_safe", "Signed half-tables"),
    ("odd_torsion_add", "Complete odd-only construction"),
)
# Fixed, user-supplied quotations, not measured or independently verified here.
LITERATURE_REFERENCES = (
    dict(construction="secp256k1, space-optimized", citation_key="Schrottenloher26",
         toffoli_log2=26.11, logical_qubits=1208),
    dict(construction="secp256k1, gate-optimized", citation_key="Schrottenloher26",
         toffoli_log2=25.78, logical_qubits=1462),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def scalar_job(directory, field, variant, name):
    bits = 253 if field == "25519" else 256
    target = directory / (name+".json")
    command = [sys.executable, "-I", "-B", "-u", "-m", "qsm", "--field", field,
               "audit", "full-scalar", "--variants", variant,
               "--u-bits", str(bits), "--v-bits", "208", "--window-bits", "16",
               "--samples", "4", "--seed", str(SEED), "--json", str(target)]
    return dict(name=name, field=field, variant=variant, u_bits=bits, command=command, output=str(target))


def jobs(directory, fields=FIELDS):
    return [scalar_job(directory, field, "signed_zero_safe", field) for field in fields]


def schedule_jobs(directory):
    return [scalar_job(directory, "25519", variant, variant) for variant, _ in SCHEDULES]


def comparison_jobs(directory):
    return [scalar_job(directory, "25519", "odd_torsion_add", "odd_torsion_add")]


def verify(data, field, source, *, variant="signed_zero_safe"):
    if variant not in dict(SCHEDULES) or field not in FIELDS:
        raise ValueError("unknown scalar benchmark construction or field")
    row, = data["rows"]
    bits = 253 if field == "25519" else 256
    if (data["field"], row["variant"], row["u_bits"], row["v_bits"], row["window_bits"],
        row["source_digest"], row["seed"], row["recycled"], row["full_width_test_settings"]) != (
        field, variant, bits, 208, 16, source, SEED, True, False):
        raise ValueError("scalar benchmark identity mismatch")
    if variant == "signed_zero_safe" and row.get("zero_safe") is not True:
        raise ValueError("signed benchmark does not use the selected zero-safe policy")
    if variant in ("signed_zero_safe", "odd_torsion_add"):
        if row.get("restoration_exact_reduction") is not True:
            raise ValueError("scalar benchmark does not use exact restoration reduction")
        if row.get("coordinate_restoration") != "add" or row.get("restoration_unlookup") != "paired":
            raise ValueError("scalar benchmark restoration policy mismatch")
    if len(row["stages"]) != 29 or row["point_additions"] != 28 or len(row["diagnostics"]) != 6:
        raise ValueError("incomplete signed schedule or diagnostics")
    if row["stages"][0]["kind"] != "initial_lookup" or row["stages"][0]["bits"] != bits-240:
        raise ValueError("unexpected initializer")
    for metric in ("toffoli", "clifford", "measurements"):
        if sum(s[metric] for s in row["stages"]) != row["resources"][metric]:
            raise ValueError("stage/full resource mismatch")
    if max(s["global_peak_so_far"] for s in row["stages"]) != row["resources"]["peak_qubits"]:
        raise ValueError("stage/full peak mismatch")
    additions = {s["toffoli"] for s in row["stages"][1:]}
    if len(additions) != 1 or any(s["kind"] != "point_addition" for s in row["stages"][1:]):
        raise ValueError("addition costs are not uniform; do not print a single per-addition value")
    strict = sum(bool(d["observed_pass"] and d["active"] and d["phase"] == 0
                      and d["output"] == d["expected"] and not d["phase_stages"]
                      and not d["noncanonical_stages"] and not any(d[k] for k in
                      ("value_failure", "control_failure", "domain_failure", "dirty_failure")))
                 for d in row["diagnostics"])
    return dict(field=field, variant=variant, allocation=f"{bits}/208", addition_ccx=additions.pop(),
                point_additions=row["point_additions"], initializer_ccx=row["stages"][0]["toffoli"],
                total_ccx=row["resources"]["toffoli"], peak_qubits=row["resources"]["peak_qubits"],
                strict_clean=strict, diagnostics=6)


def schedule_table(results):
    by_variant = {r["variant"]: r for r in results}
    if len(results) != len(SCHEDULES) or set(by_variant) != dict(SCHEDULES).keys():
        raise ValueError("Table 4 requires exactly the three selected constructions")
    if any(r["field"] != "25519" or r["allocation"] != "253/208" for r in results):
        raise ValueError("Table 4 requires Wei25519 253/208 schedules")
    shared = {(r["initializer_ccx"], r["point_additions"], r["peak_qubits"]) for r in results}
    if len(shared) != 1:
        raise ValueError("Table 4 caption requires matching initializer costs, addition counts and peaks")
    initial, additions, peak = shared.pop()
    caption = (f"Complete emitted Wei25519 scalar circuits. Every row uses one {initial:,}-CCX "
               f"initialization lookup and {additions} additions, and has a peak width of {peak:,} logical qubits. "
               "The signed and odd-only rows use zero-safe replay and canonical positive-coordinate restoration; "
               "the unsigned row uses the baseline arithmetic.")
    tex = [r"\begin{table}[htbp]", r"\centering", r"\caption{"+caption+"}",
           r"\label{tab:schedules}", r"\begin{tabularx}{\linewidth}{Xrr}", r"\toprule",
           r"Construction & CCX per addition & Total CCX\\", r"\midrule"]
    md = ["| Construction | CCX per addition | Total CCX |", "|---|---:|---:|"]
    for variant, label in SCHEDULES:
        r = by_variant[variant]
        tex.append(f"{label} & {r['addition_ccx']:,} & {r['total_ccx']:,}"+r"\\")
        md.append(f"| {label} | {r['addition_ccx']:,} | {r['total_ccx']:,} |")
    tex += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(tex)+"\n", "\n".join(md)+"\n"


def comparison_table(results):
    if len(results) != 1:
        raise ValueError("Table 6 requires exactly one measured Wei25519 construction")
    result, = results
    if (result["field"], result["variant"], result["allocation"]) != (
            "25519", "odd_torsion_add", "253/208"):
        raise ValueError("Table 6 requires the Wei25519 253/208 odd-only construction")
    if result["strict_clean"] != 6 or result["diagnostics"] != 6:
        raise ValueError("Table 6 requires six strict-clean diagnostics")
    if any(type(result[k]) is not int or result[k] <= 0 for k in ("total_ccx", "peak_qubits")):
        raise ValueError("Table 6 requires positive integer measured resources")
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Overall resource comparison with published \texttt{secp256k1} estimates.}",
           r"\label{tab:full-comparison}", r"\begin{tabularx}{\linewidth}{Xrr}", r"\toprule",
           r"Construction & Toffoli count & Logical qubits\\", r"\midrule"]
    md = ["| Construction | Toffoli count | Logical qubits |", "|---|---:|---:|"]
    for reference in LITERATURE_REFERENCES:
        label = reference["construction"].replace("secp256k1", r"\texttt{secp256k1}", 1)
        label += rf"~\cite{{{reference['citation_key']}}}"
        count = rf"$2^{{{reference['toffoli_log2']:.2f}}}$"
        tex.append(f"{label} & {count} & {reference['logical_qubits']:,}"+r"\\")
        md.append(f"| {reference['construction']} ({reference['citation_key']}) | {count} | {reference['logical_qubits']:,} |")
    count = rf"${result['total_ccx']:,}\approx 2^{{{math.log2(result['total_ccx']):.2f}}}$"
    tex.append(f"Wei25519, this work & {count} & {result['peak_qubits']:,}"+r"\\")
    md.append(f"| Wei25519, this work | {count} | {result['peak_qubits']:,} |")
    tex += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(tex)+"\n", "\n".join(md)+"\n"


def stream(process, handle, field):
    for line in process.stdout:
        handle.write(line)
        handle.flush()
        print(f"[{field}] {line.rstrip()}", flush=True)


def run(output=None, field=None, *, table4=False, table6=False):
    if table4 and table6:
        raise ValueError("--table4 and --table6 are mutually exclusive")
    if (table4 or table6) and field is not None:
        raise ValueError(f"Table {6 if table6 else 4} is fixed to Wei25519; --field applies only to signed comparisons")
    sys.path.insert(0, str(ROOT))
    from qsm.bench.calibration import source_digest
    mode = "table6" if table6 else "table4" if table4 else "signed"
    directory = benchmark_files.create_directory(ROOT, mode, output)
    outputs = benchmark_files.paths(directory, mode)
    naming_sha = benchmark_files.digest()
    source = source_digest()
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selected_jobs = (comparison_jobs(directory) if table6 else schedule_jobs(directory) if table4
                     else jobs(directory, (field,) if field else FIELDS))
    versions = dict(table6="fresh-paper-comparison-v1", table4="fresh-paper-schedules-v1",
                    signed="fresh-zero-safe-signed-v1")
    manifest = dict(version=versions[mode], table=mode, started=now(), complete=False,
                    measurement_mode="fresh full emitted circuits; no archived counts read",
                    source_digest=source, runner_sha256=runner_sha, output_naming_sha256=naming_sha,
                    jobs=selected_jobs, results=[])
    if table6:
        manifest["measurement_mode"] = ("fresh full emitted Wei25519 circuit only; no archived counts read; "
                                        "secp256k1 rows are fixed user-supplied literature quotations")
        manifest["reference_rows"] = [dict(reference,
            provenance="quoted user-supplied literature reference values",
            fresh_measurement=False, independently_verified=False) for reference in LITERATURE_REFERENCES]
    running = []
    try:
        if table6:
            print("Fresh Wei25519 Table 6: only the Wei25519 odd-only circuit is built. "
                  "External secp256k1 rows are fixed quoted user-supplied literature reference values, "
                  "not fresh measurements or independently verified. No archived counts are read.", flush=True)
        else:
            print("Fresh Wei25519 Table 4: all three constructions run in parallel. No archived counts are read." if table4 else
                  "Fresh zero-safe signed schedules only; fields run in parallel. No archived counts are read.", flush=True)
        for job in manifest["jobs"]:
            handle = (directory/(job["name"]+".log")).open("x")
            try:
                process = subprocess.Popen(job["command"], cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1,
                                           start_new_session=os.name == "posix")
            except BaseException:
                handle.close()
                raise
            thread = threading.Thread(target=stream, args=(process, handle, job["name"]), daemon=True)
            running.append((process, handle, thread, job))
            thread.start()
        for process, _, thread, job in running:
            job["exit_code"] = process.wait()
            thread.join()
            if job["exit_code"] not in (0, 1) or not Path(job["output"]).exists():
                raise RuntimeError(f"{job['name']} build failed; inspect its log")
            raw = Path(job["output"]).read_bytes()
            summary = verify(json.loads(raw), job["field"], source, variant=job["variant"])
            if job["exit_code"] != 0 and summary["strict_clean"] == summary["diagnostics"]:
                raise RuntimeError("worker failed despite an apparently clean output; inspect its log")
            job["output_sha256"] = hashlib.sha256(raw).hexdigest()
            manifest["results"].append(summary)
            if table4:
                print(f"Measured {dict(SCHEDULES)[job['variant']]}: {summary['addition_ccx']:,} CCX/addition; "
                      f"{summary['total_ccx']:,} CCX total (provisional until completion)", flush=True)
            elif table6:
                print(f"Measured Wei25519, this work: {summary['total_ccx']:,} CCX total; "
                      f"{summary['peak_qubits']:,} logical qubits (provisional until completion)", flush=True)
        if (source_digest() != source or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runner_sha
                or benchmark_files.digest() != naming_sha):
            raise RuntimeError("source, runner or output naming code changed while measuring")
        if table4 or table6:
            if any(r["strict_clean"] != r["diagnostics"] for r in manifest["results"]):
                raise RuntimeError(f"Table {6 if table6 else 4} diagnostics failed; raw measurements retained, no completed table emitted")
            tex, text = (comparison_table if table6 else schedule_table)(manifest["results"])
            outputs["tex"].write_text(tex)
        else:
            text = "| Field | Allocation | CCX per addition | Total CCX | Peak qubits | Strict diagnostics |\n|---|---|---:|---:|---:|---:|\n"
            for r in manifest["results"]:
                text += f"| {r['field']} | {r['allocation']} | {r['addition_ccx']:,} | {r['total_ccx']:,} | {r['peak_qubits']:,} | {r['strict_clean']}/{r['diagnostics']} |\n"
        outputs["md"].write_text(text)
        manifest["complete"] = True
        print(text, flush=True)
        print(f"Completed fresh {mode} measurements: {directory}", flush=True)
    except BaseException as exc:
        manifest["error"] = str(exc)
        raise
    finally:
        for process, handle, thread, _ in running:
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            thread.join(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            handle.close()
        manifest["finished"] = now()
        outputs["json"].write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return int(any(r["strict_clean"] != r["diagnostics"] for r in manifest["results"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--field", choices=FIELDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--table4", action="store_true", help="rebuild all three Wei25519 schedules instead of the signed-only comparison")
    mode.add_argument("--table6", action="store_true", help="rebuild only the Wei25519 odd-only row; secp256k1 literature rows are fixed quotations")
    args = parser.parse_args(argv)
    if (args.table4 or args.table6) and args.field is not None:
        parser.error(f"--field does not apply to Table {6 if args.table6 else 4}")
    try:
        return run(args.out, args.field, table4=args.table4, table6=args.table6)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"scalar benchmark: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
