"""The whole stack, over 25519 and over secp256k1. The controlled experiment.

`audit curves` runs this comparison at Layer 0, where a reduction IS the circuit and the
field's shape shows up at its most flattering: ~28 Toffoli a call, ~38% on a `moddouble`.
This runs it at Layer 3, where the answer turns out to be very different, and Layer 3 is
the one the paper's claim is made at.

WHY IT NEEDS SUBPROCESSES. Every module binds `U`/`F`/`PRIME` with `from ..params import`
at import time, so a field is chosen once per process and cannot be swapped underneath a
registered circuit. `curves` avoids that by calling the builders directly with explicit
`(u, f, p)`; a registered version has no such handle, so this forks instead.

The fixed-iteration comparison controls the GCD loop count, not the total failure rate.
The Gaussian iteration-tail estimate depends on field width even at an identical iteration
count. Both estimated errors are reported; these are conditional models for the experimental
circuits, not guarantees for a future scalar-multiplication input distribution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

#: Registered versions worth a row, coarse to fine. Missing ones are skipped, so this
#: list can name circuits that do not exist yet without breaking the audit.
VERSIONS = [
    "field.modsub_direct_trunc.v2",
    "field.modadd_lazy.v4",
    "mul.modmul_const_f.v1",
    "mul.modmul_acc_trunc.v1",
    "mul.modsqr_acc_trunc.v1",
    "mul.modsqrsub_inplace_trunc.v1",
    "mul.modsqrsub_inplace_trunc_flat.v1",
    "mul.modsqrsub_inplace_trunc_flat.v2",
    "inv.modmul_inplace.v1",
    "inv.moddiv_inplace.v1",
    "point.add_const_affine_inplace.v1",
    "point.add_const_affine_inplace_ctrl.v1",
    "point.add_const_affine_inplace.v2",
    "point.add_const_affine_inplace_ctrl.v2",
]

FIELDS = ("25519", "secp256k1")

_COUNT = r"""
import json, sys
import qsm.field, qsm.mul, qsm.inversion, qsm.point
from qsm.bench.registry import get_builder, all_versions
from qsm.core.resources import count
from qsm import params as P
known, out = set(all_versions()), {}
for v in sys.argv[1:]:
    if v in known:
        r = count(get_builder(v)().circuit)
        out[v] = [r.toffoli, r.peak_qubits]
print("JSON" + json.dumps(
    {"field": P.FIELD, "u": P.U, "f_bits": P.F.bit_length(), "L": P.CARRY_WINDOW,
     "ripple": P.SQUARE_RIPPLE, "iters": P.gcd_j2_iterations(), "rows": out}))
"""

# The matched-budget rebuild. Not a registered version -- it is `add_const_affine_inplace`
# with the iteration count forced, which is exactly the one knob the block rounding moves.
_MATCHED = r"""
import json, sys
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm import params as P
from qsm.inversion.config import InversionConfig
from qsm.point import affine

iters = int(sys.argv[1])
legacy = sys.argv[2] == "legacy"
config = InversionConfig.tuned(P.U, P.PRIME, iters=iters)
case = affine._affine_case(False, config, legacy_square_ripple=legacy)
r = count(case.circuit)
print("JSON" + json.dumps({"toffoli": r.toffoli, "peak": r.peak_qubits,
    "fail_analytic": case.meta["fail_analytic"], "square_ripple": case.meta["square_ripple"],
    "iteration_tail": P.gcd_j2_iteration_fail_prob(P.U, iters)}))
"""


def _fork(field: str, src: str, *args: str) -> dict:
    """Run `src` in a child process with QSM_FIELD=field, return its JSON line."""
    p = subprocess.run([sys.executable, "-c", src, *args], capture_output=True,
                       text=True, env=dict(os.environ, QSM_FIELD=field))
    if p.returncode:
        raise RuntimeError(f"{field}: child failed\n{p.stderr[-2000:]}")
    for line in p.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    raise RuntimeError(f"{field}: child produced no result\n{p.stdout[-2000:]}")


def run() -> int:
    meta = {f: _fork(f, _COUNT, *VERSIONS) for f in FIELDS}
    a, b = (meta[f] for f in FIELDS)

    print(f"\n  {'field':11s} {'U':>4s} {'log2 F':>7s} {'L':>4s} {'J2 iters':>9s}")
    for m in (a, b):
        print(f"  {m['field']:11s} {m['u']:4d} {m['f_bits']:7d} {m['L']:4d} "
              f"{m['iters']:9d}")
    print(f"\n  L is derived from ERROR_BUDGET: the same per-operation truncation target,"
          f"\n  with a {b['L'] - a['L']}-bit window gap. Field width and GCD error also differ.")

    print(f"\n[as designed — each field at its own block-rounded GCD budget]\n")
    print(f"  {'version':40s} {'25519':>12s} {'peak':>6s} {'secp256k1':>12s} "
          f"{'peak':>6s} {'ratio':>7s} {'dq':>5s}")
    print("  " + "-" * 94)
    for v in VERSIONS:
        x, y = a["rows"].get(v), b["rows"].get(v)
        if not (x and y):
            continue
        print(f"  {v:40s} {x[0]:12,} {x[1]:6d} {y[0]:12,} {y[1]:6d} "
              f"{y[0] / x[0]:7.4f} {y[1] - x[1]:+5d}")

    # --- matched budget ----------------------------------------------------
    iters = min(m["iters"] for m in meta.values())
    print(f"\n[fixed iteration count — both fields at {iters} jump-2 iterations; "
          f"total accuracy is not matched]\n")
    for label, mode in (("point, legacy ripple=L (v1)", "legacy"),
                        ("point, ripple=SQUARE_RIPPLE (v2)", "current")):
        mt = {f: _fork(f, _MATCHED, str(iters), mode) for f in FIELDS}
        ma, mb = (mt[f] for f in FIELDS)
        print(f"  {label:40s} {ma['toffoli']:12,} "
              f"{ma['peak']:6d} {mb['toffoli']:12,} {mb['peak']:6d} "
              f"{mb['toffoli'] / ma['toffoli']:7.4f} {mb['peak'] - ma['peak']:+5d}")
        for field, row in mt.items():
            print(f"    {field}: ripple={row['square_ripple']}; iteration tail estimate="
                  f"{row['iteration_tail']:.6e}; point P(fail) estimate={row['fail_analytic']:.6e}")
    delta = mb["toffoli"] - ma["toffoli"]
    pct = 100.0 * (mb["toffoli"] / ma["toffoli"] - 1.0)
    print(f"""
Reading this:

  The experimental point circuit costs {delta:,} more Toffoli over secp256k1
  ({pct:.1f}%) at a fixed iteration count. This is not an equal-total-error comparison.
  The one-bit field-width change also affects field-operation costs; the carry-window
  gap alone is not the complete subtraction difference.
  These counts describe the named tuned circuits. Their conditional error models
  must not be transferred to correlated on-curve window trajectories without analysis.
  Resource comparisons with external circuits additionally require matching domains,
  error targets and lookup accounting.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
