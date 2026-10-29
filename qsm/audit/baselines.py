"""Cost against the curve-agnostic baselines, separated by CONTRACT.

A single "3x below Litinski" hides three different wins stacked on top of each other:
the pseudo-Mersenne structure, the lazy output contract, and the truncation. They are
not interchangeable -- only the first is curve-specific, and only the first two keep an
exact answer. This prints the ladder so each row states what it gave up to get there.

Litinski (arXiv:2306.08585) is quoted as 4n / 6n / 2n for modular add / sub / negate.
Those figures are ASSUMED here to be exact and canonical; if that is wrong, the
`like-for-like` block below is the thing that changes.
"""

from __future__ import annotations

import math

from ..bench.registry import all_versions, get_builder
from ..core.resources import count

LITINSKI = {"modadd": 4 * 255, "modsub": 6 * 255, "modneg": 2 * 255}

LADDER = [
    ("modadd", "field.modadd.v3", "canonical, exact"),
    ("modadd", "field.modadd_lazy_exact.v4", "lazy, exact"),
    ("modadd", "field.modadd_lazy.v4", "lazy, truncated + vented"),
    ("modsub", "field.modsub.v1", "lazy, exact (sandwich)"),
    ("modsub", "field.modsub_direct.v2", "canonical, untruncated (phase term)"),
    ("modsub", "field.modsub_direct_trunc.v2", "canonical, truncated + phase"),
    ("modneg", "field.modneg.v1", "exact"),
    ("modneg", "field.modneg_trunc.v1", "truncated"),
]

# the cheapest version of each op that is BOTH exact and canonical
LIKE_FOR_LIKE = {"modadd": "field.modadd.v3", "modsub": "field.modsub.v1",
                 "modneg": "field.modneg.v1"}


def _row(name):
    case = get_builder(name)()
    q = count(case.circuit)
    fa = case.meta.get("fail_analytic", 0.0)
    ph = case.meta.get("tolerates_phase", False)
    if not fa and not ph:
        err = "exact"
    else:
        err = (f"2^{math.log2(fa):.1f}" if fa else "") + (" +phase" if ph else "")
    return q, err.strip()


def run() -> int:
    import qsm.bench.__main__  # noqa: F401  -- ensure every layer is registered

    known = set(all_versions())
    print(f"{'op':7s} {'version':30s} {'contract':34s} {'Toff':>5s} "
          f"{'peak':>5s} {'vs base':>8s}  error")
    print("-" * 104)
    for op, name, contract in LADDER:
        if name not in known:
            print(f"{op:7s} {name:30s} NOT REGISTERED")
            continue
        q, err = _row(name)
        print(f"{op:7s} {name:30s} {contract:34s} {q.toffoli:5d} "
              f"{q.peak_qubits:5d} {LITINSKI[op] / q.toffoli:7.2f}x  {err}")

    print("\nlike-for-like (exact AND canonical on both sides):")
    for op, name in LIKE_FOR_LIKE.items():
        q, _ = _row(name)
        base = LITINSKI[op]
        print(f"  {op:7s} ours {q.toffoli:5d}  vs  baseline {base:5d}   "
              f"-> {base / q.toffoli:.2f}x")
    print("\nAnything above that ratio is bought with laziness or truncation, not with")
    print("the field's structure. Quote it with the contract attached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
