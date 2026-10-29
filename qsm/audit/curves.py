"""The same Layer-0 construction, instantiated over 25519 and over secp256k1.

This is the controlled experiment behind "which curve fails first". Comparing our numbers
to someone else's codebase confounds the field's shape with every other design choice
they made -- adder, scheduling, venting, peak strategy. Here the ONLY thing that changes
is `(U, F)` and the carry window each prime needs for the same accuracy, so whatever
difference appears is attributable to the field.

    25519      p = 2^255 - 19              F =         19   (5 bits)   -> L = 45
    secp256k1  p = 2^256 - 2^32 - 977      F = 2^32 + 977   (33 bits)  -> L = 73

`F` does triple duty -- the deficit, the reduction constant, and the width of the carry
window needed for a target accuracy -- so a 28-bit gap in `log2(F)` is a 28-bit gap in
the window, and the window is what a truncated reduction actually costs.

Both windows are chosen by the same rule, `L = ceil(40 + log2(F))`, so the two columns
are at EQUAL accuracy (~2^-40 unconditional / ~2^-41 controlled). Comparing 25519 at
L=45 against secp256k1 at L=73 is therefore like-for-like, not a handicap.
"""

from __future__ import annotations

import math
import random

from ..core.circuit import Circuit
from ..core.resources import count
from ..core.simulator import simulate
from ..field.add import gidney_add
from ..field.modarith import (
    modadd_lazy, moddouble, modneg, modsub, modsub_direct,
)

CURVES = {
    "25519":     dict(u=255, f=19),
    "secp256k1": dict(u=256, f=(1 << 32) + 977),
}
for _c in CURVES.values():
    _c["p"] = (1 << _c["u"]) - _c["f"]

# Per-operation error budgets. Ours is 2^-40. ecdsa.fail has shipped BOTH, using the
# same `window = budget + log2(F)` rule we do:
#
#   2^-40  trailmix_port/rfold_mbu.rs:42   RFOLD_WINDOW = 73 = 40 + 33  (Jun 2026)
#   2^-20  trailmix_ludicrous/arith.rs:452 LSBS = 20 + F_BITLEN = 53    (HEAD)
#
# So 73-vs-45 is the like-for-like pair, and their current HEAD runs ~20 bits looser
# than we do. Both rows are printed so a comparison always states which budget it is at.
BUDGETS = {40: "ours; ecdsa.fail's Jun-2026 RFOLD_WINDOW=73 is the same budget",
           20: "ecdsa.fail at HEAD (LSBS=53, MSBS=19)"}


def _windows(cfg, budget):
    """The equal-accuracy windows for a prime at a target budget."""
    return math.ceil(budget + math.log2(cfg["f"])), budget


def _build(name, u, p, L, k, exact):
    """One Layer-0 circuit, at whichever prime."""
    w, kk = (None, None) if exact else (L, k)
    c = Circuit()
    if name == "modadd":
        a, b = c.input("a", u), c.input("b", u)
        modadd_lazy(c, a, b, p, gidney_add, w, kk, vent=True)
    elif name == "modsub_sandwich":
        a, b = c.input("a", u), c.input("b", u)
        modsub(c, a, b, p, gidney_add, w, kk)
    elif name == "modsub_direct":
        a, b = c.input("a", u), c.input("b", u)
        modsub_direct(c, a, b, p, gidney_add, w, kk)
    elif name == "modneg":
        a = c.input("a", u)
        modneg(c, a, p, window=w)
    elif name == "moddouble":
        a = c.input("a", u)
        c.rebind("a", moddouble(c, a, p, w))
    else:
        raise KeyError(name)
    return c


OPS = ["modadd", "modsub_sandwich", "modsub_direct", "modneg", "moddouble"]


def _correctness_at_small_width() -> int:
    """The comparison is worthless if the construction is only correct for small F.

    Model both shapes at a width small enough to enumerate: a 5-bit-ish deficit and a
    deficit that is a large fraction of the register, mirroring 19 vs 2^32+977.
    """
    bad = 0
    for tag, (u, f) in (("small-F (25519-like)", (12, 19)),
                        ("large-F (secp-like)", (12, 1029))):
        p = (1 << u) - f
        if not f & 1:
            continue
        for name in OPS:
            c = _build(name, u, p, None, None, exact=True)
            rng = random.Random(4)
            for _ in range(300):
                vals = {n: rng.randrange(p) for n in c.inputs}
                out = simulate(c, dict(vals), rng=random.Random(9)).outputs
                if name == "modadd":
                    s = vals["a"] + vals["b"]
                    want = s - p if s >= (1 << u) else s
                elif name == "modsub_sandwich":
                    s = (p - vals["a"]) + vals["b"]
                    want = s - p if s >= (1 << u) else s
                elif name == "modsub_direct":
                    want = (vals["b"] - vals["a"]) % p
                elif name == "modneg":
                    want = p - vals["a"]
                else:
                    d = 2 * vals["a"]
                    want = d - p if d >= (1 << u) else d
                key = "b" if name.startswith("mods") or name == "modadd" else "a"
                if out[key] != want:
                    bad += 1
        print(f"  {tag:22s} {'OK' if not bad else 'FAIL'}")
    return bad


def run() -> int:
    bad = _correctness_at_small_width()
    if bad:
        print(f"  {bad} correctness failures -- comparison below is meaningless")
        return bad

    # --- exact: the window is irrelevant, so this isolates "does the prime matter?" ---
    print(f"\n[exact — no truncation, so F's width cannot show up]")
    print(f"{'operation':22s} {'25519':>10s} {'secp256k1':>12s} {'edge':>8s}")
    print("-" * 56)
    tot = [0, 0]
    for op in OPS:
        t = [count(_build(op, c["u"], c["p"], None, None, True)).toffoli
             for c in CURVES.values()]
        tot[0] += t[0]
        tot[1] += t[1]
        print(f"  {op:20s} {t[0]:10d} {t[1]:12d} {t[1] - t[0]:8d}")
    print(f"  {'SUM':20s} {tot[0]:10d} {tot[1]:12d} {tot[1] - tot[0]:8d}"
          f"   ({100.0 * (tot[1] - tot[0]) / tot[1]:+.1f}%)")

    # --- truncated, at each budget ------------------------------------------
    for budget, who in BUDGETS.items():
        w = {n: _windows(c, budget) for n, c in CURVES.items()}
        print(f"\n[truncated @ 2^-{budget} per op — {who}]")
        print(f"  windows: 25519 L={w['25519'][0]}, secp256k1 L={w['secp256k1'][0]}"
              f"  (gap {w['secp256k1'][0] - w['25519'][0]} = log2(F) difference)")
        print(f"{'operation':22s} {'25519':>10s} {'secp256k1':>12s} {'edge':>8s} {'%':>7s}")
        print("-" * 64)
        tot = [0, 0]
        for op in OPS:
            t = []
            for n, c in CURVES.items():
                L, k = w[n]
                t.append(count(_build(op, c["u"], c["p"], L, k, False)).toffoli)
            tot[0] += t[0]
            tot[1] += t[1]
            print(f"  {op:20s} {t[0]:10d} {t[1]:12d} {t[1] - t[0]:8d} "
                  f"{100.0 * (t[1] - t[0]) / t[1]:6.1f}%")
        print(f"  {'SUM':20s} {tot[0]:10d} {tot[1]:12d} {tot[1] - tot[0]:8d} "
              f"{100.0 * (tot[1] - tot[0]) / tot[1]:6.1f}%")

    print("""
Reading this:

  * EXACT: the two primes cost the same. Pseudo-Mersenne structure on its own buys
    nothing -- there is no truncation for a small F to make cheap.
  * TRUNCATED: 25519 wins a FLAT ~28 Toffoli per truncated reduction, at either budget,
    because 28 = log2(2^32+977) - log2(19) is the extra carry window secp256k1 needs for
    the same accuracy. It is a constant, not a factor.
  * So the edge is large in % terms exactly where the reduction IS the circuit
    (moddouble, modneg: ~38%) and small where a full-width register addition dominates
    (modadd, modsub: ~8%).
  * The budget row matters for any comparison against published ecdsa.fail numbers:
    their shipped constants target ~2^-20 per operation and ours target 2^-40, so our
    headline figures are ~20 bits more conservative than theirs and are NOT
    directly comparable.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
