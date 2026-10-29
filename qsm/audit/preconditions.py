"""What ACTUALLY happens when a Layer-0 value precondition is violated.

Several circuits carry a precondition a quantum circuit cannot check -- "requires a < p"
-- because checking it is the full-width comparison the design exists to avoid. A bare
`Requires a < p` in a docstring is folklore: it says a rule exists without saying what
breaks, so a caller cannot weigh the risk and a scheduler cannot budget it.

This measures the consequence and CLASSIFIES it, because the classes are not equally bad:

    EXACT       violating the precondition changes nothing
    CONGRUENT   right value mod p, wrong representative -- a final `% p` repairs it.
                This is ordinary laziness and costs nothing but a wider contract.
    WRONG       not congruent -- NO downstream reduction can repair it. Corruption.
    WRAPS       does not fit the register at all.
    DIRTY       a freed ancilla comes back |1>: entangled garbage, not a wrong value.
                Strictly the worst class, and the one a value test would never notice
                because the *returned value* can still be right.

`moddouble` got this treatment first (predicate `moddouble_overflows`, a test that feeds
it lazy inputs, and a documented scheduler rule). Everything with an unchecked value
precondition should get the same, so this enumerates them all in one place.
"""

from __future__ import annotations

import random

from ..core.circuit import Circuit, DirtyAncillaError
from ..core.simulator import simulate
from ..field.add import gidney_add
from ..field.modarith import (
    modadd, modadd_cmp, modadd_lazy, moddouble, modneg, modsub, modsub_direct,
)

WIDTHS = [(8, 3, 253), (8, 5, 251), (10, 5, 1019), (11, 19, 2029)]


def _cases(u, p):
    """(label, builder, output register, expected-value-mod-p, which operands are lazy)"""
    def two(fn, **kw):
        def build():
            c = Circuit()
            a, b = c.input("a", u), c.input("b", u)
            fn(c, a, b, p, gidney_add, **kw)
            return c
        return build

    def one(fn, relabel=False, **kw):
        def build():
            c = Circuit()
            a = c.input("a", u)
            r = fn(c, a, p, **kw)
            if relabel and r is not None:
                c.rebind("a", r)
            return c
        return build

    return [
        ("modadd.v1/v2 (canonical)", two(modadd), "b", lambda a, b: (a + b) % p, "ab"),
        ("modadd_cmp.v3 (canonical)", two(modadd_cmp), "b", lambda a, b: (a + b) % p, "ab"),
        ("modadd_lazy.v4",
         two(modadd_lazy, carry_window=None, cmp_bits=None, vent=True),
         "b", lambda a, b: (a + b) % p, "ab"),
        ("modsub.v1 (sandwich)",
         two(modsub, carry_window=None, cmp_bits=None), "b",
         lambda a, b: (b - a) % p, "ab"),
        ("modsub_direct.v2",
         two(modsub_direct, carry_window=None, cmp_bits=None), "b",
         lambda a, b: (b - a) % p, "ab"),
        ("modneg.v1", one(modneg, window=None), "a", lambda a, b: (-a) % p, "a"),
        ("moddouble.v1", one(moddouble, relabel=True, window=None), "a",
         lambda a, b: (2 * a) % p, "a"),
    ]


def _classify(u, p, build, out, want, lazy_on, trials=260, seed=1, how="all"):
    """Bucket the outcome when operands in `lazy_on` are pushed into [p, 2^u).

    `how` selects WHICH operands are violated, and the distinction is the whole point:

        "one"  exactly one operand lazy -- frequency ~F/2^U = 2^-250.8 per call
        "all"  every eligible operand lazy -- frequency ~(F/2^U)^2 = 2^-501.6

    A circuit that survives "one" and breaks on "all" is fine in practice; a circuit
    that breaks on "one" is not. Reporting them together hides a 250-bit difference.
    """
    circ = build()
    rng = random.Random(seed)
    n = {"EXACT": 0, "CONGRUENT": 0, "WRONG": 0, "WRAPS": 0, "DIRTY": 0}
    eligible = [x for x in circ.inputs if x in lazy_on]
    if how == "one" and not eligible:
        return n
    for t in range(trials):
        vals = {}
        violate = set(eligible) if how == "all" else {eligible[t % len(eligible)]}
        for name in circ.inputs:
            if name in violate:
                vals[name] = rng.randrange(p, 1 << u)      # the violating band
            else:
                vals[name] = rng.randrange(p)
        try:
            got = simulate(circ, dict(vals), rng=random.Random(7)).outputs[out]
        except DirtyAncillaError:
            n["DIRTY"] += 1
            continue
        expect = want(vals.get("a", 0), vals.get("b", 0))
        if got >= (1 << u):
            n["WRAPS"] += 1
        elif got % p != expect:
            n["WRONG"] += 1
        elif got < p:
            n["EXACT"] += 1
        else:
            n["CONGRUENT"] += 1
    return n


def run() -> int:
    print("Behaviour when a value precondition is VIOLATED (an operand in [p, 2^u)).\n")
    print("  CONGRUENT = repairable by a final `% p`   WRONG = not repairable\n")
    def verdict(n):
        if n["DIRTY"]:
            return "DIRTIES ANCILLA"
        if n["WRONG"] or n["WRAPS"]:
            return "CORRUPTS"
        if n["CONGRUENT"]:
            return "lazy-tolerant"
        return "unaffected"

    worst: dict[str, dict] = {}
    for how, title, freq in (
        ("one", "ONE operand lazy", "~F/2^U = 2^-250.8 per call"),
        ("all", "ALL operands lazy", "~(F/2^U)^2 = 2^-501.6 per call"),
    ):
        print(f"[{title} -- frequency {freq}]")
        print(f"{'circuit':28s} {'exact':>6} {'congr':>6} {'WRONG':>6} "
              f"{'wraps':>6} {'DIRTY':>6}   verdict")
        print("-" * 90)
        for label, build, out, want, lazy_on in _cases(*WIDTHS[-1][::2]):
            agree = set()
            for u, f, p in WIDTHS:
                for lab2, b2, o2, w2, lz2 in _cases(u, p):
                    if lab2 == label:
                        n = _classify(u, p, b2, o2, w2, lz2, how=how)
                        agree.add(verdict(n))
                        if u == WIDTHS[-1][0]:
                            shown = n
            v = "DIRTIES ANCILLA" if "DIRTIES ANCILLA" in agree else (
                "CORRUPTS" if "CORRUPTS" in agree else
                ("lazy-tolerant" if "lazy-tolerant" in agree else "unaffected"))
            worst.setdefault(label, {})[how] = v
            print(f"{label:28s} {shown['EXACT']:6d} {shown['CONGRUENT']:6d} "
                  f"{shown['WRONG']:6d} {shown['WRAPS']:6d} {shown['DIRTY']:6d}   {v}")
        print()

    need = [k for k, v in worst.items() if v["one"] in ("CORRUPTS", "DIRTIES ANCILLA")]
    ok1 = [k for k, v in worst.items()
           if v["one"] not in ("CORRUPTS", "DIRTIES ANCILLA")
           and v["all"] in ("CORRUPTS", "DIRTIES ANCILLA")]

    print("NEEDS THE `moddouble` TREATMENT -- breaks with a SINGLE lazy operand, which is")
    print("the case a lazy-throughout schedule actually produces (2^-250.8 per call):")
    for k in need:
        print(f"  - {k:28s} {worst[k]['one']}")
    print()
    print("Closed for one lazy operand; only breaks when EVERY operand is lazy at once")
    print("(2^-501.6 -- below any budget that matters). Documented, not guarded:")
    for k in ok1:
        print(f"  - {k:28s} ({worst[k]['all']} only when all-lazy)")
    print()
    print("The treatment is: an explicit predicate in qsm/params.py, a test that feeds")
    print("the violating band, and a stated scheduler rule.")
    print()
    print("CAUTION on the 'unaffected' rows. This samples the violating band for the")
    print("named operands and leaves the others uniform, so it finds hazards that need")
    print("ONE rare condition and can miss ones needing a CONJUNCTION. modsub_direct")
    print("reads unaffected here yet does corrupt when a - b > p -- f(f-1)/2 pairs,")
    print("2^-502.6. Exhaustive small-width tests, not this sweep, settle those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
