"""Measure every truncation rate against the formula that is quoted for it.

The distinction the whole table turns on:

    UNCONTROLLED  +c always runs           ->  P = c / 2^L
    CONTROLLED    +c runs on a flag        ->  P = P(flag) * c / 2^L

`modadd_lazy` and `modsub_direct` add F only on the reduction branch, and that branch
is taken with probability -> 1/2, so their true rate is F/2^(L+1) -- half what a bare
`carry_fail_prob` says. `modneg` and the standalone constant adders are unconditional
and keep the full c/2^L. `moddouble` is controlled AND its low limb is always even, so
it gets a second factor and lands at floor(F/2)/2^L.

Getting this wrong is safe (it over-states the error) but it is still wrong, and it is
the kind of number a referee recomputes.
"""

from __future__ import annotations

import random

from ..core.circuit import Circuit
from ..core.simulator import simulate
from ..field.add import gidney_add
from ..field.add_chunked import (
    gidney_add_ctrl_chunked, gidney_add_ctrl_chunked_threaded,
)
from ..field.constadd import add_const_inplace
from ..field.modarith import (
    modadd_lazy, moddouble, modhalve, modneg, modsub_direct,
)
from ..params import (
    F, carry_fail_prob, chunked_add_boundary_fail_prob, controlled_carry_fail_prob,
    moddouble_carry_fail_prob, modhalve_carry_fail_prob,
)

# (u, f, p) triples; f = 19 mirrors the real deficit.
WIDTHS = [(11, 19, 2029), (12, 19, 4077), (13, 19, 8173), (14, 19, 16365)]


def _rate(build_exact, build_trunc, u, p, samples, seed=5):
    """P(truncated output != exact output) over uniform canonical operands."""
    ex, tr = build_exact(u, p), build_trunc(u, p)
    rng = random.Random(seed)
    bad = 0
    for _ in range(samples):
        vals = {name: rng.randrange(p) for name in ex.inputs}
        a = simulate(ex, vals, rng=random.Random(11)).outputs
        b = simulate(tr, dict(vals), rng=random.Random(11)).outputs
        bad += any(a[k] != b[k] for k in a)
    return bad / samples


# --- the constructions, each as (exact builder, truncated builder) ----------
def _modadd(L):
    def mk(w):
        def f(u, p):
            c = Circuit()
            a, b = c.input("a", u), c.input("b", u)
            modadd_lazy(c, a, b, p, gidney_add, w, None, vent=True)
            return c
        return f
    return mk(None), mk(L)


def _modsub(L):
    def mk(w):
        def f(u, p):
            c = Circuit()
            a, b = c.input("a", u), c.input("b", u)
            modsub_direct(c, a, b, p, gidney_add, w, None)
            return c
        return f
    return mk(None), mk(L)


def _modneg(L):
    def mk(w):
        def f(u, p):
            c = Circuit()
            a = c.input("a", u)
            modneg(c, a, p, window=w)
            return c
        return f
    return mk(None), mk(L)


def _moddouble(L):
    def mk(w):
        def f(u, p):
            c = Circuit()
            a = c.input("a", u)
            c.rebind("a", moddouble(c, a, p, w))
            return c
        return f
    return mk(None), mk(L)


def _modhalve(L):
    def mk(w):
        def f(u, p):
            c = Circuit()
            a = c.input("a", u)
            c.rebind("a", modhalve(c, a, p, w))
            return c
        return f
    return mk(None), mk(L)


def _addconst(L):
    def mk(w):
        def f(u, p):
            c = Circuit()
            b = c.input("b", u)
            add_const_inplace(c, b, F, window=w)
            return c
        return f
    return mk(None), mk(L)


CASES = [
    # label                    builders     declared formula                 controlled?
    ("modadd_lazy   (+F on ovf)",  _modadd,
     lambda L, f, u, p: controlled_carry_fail_prob(L, f), True),
    ("modsub_direct (+F on borrow)", _modsub,
     lambda L, f, u, p: controlled_carry_fail_prob(L, f), True),
    ("moddouble     (+F on msb)", _moddouble,
     lambda L, f, u, p: moddouble_carry_fail_prob(L, f, u, p), True),
    ("modhalve      (-F on parity)", _modhalve,
     lambda L, f, u, p: modhalve_carry_fail_prob(L, f, u, p), True),
    ("modneg        (+F-1 always)", _modneg,
     lambda L, f, u, p: carry_fail_prob(L, f - 1), False),
    ("addconst_19   (+F always)", _addconst,
     lambda L, f, u, p: carry_fail_prob(L, f), False),
]


# --- the chunked adders' boundary vent -------------------------------------
# A different shape of claim, and it needs its own measurement: everything above fails
# by returning a WRONG VALUE, so `_rate` finds it by comparing outputs. The chunked
# boundary vent fails by leaking a PHASE -- the value is always right -- so an output
# comparison is blind to it and would report a flat 0.00 for a rate that is really 2^-22.
# See `add_chunked`'s module docstring for why the boundary is vented at all.

#: (n, chunk, k). Widths small enough to simulate, cut so that the last block is partial
#: at least once, since a boundary the operand does not reach is the cheap case.
BOUNDARY_SHAPES = [(8, 4, 2), (12, 4, 3), (12, 6, 3), (9, 3, 2)]


def _boundary_phase_rate(add_fn, n, chunk, k, samples, seed=0x5EED):
    """P(the vent leaks a phase) over uniform operands, with the control ON.

    Control off is the other half of the claim and is asserted in the tests rather than
    sampled here: with ctrl = 0 the rate must be exactly 0, not merely small.
    """
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    a, b = c.input("a", n), c.input("b", n)
    add_fn(c, ctrl[0], a, b, chunk=chunk, cmp_bits=k)
    rng = random.Random(seed)
    phased = 0
    for _ in range(samples):
        av, bv = rng.getrandbits(n), rng.getrandbits(n)
        r = simulate(c, {"ctrl": 1, "a": av, "b": bv}, rng=rng)
        assert r.outputs["b"] == (av + bv) % (1 << n), "vent corrupted a value"
        phased += int(r.phase != 0)
    return phased / samples


def run_boundary(samples: int = 4000) -> int:
    """Measured phase rate vs `chunked_add_boundary_fail_prob`. Returns mismatches."""
    print("chunked-adder boundary vent: measured PHASE rate vs the declared bound")
    print(f"({samples} samples per cell, control on; the bound is an upper bound, so")
    print(" the ratio should sit at or just under 1.00, never above)\n")
    print(f"{'adder':28s} {'n':>4} {'chunk':>6} {'k':>3} {'measured':>10} "
          f"{'bound':>10} {'ratio':>7}")
    print("-" * 74)
    bad = 0
    for label, fn in [("gidney_add_ctrl_chunked", gidney_add_ctrl_chunked),
                      ("  ...  _threaded", gidney_add_ctrl_chunked_threaded)]:
        for n, chunk, k in BOUNDARY_SHAPES:
            m = _boundary_phase_rate(fn, n, chunk, k, samples)
            d = chunked_add_boundary_fail_prob(n, chunk, k)
            ratio = m / d if d else float("nan")
            flag = "" if 0.15 <= ratio <= 1.05 else "   <-- MISMATCH"
            if flag:
                bad += 1
            print(f"{label:28s} {n:4d} {chunk:6d} {k:3d} {m:10.5f} {d:10.5f} "
                  f"{ratio:7.2f}{flag}")
        print()
    print("Threading the control changes the ancilla count, not the approximation, so")
    print("both rows are held to the same bound. A ratio near 0 would mean the vent is")
    print("not firing at all -- which passes a one-sided check while having lost the")
    print("trade the construction exists to make.")
    return bad


def run(samples: int = 6000, windows=(6, 7)) -> int:
    """Print measured-vs-declared for each construction. Returns count of mismatches."""
    print("truncation rates: measured vs the formula quoted for that circuit")
    print(f"(f = {F}, {samples} samples per cell, ratio should be 1.00)\n")
    print(f"{'construction':32s} {'u':>3} {'L':>3} {'measured':>10} "
          f"{'declared':>10} {'ratio':>7}  {'ctrl':>5}")
    print("-" * 82)
    bad = 0
    for label, mk, formula, controlled in CASES:
        for u, f, p in WIDTHS[:2]:
            for L in windows:
                ex, tr = mk(L)
                m = _rate(ex, tr, u, p, samples)
                d = formula(L, f, u, p)
                ratio = m / d if d else float("nan")
                flag = "" if 0.85 <= ratio <= 1.18 else "   <-- MISMATCH"
                if flag:
                    bad += 1
                print(f"{label:32s} {u:3d} {L:3d} {m:10.5f} {d:10.5f} "
                      f"{ratio:7.2f}  {str(controlled):>5}{flag}")
        print()
    print("A controlled +c fires only on the reduction branch, so its rate carries an")
    print("extra P(flag) -> 1/2 that an unconditional +c does not.")
    print()
    bad += run_boundary()
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
