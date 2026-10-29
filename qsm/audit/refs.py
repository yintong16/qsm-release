"""Exhaustive correctness against references written from the DEFINITIONS.

Deliberately does not import qsm's own `*_ref` helpers. Checking a circuit against a
reference that ships beside it proves they agree, not that either is right; if the
reference encodes the same misunderstanding as the circuit, both pass. These references
are written from the arithmetic definition alone.

Also drives EVERY measurement outcome rather than a sampled one, so a phase leak that
only appears on one branch cannot hide.
"""

from __future__ import annotations

import itertools

from ..core import ops as O
from ..core.circuit import Circuit
from ..core.simulator import simulate
from ..field.add import cdkm_add, gidney_add
from ..field.compare import carry_out, carry_out_const, cmp_ge_const, cmp_lt
from ..field.constadd import add_const_inplace, addsub_p, sub_const_inplace
from ..field.modarith import (
    modadd, modadd_cmp, modadd_lazy, moddouble, moddouble_inplace,
    moddouble_lazy_wide, modneg, modsub, modsub_direct,
)
from ..field.sub import sub

SMALL = [(6, 3, 61), (8, 3, 253), (8, 5, 251), (10, 5, 1019)]


def _outcomes(circ, inputs, cap=10):
    """Simulate under every measurement outcome when there are few enough."""
    k = sum(1 for op in circ.ops if op.kind == O.MEASURE)
    if k == 0:
        return [simulate(circ, inputs)]
    if k > cap:
        import random
        return [simulate(circ, inputs, rng=random.Random(s)) for s in range(6)]
    return [simulate(circ, inputs, measurements=list(m))
            for m in itertools.product((0, 1), repeat=k)]


def _check(fails, label, cond, detail=""):
    if not cond:
        fails.append(f"{label}: {detail}")


def _primitives(fails):
    for add_fn, tag in ((gidney_add, "gidney"), (cdkm_add, "cdkm")):
        for n in (2, 3, 4):
            for na in range(1, n + 1):
                c = Circuit(); a = c.input("a", na); b = c.input("b", n)
                add_fn(c, a, b)
                d = Circuit(); a2 = d.input("a", na); b2 = d.input("b", n)
                sub(d, a2, b2, add_fn)
                for av in range(1 << na):
                    for bv in range(1 << n):
                        for r in _outcomes(c, {"a": av, "b": bv}):
                            _check(fails, f"{tag}_add n={n}",
                                   r.outputs["b"] == (av + bv) % (1 << n)
                                   and r.outputs["a"] == av and r.phase == 0,
                                   f"a={av} b={bv}")
                        for r in _outcomes(d, {"a": av, "b": bv}):
                            _check(fails, f"{tag}_sub n={n}",
                                   r.outputs["b"] == (bv - av) % (1 << n)
                                   and r.phase == 0, f"a={av} b={bv}")


def _const_adders(fails):
    for n in (3, 4):
        for cst in range(1 << n):
            for win in [None] + list(range(1, n + 1)):
                if win is not None and cst >= (1 << win):
                    continue
                for fn, sign, tag in ((add_const_inplace, +1, "addci"),
                                      (sub_const_inplace, -1, "subci")):
                    for ctrl_on in (False, True):
                        c = Circuit()
                        ctl = c.input("ctl", 1) if ctrl_on else None
                        b = c.input("b", n)
                        fn(c, b, cst, ctrl=(ctl[0] if ctrl_on else None), window=win)
                        for cv in ((0, 1) if ctrl_on else (None,)):
                            for bv in range(1 << n):
                                ins = {"b": bv} | ({"ctl": cv} if ctrl_on else {})
                                fire = (cv == 1) if ctrl_on else True
                                if not fire:
                                    want = bv
                                elif win is None or win >= n:
                                    want = (bv + sign * cst) % (1 << n)
                                else:
                                    lo = (bv % (1 << win) + sign * cst) % (1 << win)
                                    want = bv - bv % (1 << win) + lo
                                for r in _outcomes(c, ins):
                                    _check(fails, f"{tag} n={n} c={cst} w={win}",
                                           r.outputs["b"] == want and r.phase == 0,
                                           f"b={bv} ctl={cv}")
    for u, f in ((3, 3), (4, 5)):
        p = (1 << u) - f
        for adding in (False, True):
            for inplace in (True, False):
                c = Circuit(); B = c.input("B", u + 1)
                addsub_p(c, B, u, f, adding=adding, inplace=inplace)
                for bv in range(1 << (u + 1)):
                    want = (bv + (p if adding else -p)) % (1 << (u + 1))
                    for r in _outcomes(c, {"B": bv}):
                        _check(fails, f"addsub_p u={u} add={adding}",
                               r.outputs["B"] == want and r.phase == 0, f"B={bv}")


def _comparators(fails):
    for n in (3, 4):
        for cst in range(1 << n):
            c = Circuit(); B = c.input("B", n); fl = c.input("f", 1)
            carry_out_const(c, B, cst, fl[0])
            d = Circuit(); B2 = d.input("B", n); fl2 = d.input("f", 1)
            cmp_ge_const(d, B2, cst, fl2[0], k=None)
            for bv in range(1 << n):
                for r in _outcomes(c, {"B": bv, "f": 0}):
                    _check(fails, f"carry_out_const n={n}",
                           r.outputs["f"] == (bv + cst >= (1 << n))
                           and r.outputs["B"] == bv and r.phase == 0, f"B={bv}")
                for r in _outcomes(d, {"B": bv, "f": 0}):
                    _check(fails, f"cmp_ge_const n={n}",
                           r.outputs["f"] == (bv >= cst) and r.phase == 0, f"B={bv}")
        for na in range(1, n + 1):
            c = Circuit(); a = c.input("a", na); B = c.input("B", n); fl = c.input("f", 1)
            carry_out(c, a, B, fl[0])
            d = Circuit(); a2 = d.input("a", na); B2 = d.input("B", n); f2 = d.input("f", 1)
            cmp_lt(d, a2, B2, f2[0], k=None)
            for av in range(1 << na):
                for bv in range(1 << n):
                    for r in _outcomes(c, {"a": av, "B": bv, "f": 0}):
                        _check(fails, f"carry_out n={n}",
                               r.outputs["f"] == (av + bv >= (1 << n))
                               and r.phase == 0, f"a={av} B={bv}")
                    for r in _outcomes(d, {"a": av, "B": bv, "f": 0}):
                        _check(fails, f"cmp_lt n={n}",
                               r.outputs["f"] == (bv < av) and r.phase == 0,
                               f"a={av} B={bv}")


def _modular(fails):
    def lazy(a, b, p, u):
        s = a + b
        return s - p if s >= (1 << u) else s

    for u, f, p in SMALL:
        step = max(1, p // 25)
        for ipc in (False, True):
            c = Circuit(); a = c.input("a", u); b = c.input("b", u)
            modadd(c, a, b, p, gidney_add, inplace_const=ipc)
            for av in range(0, p, step):
                for bv in range(0, p, step):
                    for r in _outcomes(c, {"a": av, "b": bv}):
                        _check(fails, f"modadd ipc={ipc} p={p}",
                               r.outputs["b"] == (av + bv) % p and r.phase == 0,
                               f"a={av} b={bv}")
        c = Circuit(); a = c.input("a", u); b = c.input("b", u)
        modadd_cmp(c, a, b, p, gidney_add)
        d = Circuit(); a2 = d.input("a", u); b2 = d.input("b", u)
        modadd_lazy(d, a2, b2, p, gidney_add, None, None, vent=True)
        e = Circuit(); a3 = e.input("a", u); b3 = e.input("b", u)
        modsub(e, a3, b3, p, gidney_add, None, None)
        g = Circuit(); a4 = g.input("a", u); b4 = g.input("b", u)
        modsub_direct(g, a4, b4, p, gidney_add, None, None)
        for av in range(0, p, step):
            for bv in range(0, p, step):
                for r in _outcomes(c, {"a": av, "b": bv}):
                    _check(fails, f"modadd_cmp p={p}",
                           r.outputs["b"] == (av + bv) % p and r.phase == 0, "")
                for r in _outcomes(d, {"a": av, "b": bv}):
                    _check(fails, f"modadd_lazy p={p}",
                           r.outputs["b"] == lazy(av, bv, p, u) and r.phase == 0, "")
                for r in _outcomes(e, {"a": av, "b": bv}):
                    _check(fails, f"modsub_sandwich p={p}",
                           r.outputs["b"] == lazy(p - av, bv, p, u) and r.phase == 0, "")
                for r in _outcomes(g, {"a": av, "b": bv}):
                    _check(fails, f"modsub_direct p={p}",
                           r.outputs["b"] == (bv - av) % p, "")

        c = Circuit(); a = c.input("a", u)
        modneg(c, a, p, window=None)
        d = Circuit(); a2 = d.input("a", u); d.rebind("a", moddouble(d, a2, p, None))
        e = Circuit(); a3 = e.input("a", u); moddouble_inplace(e, a3, p, None)
        g = Circuit(); a4 = g.input("a", u)
        g.rebind("a", moddouble_lazy_wide(g, a4, p))
        for av in range(p):
            for r in _outcomes(c, {"a": av}):
                _check(fails, f"modneg p={p}", r.outputs["a"] == p - av
                       and r.phase == 0, f"a={av}")
            dbl = 2 * av - p if 2 * av >= (1 << u) else 2 * av
            for r in _outcomes(d, {"a": av}):
                _check(fails, f"moddouble p={p}", r.outputs["a"] == dbl
                       and r.phase == 0, f"a={av}")
            for r in _outcomes(e, {"a": av}):
                _check(fails, f"moddouble_inplace p={p}", r.outputs["a"] == dbl
                       and r.phase == 0, f"a={av}")
        for av in range(1 << u):
            want = 2 * av - p if av >= (1 << (u - 1)) else 2 * av
            for r in _outcomes(g, {"a": av}):
                _check(fails, f"lazy_wide p={p}", r.outputs["a"] == want
                       and r.phase == 0, f"a={av}")


def run() -> int:
    fails: list[str] = []
    for stage in (_primitives, _const_adders, _comparators, _modular):
        before = len(fails)
        stage(fails)
        name = stage.__name__.lstrip("_")
        print(f"  {name:16s} {'OK' if len(fails) == before else f'{len(fails)-before} FAIL'}")
    if fails:
        seen = set()
        print()
        for x in fails:
            key = x.split(":")[0]
            if key not in seen:
                seen.add(key)
                print("   ", x)
    else:
        print("\n  all constructions exact, ancillas clean, phase clean on every outcome")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
