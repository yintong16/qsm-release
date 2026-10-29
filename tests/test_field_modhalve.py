"""Halving, and the controlled lazy subtraction -- Layer 0's two reversed circuits.

Both exist to undo something rather than to compute something, and the interesting
question in each case is what the reversal costs.

`modhalve` costs nothing extra, and the reason is worth pinning: `moddouble` erased its
reduction flag with the PARITY of its output, and parity is exact at every width, so the
reverse reads the flag straight back with one CX. Gate-for-gate parity with doubling is
therefore a claim about the construction, not a coincidence -- see
`test_halving_costs_exactly_what_doubling_costs`.

`modsub_lazy_ctrl` is the opposite lesson. `modadd_lazy_ctrl`'s flag was erased by
MEASUREMENT plus an approximate phase comparison, and a measurement has no inverse, so the
reverse has to materialise the flag with a real comparator. Same gates, but the forward's
phase-only failure mode becomes a value failure -- which is checked here by showing that
the composition really is the identity on the whole canonical domain, and by pinning where
it stops being one.
"""

from __future__ import annotations

import random

import pytest

from qsm import params
from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl
from qsm.field.compare import cmp_lt, cmp_lt_phase
from qsm.field.modarith import (
    PRIME,
    modadd_lazy_ctrl,
    moddouble,
    moddouble_ctrl_inplace,
    modhalve,
    modhalve_ctrl_inplace,
    modhalve_inplace,
    modhalve_ref,
    modsub_lazy_ctrl,
)

PRIMES = [29, 61, 127, 251, 1021]


def _halve(p: int, inplace: bool = False, window: int | None = None) -> Circuit:
    c = Circuit()
    b = c.input("b", p.bit_length())
    if inplace:
        modhalve_inplace(c, b, p, window)
    else:
        c.rebind("b", modhalve(c, b, p, window))
    return c


# ---------------------------------------------------------------------------
# modhalve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("p", PRIMES)
def test_modhalve_matches_the_reference_exhaustively(p: int, inplace: bool):
    """Every value in [0, 2^u), including the out-of-domain ones."""
    n = p.bit_length()
    c = _halve(p, inplace)
    for v in range(1 << n):
        r = simulate(c, {"b": v}, rng=random.Random(v))
        assert r.outputs["b"] == modhalve_ref(v, p, n), f"p={p} b={v}"
        assert r.phase == 0, f"p={p} b={v}: halving has no vent and must never leak phase"


@pytest.mark.parametrize("p", PRIMES)
def test_halving_undoes_doubling_on_every_canonical_input(p: int):
    """The left-inverse claim, over `moddouble`'s whole precondition `a < p`."""
    n = p.bit_length()
    c = Circuit()
    a = c.input("a", n)
    c.rebind("a", modhalve(c, moddouble(c, a, p), p))
    for v in range(p):
        r = simulate(c, {"a": v}, rng=random.Random(v))
        assert r.outputs["a"] == v, f"p={p}: 2a/2 != a at a={v}"
        assert r.phase == 0


@pytest.mark.parametrize("p", PRIMES)
def test_the_domain_is_exactly_the_odd_values_below_f(p: int):
    """`modhalve_corrupts` names the failure set; check it is neither loose nor tight.

    Also checks the interesting half: those values are NOT congruent to b/2. A lazy result
    would be repairable by a final reduction, and this one is not -- the discarded borrow
    is worth 2^(u-1), so it is in `modneg_corrupts`'s class, not laziness'.
    """
    n = p.bit_length()
    f = (1 << n) - p
    c = _halve(p)
    inv2 = pow(2, -1, p)
    measured = set()
    for v in range(1 << n):
        got = simulate(c, {"b": v}, rng=random.Random(v)).outputs["b"]
        if got % p != (v * inv2) % p:
            measured.add(v)
    assert measured == {v for v in range(1 << n) if params.modhalve_corrupts(v, n, f)}
    assert len(measured) == f // 2, f"p={p}: expected floor(F/2) = {f // 2} bad values"


@pytest.mark.parametrize("p", PRIMES)
def test_doublings_image_never_lands_in_the_corrupt_set(p: int):
    """Why the domain is a precondition and not a check: the caller cannot violate it.

    `moddouble`'s odd outputs are `2a - p` for a in [2^(u-1), p), which is exactly [F, p).
    So a halving that undoes a doubling is inside the domain by construction, whatever the
    input was -- there is nothing for the circuit to test.
    """
    n = p.bit_length()
    f = (1 << n) - p
    c = Circuit()
    a = c.input("a", n)
    c.rebind("a", moddouble(c, a, p))
    for v in range(p):
        img = simulate(c, {"a": v}, rng=random.Random(v)).outputs["a"]
        assert not params.modhalve_corrupts(img, n, f), f"p={p}: 2*{v} = {img} is undoable"


@pytest.mark.parametrize("inplace", [False, True])
def test_halving_costs_exactly_what_doubling_costs(inplace: bool):
    """Parity with `moddouble`, at 255 bits, exact and truncated.

    The claim `modhalve`'s docstring rests on. If this drifts, the parity trick has
    stopped being the erasure mechanism in one of the two directions and the reverse has
    grown a comparator.
    """
    from qsm.bench import registry

    suffix, dn = ("_inplace", "v2") if inplace else ("", "v1")
    for kind, dv, hv in [("exact", f"field.moddouble{suffix}.{dn}",
                          f"field.modhalve{suffix}.v1"),
                         ("trunc", f"field.moddouble{suffix}_trunc.{dn}",
                          f"field.modhalve{suffix}_trunc.v1")]:
        d = count(registry.get_builder(dv)().circuit)
        h = count(registry.get_builder(hv)().circuit)
        assert (h.toffoli, h.peak_qubits) == (d.toffoli, d.peak_qubits), (
            f"{kind}: {hv} costs {h.toffoli}@{h.peak_qubits}, "
            f"{dv} costs {d.toffoli}@{d.peak_qubits}"
        )


@pytest.mark.parametrize("p,L", [(1021, 4), (2029, 5), (4077, 6)])
def test_the_truncated_borrow_rate_matches_the_formula(p: int, L: int):
    """Exhaustive over the lazy range, against `modhalve_carry_fail_prob` on lazy input.

    On a uniform lazy draw P(odd) is exactly 1/2, so the rate collapses to
    floor(F/2)/2^L -- numerically the same number doubling gets from
    `moddouble_lazy_input_carry_fail_prob`, which is the symmetry the helper claims.
    """
    n = p.bit_length()
    f = (1 << n) - p
    ex, tr = _halve(p), _halve(p, window=L)
    wrong = 0
    for v in range(1 << n):
        a = simulate(ex, {"b": v}, rng=random.Random(1)).outputs["b"]
        b = simulate(tr, {"b": v}, rng=random.Random(1)).outputs["b"]
        wrong += a != b
    assert wrong / (1 << n) == params.moddouble_lazy_input_carry_fail_prob(L, f)
    assert wrong == (f // 2) * (1 << (n - L))


# ---------------------------------------------------------------------------
# modhalve_ctrl_inplace -- the reverse of jump 2's conditional doubling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", PRIMES)
def test_the_gated_halving_undoes_the_gated_doubling_on_both_branches(p: int):
    """Exhaustive over (ctrl, b) in {0,1} x [0,p): the claim the jump-2 reverse rests on.

    Both branches matter and for different reasons. With ctrl = 1 this is the ordinary
    halve-undoes-double statement, moved onto `b`'s own wires. With ctrl = 0 it is the
    statement that BOTH halves collapse to the identity -- which is not automatic: the
    forward's parity erasure is a Toffoli rather than a bare CX precisely so the flag stays
    |0> when the control is down, and the reverse has to make the same choice or the free
    strands a qubit on every unshifted iteration.
    """
    n = p.bit_length()
    c = Circuit()
    ctl, b = c.input("c", 1), c.input("b", n)
    moddouble_ctrl_inplace(c, ctl[0], b, p)
    modhalve_ctrl_inplace(c, ctl[0], b, p)
    for cv in (0, 1):
        for v in range(p):
            r = simulate(c, {"c": cv, "b": v}, rng=random.Random(v))
            assert r.outputs["b"] == v, f"p={p} ctrl={cv}: 2b/2 != b at b={v}"
            assert r.phase == 0, "an exact round trip must not leak phase"


@pytest.mark.parametrize("p", PRIMES)
def test_the_gated_halving_is_the_ungated_one_when_the_control_fires(p: int):
    """Over the whole register, in-domain and out: the gate changes nothing but the gating.

    Checked against `modhalve_ref` rather than against b/2 mod p, so the out-of-domain
    values are pinned too -- the gated form must inherit `modhalve`'s corruption set
    exactly, not a different one. It is also TOTAL: unlike `modsub_lazy_ctrl`, a bad input
    here produces a wrong value rather than a stranded ancilla, because the flag is
    recovered from a parity that is defined everywhere.
    """
    n = p.bit_length()
    c = Circuit()
    ctl, b = c.input("c", 1), c.input("b", n)
    modhalve_ctrl_inplace(c, ctl[0], b, p)
    for v in range(1 << n):
        on = simulate(c, {"c": 1, "b": v}, rng=random.Random(v))
        off = simulate(c, {"c": 0, "b": v}, rng=random.Random(v))
        assert on.outputs["b"] == modhalve_ref(v, p, n), f"p={p} b={v}"
        assert off.outputs["b"] == v, f"p={p} b={v}: ctrl=0 must be the identity"
        assert on.phase == 0 and off.phase == 0


@pytest.mark.parametrize("window", [None, 45])
def test_the_gated_halving_costs_exactly_what_the_gated_doubling_costs(window):
    """Parity at 255 bits, exact and truncated -- the jump-2 analogue of the ungated claim.

    This is why jump 2's surcharge over jump 1 is direction-independent. The conditional
    doubling's erasure predicate is a PARITY and it is erased with a Toffoli rather than a
    measurement, so unlike `modsub_lazy_ctrl` (+2 Toffoli) and `un_to_dialog` (a whole
    comparator) the reversal here is free. If this drifts, one of the two directions has
    grown structure the other does not have.
    """
    def one(fn):
        c = Circuit()
        ctl, b = c.input("c", 1), c.input("b", 255)
        fn(c, ctl[0], b, PRIME, window=window)
        return count(c)

    d, h = one(moddouble_ctrl_inplace), one(modhalve_ctrl_inplace)
    assert (h.toffoli, h.peak_qubits) == (d.toffoli, d.peak_qubits), (
        f"window={window}: halve {h.toffoli}@{h.peak_qubits}, "
        f"double {d.toffoli}@{d.peak_qubits}"
    )


# ---------------------------------------------------------------------------
# modsub_lazy_ctrl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("add_ctrl", [gidney_add_ctrl, cdkm_add_ctrl])
@pytest.mark.parametrize("p", [29, 61, 127])
def test_the_controlled_add_and_subtract_compose_to_the_identity(p: int, add_ctrl):
    """Exhaustive over (ctrl, a, b) in {0,1} x [0,p) x [0,p), both adders.

    Neither half is exact on its own contract -- the forward vents a phase and the reverse
    materialises a flag -- so composing them and getting the identity is the check that
    the reverse really is the forward's inverse and not merely a subtraction.
    """
    n = p.bit_length()
    c = Circuit()
    ctl, a, b = c.input("c", 1), c.input("a", n), c.input("b", n)
    modadd_lazy_ctrl(c, ctl[0], a, b, p, add_ctrl_fn=add_ctrl)
    modsub_lazy_ctrl(c, ctl[0], a, b, p, add_ctrl_fn=add_ctrl)

    for cv in (0, 1):
        for av in range(p):
            for bv in range(p):
                r = simulate(c, {"c": cv, "a": av, "b": bv},
                             rng=random.Random(av * p + bv))
                assert (r.outputs["a"], r.outputs["b"]) == (av, bv), (p, cv, av, bv)
                assert r.phase == 0, f"p={p}: an exact round trip must not leak phase"


@pytest.mark.parametrize("p", [29, 61, 127, 251, 1021, 8191])
def test_the_reverse_costs_two_toffoli_more_than_the_forward_at_every_width(p: int):
    """The cost claim, and where the two Toffoli go.

    Reversing is not free, but the whole surcharge is the comparator changing form.
    `modadd_lazy_ctrl` erases its flag by MEASURING it and venting the disagreement into a
    phase, so `cmp_lt_phase` never has to materialise the top of its carry chain.
    `modsub_lazy_ctrl` needs the flag as a value to drive `-F`, so it runs `cmp_lt` -- the
    same chain, plus the top carry, plus the controlled write into the flag qubit. That is
    exactly two Toffoli, and it is a CONSTANT: the difference does not grow with n, which
    is the whole reason a 255-bit reversal is affordable.

    Everything else -- the controlled adder, the constant subtraction, the window -- is
    the forward's, run backwards, at the forward's price.
    """
    n = p.bit_length()

    def one(fn):
        c = Circuit()
        ctl, a, b = c.input("c", 1), c.input("a", n), c.input("b", n)
        fn(c, ctl[0], a, b, p)
        return count(c)

    fwd, rev = one(modadd_lazy_ctrl), one(modsub_lazy_ctrl)
    assert rev.toffoli == fwd.toffoli + 2, f"p={p}: forward {fwd.toffoli}, reverse {rev.toffoli}"
    assert rev.peak_qubits == fwd.peak_qubits, "the reverse must not need a wider register"

    # ...and the surcharge is the comparator, not something that crept in elsewhere.
    def cmp_only(build):
        c = Circuit()
        a, b, ctl = c.input("a", n), c.input("b", n), c.input("c", 1)
        build(c, a, b, ctl[0])
        return count(c).toffoli

    value = cmp_only(lambda c, a, b, k: cmp_lt(c, a, b, c.alloc(1)[0], k=None, ctrl=k))
    phase = cmp_only(
        lambda c, a, b, k: cmp_lt_phase(c, a, b, c.measure(c.alloc(1)[0]), k=None, ctrl=k)
    )
    assert value - phase == 2, f"p={p}: cmp_lt {value}, cmp_lt_phase {phase}"


@pytest.mark.parametrize("p", [29, 61, 127])
def test_the_reverses_domain_is_exactly_the_forwards_image(p: int):
    """A precondition, like `modhalve`'s, and reachable for the same reason: it isn't.

    Run on an arbitrary `(a, b)` the reverse is NOT total. The re-derived flag is
    `b < a`, and when that fires on a `b` below F the `-F` borrows out of the register,
    so the subsequent `-a` cannot bring `ovf` back down and the free reports it.

    But the forward can never hand over such a `b`. Its flag is the carry out of n bits,
    not a comparison against p, so `ovf = 1` means `a + b >= 2^u` and the output is
    `a + b - p >= F` -- above the borrow band by construction; and `ovf = 0` means the
    output is `a + b >= a`, which re-derives as 0. Both branches land in the domain, so
    there is nothing for the circuit to test. The measured bad set is stated here so a
    future caller that feeds the reverse something else knows what it is walking into.
    """
    n = p.bit_length()
    f = (1 << n) - p
    c = Circuit()
    ctl, a, b = c.input("c", 1), c.input("a", n), c.input("b", n)
    modsub_lazy_ctrl(c, ctl[0], a, b, p)

    dirty = set()
    for av in range(1 << n):
        for bv in range(1 << n):
            try:
                simulate(c, {"c": 1, "a": av, "b": bv}, rng=random.Random(1))
            except DirtyAncillaError:
                dirty.add((av, bv))

    # b < a fires the flag; b < f makes the -F borrow out; a > b + p wraps a second time
    # and lands ovf back on |0>, which is the silent corner of the same bad region.
    assert dirty == {(av, bv) for av in range(1 << n) for bv in range(1 << n)
                     if bv < av and bv < f and av <= bv + p}
    assert dirty, "at these widths the bad region is nonempty; the test would be vacuous"

    # ...and none of it is reachable through the forward.
    fwd = Circuit()
    fc, fa, fb = fwd.input("c", 1), fwd.input("a", n), fwd.input("b", n)
    modadd_lazy_ctrl(fwd, fc[0], fa, fb, p)
    for av in range(p):
        for bv in range(p):
            out = simulate(fwd, {"c": 1, "a": av, "b": bv}, rng=random.Random(1)).outputs
            assert (av, out["b"]) not in dirty, f"p={p}: forward({av},{bv}) is not undoable"


@pytest.mark.parametrize("p,k", [(127, 3), (251, 3)])
def test_truncating_the_comparator_fails_loudly_backwards_not_silently(p: int, k: int):
    """The phase-to-value promotion, measured.

    Forward, a truncated `cmp_lt_phase` disagreeing with the measured flag costs a PHASE
    and nothing else -- the value is still right, and the disagreement is invisible to
    any value-only check. Backward the same predicate has to be materialised to drive
    `-F`, so the same disagreement is an arithmetic error.

    What that error looks like matters more than that it exists: it strands `ovf` at |1>
    and the free reports it. So truncating the reverse is not a silent-corruption knob --
    over the whole canonical domain the round trip is either exact or an exception, never
    quietly wrong. That is what makes `CMP_BITS` a budget question in the replay rather
    than a correctness one.
    """
    n = p.bit_length()
    c = Circuit()
    ctl, a, b = c.input("c", 1), c.input("a", n), c.input("b", n)
    modadd_lazy_ctrl(c, ctl[0], a, b, p)
    modsub_lazy_ctrl(c, ctl[0], a, b, p, cmp_bits=k)

    loud = 0
    for av in range(p):
        for bv in range(p):
            try:
                r = simulate(c, {"c": 1, "a": av, "b": bv}, rng=random.Random(1))
            except DirtyAncillaError:
                loud += 1
                continue
            assert (r.outputs["a"], r.outputs["b"]) == (av, bv), \
                f"p={p} k={k}: a truncated reverse returned a wrong value silently"
    assert loud > 0, "k was too generous to exercise a disagreement at this width"
