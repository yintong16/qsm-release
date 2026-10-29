"""Modular multiplication: product, fold, uncompute -- and the cost table that follows.

The layer's four registered shapes, and the one number each exists to establish:

    modmul          2 * U^2      the baseline; the product is paid for TWICE
    modsqr          1 * U^2      squaring is half a multiply, and the half survives
    modmul_const    weight(c)*U  a CLASSICAL constant needs no product register at all
    modmul_canonical  +0.6%      leaving the lazy contract is nearly free at this layer

The uncomputation in the first line is not an implementation choice. `pm_fold` is
many-to-one, so the 2U-bit product cannot be reduced in place and cannot be freed any
other way -- `test_the_product_must_be_uncomputed_because_the_fold_is_many_to_one` states
that as a counting argument rather than an assertion about our code.
"""

import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.mul.fold import canonicalise_fail_prob, fold2_overflow_prob
from qsm.mul.modmul import (A24, modmul, modmul_const, modmul_const_overflow_prob,
                            modmul_const_ref, modmul_karatsuba, modmul_ref, modsqr)
from qsm.params import F, PRIME, U

SMALL_PM = [(5, 1), (5, 3), (6, 3), (6, 5), (7, 5), (7, 19)]
#: Measurement outcomes are drawn at RANDOM, not fixed at zero: these circuits erase
#: with `_unand`, which restores exactly, so `phase == 0` must hold for EVERY outcome
#: vector. Under an all-zero one it holds whether that is true or not, since an erasure's
#: correction is conditioned on the measured bit -- a vacuity that let a ~25% phase leak
#: ship from `qsm/mul/accumulate.py`. See that module's test file.
_MEAS_RNG = random.Random(0xC0FFEE)


def _mul_circ(u, p, **kw):
    c = Circuit()
    x, y, o = c.input("x", u), c.input("y", u), c.input("out", u)
    modmul(c, x, y, o, p, **kw)
    return c


def _sqr_circ(u, p, **kw):
    c = Circuit()
    x, o = c.input("x", u), c.input("out", u)
    modsqr(c, x, o, p, **kw)
    return c


def _kara_circ(u, p, **kw):
    c = Circuit()
    x, y, o = c.input("x", u), c.input("y", u), c.input("out", u)
    modmul_karatsuba(c, x, y, o, p, **kw)
    return c


def _const_circ(u, p, k, **kw):
    c = Circuit()
    x, o = c.input("x", u), c.input("out", u)
    modmul_const(c, x, k, o, p, **kw)
    return c


# ---------------------------------------------------------------------------
# correctness, exhaustive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f", SMALL_PM)
def test_modmul_is_right_on_every_pair_of_canonical_operands(u, f):
    p = (1 << u) - f
    c = _mul_circ(u, p)
    for a in range(p):
        for b in range(p):
            r = simulate(c, {"x": a, "y": b, "out": 0}, rng=_MEAS_RNG)
            assert r.outputs == {"x": a, "y": b, "out": modmul_ref(a, b, p, u)}, (a, b)
            assert r.phase == 0


@pytest.mark.parametrize("u,f", SMALL_PM)
def test_modsqr_agrees_with_modmul_on_equal_operands(u, f):
    """Different circuit, same function. If the symmetric square's correction were wrong
    this is where it shows, because the multiply's correction is independent of it."""
    p = (1 << u) - f
    c = _sqr_circ(u, p)
    for a in range(p):
        r = simulate(c, {"x": a, "out": 0}, rng=_MEAS_RNG)
        assert r.outputs == {"x": a, "out": modmul_ref(a, a, p, u)}, a


@pytest.mark.parametrize("u,f", SMALL_PM)
def test_modmul_tolerates_lazy_operands_because_the_fold_does_not_care(u, f):
    """The fold reduces whatever product it is handed, so an operand in [p, 2^u) is not a
    precondition violation here -- unlike `modneg` or the canonical `modadd`. The product
    is just bigger, and the two rounds still land it in u bits."""
    p = (1 << u) - f
    c = _mul_circ(u, p)
    for a in range(p, 1 << u):                      # every lazy representative
        for b in ((1 << u) - 1, p, 0, 1, a):
            r = simulate(c, {"x": a, "y": b, "out": 0}, rng=_MEAS_RNG)
            assert r.outputs["out"] == modmul_ref(a, b, p, u), (a, b)
            assert r.outputs["out"] % p == (a * b) % p or \
                fold2_overflow_prob(u, f) > 0       # congruent unless the carry dropped


@pytest.mark.parametrize("u,f", [(5, 3), (6, 5), (7, 19)])
def test_modmul_const_is_right_for_every_constant_and_every_input(u, f):
    p = (1 << u) - f
    for k in range(1, p):
        c = _const_circ(u, p, k)
        for a in range(p):
            r = simulate(c, {"x": a, "out": 0}, rng=_MEAS_RNG)
            assert r.outputs == {"x": a, "out": modmul_const_ref(a, k, p, u)}, (k, a)
            assert r.phase == 0


@pytest.mark.parametrize("u,f", [(5, 3), (6, 5), (7, 19)])
def test_naf_and_binary_expansions_agree_as_circuits(u, f):
    """NAF trades adds for subtracts. It must not change the function -- only the count."""
    p = (1 << u) - f
    for k in (1, 2, 3, 7, 15, p - 1, F % p or 1):
        cb = _const_circ(u, p, k, use_naf=False)
        cn = _const_circ(u, p, k, use_naf=True)
        for a in range(p):
            rb = simulate(cb, {"x": a, "out": 0}, rng=_MEAS_RNG)
            rn = simulate(cn, {"x": a, "out": 0}, rng=_MEAS_RNG)
            assert rb.outputs == rn.outputs, (k, a)


# ---------------------------------------------------------------------------
# the shape of the computation
# ---------------------------------------------------------------------------
def test_the_product_must_be_uncomputed_because_the_fold_is_many_to_one():
    """A counting argument, not a claim about this code. A 2U-bit register holds 2^(2U)
    states and a reduced value has fewer than 2^U; no injective map exists, so no
    reversible circuit reduces in place. The reduced value must go somewhere fresh and the
    product must be unwound -- which is exactly why a modular multiply costs 2x a product.
    """
    assert (1 << (2 * U)) > PRIME
    m = count(_mul_circ(U, PRIME)).toffoli

    from qsm.mul.product import schoolbook
    pc = Circuit()
    x, y, pr = pc.input("x", U), pc.input("y", U), pc.input("prod", 2 * U)
    schoolbook(pc, x, y, pr)
    one_product = count(pc).toffoli

    assert 2.0 < m / one_product < 2.1, m / one_product


def test_the_product_register_is_actually_freed():
    """If it were not, every multiply would leak 2U qubits and the peak would grow with
    the length of the formula rather than staying flat."""
    u, f = 6, 5
    c = _mul_circ(u, (1 << u) - f)
    assert c.peak_qubits >= 2 * u
    # peak, not final: a leaked product register would push the final live count up
    live = sum(len(r) for r in c.inputs.values())
    assert c._live == live, (c._live, live)


@pytest.mark.parametrize("u,f", SMALL_PM)
def test_nothing_is_freed_dirty_by_the_lazy_versions(u, f):
    """The lazy path is EXACT apart from the declared round-2 carry drop, which is a
    value error and never an entangled one. Canonicalisation is the only thing in this
    layer that can free a dirty qubit."""
    p = (1 << u) - f
    for c in (_mul_circ(u, p), _sqr_circ(u, p), _const_circ(u, p, 3)):
        keys = list(c.inputs)
        for a in range(p):
            vals = {k: (0 if k == "out" else a) for k in keys}
            simulate(c, vals, rng=_MEAS_RNG)     # raises if any free is not |0>


@pytest.mark.parametrize("u,f", [(5, 3), (6, 5), (7, 19)])
def test_canonical_modmul_is_right_and_its_dirty_set_is_exactly_the_declared_one(u, f):
    p = (1 << u) - f
    c = _mul_circ(u, p, canonical=True)
    dirty = 0
    for a in range(p):
        for b in range(p):
            try:
                r = simulate(c, {"x": a, "y": b, "out": 0}, rng=_MEAS_RNG)
            except DirtyAncillaError:
                dirty += 1
                assert modmul_ref(a, b, p, u) < f      # only the below-f residues
                continue
            assert r.outputs["out"] == modmul_ref(a, b, p, u) % p, (a, b)
            assert r.outputs["out"] < p                # the whole point: canonical
    expected = sum(1 for a in range(p) for b in range(p) if modmul_ref(a, b, p, u) < f)
    assert dirty == expected, (dirty, expected)


# ---------------------------------------------------------------------------
# the cost table
# ---------------------------------------------------------------------------
def test_squaring_is_half_a_modular_multiply():
    m = count(_mul_circ(U, PRIME)).toffoli
    s = count(_sqr_circ(U, PRIME)).toffoli
    assert 0.50 < s / m < 0.55, (s, m, s / m)


def test_a_constant_multiply_is_two_orders_below_a_quantum_one():
    """The finding that decides Layer 2: any point formula able to trade a general
    multiply for a multiply by a curve constant should. a24 = 121665 is the Montgomery
    ladder's only constant, and it costs ~1/35 of a `modmul`."""
    m = count(_mul_circ(U, PRIME)).toffoli
    a24 = count(_const_circ(U, PRIME, A24)).toffoli
    f19 = count(_const_circ(U, PRIME, F)).toffoli
    assert m / a24 > 30, m / a24
    assert m / f19 > 50, m / f19
    assert a24 < 5000 and f19 < 3000, (a24, f19)


def test_a_constant_multiply_has_no_product_register_so_its_peak_is_far_lower():
    """Under a qubit x Toffoli target this is half the win, and it is invisible in a
    Toffoli-only comparison."""
    mul = count(_mul_circ(U, PRIME))
    const = count(_const_circ(U, PRIME, A24))
    assert const.peak_qubits < mul.peak_qubits - 2 * U
    assert const.circuit_volume * 50 < mul.circuit_volume


@pytest.mark.parametrize("u,f", [(4, 3), (5, 3), (6, 5), (7, 19)])
def test_karatsuba_computes_the_same_function_as_the_schoolbook_path(u, f):
    p = (1 << u) - f
    c = _kara_circ(u, p)
    for a in range(p):
        for b in range(p):
            r = simulate(c, {"x": a, "y": b, "out": 0}, rng=_MEAS_RNG)
            assert r.outputs == {"x": a, "y": b, "out": modmul_ref(a, b, p, u)}, (a, b)
            assert r.phase == 0


def test_karatsuba_wins_on_toffoli_and_loses_on_volume():
    """THE Layer-1 result, and the reason `qubit x Toffoli` was adopted as the target.

    One Karatsuba level does exactly what the arithmetic promises -- three quarters of the
    partial products, ~19% off the Toffoli count once the linear overheads are paid. It
    also has to hold z0, z2, c, two half sums and the assembled 2U register at once, and
    that pushes the peak up by more than half. The two effects point opposite ways and the
    target decides:

        toffoli         109,293  vs  135,417     Karatsuba by 19%
        peak_qubits       2,820  vs    1,794     schoolbook by 36%
        circuit_volume     308M  vs      243M    schoolbook by 21%

    Under an active-volume architecture, where cost tracks non-Clifford gates and not
    qubits, the answer flips. Under the 2D surface-code model this project and the
    ecdsa.fail benchmark both score, it does not.
    """
    k = count(_kara_circ(U, PRIME))
    s = count(_mul_circ(U, PRIME))
    assert k.toffoli < s.toffoli * 0.85, (k.toffoli, s.toffoli)
    assert k.peak_qubits > s.peak_qubits * 1.4, (k.peak_qubits, s.peak_qubits)
    assert k.circuit_volume > s.circuit_volume * 1.15, (k.circuit_volume,
                                                        s.circuit_volume)
    # the peak is a TRANSIENT, so the sharper metric is much closer than the crude one
    assert 1.0 < k.toffoli_volume / s.toffoli_volume < 1.10


def test_karatsuba_is_only_worth_measuring_because_it_is_fused_with_the_fold():
    """As a drop-in product it would lose outright: 1.5*U^2 computed AND uncomputed is
    3*U^2 against the schoolbook path's 2*U^2. Stated as arithmetic, since the losing
    circuit is deliberately not built."""
    from qsm.mul.product import schoolbook
    h = U // 2
    pc = Circuit()
    x, y, pr = pc.input("x", h), pc.input("y", h), pc.input("prod", 2 * h)
    schoolbook(pc, x, y, pr)
    half = count(pc).toffoli

    full = count(_mul_circ(U, PRIME)).toffoli / 2      # one product, compute only
    assert 3 * half < full * 0.85          # Karatsuba's product IS cheaper...
    assert 2 * (3 * half) > full * 1.4     # ...but paying for it twice is not


def test_canonicalising_costs_under_one_percent_of_the_multiply():
    """docs/decisions/05 left this open and guessed ~10%. It is 0.6%."""
    lazy = count(_mul_circ(U, PRIME)).toffoli
    canon = count(_mul_circ(U, PRIME, canonical=True)).toffoli
    assert 0.0 < (canon - lazy) / lazy < 0.01, (canon - lazy) / lazy


def test_the_declared_failure_rates_are_the_ones_the_versions_carry():
    """Guards against a version's `fail_analytic` drifting away from the formula that
    justifies it -- the exact drift `qsm.audit.rates` was written to catch in Layer 0."""
    from qsm.bench import registry

    for name, want in (
        ("mul.modmul.v1", fold2_overflow_prob()),
        ("mul.modsqr.v1", fold2_overflow_prob()),
        ("mul.modmul_canonical.v1", fold2_overflow_prob() + canonicalise_fail_prob()),
        ("mul.modmul_const_a24.v1", modmul_const_overflow_prob(A24)),
    ):
        got = registry.get_builder(name)().meta["fail_analytic"]
        assert got == pytest.approx(want, rel=1e-12), (name, got, want)
