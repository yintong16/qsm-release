import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.params import CARRY_WINDOW, F, carry_fail_prob
from qsm.field.constadd import (
    PRIME,
    add_const,
    add_const_inplace,
    addsub_p,
    sub_const_inplace,
)


def _circ(n: int, c: int) -> Circuit:
    circ = Circuit()
    b = circ.input("b", n)
    add_const_inplace(circ, b, c)
    return circ


def _all_meas_vectors(c: Circuit):
    k = sum(1 for op in c.ops if op.kind == ops.MEASURE)
    if k > 8:
        return [None]
    return [[(m >> j) & 1 for j in range(k)] for m in range(1 << k)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_add_const_inplace_exhaustive(n: int):
    """Every constant x every input x every measurement branch."""
    mask = (1 << n) - 1
    for cval in range(1 << n):
        circ = _circ(n, cval)
        for mv in _all_meas_vectors(circ):
            for bv in range(1 << n):
                r = simulate(circ, {"b": bv}, measurements=mv)
                assert r.outputs["b"] == (bv + cval) & mask, \
                    f"n={n}: {bv}+{cval} -> {r.outputs['b']}"
                assert r.phase == 0, f"dirty phase n={n} c={cval} b={bv} meas={mv}"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_sub_const_inplace_exhaustive(n: int):
    mask = (1 << n) - 1
    for cval in range(1 << n):
        circ = Circuit()
        b = circ.input("b", n)
        sub_const_inplace(circ, b, cval)
        for bv in range(1 << n):
            r = simulate(circ, {"b": bv})
            assert r.outputs["b"] == (bv - cval) & mask
            assert r.phase == 0


@pytest.mark.parametrize("c", [0, 1, 2, 19, 255, 256, PRIME, PRIME - 1, (1 << 254)])
def test_add_const_inplace_255bit(c: int):
    n = 255
    circ = _circ(n, c)
    rng = random.Random(7)
    mask = (1 << n) - 1
    for _ in range(16):
        bv = rng.getrandbits(n)
        r = simulate(circ, {"b": bv}, rng=rng)
        assert r.outputs["b"] == (bv + c) & mask
        assert r.phase == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_add_const_inplace_controlled_exhaustive(n: int):
    """ctrl=1 must add; ctrl=0 must be the exact identity, on every branch."""
    mask = (1 << n) - 1
    for cval in range(1 << n):
        circ = Circuit()
        ctrl = circ.input("ctrl", 1)
        b = circ.input("b", n)
        add_const_inplace(circ, b, cval, ctrl=ctrl[0])
        for mv in _all_meas_vectors(circ):
            for cv in (0, 1):
                for bv in range(1 << n):
                    r = simulate(circ, {"ctrl": cv, "b": bv}, measurements=mv)
                    want = ((bv + cval) & mask) if cv else bv
                    assert r.outputs["b"] == want, \
                        f"n={n} c={cval} ctrl={cv} b={bv} -> {r.outputs['b']}"
                    assert r.outputs["ctrl"] == cv, "control must be preserved"
                    assert r.phase == 0, f"dirty phase n={n} c={cval} ctrl={cv} b={bv}"


def test_controlled_costs_only_two_extra_toffoli():
    n = 255
    plain = _circ(n, PRIME)
    c2 = Circuit()
    ctrl = c2.input("ctrl", 1)
    b2 = c2.input("b", n)
    add_const_inplace(c2, b2, PRIME, ctrl=ctrl[0])
    assert count(c2).toffoli == count(plain).toffoli + 2


@pytest.mark.parametrize("n,L", [(8, 3), (8, 5), (10, 4), (10, 6)])
def test_truncated_const_add_error_rate_matches_formula(n: int, L: int):
    """Exhaustive: measured rate must equal c/2^L exactly, for several constants."""
    for cval in (1, 3, 5, 7):
        if cval >= (1 << L):
            continue
        circ = Circuit()
        b = circ.input("b", n)
        add_const_inplace(circ, b, cval, window=L)
        mask = (1 << n) - 1
        wrong = 0
        for bv in range(1 << n):
            r = simulate(circ, {"b": bv})
            wrong += r.outputs["b"] != ((bv + cval) & mask)
            assert r.phase == 0, "truncation must not leak phase"
        rate = wrong / (1 << n)
        assert abs(rate - carry_fail_prob(L, cval)) < 1e-9, \
            f"n={n} L={L} c={cval}: measured {rate} vs {carry_fail_prob(L, cval)}"


@pytest.mark.parametrize("n,L", [(8, 3), (10, 5)])
def test_truncated_const_add_is_exact_when_no_carry_escapes(n: int, L: int):
    """The safety property: only inputs whose low part is within c of the window
    ceiling can be wrong. Everything else is exact."""
    cval = 3
    circ = Circuit()
    b = circ.input("b", n)
    add_const_inplace(circ, b, cval, window=L)
    mask = (1 << n) - 1
    for bv in range(1 << n):
        if (bv % (1 << L)) + cval >= (1 << L):
            continue                                  # allowed to be wrong
        r = simulate(circ, {"b": bv})
        assert r.outputs["b"] == ((bv + cval) & mask), f"n={n} L={L} b={bv}"
        assert r.phase == 0


def test_truncated_const_add_is_much_cheaper():
    n = 255
    exact = Circuit()
    add_const_inplace(exact, exact.input("b", n), F)
    trunc = Circuit()
    add_const_inplace(trunc, trunc.input("b", n), F, window=CARRY_WINDOW)
    re_, rt = count(exact), count(trunc)
    assert rt.toffoli == CARRY_WINDOW - 2            # one per windowed bit, minus seeds
    assert rt.toffoli < re_.toffoli // 5             # 43 vs 253
    assert rt.peak_qubits < re_.peak_qubits


def test_truncated_const_add_rejects_oversized_constant():
    circ = Circuit()
    b = circ.input("b", 64)
    with pytest.raises(AssertionError):
        add_const_inplace(circ, b, 1 << 20, window=8)


# ---------------------------------------------------------------------------
# addsub_p: +/- p via the (B ^ 2^u) -+ f decomposition
# ---------------------------------------------------------------------------
SMALL_PM = [(3, 3, 5), (3, 1, 7), (4, 3, 13), (5, 3, 29), (6, 3, 61)]   # (u, f, p)


@pytest.mark.parametrize("adding", [False, True], ids=["sub_p", "add_p"])
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_addsub_p_exact(u: int, f: int, p: int, adding: bool):
    """Exhaustive over a full (u+1)-bit register, both signs."""
    assert p == (1 << u) - f
    n = u + 1
    circ = Circuit()
    b = circ.input("b", n)
    addsub_p(circ, b, u, f, adding=adding)
    mod = 1 << n
    delta = p if adding else -p
    for bv in range(mod):
        r = simulate(circ, {"b": bv})
        assert r.outputs["b"] == (bv + delta) % mod, \
            f"u={u} f={f} adding={adding}: b={bv}"
        assert r.phase == 0


@pytest.mark.parametrize("adding", [False, True], ids=["sub_p", "add_p"])
@pytest.mark.parametrize("u,f,p,L", [(8, 3, 253, 3), (8, 3, 253, 5),
                                     (10, 5, 1019, 4), (10, 5, 1019, 6)])
def test_addsub_p_truncated_error_rate_is_f_over_2L(u, f, p, L, adding):
    """The whole point: p can't be truncated, but f can -- at exactly f/2^L."""
    n = u + 1
    circ = Circuit()
    b = circ.input("b", n)
    addsub_p(circ, b, u, f, adding=adding, window=L)
    mod = 1 << n
    delta = p if adding else -p
    wrong = 0
    for bv in range(mod):
        r = simulate(circ, {"b": bv})
        wrong += r.outputs["b"] != (bv + delta) % mod
        assert r.phase == 0, "truncation must not leak phase"
    rate = wrong / mod
    assert abs(rate - carry_fail_prob(L, f)) < 1e-9, \
        f"u={u} f={f} L={L} adding={adding}: {rate} vs {carry_fail_prob(L, f)}"


def test_addsub_p_truncation_is_only_possible_via_the_decomposition():
    """Adding p into a small window is impossible; adding f into one is fine."""
    circ = Circuit()
    b = circ.input("b", _N := 255)
    with pytest.raises(AssertionError):
        add_const_inplace(circ, b, PRIME, window=CARRY_WINDOW)   # p does not fit
    # but the decomposed form does, on the same window
    circ2 = Circuit()
    b2 = circ2.input("b", 256)
    addsub_p(circ2, b2, 255, F, adding=False, window=CARRY_WINDOW)
    assert count(circ2).toffoli < 50


def test_addsub_p_bit_flip_stays_exact_under_truncation():
    """The window covers the low bits only; the bit-u flip is never approximated."""
    u, f, p = 10, 5, (1 << 10) - 5
    n = u + 1
    circ = Circuit()
    b = circ.input("b", n)
    addsub_p(circ, b, u, f, adding=False, window=4)
    # inputs whose low 4 bits cannot overflow the window are exact, including ones
    # that flip bit u
    for bv in range(1 << n):
        if (bv % 16) + f >= 16:
            continue
        r = simulate(circ, {"b": bv})
        assert r.outputs["b"] == (bv - p) % (1 << n), f"b={bv}"


def test_carry_ripple_worst_case():
    """all-ones + 1 wraps to zero: the maximal carry propagation."""
    n = 32
    circ = _circ(n, 1)
    r = simulate(circ, {"b": (1 << n) - 1})
    assert r.outputs["b"] == 0
    assert r.phase == 0


def test_matches_register_baseline():
    """v2 must agree with the verified v1 baseline on the same inputs."""
    n, rng = 40, random.Random(8)
    for c in [1, 19, 12345, (1 << 39), PRIME % (1 << n)]:
        v1 = Circuit()
        b1 = v1.input("b", n)
        add_const(v1, b1, c)
        v2 = _circ(n, c)
        for _ in range(8):
            bv = rng.getrandbits(n)
            assert (simulate(v1, {"b": bv}, rng=rng).outputs["b"]
                    == simulate(v2, {"b": bv}, rng=rng).outputs["b"])


def test_inplace_removes_the_constant_register():
    """v2 materialises no constant qubits at all; v1 materialises c.bit_length()."""
    n = 255
    w = (19).bit_length()                       # 5 significant bits

    v1 = Circuit()
    add_const(v1, v1.input("b", n), 19)
    v2 = _circ(n, 19)
    r1, r2 = count(v1), count(v2)

    # v1 now allocates only the constant's significant bits (was n before the fix).
    assert r1.peak_qubits == w + n + (n - 1)    # 514
    assert r2.peak_qubits == n + (n - 1)        # 509
    assert r1.peak_qubits - r2.peak_qubits == w

    # For the full-width prime, v1 pays the whole register and v2 still pays none.
    v1p = Circuit()
    add_const(v1p, v1p.input("b", n), PRIME)
    assert count(v1p).peak_qubits - count(_circ(n, PRIME)).peak_qubits == n

    # A constant whose low bits are clear costs strictly fewer Toffoli, because
    # carries below its lowest set bit are known to be |0>.
    low = count(_circ(n, 1 << 100))
    assert low.toffoli < r2.toffoli
