"""Modular multiplication mod p = 2^255 - 19: product, fold, uncompute.

    out <- x*y mod p          out arrives |0>, x and y are preserved

The shape is forced, not chosen. `pm_fold` is many-to-one, so the reduced value must be
written into fresh space and the 2U-bit product uncomputed afterwards -- which means
**every modular multiply pays for its product twice**. That doubling is uniform, so it
does not change which product circuit wins; it does mean the fold, at ~2% of one product,
is ~1% of the operation and is not where the optimisation lives.

What the layer costs, at U = 255 (measured, not estimated -- `python -m qsm.bench mul.*`)

    modsqr     ~ U^2   + fold      squaring is HALF a multiply, structurally
    modmul     ~ 2*U^2 + fold
    modmul_c   ~ weight(c) * U     a CLASSICAL constant needs no product at all

The third line is the one that matters for the paper. A multiply by a curve constant is
`weight(c)` additions and no product register, so it is two orders of magnitude below a
quantum-by-quantum multiply. Any point formula that can trade a general multiply for a
constant multiply should. That is a Layer-2 decision, but it is decided by this table.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add
from ..field.sub import sub
from ..params import CARRY_WINDOW, F, PRIME, U
from .fold import (FOLD2_OVERFLOW, add_times_const, canonicalise_fail_prob,
                   fold_once_ref, pm_fold, pm_fold_ref, set_bits)
from .product import (schoolbook, schoolbook_inverse, square_symmetric,
                      square_symmetric_inverse)


def modmul(circ: Circuit, x: Register, y: Register, out: Register,
           p: int = PRIME, add_fn=gidney_add, window: int | None = None,
           canonical: bool = False) -> None:
    """out <- x*y mod p, lazily reduced into [0, 2^U). x, y preserved; out arrives |0>.

    `canonical=True` narrows the output to [0, p) for ~0.6% more Toffoli; see
    `fold.canonicalise` for why that is legal here and nowhere else.
    """
    u = len(x)
    assert len(y) == u and len(out) == u, "operands and output must all be u bits"
    f = (1 << u) - p
    prod = circ.alloc(2 * u)
    schoolbook(circ, x, y, prod, add_fn)
    pm_fold(circ, prod, out, u, f, add_fn, window, canonical)
    schoolbook_inverse(circ, x, y, prod, add_fn)
    circ.free(prod)


def modsqr(circ: Circuit, x: Register, out: Register,
           p: int = PRIME, add_fn=gidney_add, window: int | None = None) -> None:
    """out <- x^2 mod p, lazily reduced into [0, 2^U). x preserved; out arrives |0>.

    Half a `modmul`, and the halving survives the compute/uncompute doubling because the
    doubling applies to both.
    """
    u = len(x)
    assert len(out) == u, "output must be u bits"
    f = (1 << u) - p
    prod = circ.alloc(2 * u)
    square_symmetric(circ, x, prod, add_fn)
    pm_fold(circ, prod, out, u, f, add_fn, window)
    square_symmetric_inverse(circ, x, prod, add_fn)
    circ.free(prod)


def _build_sum(circ: Circuit, a: Register, b: Register, s: Register,
               add_fn=gidney_add, inverse: bool = False) -> None:
    """s <- a + b for len(a) <= len(b) = len(s)-1. `s` arrives |0>; a, b preserved.

    The wider half is copied in with CNOTs -- free, because the target is zeroed -- so
    only the narrower half costs an addition.
    """
    if inverse:
        sub(circ, a, s, add_fn)
        for i in range(len(b)):
            circ.cx(b[i], s[i])
        return
    for i in range(len(b)):
        circ.cx(b[i], s[i])
    add_fn(circ, a, s)


def modmul_karatsuba(circ: Circuit, x: Register, y: Register, out: Register,
                     p: int = PRIME, add_fn=gidney_add,
                     window: int | None = None) -> None:
    """out <- x*y mod p by one level of Karatsuba, FUSED with the reduction.

    The fusion is not a refinement, it is the only arrangement that wins. Karatsuba's
    three half-products are scratch and have to be uncomputed, so as a drop-in replacement
    for `schoolbook` inside `modmul` it would cost `2 * 1.5*U^2 = 3*U^2` against the
    schoolbook path's `2*U^2` -- strictly worse. Keeping all three ALIVE across the fold
    and unwinding them once afterwards pays for each exactly twice, giving `1.5*U^2`:

        z0 = x0*y0      z2 = x1*y1      c = (x0+x1)(y0+y1)
        x*y = z0 + (c - z0 - z2)*2^h + z2*2^2h

    So the interesting question was never "is Karatsuba cheaper in Toffoli" -- 3/4 of the
    partial products, it plainly is -- but whether holding three half-products, two half
    sums and the assembled 2U-bit register at once is worth it under `peak x Toffoli`.
    That is what `mul.modmul_karatsuba.v1` exists to measure.
    """
    u = len(x)
    assert len(y) == u and len(out) == u
    assert u >= 4, "one Karatsuba level needs both halves at least two bits wide"
    f = (1 << u) - p
    h, m = u // 2, u - u // 2
    x0, x1, y0, y1 = x[0:h], x[h:u], y[0:h], y[h:u]

    z0 = circ.alloc(2 * h)
    z2 = circ.alloc(2 * m)
    sx = circ.alloc(m + 1)
    sy = circ.alloc(m + 1)
    cc = circ.alloc(2 * m + 2)
    W = circ.alloc(2 * u)

    schoolbook(circ, x0, y0, z0, add_fn)
    schoolbook(circ, x1, y1, z2, add_fn)
    _build_sum(circ, x0, x1, sx, add_fn)
    _build_sum(circ, y0, y1, sy, add_fn)
    schoolbook(circ, sx, sy, cc, add_fn)

    add_fn(circ, z0, W)                       # + z0
    sub(circ, z0, W[h:], add_fn)              # - z0 * 2^h
    add_fn(circ, cc, W[h:])                   # + c  * 2^h
    sub(circ, z2, W[h:], add_fn)              # - z2 * 2^h
    add_fn(circ, z2, W[2 * h:])               # + z2 * 2^2h

    pm_fold(circ, W, out, u, f, add_fn, window)

    sub(circ, z2, W[2 * h:], add_fn)
    add_fn(circ, z2, W[h:])
    sub(circ, cc, W[h:], add_fn)
    add_fn(circ, z0, W[h:])
    sub(circ, z0, W, add_fn)
    circ.free(W)

    schoolbook_inverse(circ, sx, sy, cc, add_fn)
    circ.free(cc)
    _build_sum(circ, y0, y1, sy, add_fn, inverse=True)
    circ.free(sy)
    _build_sum(circ, x0, x1, sx, add_fn, inverse=True)
    circ.free(sx)
    schoolbook_inverse(circ, x1, y1, z2, add_fn)
    circ.free(z2)
    schoolbook_inverse(circ, x0, y0, z0, add_fn)
    circ.free(z0)


def modmul_const(circ: Circuit, x: Register, c: int, out: Register,
                 p: int = PRIME, add_fn=gidney_add, window: int | None = None,
                 use_naf: bool = True) -> None:
    """out <- c*x mod p for CLASSICAL c. x preserved; out arrives |0>.

    No product register and no uncomputation of one: `c*x` is `weight(c)` shifted adds
    into a `u + bitlen(c)`-bit scratch, folded out, and the scratch reversed. Cost is
    linear in u, not quadratic -- for Curve25519's a24 = 121665 (NAF weight 7) that is
    ~2 orders of magnitude below `modmul`.

    The scratch is folded with the same two-round `pm_fold` machinery, generalised to a
    high part of `bitlen(c)` bits rather than u.
    """
    u = len(x)
    assert len(out) == u
    f = (1 << u) - p
    c %= p
    if c == 0:
        return

    wide = circ.alloc(u + max(1, c.bit_length()))
    add_times_const(circ, x, wide, c, add_fn=add_fn, use_naf=use_naf)

    _fold_wide_into(circ, wide, out, u, f, add_fn, window)

    add_times_const(circ, x, wide, c, add_fn=add_fn, use_naf=use_naf, inverse=True)
    circ.free(wide)


def _fold_wide_into(circ: Circuit, wide: Register, out: Register,
                    u: int, f: int, add_fn, window: int | None) -> None:
    """out <- (wide mod p) for a `wide` only a little longer than u. `wide` preserved.

    The narrow-high-part case of `pm_fold`: `wide` is u + k bits with k small, so
    `f * wide_hi` is at most `f * 2^k` and one extra round suffices. Used by
    `modmul_const`, where k = bitlen(c), and it needs no scratch register of its own --
    the free CNOT copy into `out` is the whole of round 1.
    """
    for j in range(u):
        circ.cx(wide[j], out[j])
    add_times_const(circ, wide[u:], out, f, add_fn=add_fn, window=window)


# --- exact semantics -------------------------------------------------------
def modmul_ref(x: int, y: int, p: int = PRIME, u: int = U) -> int:
    """Exact semantics of `modmul`/`modsqr`: two fold rounds, then a u-bit truncation."""
    return pm_fold_ref(x * y, u, (1 << u) - p)


def modmul_const_ref(x: int, c: int, p: int = PRIME, u: int = U) -> int:
    """Exact semantics of `modmul_const`: ONE fold round, then a u-bit truncation.

    One round rather than two, because `c*x` has only bitlen(c) bits above 2^u to begin
    with -- there is no wide product to walk down.
    """
    f = (1 << u) - p
    return fold_once_ref((c % p) * x, u, f) & ((1 << u) - 1)


def modmul_const_overflow_prob(c: int, p: int = PRIME, u: int = U) -> float:
    """P(`modmul_const` drops its round-1 carry), for x uniform on [0, 2^u).

    `c*x < c*2^u` gives `hi <= c`, so the fold adds at most `f*c` to a u-bit low part and
    overflows when that low part is within `f*c` of 2^u. For a24 = 121665 this is
    19*121665/2^255 = 2^-233.9 -- rarer than anything in the point layer will care about,
    and NON-CONGRUENT like `fold2_overflow_prob`.
    """
    f = (1 << u) - p
    return (f * (c % p)) / float(1 << u)


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
# ---------------------------------------------------------------------------
_N = U
A24 = 121665            # Curve25519's (A-2)/4; the Montgomery ladder's only constant


def _binary_case(build, ref, gen, fail: float) -> Case:
    circ = Circuit()
    x = circ.input("x", _N)
    y = circ.input("y", _N)
    out = circ.input("out", _N)
    build(circ, x, y, out)
    return Case(circ, ref, gen, meta={"fail_analytic": fail} if fail else {})


def _unary_case(build, ref, gen, fail: float) -> Case:
    circ = Circuit()
    x = circ.input("x", _N)
    out = circ.input("out", _N)
    build(circ, x, out)
    return Case(circ, ref, gen, meta={"fail_analytic": fail} if fail else {})


@circuit_version("mul.product_schoolbook.v1")
def _build_product_schoolbook_v1() -> Case:
    """
    EXACT, NOT modular. The raw 2U-bit product, via the add/subtract trick.

    n^2 + ~5n Toffoli: n unconditional adds instead of n controlled ones. The proven
    floor for an uncontrolled n-bit add is n-1 and it is achieved, so each row is
    already at the floor -- the remaining factor of two against a controlled-add
    schoolbook is bought by the correction, not by a better adder.
    """
    circ = Circuit()
    x = circ.input("x", _N)
    y = circ.input("y", _N)
    prod = circ.input("prod", 2 * _N)
    schoolbook(circ, x, y, prod)

    def ref(v):
        return {"x": v["x"], "y": v["y"], "prod": v["x"] * v["y"]}

    def gen(rng):
        return {"x": rng.randrange(1 << _N), "y": rng.randrange(1 << _N), "prod": 0}

    return Case(circ, ref, gen)


@circuit_version("mul.product_square.v1")
def _build_product_square_v1() -> Case:
    """
    EXACT, NOT modular. The raw 2U-bit square: n(n-1)/2 + ~5n Toffoli.

    HALF the multiply, and structurally so -- each unordered pair {i,j} is touched once
    and weighted twice, so the cross terms form a triangle. The diagonal is free.
    """
    circ = Circuit()
    x = circ.input("x", _N)
    prod = circ.input("prod", 2 * _N)
    square_symmetric(circ, x, prod)

    def ref(v):
        return {"x": v["x"], "prod": v["x"] * v["x"]}

    def gen(rng):
        return {"x": rng.randrange(1 << _N), "prod": 0}

    return Case(circ, ref, gen)


def _modmul_gen(rng: random.Random) -> dict[str, int]:
    return {"x": rng.randrange(PRIME), "y": rng.randrange(PRIME), "out": 0}


def _modmul_ref(v: dict[str, int]) -> dict[str, int]:
    return {"x": v["x"], "y": v["y"], "out": modmul_ref(v["x"], v["y"])}


@circuit_version("mul.modmul.v1")
def _build_modmul_v1() -> Case:
    """
    out <- x*y mod p, lazily reduced. Product, fold, uncompute the product.

    The uncomputation is not overhead that better engineering removes: the fold is
    many-to-one, so the product register cannot be reduced in place and cannot be freed
    any other way. Every modular multiply in the literature pays this.
    """
    return _binary_case(lambda c, x, y, o: modmul(c, x, y, o),
                        _modmul_ref, _modmul_gen, FOLD2_OVERFLOW)


@circuit_version("mul.modmul_trunc.v1")
def _build_modmul_trunc_v1() -> Case:
    """
    Same, with the fold's second round confined to CARRY_WINDOW bits.

    Only the round-2 `+F*w_hi` terms are windowed -- round 1 adds a full-width operand
    and has no window to speak of. Saves ~3*(U-CARRY_WINDOW) Toffoli out of ~2U^2, i.e.
    almost nothing, which is itself the finding: at this layer the fold is not the cost.
    """
    return _binary_case(
        lambda c, x, y, o: modmul(c, x, y, o, window=CARRY_WINDOW),
        _modmul_ref, _modmul_gen,
        FOLD2_OVERFLOW + 3 * F / float(1 << CARRY_WINDOW))


@circuit_version("mul.modmul_canonical.v1")
def _build_modmul_canonical_v1() -> Case:
    """
    out <- x*y mod p, CANONICALLY reduced into [0, p) rather than [0, 2^U).

    This is the answer to the trade doc 05 left open. Canonicalising costs a comparison
    and a conditioned `+F` -- ~760 Toffoli, 0.6% -- and in exchange every canonical-input
    obligation Layer 0 handed forward is discharged: `modneg` (42 Toffoli) and
    `moddouble` (45) become usable, and `modadd`'s accumulator hazard goes away. At 0.6%
    the lazy-throughout schedule has to justify itself, not the other way round.

    Not free of consequences: the flag erasure is many-to-one, so it fails DIRTY at
    F/2^U = 2^-250.8. Venting it would move that to the phase class.
    """
    return _binary_case(
        lambda c, x, y, o: modmul(c, x, y, o, canonical=True),
        lambda v: {"x": v["x"], "y": v["y"],
                   "out": modmul_ref(v["x"], v["y"]) % PRIME},
        _modmul_gen,
        FOLD2_OVERFLOW + canonicalise_fail_prob())


@circuit_version("mul.modmul_karatsuba.v1")
def _build_modmul_karatsuba_v1() -> Case:
    """
    out <- x*y mod p, one Karatsuba level fused with the reduction.

    Three quarters of the partial products, and the fusion keeps that saving through the
    compute/uncompute doubling. The question it is registered to answer is not the Toffoli
    count but the VOLUME: it holds z0, z2, c, two half sums and the assembled 2U register
    live simultaneously, where the schoolbook path holds one 2U product. Read
    `circuit_volume`, not `toffoli`.
    """
    return _binary_case(lambda c, x, y, o: modmul_karatsuba(c, x, y, o),
                        _modmul_ref, _modmul_gen, FOLD2_OVERFLOW)


@circuit_version("mul.modsqr.v1")
def _build_modsqr_v1() -> Case:
    """
    out <- x^2 mod p. Half of `mul.modmul.v1`, for the same fold.

    Worth registering separately rather than calling `modmul(x,x)`: the point formulas
    are written in terms of squarings, and a design that spends a general multiply where
    a squaring would do is paying twice for nothing.
    """
    return _unary_case(
        lambda c, x, o: modsqr(c, x, o),
        lambda v: {"x": v["x"], "out": modmul_ref(v["x"], v["x"])},
        lambda rng: {"x": rng.randrange(PRIME), "out": 0},
        FOLD2_OVERFLOW)


@circuit_version("mul.modmul_const_a24.v1")
def _build_modmul_const_a24_v1() -> Case:
    """
    out <- 121665 * x mod p, the Montgomery ladder's only constant multiplication.

    NAF weight 7 against a full multiply's 255 rows, and no 2U-bit product register to
    build and unbuild. This is the cheapest non-trivial circuit in the layer and it is
    the concrete reason an x-only ladder is worth costing at Layer 2.
    """
    return _unary_case(
        lambda c, x, o: modmul_const(c, x, A24, o),
        lambda v: {"x": v["x"], "out": modmul_const_ref(v["x"], A24)},
        lambda rng: {"x": rng.randrange(PRIME), "out": 0},
        modmul_const_overflow_prob(A24))


@circuit_version("mul.modmul_const_f.v1")
def _build_modmul_const_f_v1() -> Case:
    """
    out <- F * x mod p. The reduction constant itself, priced as a circuit.

    Registered because it is the unit the whole fold is built from: `weight(F)` adds --
    3 for 25519's F = 19, 5 for secp256k1's NAF-weight-5 F = 2^32 + 977. Running this one
    row under both `QSM_FIELD` values is the field-level advantage in its most direct form.
    """
    return _unary_case(
        lambda c, x, o: modmul_const(c, x, F, o),
        lambda v: {"x": v["x"], "out": modmul_const_ref(v["x"], F)},
        lambda rng: {"x": rng.randrange(PRIME), "out": 0},
        modmul_const_overflow_prob(F))


#: The reduction constant's shape, per field. The fold costs one shifted add per set bit,
#: so this is the one place F's *weight* (as opposed to its width) is pinned down.
_F_SHAPE = {19: [0, 1, 4],                       # 2^4 + 2^1 + 2^0
            (1 << 32) + 977: [0, 4, 6, 7, 8, 9, 32]}   # 2^32 + 977
assert set_bits(F) == _F_SHAPE[F], f"F = {F} decomposes as {set_bits(F)}"
