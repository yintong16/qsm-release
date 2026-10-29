"""Pseudo-Mersenne reduction: a 2U-bit product down to a U-bit field element.

Everything here rests on one congruence:

    2^U == F   (mod p),        p = 2^U - F

so splitting a value `v = v_lo + 2^U * v_hi` gives `v == v_lo + F*v_hi (mod p)`, and a
reduction is a MULTIPLICATION BY A SMALL CONSTANT plus an addition. F's *bit weight* is
therefore the price of a reduction, and that is where 25519 and secp256k1 separate:

    25519       F = 19          = 2^4 + 2^1 + 2^0            3 shifted adds
    secp256k1   F = 2^32 + 977  (NAF weight 5)               5 shifted adds

with each add running the width of the register. The 28-bit gap in log2(F) that
`qsm/params.py` prices in the carry window shows up a second time here, as a term count.

The fold cannot be done in place
--------------------------------
`v -> v_lo + F*v_hi` maps 2U bits onto U bits, so it is many-to-one and NO reversible
circuit computes it in place. The reduced value has to be written into fresh space and
the source uncomputed by whoever produced it. That is not an inefficiency to be
engineered away, it is the same obstruction as canonicalisation
(docs/decisions/04-no-in-circuit-canonicalization.md), and it is the reason a modular
multiply costs 2x its product: compute, reduce OUT, uncompute.

Two rounds, and exactly two
---------------------------
Round 1 sends a 2U-bit product to `w < (F+1)*2^U`, i.e. U + ceil(log2(F+1)) bits.
Round 2 sends `w` to `w_lo + F*w_hi <= 2^U - 1 + F^2`, which is U bits plus a whisker.
A third round would be needed only for that whisker, and it is dropped instead -- see
`FOLD2_OVERFLOW` below. The two-round bound matches the closed form and the exhaustive
small-width check in `docs/decisions/03-approximate-arithmetic.md`: the fold converges to
exactly `2^U - 1`, which is the LAZY contract Layer 0 already speaks.
"""

from __future__ import annotations

from ..core.circuit import Circuit, Register
from ..field.add import gidney_add
from ..field.compare import carry_out_const
from ..field.constadd import add_const_inplace
from ..field.sub import sub
from ..params import F, U


def set_bits(c: int) -> list[int]:
    """Positions of the set bits of a non-negative classical constant, low to high."""
    return [i for i in range(c.bit_length()) if (c >> i) & 1]


def naf(c: int) -> list[tuple[int, int]]:
    """Non-adjacent form of `c` as [(shift, sign), ...] with sign in {+1, -1}.

    Minimises the number of shifted add/subtract terms, which is exactly the cost of a
    multiplication by a classical constant. For F = 19 the NAF is `2^4 + 2^2 - 2^0`,
    weight 3 -- the same as its binary weight, so 25519's reduction constant gains
    nothing from NAF. It matters for the curve constants (Layer 2), not for F.
    """
    out: list[tuple[int, int]] = []
    i = 0
    while c:
        if c & 1:
            d = 2 - (c % 4)          # +1 or -1, chosen so c-d is divisible by 4
            out.append((i, d))
            c -= d
        c >>= 1
        i += 1
    return out


def add_times_const(
    circ: Circuit,
    v: Register,
    target: Register,
    c: int,
    offset: int = 0,
    add_fn=gidney_add,
    window: int | None = None,
    inverse: bool = False,
    use_naf: bool = False,
) -> None:
    """target += c * v * 2^offset, for CLASSICAL c and quantum v. One add per term of c.

    Cost is `weight(c)` additions -- the entire reason a multiply by a curve constant is
    cheap and a multiply by a quantum register is not. Terms are emitted low-shift-first
    (and reversed when `inverse`), so the routine is its own inverse under `inverse=True`.

    `use_naf` spends subtractions to lower the term count; it never helps for F = 19
    (NAF weight equals binary weight, 3) and does help for the wide curve constants.

    `window` caps each term's target slice at `window` bits, dropping the carry beyond it.
    That is the usual truncation trade: it only makes sense when `v` is much narrower than
    `target`, which is the case for the second fold round (a <=5-bit `v` into 255 bits).
    """
    terms = naf(c) if use_naf else [(s, 1) for s in set_bits(c)]
    if inverse:
        terms = terms[::-1]
    for s, sign in terms:
        lo = offset + s
        if lo >= len(target):
            continue
        hi = len(target) if window is None else min(len(target), lo + window)
        slice_ = target[lo:hi]
        operand = v if len(v) <= len(slice_) else v[0:len(slice_)]
        if (sign > 0) != inverse:
            add_fn(circ, operand, slice_)
        else:
            sub(circ, operand, slice_, add_fn)


def fold_width(u: int = U, f: int = F) -> int:
    """Width of the round-1 accumulator: `v_lo + f*v_hi < (f+1)*2^u` needs this many bits."""
    return u + f.bit_length()


def pm_fold(
    circ: Circuit,
    prod: Register,
    out: Register,
    u: int = U,
    f: int = F,
    add_fn=gidney_add,
    window: int | None = None,
    canonical: bool = False,
) -> None:
    """out <- (prod mod p) lazily, for `p = 2^u - f`. `prod` is 2u bits and PRESERVED.

    `out` must be u bits and arrive |0>. Roughly `2*(f_weight+1)*u` Toffoli, about 2% of
    the multiply that produced `prod`.

    Round 1 builds `w = P_lo + f*P_hi` in a borrowed `u + bitlen(f)`-bit register; round 2
    copies `w`'s low u bits into `out` (free -- CNOTs onto |0>) and adds `f*w_hi` on top;
    then round 1 is reversed against `prod`, which is still live. `out` survives because
    it is never an input to the uncomputation.

    APPROXIMATE, at `FOLD2_OVERFLOW` = f^2/2^u: round 2 can produce `w_lo + f*w_hi >= 2^u`
    (needs `w_lo` within f^2 of the top), and the carry out of bit u-1 has nowhere to go.
    Dropping it loses 2^u == f, so the result is NON-CONGRUENT, not merely lazy. At
    2^-246.5 for 25519 that is four orders below the lazy contract's own 2^-250.8 and is
    accepted deliberately; the alternative -- recomputing the carry as an AND over the
    result's top u-9 bits -- costs ~u Toffoli to turn a non-congruent value into a DIRTY
    ANCILLA at the same rate, which is the worse class (03-approximate-arithmetic.md).
    """
    n = len(prod)
    assert n == 2 * u, "the fold consumes a full 2u-bit product"
    assert len(out) == u, "the folded output is u bits (lazy: [0, 2^u), not [0, p))"
    lo, hi = prod[0:u], prod[u:2 * u]
    w = circ.alloc(fold_width(u, f))

    add_fn(circ, lo, w)                                        # round 1: w = P_lo ...
    add_times_const(circ, hi, w, f, add_fn=add_fn)             # ... + f * P_hi

    for j in range(u):                                         # round 2: free copy
        circ.cx(w[j], out[j])
    add_times_const(circ, w[u:], out, f, add_fn=add_fn, window=window)
    if canonical:
        canonicalise(circ, out, u, f, window)

    add_times_const(circ, hi, w, f, add_fn=add_fn, inverse=True)   # uncompute round 1
    sub(circ, lo, w, add_fn)
    circ.free(w)


def canonicalise(circ: Circuit, out: Register, u: int = U, f: int = F,
                 window: int | None = None) -> None:
    """out <- out mod p, turning the LAZY contract [0, 2^u) into the canonical [0, p).

    ~2u Toffoli: one carry chain to decide `out >= p`, one conditioned constant add to
    act on it. Subtracting p needs no wide constant -- on a u-bit register
    `out - p == out + f (mod 2^u)`, so the whole reduction is the same `+F` the rest of
    the layer is built from. `carry_out_const(out, f)` answers `out + f >= 2^u`, which
    IS `out >= p`, so even the comparison is the pseudo-Mersenne shape.

    Why it is priced here rather than being the default
    ---------------------------------------------------
    `docs/decisions/05-optimisation-target-and-inversion.md` left this as Layer 1's first
    real trade and guessed it at "~250 Toffoli, ~10% of a schoolbook multiply". The
    denominator was wrong: measured, this is ~760 Toffoli against `mul.modmul.v1`'s
    135,417, i.e. **0.6%**. Canonicalising once per multiply discharges every
    canonical-input obligation in that document's table -- `modneg`, `moddouble`, and the
    canonical `modadd` accumulator all become usable -- for well under one percent.

    THE FLAG CANNOT BE ERASED CLEANLY, and that is a theorem, not a gap. `flag` is
    exactly `out_new < f` after the add, except when the input was already below f and no
    reduction happened; those two cases are indistinguishable in the output, so the map
    (flag, out) -> out is many-to-one. This is `docs/decisions/04-no-in-circuit-
    canonicalization.md` showing up at the smallest possible scale, and it is why this
    routine is only legal where `out` is being written into FRESH space -- as it is
    inside `pm_fold`, whose source register is uncomputed afterwards anyway.

    The residual is `f/2^u = 2^-250.8` and its class is DIRTY ANCILLA, which the project
    ranks below a non-congruent value. Venting it (measure the flag, repair the phase with
    the same predicate) would move it to the phase class for one fewer Toffoli, exactly as
    `modadd_vented.v3` does; that needs a phase-mode `carry_out_const`, which does not
    exist yet. Declared rather than hidden: the registered versions carry it in
    `fail_analytic`.
    """
    n = len(out)
    flag = circ.alloc(1)
    carry_out_const(circ, out, f, flag[0])              # flag = (out + f >= 2^u) = out >= p
    add_const_inplace(circ, out, f, ctrl=flag[0], window=window)   # out -= p when flagged
    carry_out_const(circ, out, (1 << n) - f, flag[0])   # flag ^= (out >= f)
    circ.x(flag[0])                                     # ... so flag ^= (out < f): erased
    circ.free(flag)


def canonicalise_fail_prob(u: int = U, f: int = F) -> float:
    """P(`canonicalise` leaves its flag dirty) = f/2^u.

    The erasure reads `out < f`, which is the reduction's signature except on inputs that
    were already below f without needing one -- exactly f of the 2^u values.
    """
    return f / float(1 << u)


# --- exact semantics -------------------------------------------------------
def fold_once_ref(v: int, u: int = U, f: int = F) -> int:
    """One reduction round: `v_lo + f*v_hi`, congruent to v mod p and strictly smaller."""
    return (v & ((1 << u) - 1)) + f * (v >> u)


def pm_fold_ref(prod: int, u: int = U, f: int = F) -> int:
    """Exact semantics of `pm_fold`, INCLUDING the dropped round-2 carry.

    Two rounds, then a truncation to u bits. The truncation is the circuit's declared
    approximation, so it belongs in the reference rather than being hidden by a `% p`:
    a reference that reduced properly would silently absolve the very failure the
    version's `fail_analytic` prices.
    """
    return fold_once_ref(fold_once_ref(prod, u, f), u, f) & ((1 << u) - 1)


def fold2_overflow_prob(u: int = U, f: int = F) -> float:
    """P(`pm_fold`'s second round carries out of bit u-1 and the carry is dropped).

        P = f^2 / 2^u          (an upper bound; the mean over w_hi is about half that)

    Round 1 leaves `w < (f+1)*2^u`, so `w_hi <= f` and round 2 adds at most `f^2` to a
    u-bit `w_lo`. Overflow needs `w_lo >= 2^u - f*w_hi`, i.e. `w_lo` in the top `f*w_hi`
    values, and `f*w_hi <= f^2 = 361` for 25519.

    Consequence class: NON-CONGRUENT. The lost bit is worth `2^u == f (mod p)`, so the
    result is short by exactly f -- no downstream reduction repairs it. Compare
    `lazy_unreduced_prob` (2^-250.8), which is congruent and therefore not an error at
    all. This one is a real error, four orders rarer.

        25519:      361 / 2^255           = 2^-246.5
        secp256k1:  (2^32+977)^2 / 2^256  = 2^-190.0

    The 56-bit gap is 2*log2(F) and is the same field-level advantage that
    `qsm/params.py` prices in the carry window, appearing here squared because the
    constant is applied twice.
    """
    return (f * f) / float(1 << u)


FOLD2_OVERFLOW = fold2_overflow_prob()
