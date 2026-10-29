"""Reduce-as-you-accumulate modular multiplication: no product register, ever.

`modmul.py` builds the full 2U-bit product, folds it once, and uncomputes it. That shape
is forced *only* if you insist on having a product. This module takes the other road --
qarton's `ModularProductAdd` (double-and-add with a modular accumulator) -- where the
value never grows past U bits and there is nothing to uncompute.

    modmul.py     x*y into 2U bits, fold once,  uncompute      peak ~ 4U + carries
    this module   u rows of `z = 2z +- y (mod p)`              peak ~ 3U + carries

Why this is a 25519 circuit and not a generic one
-------------------------------------------------
Reducing on every row means paying a fold on every row, and that is only affordable
because F = 19:

    25519      F = 19          fold = 3 shifted adds, lands at U+5 bits
    secp256k1  F = 2^32 + 977  fold = 5 shifted adds, lands at U+33 bits

ecdsa.fail builds the wide product precisely because their fold is too expensive and too
weakly contracting to run 256 times. Ours is neither. The cheap reduction is not a
cheaper line item, it is a licence to change the schedule -- and the schedule is where
the peak lives. Concretely, our per-row double is `field.moddouble_trunc.v1` at 45
Toffoli, against qarton's generic `ModularDouble` (a full `-p` plus a controlled `+p`,
~2U) -- an ~11x gap, taken 255 times.

Removing the control, which is where the factor of two is
---------------------------------------------------------
qarton's row is a CONTROLLED modular add. ecdsa.fail's own floor table says an
uncontrolled n-bit add costs n-1 and is TIGHT, while a controlled one costs n at the
floor and 2n in practice. So the row is paying the factor-two gap 255 times.

The add/subtract trick removes it: add `+y` when `x_i = 1` and `-y` when `x_i = 0`, an
UNCONDITIONAL add either way, with a closed-form aggregate correction. In the modular
domain the sign flip is free, because complementing all U bits conjugates addition into
subtraction:

    N(z) = 2^U - 1 - z == F - 1 - z   (mod p)
    N(N(z) + y)        == F - 1 - (F - 1 - z + y) == z - y   (mod p)

The `F-1` cancels, so N is an involution mod p that turns `+y` into `-y` with a layer of
X gates and no arithmetic at all. Conditioning it on `x_i` costs one CX layer:

    X layer on z            unconditional N
    CX layer from x_i       undoes it when x_i = 1

which is N exactly when `x_i = 0`, and NEVER TOUCHES `x_i`. That last point is what lets
the square reuse `y`'s own bits as the controls with no ancilla, where qarton needs one
(`modular_product.py:730`) because its version flips the control qubit.

The aggregate correction closes classically
-------------------------------------------
Rows contribute `s_i = 2*x_i - 1`, so

    sum_i s_i 2^i y  =  y * (2x - (2^U - 1))        =: S
    =>   x*y = (S + (2^U - 1) * y) / 2              (mod p)

and `(2^U - 1) == F - 1 = 18 (mod p)` -- tiny only because F is. So every control in the
circuit is bought back by ONE multiply-by-18 (four doublings and two adds) and ONE
modular halving, together ~1% of the multiply. Verified exhaustively at six (u, F, p) in
`tests/test_mul_accumulate.py`.

What it costs, and the one place it loses
-----------------------------------------
    rows        (U-1) * modadd_lazy      254 * 339
    top row     1 controlled +(F-1)               ~45   <- into a FRESH target; see below
    doubles     (U-1) * moddouble_trunc  254 *  45
    correction  4 dbl + 4 halve + 2 add       ~1038
    halve       1                              ~45

Against `mul.modmul.v1` (135,417 Toffoli, peak 1,794) this trades ~30% fewer Toffoli for
~40% less peak. It does NOT halve for squaring: `modsqr_acc` costs the same as
`modmul_acc`, because every bit of the operand still drives a row, whereas the schoolbook
square is structurally a triangle. So the schoolbook path may well keep the square while
this one takes the general product -- which is exactly what the benchmark is for.

Approximation budget
--------------------
THREE terms, and they span 210 orders. Only the last one matters -- but the middle one is
the one that nearly went unnoticed, so it is written up first.

1.  NON-CLOSURE, ~2^-243.8. `moddouble` is the one Layer-0 circuit not closed under the
    lazy contract (`moddouble_lazy_overflow_prob` = 9/2^255 = 2^-251.8 per call, a
    NON-CONGRUENT wrap), and `modhalve` mirrors it at the other end of the register with
    the same 9. 259 exposed scalings. `scaling_chain_prob`.

2.  VENTED ERASURE, ~2^-242.8, and this one is PHASE. Every `modadd_lazy` here is handed a
    LAZY accumulator -- the rows hand it `2^u - 1 - z`, the correction hands it `z` -- and
    `modadd_lazy` erases its overflow flag against a `b < a` predicate that assumes the
    accumulator was canonical. It is wrong at `f/2^u` per call and it fails as a relative
    phase, which no comparison of OUTPUT VALUES can see. `vent_phase_prob`.

    That rate is only true because of `_first_row_from_zero`. Before it, the top row
    complemented a fresh |0> target into `2^u - 1` -- the worst value in the register, on
    every single shot -- and the multiply leaked phase on ~26% of inputs while returning
    the right answer. It survived a green 83-test suite (which simulated with an all-zero
    measurement vector, under which the check is vacuous) and was caught by the benchmark.
    The repair is also 510 Toffoli CHEAPER than the row it replaces, which is the usual
    sign that the original was doing something it did not need to do.

    It is not free, though, and the price is not a gate count: that row is `modneg`, so it
    needs `y <= p` where the `modadd_lazy` row it replaced accepted a lazy operand. See
    `_first_row_from_zero` for the exact failure set (the F-1 = 18 values above p) and for
    why doc 05 section 4 is what pays for it.

3.  TRUNCATION, ~2^-31.9, and the only term that binds. Reducing on every row means ~520
    windowed primitives instead of `mul.modmul_trunc.v1`'s handful, so at the same
    CARRY_WINDOW this reports 2^-31.9 against modmul's 2^-39.2 -- SEVEN BITS WORSE for a
    2.4x volume win. `truncation_prob`. Nothing here hides that: it is the arithmetic
    consequence of 255 reductions, it is bought back by ~7 more window bits at ~4% of the
    circuit, and the choice belongs to whoever knows the target rate.
"""

from __future__ import annotations

import functools
import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import _and, _unand, gidney_add
from ..field.add_chunked import gidney_add_chunked
from ..field.constadd import add_const_inplace, sub_const_inplace
from ..field.modarith import (AddFn, modadd_lazy, moddouble,
                              moddouble_inplace, modsub_direct)
from ..params import (CARRY_WINDOW, CMP_BITS, F, PRIME, SQUARE_RIPPLE, U, cmp_fail_prob,
                      controlled_carry_fail_prob, modadd_lazy_vent_phase_prob,
                      moddouble_lazy_input_carry_fail_prob,
                      moddouble_lazy_overflow_prob, modhalve_borrow_fail_prob,
                      modhalve_underflow_prob, narrow_row_carry_fail_prob)


# ---------------------------------------------------------------------------
# Modular halving -- the exact mirror of `moddouble`, and the piece Layer 0 lacked.
# ---------------------------------------------------------------------------
def modhalve(circ: Circuit, a: Register, p: int = PRIME,
             window: int | None = None) -> Register:
    """a <- a/2 mod p. Returns the result register (RELABELLED, like `moddouble`).

    The mirror image of `moddouble`, and free for the same reason -- the shift is a
    renaming of wires, so the entire cost is one controlled constant `-F`.

        z even:   z/2                    a right shift
        z odd:    (z + p)/2 = (z - F)/2 + 2^(U-1)

    F is odd, so `z - F` is even exactly when `z` is odd: subtracting F conditioned on
    the parity makes the value even in both branches, and the shift is then exact. The
    parity bit that controlled the subtraction becomes the new TOP bit, which is where
    the `+ 2^(U-1)` comes from -- it is not an extra addition, it is the same qubit
    read at the other end of the register.

    Ancilla accounting is exactly neutral: bit 0 is provably |0> after the subtraction
    and is freed, and the parity copy takes its place at the top. Net zero qubits.

    NOT CLOSED UNDER THE LAZY CONTRACT -- it inherits `moddouble`'s non-closure at the
    other end of the register, and by the same count. This circuit is the exact inverse of
    `moddouble`, so it is correct exactly on `moddouble`'s image, and that image is missing
    the odd values below F: for odd `z < F` the `-F` borrows out of the register and the
    result comes back too large by `2^(U-1)`. `modhalve_underflows` is the predicate,
    `modhalve_underflow_prob` the rate -- 9/2^255 = 2^-251.8, the SAME 9 as doubling.

    An exact halving on the full lazy range does not exist at this width, and the reason is
    counting rather than cleverness: `z` and `z + p` are congruent and halve to values `p`
    apart, so (F-1)/2 pairs of inputs would have to share an output. Reversibility forbids
    it. The choice is therefore between a wrong representative with clean ancillas (this)
    and the right representative with a dirty one (an extra top qubit carrying the borrow,
    which then cannot be erased). The first is cheaper and composes with the rest of the
    lazy accounting; the second turns a value error into a phase error, which is worse.

    A halving that UNDOES a doubling cannot trip this, since the doubling's image excludes
    the danger set by construction. That is why the four halvings in `_add_correction` are free
    of it and only the accumulator halvings pay.
    """
    res, low = _modhalve_core(circ, a, p, window)
    circ.free(Register([low]))
    return res


def _modhalve_core(circ: Circuit, a: Register, p: int,
                   window: int | None) -> tuple[Register, int]:
    """Shared body: returns (result register, the now-|0> low qubit, NOT yet freed).

    The low qubit is handed back rather than freed so `modhalve_inplace` can use it as a
    move target while it is still legitimately owned. This mirrors `_moddouble_core`, and
    for exactly the reason its docstring gives: freeing first and writing afterwards
    happens to work only while nothing else allocates. Something does -- `_add_correction`
    scales `y` in place and halves it back, and the recycled qubit corrupted both the
    value and the ancilla. The first caller that allocates, as predicted.
    """
    n = len(a)
    assert p < (1 << n), "p must fit in n bits"
    assert n >= 2, "need at least two bits to shift"
    f = (1 << n) - p
    assert f % 2 == 1, "F must be odd for the parity trick (p odd)"

    par = circ.alloc(1)
    circ.cx(a[0], par[0])                                   # parity, out of the way
    sub_const_inplace(circ, a, f, ctrl=par[0], window=window)
    # `a` is now even, so a[0] is |0> and can be recycled; the shift is a relabelling.
    return Register(list(a.ids[1:]) + [par[0]], a.name), a.ids[0]


def modhalve_inplace(circ: Circuit, a: Register, p: int = PRIME,
                     window: int | None = None) -> None:
    """`modhalve`, with the result slid back onto the qubits `a` came in on.

    A rotation the other way from `moddouble_inplace`, and the same 2-CNOT moves: each
    destination has already been vacated, so no SWAP is needed. 2U CNOTs, zero Toffoli.
    """
    n = len(a)
    res, _low = _modhalve_core(circ, a, p, window)   # a[0] stays OWNED; it is a target
    # res holds [a_1, ..., a_{n-1}, par]; we want those values on [a_0, ..., a_{n-1}].
    # Walk UPWARDS: a_0 was just zeroed, and every step vacates the next destination.
    for i in range(n):
        src, dst = res[i], a[i]
        if src == dst:
            continue
        circ.cx(src, dst)      # dst is provably |0>, so this COPIES
        circ.cx(dst, src)      # ...and this CLEARS the source. 2 CNOTs, not 3.
    circ.free(Register([res[n - 1]]))   # the parity qubit, now vacated


# ---------------------------------------------------------------------------
# The row: an UNCONDITIONAL modular add, signed by a CNOT layer.
# ---------------------------------------------------------------------------
def _signed_row(circ: Circuit, y: Register, z: Register, ctrl: int,
                p: int, add_fn: AddFn, carry_window: int | None,
                cmp_bits: int | None, negate: bool = False) -> None:
    """z <- z + y (mod p) if `ctrl` is |1>, else z <- z - y (mod p). `y`, `ctrl` preserved.

    The X layer complements `z` unconditionally and the CX layer undoes that when
    `ctrl = 1`, so the pair is `N` exactly on the `ctrl = 0` branch. Both layers are
    Clifford, so the ENTIRE cost of this row is one uncontrolled `modadd_lazy` -- the
    control is free, which is the whole point of the module.

    `ctrl` is only ever read. That is what lets `modsqr_acc` pass a bit of `y` itself.

    `negate` drops the X layer, leaving the CX layer to apply `N` on `ctrl = 1` instead of
    `ctrl = 0`: the row becomes `-y` when the control is set and `+y` when it is not. The
    whole signed sum flips, which is how `modsqrsub_inplace` subtracts a square without
    computing one first. It costs nothing -- `u` X gates fewer, in fact.
    """
    if not negate:
        for q in z:
            circ.x(q)
    for q in z:
        circ.cx(ctrl, q)
    modadd_lazy(circ, y, z, p, add_fn, carry_window, cmp_bits)
    for q in z:
        circ.cx(ctrl, q)
    if not negate:
        for q in z:
            circ.x(q)


def _first_row_from_zero(circ: Circuit, y: Register, z: Register, ctrl: int,
                         p: int, window: int | None) -> None:
    """z (which MUST arrive |0>) <- y if `ctrl` else -y (mod p). y, ctrl preserved.

    The top row of a multiply into a FRESH target, special-cased. It is not an
    optimisation that happened to be available -- the general `_signed_row` is WRONG here,
    deterministically, and this is the repair.

    Why the general row breaks. `_signed_row` complements the accumulator, handing
    `modadd_lazy` a `b` of `2^u - 1 - z`. That exceeds `p` exactly when `z <= f - 1`, and
    `modadd_lazy`'s vented erasure needs `b < p` (`modadd_lazy_vent_phase_prob`). For a
    running accumulator `z <= f-1` has probability `f/2^u = 2^-250.8` and is a rounding
    error in the budget. On the FIRST row `z` is not running -- it is |0>, so
    `2^u - 1 - z` is `2^u - 1`, the single worst value in the register, on EVERY shot.
    The sum is then correct (step 2 does not consult the flag) but the flag is erased
    against a false predicate, so the multiply leaks a relative phase whenever the top
    control bit is 0 and the measurement comes back 1: ~25% of all inputs, silent to any
    check that compares values. Measured at 12/40 before this function existed.

    Why it is also CHEAPER, which is the part worth keeping. Adding into |0> is a copy, so
    the row's arithmetic collapses to a sign:

        ctrl = 1    z <- y                    255 CNOTs, ZERO Toffoli
        ctrl = 0    z <- p - y = ~(y + f - 1) one CONTROLLED constant add of f-1

    That is `modneg`, whose constant is small for the same reason the whole module works:
    f = 19, so f-1 = 18 has two set bits. A full `modadd_lazy` is replaced by a controlled
    `+18`.

    PRECONDITION, and it is a genuine NARROWING -- do not read the paragraph above as a
    free lunch. `modneg` needs `y <= p`; `_signed_row` did not, because `modadd_lazy`
    accepts a lazy `a`. So this row buys 510 Toffoli at the price of an operand contract:

        y <= p          required here, NOT required by the row it replaces

    Above `p` the constant add wraps (`y + f - 1 >= 2^u`) and the result is congruent to
    `f - y` rather than `-y`. The damage is exactly the `f - 1` values in `(p, 2^u)` -- 18
    of 2^255, verified exhaustively at four widths against the `bit = 0` branch, and zero
    failures for every `y <= p`. Restructuring does not help: `p - y` is negative for
    `y > p`, so no `u`-bit register can hold it and only a modular add can absorb it.

    What supplies the precondition is doc 05 section 4, which settled canonicalisation at
    the fold (763 Toffoli, 0.56% of a multiply) precisely so downstream circuits could use
    the cheap `modneg`-class primitives. This row is one of them. A caller that feeds this
    a LAZY operand -- straight from an unreduced fold, say -- is outside the contract, and
    `test_the_zero_row_narrows_the_operand_contract_to_canonical` pins the exact set so the
    narrowing cannot be quietly forgotten or quietly "fixed".

    Both conditionals ride the same Clifford trick as `_signed_row` and neither writes
    `ctrl`, so `modsqr_acc` can still drive this from a bit of `y` itself.
    """
    n = len(y)
    assert len(z) == n, "operand and target must match"
    f = (1 << n) - p
    for i in range(n):
        circ.cx(y[i], z[i])                  # z = y. Exact: z was |0>.
    # --- conditional modneg, applied iff ctrl = 0 ---
    circ.x(ctrl)                             # so the constant add fires on ctrl = 0 ...
    add_const_inplace(circ, z, f - 1, ctrl=ctrl, window=window)
    circ.x(ctrl)                             # ... and ctrl comes back untouched.
    for q in z:                              # complement unconditionally ...
        circ.x(q)
    for q in z:                              # ... and undo it when ctrl = 1.
        circ.cx(ctrl, q)


def _add_correction(circ: Circuit, y: Register, z: Register, p: int, add_fn: AddFn,
                    carry_window: int | None, cmp_bits: int | None,
                    window: int | None) -> None:
    """z += (2^n - 1) * y == (F - 1) * y (mod p). The aggregate correction, in full.

    For 25519 the constant is 18 = 0b10010, so `y` is doubled in place and added into `z`
    at bit 1 and bit 4, then halved back. There is no scratch register and nothing to
    erase: the halvings exactly invert the doublings on the whole lazy range, wrap
    included, so `y` returns bit-for-bit rather than by an erasure argument.

    These two adds are the ONLY lazy+lazy additions in the schedule -- `2^k y` is lazy and
    so is the accumulator -- so they are the only place `modadd_lazy`'s 2^-502.6 hole is
    exposed. The rows are all canonical+lazy and closed.

    Cost is set by the BINARY WEIGHT AND LENGTH OF F-1, not by the field size:

        F = 19   F-1 = 18 = 0b10010    4 doublings, 2 adds, 4 halvings
        F = 3    F-1 =  2 = 0b10       1 doubling,  1 add,  1 halving

    On a curve with a large F this constant would be a dense multi-hundred-bit multiply
    and the correction would cost more than the 255 controls it buys back -- another place
    the schedule is specific to a small pseudo-Mersenne F, and the reason this is not a
    generic trick with a 25519 flavour but a 25519 trick.
    """
    n = len(y)
    c = ((1 << n) - 1) % p                        # == F - 1
    if c == 0:                                   # F = 1: nothing to correct
        return
    hi = c.bit_length() - 1

    # No scratch copy. `y` itself is scaled up and scaled back down, which is exact
    # because halving inverts doubling on the whole lazy range -- so `y` is restored
    # bit-for-bit and an entire u-qubit register is saved off the peak. The in-place
    # variants cost 2u CNOTs apiece and no Toffoli; buying 255 qubits of peak for
    # ~4k Cliffords is the trade this whole module exists to make.
    for k in range(hi + 1):
        if (c >> k) & 1:                          # y currently holds 2^k * y_0
            modadd_lazy(circ, y, z, p, add_fn, carry_window, cmp_bits)
        if k < hi:
            moddouble_inplace(circ, y, p, window)
    for _ in range(hi):
        modhalve_inplace(circ, y, p, window)      # y = y_0 again, exactly


def _rows(circ: Circuit, ctrl_reg: Register, y: Register, z: Register,
          p: int, add_fn: AddFn, carry_window: int | None, cmp_bits: int | None,
          window: int | None, zero_target: bool = False) -> Register:
    """The double-and-add loop, MSB first. Returns `z` (RELABELLED by the doublings).

    `zero_target` asserts that `z` arrives |0>, which is a fact about the CALLER and not
    checkable here -- a register's value is quantum. It is not a hint: pass it and the top
    row becomes `_first_row_from_zero`, which is both correct and ~300 Toffoli cheaper;
    withhold it and that row leaks phase on ~25% of inputs. See `_first_row_from_zero`.
    """
    u = len(ctrl_reg)
    for i in range(u - 1, -1, -1):
        if zero_target and i == u - 1:
            _first_row_from_zero(circ, y, z, ctrl_reg[i], p, window)
        else:
            _signed_row(circ, y, z, ctrl_reg[i], p, add_fn, carry_window, cmp_bits)
        if i > 0:
            z = moddouble(circ, z, p, window)
    return z


def modmul_acc(circ: Circuit, x: Register, y: Register, out: Register,
               p: int = PRIME, add_fn: AddFn = gidney_add,
               carry_window: int | None = None, cmp_bits: int | None = None,
               window: int | None = None) -> Register:
    """out <- x*y mod p, lazily. x, y preserved; `out` MUST arrive |0>.

    Out-of-place, in qarton's sense (`qc_mul`: |x,y> -> |x,y,xy>), but built from the
    in-place-additive core, so no 2U-bit product is ever materialised.

    Returns the result register, RELABELLED: the U-1 doublings and the final halving are
    free shifts, so the value does not end on the wires `out` came in on. The caller must
    use the return value -- same contract as `moddouble`. In a ladder the shifts compose
    into one classically-tracked offset and never need to be emitted.

    `out` arriving |0> is what makes this the cheap case: a nonzero accumulator would
    have to be pre-scaled by 2^-(U-2) to survive the loop's doublings, costing U-2 extra
    halvings. That is `modmuladd`.
    """
    u = len(x)
    assert len(y) == u and len(out) == u, "operands and output must all be u bits"
    z = _rows(circ, x, y, out, p, add_fn, carry_window, cmp_bits, window,
              zero_target=True)
    _add_correction(circ, y, z, p, add_fn, carry_window, cmp_bits, window)
    return modhalve(circ, z, p, window)


def modsqr_acc(circ: Circuit, x: Register, out: Register,
               p: int = PRIME, add_fn: AddFn = gidney_add,
               carry_window: int | None = None, cmp_bits: int | None = None,
               window: int | None = None) -> Register:
    """out <- x^2 mod p, lazily. x preserved; `out` MUST arrive |0>. Returns `out`.

    The same loop with `x` driving both the controls and the addend -- legal only because
    `_signed_row` never writes to its control. One register cheaper than `modmul_acc` and
    otherwise identical, INCLUDING the cost: unlike the schoolbook square, this one does
    not halve. Every bit of the operand still drives a full-width row, so there is no
    triangle to exploit. Measured against `mul.modsqr.v1` rather than assumed.
    """
    u = len(x)
    assert len(out) == u, "output must be u bits"
    z = _rows(circ, x, x, out, p, add_fn, carry_window, cmp_bits, window,
              zero_target=True)
    _add_correction(circ, x, z, p, add_fn, carry_window, cmp_bits, window)
    return modhalve(circ, z, p, window)


def modmuladd(circ: Circuit, x: Register, y: Register, z: Register,
              p: int = PRIME, add_fn: AddFn = gidney_add,
              carry_window: int | None = None, cmp_bits: int | None = None,
              window: int | None = None) -> Register:
    """z <- z + x*y mod p, for an ARBITRARY incoming z. x, y preserved. Returns z.

    qarton's `ModularProductAdd` with `zero_input=False`. The loop multiplies whatever
    `z` starts with by 2^(U-1) and the final halving by another 2^-1, so the incoming
    value is pre-scaled by 2^-(U-2) to compensate -- U-2 modular halvings, which roughly
    doubles the circuit's doubling cost (though not its dominant row cost).

    Prefer `modmul_acc` wherever the target is fresh. This exists because a point formula
    that can accumulate into a live register saves an entire modular addition per term,
    and that trade is Layer 2's to make with these two numbers in hand.

    PRECONDITION, and it is a real one: `z` must not be |0>, or a value below `f`. This
    circuit cannot take the `_first_row_from_zero` shortcut, because `z` is arbitrary and
    its value is not knowable at build time -- so the top row is a general `_signed_row`,
    which needs `2^u - 1 - z_scaled < p`, i.e. `z_scaled > f - 1`. For a live accumulator
    that holds at `1 - 2^-250.8` and is priced in `MULADD_OVERFLOW`; for `z = 0` it fails
    on ~1/2 of inputs, and it fails as PHASE, which no output comparison can see. Calling
    this with a fresh target is therefore not merely wasteful (it pays u-2 pointless
    halvings) but wrong. Use `modmul_acc`, which is both correct and cheaper.
    """
    u = len(x)
    assert len(y) == u and len(z) == u, "operands and accumulator must all be u bits"
    for _ in range(u - 2):
        z = modhalve(circ, z, p, window)
    z = _rows(circ, x, y, z, p, add_fn, carry_window, cmp_bits, window)
    _add_correction(circ, y, z, p, add_fn, carry_window, cmp_bits, window)
    return modhalve(circ, z, p, window)


# ---------------------------------------------------------------------------
# The TRIANGLE: a square whose rows shrink, with the accumulator still U bits.
# ---------------------------------------------------------------------------
# `modsqr_acc` costs exactly what `modmul_acc` costs, and the module header says why:
# every bit of the operand drives a full-width row, so there is no triangle. The
# schoolbook square (`mul.modsqr.v1`) gets the triangle by materialising 2U bits. The
# identity below gets it with a U-bit accumulator, which is the point.
#
#     x^2 = sum_i x_i 2^(2i) + 2 * sum_{i<j} x_i x_j 2^(i+j)
#         = 2 * sum_j x_j A_j 2^j  +  D,      A_j := x mod 2^j,  D := sum_j x_j 2^(2j)
#
# so row `j`'s addend is `A_j` -- a `j`-bit PREFIX SLICE of `x`, addressable for free as
# `x[0:j]` with no gates and no shifting. Feed that through the same sign trick
# (`s_j = 2x_j - 1`, aggregate correction `C = sum_j A_j 2^j`) and the halving that
# `modmul_acc` ends on cancels against the identity's factor of 2:
#
#     sum_j x_j A_j 2^j = (S + C)/2,   S := sum_j s_j A_j 2^j,   C = 2^U x - 2D
#     =>  x^2 = S + C + D = S + 2^U x - D                     (VERIFIED exhaustively)
#
# Two things have to be true for this to be a saving rather than a re-arrangement, and
# both are, but only the first is obvious.
#
# 1.  A `j`-bit addend has to cost `j`, not `U`. `gidney_add` already accepts `len(a) <
#     len(b)`, but it still pays one Toffoli per bit of `b` -- the carry has to ripple to
#     the top to decide the modular reduction. The ripple is what has to go, and it can:
#     run the add on `z[0 : j + ripple]` and let the carry out of THAT fall off the end.
#     That is the same windowed-carry approximation `add_const_inplace` already makes,
#     at `2^-(ripple+1)` a row (`narrow_row_carry_fail_prob`).
#
# 2.  Dropping the ripple also drops the modular reduction -- and the row does not need
#     one. `z < 2^U` going in; the carry provably cannot reach bit U (it died at
#     `j + ripple`), so bits above are untouched and `z + A_j < 2^U` comes out for free.
#     The row is a PLAIN add: no `+F`, no comparator, no vented flag. That is 85 Toffoli
#     of per-row overhead gone on top of the width saving, and it is most of the win.
#
# Rows with `j + ripple >= U` cannot make that argument and fall back to `_signed_row`.
# There are exactly `ripple` of them, at the wide end where there was nothing to save.
#
# `D` is the piece that does not come free. `D` is 2U bits wide, so `D = D_lo + 2^U D_hi`
# and `2^U == F (mod p)` would want a multiply-by-19. It is dodged instead: `D_hi` is
# injected BEFORE the loop, where the loop's own U-1 doublings (plus one) scale it by
# `2^U` at no cost, and `2^U x` rides in on the same injection. Both halves of `D` are
# pure bit-spreads of `x`, so their registers are built by INTERLEAVING `x`'s own qubits
# with |0> pads -- no copy, no arithmetic, just an id list (`_spread_register`).
def _spread_register(circ: Circuit, x: Register, hi: bool) -> tuple[Register, Register]:
    """The register holding `D_lo` (or `D_hi`), built from `x`'s own qubits. Returns (reg, pad).

    `D = sum_j x_j 2^(2j)` is `x` with its bits spread onto the EVEN positions of a 2U-bit
    word. Split it at the fold point, `D = D_lo + 2^U * D_hi`:

        D_lo    x_j at position 2j       for 2j <= U-1
        D_hi    x_j at position 2j-U     for 2j >= U

    Neither half needs to be computed. A `Register` is an id list, so the half is just
    `x`'s qubits listed at the right indices with fresh |0> qubits in the gaps -- zero
    gates, and `x` is never copied. The pads are only ever read (`modsub_direct` preserves
    its subtrahend), so they come back |0> and the caller frees them.

    For U odd the two halves land on opposite parities -- `D_lo` on the even positions,
    `D_hi` on the odd -- which is why they can share one pad allocation across the two
    calls even though they cannot share a register.
    """
    u = len(x)
    h = (u + 1) // 2                       # j < h -> D_lo, j >= h -> D_hi
    slot = ({2 * j - u: x[j] for j in range(h, u)} if hi
            else {2 * j: x[j] for j in range(h)})
    pad = circ.alloc(u - len(slot))
    spare = iter(pad)
    # `slot[i] if i in slot` -- NOT `slot.get(i) or ...`, which would swallow qubit id 0.
    return Register([slot[i] if i in slot else next(spare) for i in range(u)]), pad


def _narrow_row(circ: Circuit, y: Register, z: Register, ctrl: int,
                ripple: int, negate: bool = False) -> None:
    """z <- z +- y on the low `len(y) + ripple` bits ONLY. No reduction. y, ctrl preserved.

    `_signed_row` with both of its modular halves deleted, which is legitimate exactly
    when the carry cannot escape the window -- see point 2 in the block comment above. The
    X/CX sandwich is the same free sign flip, restricted to the bits the add can touch;
    complementing the bits above it and un-complementing them is the identity, so leaving
    them out is a saving and not a change.

    Cost is `len(y) + ripple - 1` Toffoli, against `_signed_row`'s `U + 85`. For the row
    that made this module's square a non-triangle -- `j` around U/2 -- that is 172 against
    340.

    The approximation: the carry out of bit `len(y)+ripple-1` is DISCARDED, so the row is
    wrong by that power of two when a carry reaches it, at `2^-(ripple+1)` for a uniform
    accumulator (`narrow_row_carry_fail_prob`). It is the same windowed carry the rest of
    the schedule already truncates, in the same direction, priced the same way.

    `negate` drops the X layer, which swaps the two branches: `+y` on `ctrl = 0` and `-y`
    on `ctrl = 1`. See `_signed_row`.
    """
    w = min(len(y) + ripple, len(z))
    zw = z[0:w]
    if not negate:
        for q in zw:
            circ.x(q)
    for q in zw:
        circ.cx(ctrl, q)
    gidney_add(circ, y, zw)
    for q in zw:
        circ.cx(ctrl, q)
    if not negate:
        for q in zw:
            circ.x(q)


def _tri_rows(circ: Circuit, x: Register, z: Register, p: int, add_fn: AddFn,
              carry_window: int | None, cmp_bits: int | None, window: int | None,
              ripple: int | None, negate: bool = False) -> Register:
    """The double-and-add loop with row `j`'s addend narrowed to `x[0:j]`. Returns `z`.

    `ripple=None` means every row is a full `_signed_row` -- the EXACT schedule, same
    identity, same answer, and no cheaper than `modsqr_acc`. It exists so the triangle's
    arithmetic can be tested apart from its approximation.

    Row 0 is skipped outright: `A_0` is the empty slice, so the row is the identity on
    both branches. That is structural (a zero-width register), not a value test.

    `negate` flips every row's sign, accumulating `-S` instead of `S`. Both row kinds take
    it and neither charges for it.
    """
    u = len(x)
    for j in range(u - 1, 0, -1):
        y = x[0:j]                                     # A_j -- a slice, not a copy
        if ripple is None or j + ripple >= u:
            pad = circ.alloc(u - j)                    # |0> above A_j; modadd needs U bits
            _signed_row(circ, Register(list(y.ids) + list(pad.ids)), z, x[j],
                        p, add_fn, carry_window, cmp_bits, negate)
            circ.free(pad)
        else:
            _narrow_row(circ, y, z, x[j], ripple, negate)
        z = moddouble(circ, z, p, window)
    return z


def modsqr_tri(circ: Circuit, x: Register, out: Register,
               p: int = PRIME, add_fn: AddFn = gidney_add,
               carry_window: int | None = None, cmp_bits: int | None = None,
               window: int | None = None, ripple: int | None = None,
               diag_add_fn: AddFn | None = None) -> Register:
    """out <- x^2 mod p. x preserved; `out` MUST arrive |0>. Returns `out`, RELABELLED.

    The triangle: `x^2 = S + 2^U x - D` with `S` accumulated over rows of shrinking width.
    Output is LAZY, in `[0, 2^U)`. `modsub_direct` maps canonical inputs to canonical
    outputs, but its accumulator here is the loop's own lazy `z`, and it subtracts once:
    on `z >= D_lo` -- the usual case -- it returns `z - D_lo` untouched, so a `z` that was
    already above `p` stays above `p`. Measured at u=5..8, where the residue lands in
    `[p, 2^U)` on a few inputs per width. Treat the output like every other output of this
    module. The schedule, in order:

        z <- x                      U CNOTs. Free, and `out` is |0> so it is a copy.
        z <- z - D_hi               one `modsub_direct` against an interleaved register
        z <- 2z                     the extra doubling that makes the loop's U-1 into U
        rows U-1 .. 1               `_narrow_row`, or `_signed_row` at the wide end
        z <- z - D_lo               one `modsub_direct`

    so `z = 2^U (x - D_hi) + S - D_lo = S + 2^U x - D`, since `D = D_lo + 2^U D_hi`.

    NO OPERAND PRECONDITION, including `x = 0`, and that is worth stating because the
    obvious reading says otherwise. The top row is a general `_signed_row`, which hands
    `modadd_lazy` a complemented accumulator and so wants `z > f-1` -- and at `x = 0` the
    opening `z = 2(x - D_hi) mod p` is DETERMINISTICALLY 0, since `D_hi` is built from
    `x`'s own bits. That is the shape of the `modmuladd` defect exactly. It does not fire,
    for a reason specific to this schedule: at `x = 0` every row's addend `A_j` is the zero
    register too, and `modadd_lazy` erases with `result < a`, a predicate that is EXACT
    when `a = 0` (nothing can carry, so the flag is honestly 0). The dangerous conjunction
    is a small accumulator AND a nonzero addend, and the triangle cannot produce it at the
    top row because both come from the same `x`. Measured at (u,f) = (9,3), (9,5), (10,19):
    zero inputs leak on every shot, and `x = 0` leaks on none of 400.

    So the residual is the ordinary vent budget `vent_phase_prob` already carries, and
    `TRI_OVERFLOW` charges it over `u-1` rows plus the two `modsub_direct` calls.

    `ripple=None` runs the exact schedule (every row full width), which is correct and
    pointless -- it is the triangle without the saving, kept for testing.

    THE PEAK IS THE TWO DIAGONAL SUBTRACTIONS, AND NOTHING ELSE, which is not where the
    schedule above suggests it would be. Live-qubit trace at U = 255 (`resources.count`
    reports only the max; this is per region):

        z <- x                510          ops      2 - 4,371   <- both diagonals
        z <- z - D_hi         894  PEAK    ops  4,371 - 5,908
        z <- 2z               555          ops  5,908 - 954,750 <- all 254 rows
        rows U-1 .. 1         811
        z <- z - D_lo         893          ops 954,750 - 958,864

    894 = 510 operands + 128 pad + 255 carries + 1 borrow, and it is reached at op 515
    of 958,864. The 254 rows -- 99.1% of the circuit -- run 83 qubits below it. So the
    global peak is a transient across 0.86% of the ops, and `circuit_volume` (peak x
    Toffoli) charges the whole circuit for it.

    `diag_add_fn` is the lever, defaulting to `add_fn` so nothing changes unless asked.
    Giving the two diagonals a CHUNKED adder holds only one block's carries live, which
    costs Toffoli on two additions rather than on 254 rows:

        diagonals `gidney_add`                58,549  @ 894     (the default)
        diagonals chunked 128, EXACT          58,805  @ 811     +256 Toffoli, -83 qubits
        rows chunked as well, 128             64,565  @ 767     +6,016 for 44 more

    +0.44% Toffoli for -9.3% peak, and -8.9% on `circuit_volume`. `cmp_bits=None` makes
    the chunking EXACT (`add_chunked`'s boundary carry is killed by a real comparator),
    so this buys the peak with no new approximation and no new failure term -- the only
    lever in this module that does. Chunking the rows too is a worse trade and stalls at
    763 regardless, because `_narrow_row` calls `gidney_add` directly.

    Not the diagonal pad itself: see `_spread_register`. Removing the 128 pad qubits
    needs the ADDER to accept an operand whose known-zero bits are interleaved rather
    than trailing, and `modsub_direct` compares against `~a`, which turns those zeros
    into ones. Doc 13 records why that redesign was not taken and what it would save.
    """
    u = len(x)
    assert len(out) == u, "output must be u bits"
    diag_add_fn = diag_add_fn or add_fn

    for i in range(u):
        circ.cx(x[i], out[i])                          # z <- x. Exact: out was |0>.
    dhi, pad = _spread_register(circ, x, hi=True)
    modsub_direct(circ, dhi, out, p, diag_add_fn, carry_window, cmp_bits)
    circ.free(pad)

    z = moddouble(circ, out, p, window)                # the U-th doubling
    z = _tri_rows(circ, x, z, p, add_fn, carry_window, cmp_bits, window, ripple)

    dlo, pad = _spread_register(circ, x, hi=False)
    modsub_direct(circ, dlo, z, p, diag_add_fn, carry_window, cmp_bits)
    circ.free(pad)
    return z


def modsqrsub_inplace(circ: Circuit, x: Register, acc: Register,
                      p: int = PRIME, add_fn: AddFn = gidney_add,
                      carry_window: int | None = None, cmp_bits: int | None = None,
                      window: int | None = None, ripple: int | None = None,
                      ctrl: int | None = None,
                      diag_add_fn: AddFn | None = None) -> Register:
    """acc <- acc - x^2 mod p, IN PLACE. `x` preserved. Returns `acc`, RELABELLED.

    The point-addition shape: a square is almost never wanted on its own, it is wanted
    subtracted from a running value. Doing that with `modsqr_tri` costs the square TWICE --
    once forward, once to uncompute the scratch -- plus the 255-qubit scratch. This fuses
    it, for one extra scaling chain and no extra register.

    Two changes to `modsqr_tri` and nothing else:

        1.  `acc` is pre-scaled by `2^-U` (U `modhalve_inplace`s) and injected where
            `modsqr_tri` injects nothing, so the loop's own U doublings carry it back up
            unchanged. The halvings and the doublings are mutually inverse on the WHOLE
            lazy range, wrap included -- `test_halve_and_double_are_mutually_inverse_
            bijections` -- so this is exact, not merely congruent.
        2.  every row's sign is flipped (`negate`), so the loop accumulates `-S`; the
            two `D` terms are then ADDED rather than subtracted.

            z = 2^U (acc 2^-U - x + D_hi) - S + D_lo = acc - (2^U x + S - D) = acc - x^2

    `ctrl`, if given, gates the whole thing: `acc <- acc - ctrl * x^2`. It is NOT done by
    controlling the rows, and the reason is worth recording, because the obvious approach
    fails silently. `_signed_row`'s cheap sign is a Clifford conjugation, so `ctrl = 0` is
    not "no row", it is "the OTHER sign" -- gating the rows the naive way would compute
    `acc + x^2`. What is gated instead is the OPERAND: `t = ctrl AND x`, U Toffoli by
    `_and`, erased by `_unand` for free. At `ctrl = 0` every `A_j`, every row control and
    both diagonal halves are zero, the rows are `+-0`, and the U halvings meet the U
    doublings with nothing in between -- exactly the identity. The price is U Toffoli and
    U qubits of peak, which is why it is optional rather than the only form.

    PRECONDITION: `x <= p`, as everywhere in this module. `acc` may be lazy and may be
    zero: unlike `modsqr_tri`, whose accumulator and addends both come from `x` (see its
    docstring), this one's accumulator has an independent source, so the top row sees
    `2(acc 2^-U - x + D_hi)` and the `f/2^U` vent residual is the whole story.

    `diag_add_fn` is `modsqr_tri`'s lever and means the same thing here -- read that
    docstring for the live-qubit trace that motivates it. It reaches the two `modadd_lazy`
    calls only. The `-x` injection is deliberately NOT included even though it is the same
    width: `x` is dense, so that one carries no pad and runs 128 qubits below the two that
    do. Chunking it would buy nothing and cost Toffoli.
    """
    u = len(x)
    assert len(acc) == u, "accumulator must be u bits"
    diag_add_fn = diag_add_fn or add_fn

    src = x                                            # the caller's `x`, kept for _unand
    gate = circ.alloc(u) if ctrl is not None else None
    if gate is not None:
        for i in range(u):
            _and(circ, ctrl, src[i], gate[i])          # t = ctrl & x. U Toffoli.
        x = gate

    for _ in range(u):
        modhalve_inplace(circ, acc, p, window)         # acc <- acc * 2^-U

    modsub_direct(circ, x, acc, p, add_fn, carry_window, cmp_bits)
    dhi, pad = _spread_register(circ, x, hi=True)
    modadd_lazy(circ, dhi, acc, p, diag_add_fn, carry_window, cmp_bits)
    circ.free(pad)

    z = moddouble(circ, acc, p, window)                # the U-th doubling
    z = _tri_rows(circ, x, z, p, add_fn, carry_window, cmp_bits, window, ripple,
                  negate=True)

    dlo, pad = _spread_register(circ, x, hi=False)
    modadd_lazy(circ, dlo, z, p, diag_add_fn, carry_window, cmp_bits)
    circ.free(pad)

    if gate is not None:
        for i in range(u):
            _unand(circ, ctrl, src[i], gate[i])        # free, and phase-clean
    return z


# --- exact semantics -------------------------------------------------------
# These REPLAY the schedule rather than computing `x*y % p`. The circuit is lazily
# reduced at every step, so which representative in [0, 2^u) it lands on is a fact about
# the schedule, not about the residue -- and a reference that returned the canonical
# residue would silently absolve every laziness the versions declare.
def _lazy_add(a: int, b: int, u: int, f: int, trace: list | None = None) -> int:
    """`modadd_lazy`: add, and if it reaches 2^u drop that bit and add f instead.

    The `+f` can itself carry out, and that second carry is LOST -- this is `modadd_lazy`'s
    documented lazy+lazy hole (`f(f-1)/2` pairs, 2^-502.6 at real width). `modadd_lazy_ref`
    in Layer 0 does not model it because Layer 0 never feeds two lazy operands; this
    schedule does, in `_add_correction`, so the model has to be one order deeper.

    The ROWS are not exposed: their addend is the caller's operand, which arrives canonical,
    and canonical+lazy is closed. Only the correction's adds can trip this.
    """
    s = a + b
    if s >= (1 << u):
        s = s - (1 << u) + f
        if s >= (1 << u):                          # the second carry, dropped
            if trace is not None:
                trace.append(("add", a, b))
            s -= 1 << u
    return s


def _lazy_double(a: int, u: int, f: int, trace: list | None = None) -> int:
    """`moddouble`. Wraps (NON-CONGRUENTLY) for the top floor(f/2) inputs -- modelled.

    `trace`, when given, collects one entry per NON-CONGRUENT event, so a caller can tell
    "the schedule diverged from x*y" apart from "the schedule is wrong". At real width
    those events have probability 2^-251.8 and never happen; at the widths that can be
    enumerated they happen constantly, which is precisely why the tests need to see them.
    """
    s = 2 * a
    if s >= (1 << u):
        s = s - (1 << u) + f
    if s >= (1 << u):                  # the non-closure: a lost bit, not a reduction
        if trace is not None:
            trace.append(("double", a))
        s &= (1 << u) - 1
    return s


def _lazy_halve(a: int, u: int, f: int, trace: list | None = None) -> int:
    """`modhalve`. Borrows (NON-CONGRUENTLY) for the odd inputs below f -- modelled.

    The mirror of `_lazy_double`'s wrap, and the same 9 values. `(a - f)` is taken in the
    register, so for odd `a < f` it comes back as `a - f + 2^u` and the result is off by
    `2^(u-1)`.
    """
    if not a & 1:
        return a // 2
    if a < f and trace is not None:    # the borrow: mirror of the doubling's wrap
        trace.append(("halve", a))
    return ((a - f) & ((1 << u) - 1)) // 2 + (1 << (u - 1))


def _first_row_from_zero_ref(y: int, bit: int, u: int, f: int) -> int:
    """`_first_row_from_zero`: y, or `modneg(y) = ~(y + f - 1)`.

    Not the same representative as the general row it replaces, on one input. For
    `1 <= y <= p` both give `p - y`; for `y = 0` the general row lands on 0 and this lands
    on `p`. Congruent, both inside the lazy range, and the reference tracks the CIRCUIT --
    a reference that returned the prettier representative would be reporting on a circuit
    we do not have.
    """
    if bit:
        return y
    return ((1 << u) - 1) ^ ((y + f - 1) & ((1 << u) - 1))


def _rows_ref(ctrl: int, y: int, z: int, u: int, f: int,
              trace: list | None = None, zero_target: bool = False) -> int:
    for i in range(u - 1, -1, -1):
        if zero_target and i == u - 1:
            z = _first_row_from_zero_ref(y, (ctrl >> i) & 1, u, f)
        else:
            comp = (1 << u) - 1
            zz = z if (ctrl >> i) & 1 else comp - z
            zz = _lazy_add(y, zz, u, f, trace)
            z = zz if (ctrl >> i) & 1 else comp - zz
        if i > 0:
            z = _lazy_double(z, u, f, trace)
    return z


def _add_correction_ref(y: int, z: int, u: int, f: int,
                        trace: list | None = None) -> int:
    """The RESTORING halvings are omitted deliberately: they exactly invert the doublings
    (proved on the whole lazy range in the tests), so they move no value and contribute no
    error term. Only the doublings, whose results reach `z`, are traced."""
    c = ((1 << u) - 1) % ((1 << u) - f)
    if c == 0:
        return z
    hi = c.bit_length() - 1
    t = y
    for k in range(hi + 1):
        if (c >> k) & 1:
            z = _lazy_add(t, z, u, f, trace)
        if k < hi:
            t = _lazy_double(t, u, f, trace)
    return z


def modmul_acc_ref(x: int, y: int, p: int = PRIME, u: int = U,
                   trace: list | None = None) -> int:
    """Exact semantics of `modmul_acc`, including every laziness and every scaling wrap.

    With `trace == []` the caller learns whether the result is congruent to `x*y`: it is
    iff the trace comes back empty. That is the honest form of the correctness claim --
    the circuit always equals this reference, and this reference equals `x*y mod p`
    except on the declared 2^-243.8 set.
    """
    f = (1 << u) - p
    z = _rows_ref(x, y, 0, u, f, trace, zero_target=True)
    z = _add_correction_ref(y, z, u, f, trace)
    return _lazy_halve(z, u, f, trace)


def modsqr_acc_ref(x: int, p: int = PRIME, u: int = U,
                   trace: list | None = None) -> int:
    return modmul_acc_ref(x, x, p, u, trace)


def diag_split(x: int, u: int) -> tuple[int, int]:
    """(D_lo, D_hi) with `D = sum_j x_j 2^(2j) = D_lo + 2^u * D_hi`. See `_spread_register`."""
    lo = hi = 0
    for j in range(u):
        if (x >> j) & 1:
            if 2 * j <= u - 1:
                lo |= 1 << (2 * j)
            else:
                hi |= 1 << (2 * j - u)
    return lo, hi


def _sub_direct_ref(b: int, a: int, p: int) -> int:
    """`modsub_direct(a, b)`: `b - a`, with `p` added back ONLY on a borrow.

    Not `(b - a) % p`. The circuit subtracts in U+1 bits and lets the borrow out of bit
    U-1 gate the `-F`; when `b >= a` there is no borrow, nothing is added, and a `b` that
    was already lazy comes back lazy. That is exactly what its docstring means by
    "canonical inputs give canonical outputs" -- the converse is not offered, and the
    triangle's accumulator is never canonical.

    The `a - b > p` failure set is not modelled: at U = 255 it is 2^-502.6, and at the
    widths a test enumerates `a` is a diagonal spread, so `a - b <= p` holds by
    construction on the pre-loop call (`D_hi < 2^(u-1) < p`).
    """
    d = b - a
    return d if d >= 0 else d + p


def _narrow_row_ref(z: int, y: int, bit: int, w: int, trace: list | None = None) -> int:
    """`_narrow_row`: the low `w` bits take `+-y` mod 2^w, and the carry out is GONE.

    The dropped carry is a NON-CONGRUENT event, like `_lazy_double`'s wrap, so it is
    traced -- a caller with `trace == []` learns whether this run of the schedule still
    equals `x^2`. At U = 255 with `ripple = 45` it happens at ~2^-46 a row; at the widths
    a test can enumerate it happens constantly, which is the point of tracing it.
    """
    mask = (1 << w) - 1
    low = z & mask
    if not bit:
        low = mask - low                   # N restricted to the window
    s = low + y
    if s > mask:
        if trace is not None:
            trace.append(("ripple", w, y))
        s &= mask
    if not bit:
        s = mask - s
    return (z & ~mask) | s


def modsqr_tri_ref(x: int, p: int = PRIME, u: int = U, ripple: int | None = None,
                   trace: list | None = None) -> int:
    """Exact semantics of `modsqr_tri`, every laziness and every dropped carry included.

    As with `modmul_acc_ref`: the result is congruent to `x*x` iff `trace` comes back
    empty. The two `modsub_direct` calls are `_sub_direct_ref`, NOT `% p` -- that circuit
    subtracts once and adds `p` back only on a borrow, so it reduces a canonical input and
    leaves a lazy one lazy. Getting that wrong is how the first draft of this function
    disagreed with the circuit on ~1 input per width, always by exactly `p`. Neither call
    contributes a trace entry: their own failure set is the 2^-502.6 conjunction
    `modsub_direct` pins down, not a schedule event.
    """
    f = (1 << u) - p
    comp = (1 << u) - 1
    d_lo, d_hi = diag_split(x, u)

    z = _lazy_double(_sub_direct_ref(x, d_hi, p), u, f, trace)
    for j in range(u - 1, 0, -1):
        y = x & ((1 << j) - 1)                          # A_j
        bit = (x >> j) & 1
        if ripple is None or j + ripple >= u:
            zz = z if bit else comp - z
            zz = _lazy_add(y, zz, u, f, trace)
            z = zz if bit else comp - zz
        else:
            z = _narrow_row_ref(z, y, bit, j + ripple, trace)
        z = _lazy_double(z, u, f, trace)
    return _sub_direct_ref(z, d_lo, p)


def modsqrsub_inplace_ref(x: int, acc: int, p: int = PRIME, u: int = U,
                          ripple: int | None = None, ctrl: int = 1,
                          trace: list | None = None) -> int:
    """Exact semantics of `modsqrsub_inplace`. Congruent to `acc - ctrl*x^2` iff no trace.

    The mirror of `modsqr_tri_ref` with three differences, each one a line of the circuit:
    `u` pre-halvings, the row sign flipped, and the two diagonal halves added instead of
    subtracted. The pre-halvings ARE traced -- `_lazy_halve` borrows off the bottom on
    `(f-1)/2` of the lazy range, exactly as `_lazy_double` wraps off the top -- so a
    caller reading `trace` sees the full chain and not just the loop's half of it.
    """
    f = (1 << u) - p
    comp = (1 << u) - 1
    if not ctrl:
        x = 0
    d_lo, d_hi = diag_split(x, u)

    for _ in range(u):
        acc = _lazy_halve(acc, u, f, trace)
    z = _lazy_add(d_hi, _sub_direct_ref(acc, x, p), u, f, trace)
    z = _lazy_double(z, u, f, trace)

    for j in range(u - 1, 0, -1):
        y = x & ((1 << j) - 1)
        bit = (x >> j) & 1
        if ripple is None or j + ripple >= u:
            zz = comp - z if bit else z                 # NEGATED: N on bit=1
            zz = _lazy_add(y, zz, u, f, trace)
            z = comp - zz if bit else zz
        else:
            z = _narrow_row_ref(z, y, 1 - bit, j + ripple, trace)
        z = _lazy_double(z, u, f, trace)
    return _lazy_add(d_lo, z, u, f, trace)


def modmuladd_ref(x: int, y: int, z: int, p: int = PRIME, u: int = U,
                  trace: list | None = None) -> int:
    f = (1 << u) - p
    for _ in range(u - 2):
        z = _lazy_halve(z, u, f, trace)
    z = _rows_ref(x, y, z, u, f, trace)
    z = _add_correction_ref(y, z, u, f, trace)
    return _lazy_halve(z, u, f, trace)


def scaling_chain_prob(n_doublings: int, n_halvings: int,
                       u: int = U, f: int = F) -> float:
    """Union bound on the schedule's DOMINANT error: the doubling/halving chain.

    Both directions are non-closed and, for odd f, by the SAME count of 9 values:
    `moddouble` on the top floor(f/2) of the register, `modhalve` on the odd (f-1)/2
    below f. So the rate is one number times the number of scalings, and this schedule
    runs one per row. It is the price of reducing as you go, it is NON-CONGRUENT rather
    than merely lazy, and at ~2^-243.8 for 259 scalings it is four orders worse than
    `mul.modmul.v1`'s `FOLD2_OVERFLOW`. Declared, not hidden.

    Only scalings whose result is USED count. A halving that undoes a doubling is exactly
    inverse -- `test_mul_accumulate.py` proves the round trip on the whole lazy range,
    wrap included -- so `_add_correction`'s four restoring halvings contribute nothing here and
    are not passed in.
    """
    assert moddouble_lazy_overflow_prob(u, f) == modhalve_underflow_prob(u, f), \
        "the two rates coincide only for odd f; this caller assumes it"
    return (n_doublings + n_halvings) * moddouble_lazy_overflow_prob(u, f)


def schedule_counts(u: int = U, p: int = PRIME,
                    zero_target: bool = True) -> tuple[int, int, int, int]:
    """(modadd_lazy, moddouble, modhalve, controlled const-add) counts for one multiply.

    Read off the schedule rather than written down, so the failure accounting cannot drift
    away from the circuit when the constant or the width changes:

        rows          u adds, u-1 doublings          (`_rows`)
        correction    popcount(F-1) adds,
                      bitlen(F-1)-1 doublings and the same many halvings
        finish        1 halving                      (`modhalve`)

    With `zero_target` (`modmul_acc`, `modsqr_acc`) the top row is
    `_first_row_from_zero`, so one `modadd_lazy` becomes one controlled constant add --
    which is why the fourth number exists rather than being folded into the third.
    `modmuladd` passes `zero_target=False` and adds u-2 more halvings; see `MULADD_*`.
    """
    c = ((1 << u) - 1) % p                        # == F - 1
    hi = c.bit_length() - 1 if c else 0
    n_add = u + bin(c).count("1")
    n_dbl = (u - 1) + hi
    n_halve = hi + 1
    n_const = 0
    if zero_target:
        n_add -= 1                                # the top row is no longer an add
        n_const = 1
    return n_add, n_dbl, n_halve, n_const


def vent_phase_prob(n_add: int, u: int = U, f: int = F) -> float:
    """Union bound on `modadd_lazy`'s vented erasure leaking PHASE, over the schedule.

    The term this module was missing until the benchmark simulated its measurements, and
    the reason `_first_row_from_zero` exists. Every `modadd_lazy` here receives an
    accumulator that is LAZY rather than canonical -- the rows hand it `2^u - 1 - z`, the
    correction hands it `z` -- so each one sits on `modadd_lazy_vent_phase_prob`, and a
    hit is a relative phase, not a wrong value.

    At real width 256 exposed adds give ~2^-242.8, which is the same order as
    `scaling_chain_prob` and equally irrelevant. That is only true because the ONE add
    whose accumulator was deterministically out of contract has been removed; with it the
    circuit sat at ~25%, and no union bound would have made that acceptable.
    """
    return n_add * modadd_lazy_vent_phase_prob(u, f)


def truncation_prob(n_add: int, n_dbl: int, n_halve: int, n_const: int = 0,
                    carry_window: int | None = CARRY_WINDOW,
                    cmp_bits: int | None = CMP_BITS, f: int = F) -> float:
    """Union bound on the TRUNCATION error: every windowed carry the schedule can lose.

    Distinct from `scaling_chain_prob`, and at real width it is ~210 orders LARGER. The
    non-closure is a property of the lazy contract and costs 2^-251.8 a call; truncation
    is a property of the window and costs ~2^-41 a call. Both are declared because both
    are real, but only this one is ever going to matter.

        modadd_lazy    controlled_carry_fail_prob(L) + cmp_fail_prob(k)   2^-40.3
        moddouble      moddouble_lazy_input_carry_fail_prob(L)            2^-41.8
        modhalve       modhalve_borrow_fail_prob(L)                       2^-41.8
        const-add      controlled_carry_fail_prob(L, f-1)                 2^-41.8

    The last row is `_first_row_from_zero`'s controlled `+(f-1)`: same shape as the others,
    one call, and it REPLACES a `modadd_lazy` -- so the top row got cheaper in this budget
    as well as in Toffoli.

    The doubling rate is the LAZY-input one, not `moddouble_carry_fail_prob`: after the
    first row the accumulator is uniform on [0, 2^u), not on [0, p).

    WHERE THIS SCHEDULE ACTUALLY PAYS. `mul.modmul_trunc.v1` truncates a handful of
    constant adds and reports 2^-39.2; this one truncates ~520 primitives and reports
    ~2^-31.9, seven bits worse for the same window. That is the arithmetic price of
    reducing on every row, and it is NOT hidden inside the laziness term -- reducing 255
    times means 255 opportunities to lose a carry. Buying the seven bits back is cheap
    (CARRY_WINDOW 45 -> 52 costs ~7 Toffoli per constant add, ~4% of the circuit) and is
    a Layer-2 decision once the point layer says what rate it needs.
    """
    per_add = controlled_carry_fail_prob(carry_window, f) + cmp_fail_prob(cmp_bits)
    return (n_add * per_add
            + n_dbl * moddouble_lazy_input_carry_fail_prob(carry_window, f)
            + n_halve * modhalve_borrow_fail_prob(carry_window, f)
            + n_const * controlled_carry_fail_prob(carry_window, f - 1))


def tri_schedule_counts(u: int = U, ripple: int | None = CARRY_WINDOW) -> tuple[int, int, int]:
    """(full `_signed_row`s, `_narrow_row`s, `moddouble`s) for one `modsqr_tri`.

    Read off `_tri_rows` rather than written down, same as `schedule_counts`. Rows run
    `j = u-1 .. 1`; a row is narrow iff `j + ripple < u`, so the count of full rows is
    `min(ripple, u-1)` and everything below is a triangle. The doublings are `u-1` for the
    rows plus the one `modsqr_tri` prepends, i.e. `u`.

    The two `modsub_direct` calls are NOT counted here: they are not schedule steps, they
    are two Layer-0 circuits with their own published rates, and `tri_truncation_prob`
    adds them separately so the provenance stays visible.
    """
    rows = u - 1
    full = rows if ripple is None else min(ripple, rows)
    return full, rows - full, u


def tri_truncation_prob(u: int = U, ripple: int | None = CARRY_WINDOW,
                        carry_window: int | None = CARRY_WINDOW,
                        cmp_bits: int | None = CMP_BITS, f: int = F) -> float:
    """Union bound on everything `modsqr_tri` truncates.

    Four terms, and the NEW one is third:

        full rows       controlled_carry_fail_prob + cmp_fail_prob      2^-40.3 each
        doublings       moddouble_lazy_input_carry_fail_prob            2^-41.8 each
        narrow rows     narrow_row_carry_fail_prob(ripple)              2^-46   each
        2 modsub_direct controlled_carry_fail_prob + cmp_fail_prob      2^-40.3 each

    The narrow rows are the cheap ones to be wrong about -- `2^-(ripple+1)` rather than
    `f/2^(L+1)` -- but there are 210 of them, so they land at ~2^-38.3 overall and are
    comparable to the 45 full rows. `ripple` and `carry_window` are separate knobs on
    purpose: they truncate different things (a row's own sum, versus a `+f`), and the
    triangle's whole width saving is paid for in the first of them.
    """
    full, narrow, n_dbl = tri_schedule_counts(u, ripple)
    per_add = controlled_carry_fail_prob(carry_window, f) + cmp_fail_prob(cmp_bits)
    return (full * per_add
            + n_dbl * moddouble_lazy_input_carry_fail_prob(carry_window, f)
            + narrow * narrow_row_carry_fail_prob(ripple)
            + 2 * per_add)


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
# ---------------------------------------------------------------------------
_N = U
_ADDS, _DOUBLES, _HALVES, _CONSTS = schedule_counts(_N, PRIME)         # 256, 258, 5, 1
_MA_ADDS, _MA_DOUBLES, _MA_HALVES, _MA_CONSTS = schedule_counts(
    _N, PRIME, zero_target=False)                                      # 257, 258, 5, 0

# --- the exact circuits' rate, ~2^-242.2: non-closure + vented erasure --------
# Scalings whose RESULT IS CONSUMED. `_add_correction`'s four restoring halvings are exact
# inverses of its doublings, so they are deliberately excluded here (`schedule_counts`
# counts them, because truncation does not care about provenance and they DO pay there).
# The second term is the one the benchmark found missing: every `modadd_lazy` in this
# schedule takes a LAZY accumulator, and its vented erasure fails as PHASE at f/2^u each.
_CORR_HALVES = _HALVES - 1
ACC_OVERFLOW = (scaling_chain_prob(_DOUBLES, _HALVES - _CORR_HALVES)
                + vent_phase_prob(_ADDS))
# `modmuladd` additionally pre-halves a live accumulator U-2 times, and every one of those
# is exposed. Roughly doubles the schedule's dominant error term.
MULADD_OVERFLOW = (scaling_chain_prob(_DOUBLES, _HALVES - _CORR_HALVES + (_N - 2))
                   + vent_phase_prob(_MA_ADDS))

# --- the truncation term, ~2^-31.9, and the one that actually binds -----------
ACC_TRUNC = ACC_OVERFLOW + truncation_prob(_ADDS, _DOUBLES, _HALVES, _CONSTS)
# The triangle has no halvings and no correction multiply: `u` doublings and `u-1` rows,
# of which only `ripple` are full width. Its exact-schedule term is therefore smaller than
# `ACC_OVERFLOW` (fewer scalings, far fewer exposed `modadd_lazy`s) and its truncated term
# is dominated by the narrow rows -- `tri_truncation_prob`.
_TRI_FULL, _TRI_NARROW, _TRI_DBL = tri_schedule_counts(_N, CARRY_WINDOW)   # 45, 209, 255
TRI_OVERFLOW = (scaling_chain_prob(_N, 0)
                + vent_phase_prob(tri_schedule_counts(_N, None)[0] + 2))
TRI_TRUNC = TRI_OVERFLOW + tri_truncation_prob(_N, CARRY_WINDOW)
# The fused subtract-square adds `u` pre-halvings to the triangle -- every one of them
# exposed, since the accumulator arrives live -- and swaps the two `modsub_direct`s for
# two `modadd_lazy`s, which is the same rate from a different Layer-0 circuit. So its
# budget is the triangle's plus one scaling chain of length `u` plus `u` more windowed
# borrows. Roughly doubles the exact term and moves the truncated one by ~40%.
TRISUB_OVERFLOW = (scaling_chain_prob(_N, _N)
                   + vent_phase_prob(tri_schedule_counts(_N, None)[0] + 2))
TRISUB_TRUNC = (TRISUB_OVERFLOW + tri_truncation_prob(_N, CARRY_WINDOW)
                + _N * modhalve_borrow_fail_prob(CARRY_WINDOW, F))
# The same model with the F-independent narrow-row ripple. The `v1` names above keep the
# historical ripple=CARRY_WINDOW so their recorded counts and rates stay reproducible.
TRI_TRUNC_R = TRI_OVERFLOW + tri_truncation_prob(_N, SQUARE_RIPPLE)
TRISUB_TRUNC_R = (TRISUB_OVERFLOW + tri_truncation_prob(_N, SQUARE_RIPPLE)
                  + _N * modhalve_borrow_fail_prob(CARRY_WINDOW, F))
MULADD_TRUNC = MULADD_OVERFLOW + truncation_prob(
    _MA_ADDS, _MA_DOUBLES, _MA_HALVES + (_N - 2), _MA_CONSTS)


def _ACC_META(fail: float) -> dict:
    """Case metadata shared by all eight versions. Deliberately NOT `tolerates_phase`.

    These circuits do contain vented erasures -- every `modadd_lazy` here erases its
    overflow flag by MEASUREMENT, so a wrong predicate becomes a relative phase rather
    than a wrong value, which is why Layer 0's `field.modadd_lazy.*` is on the harness's
    phase allowlist. The rate this schedule INHERITS from ~256 of them, however, is
    `vent_phase_prob` = 2^-242.2 exact / ~2^-33 truncated: unobservable in a benchmark's
    256 shots. So keeping kickback fatal costs us no false failures, and it is what
    detects the next systematic defect.

    That is not hypothetical. This module shipped leaking phase on ~25% of inputs
    (`_first_row_from_zero`), and the strict check is exactly what caught it. Declaring
    tolerance would not have hidden it -- the harness still budgets `max(1, 2 * fail *
    trials)` -- but it would have moved a real defect one step closer to looking normal.
    What DID hide it was the TEST suite, which simulated with an all-zero measurement
    vector; under one a vented erasure contributes no phase whether it is right or wrong,
    so `assert phase == 0` held vacuously. Tests here now randomise the measurements.
    """
    return {"fail_analytic": fail}


def _acc_binary_case(build, ref_fn, fail: float) -> Case:
    circ = Circuit()
    x = circ.input("x", _N)
    y = circ.input("y", _N)
    out = circ.input("out", _N)
    res = build(circ, x, y, out)
    circ.rebind("out", res)          # the doublings relabel the result register

    def ref(v):
        return {"x": v["x"], "y": v["y"], "out": ref_fn(v["x"], v["y"])}

    def gen(rng: random.Random):
        return {"x": rng.randrange(PRIME), "y": rng.randrange(PRIME), "out": 0}

    return Case(circ, ref, gen, meta=_ACC_META(fail))


def _acc_unary_case(build, fail: float) -> Case:
    circ = Circuit()
    x = circ.input("x", _N)
    out = circ.input("out", _N)
    circ.rebind("out", build(circ, x, out))

    def ref(v):
        return {"x": v["x"], "out": modsqr_acc_ref(v["x"])}

    def gen(rng: random.Random):
        return {"x": rng.randrange(PRIME), "out": 0}

    return Case(circ, ref, gen, meta=_ACC_META(fail))


def _acc_muladd_case(build, fail: float) -> Case:
    circ = Circuit()
    x = circ.input("x", _N)
    y = circ.input("y", _N)
    z = circ.input("z", _N)
    circ.rebind("z", build(circ, x, y, z))

    def ref(v):
        return {"x": v["x"], "y": v["y"], "z": modmuladd_ref(v["x"], v["y"], v["z"])}

    def gen(rng: random.Random):
        return {"x": rng.randrange(PRIME), "y": rng.randrange(PRIME),
                "z": rng.randrange(PRIME)}

    return Case(circ, ref, gen, meta=_ACC_META(fail))


@circuit_version("mul.modmul_acc.v1")
def _build_modmul_acc_v1() -> Case:
    """
    out <- x*y mod p with NO product register: 255 signed rows into a U-bit accumulator.

    The alternative to `mul.modmul.v1`'s compute/fold/uncompute, and the reason it is
    affordable is F = 19 -- a fold cheap enough to run on every row. Each row is an
    UNCONTROLLED modular add (the proven-tight case), signed by two Clifford layers
    rather than by a control, with the aggregate bought back by one multiply-by-18.
    """
    return _acc_binary_case(
        lambda c, x, y, o: modmul_acc(c, x, y, o), modmul_acc_ref, ACC_OVERFLOW)


@circuit_version("mul.modmul_acc_trunc.v1")
def _build_modmul_acc_trunc_v1() -> Case:
    """
    Same, with every constant add confined to its carry window and every comparison
    to CMP_BITS. This is the version the point layer would actually call.

    2.7x cheaper than the exact version and seven bits riskier than `mul.modmul_trunc.v1`
    at the same window, because reducing on every row means 520 truncated primitives
    instead of a handful. See `truncation_prob`.
    """
    return _acc_binary_case(
        lambda c, x, y, o: modmul_acc(c, x, y, o, carry_window=CARRY_WINDOW,
                                      cmp_bits=CMP_BITS, window=CARRY_WINDOW),
        modmul_acc_ref, ACC_TRUNC)


@circuit_version("mul.modsqr_acc.v1")
def _build_modsqr_acc_v1() -> Case:
    """
    out <- x^2 mod p by the same schedule, one register cheaper and NO ancilla.

    The head-to-head against `mul.modsqr.v1` is the point: this schedule does not halve
    for squaring, so it should LOSE on Toffoli and win on peak. Which of those decides
    the point layer depends on whether the multiply owns the global peak.
    """
    return _acc_unary_case(lambda c, x, o: modsqr_acc(c, x, o), ACC_OVERFLOW)


@circuit_version("mul.modsqr_acc_trunc.v1")
def _build_modsqr_acc_trunc_v1() -> Case:
    """
    The windowed square: the head-to-head partner for `mul.modsqr.v1`.

    Registered because comparing the EXACT accumulate square against the TRUNCATED
    schoolbook one flatters the schoolbook by the whole 2.7x truncation gap -- which is
    exactly what the first version of this table did.
    """
    return _acc_unary_case(
        lambda c, x, o: modsqr_acc(c, x, o, carry_window=CARRY_WINDOW,
                                   cmp_bits=CMP_BITS, window=CARRY_WINDOW),
        ACC_TRUNC)


#: The diagonals' chunked adder. `cmp_bits=None` kills the boundary carry with a real
#: comparator, so this is EXACT -- it moves peak without moving any failure rate. 128 is
#: the measured optimum: one boundary is the cheapest chunking that chunks at all, and
#: smaller blocks cost Toffoli for a peak the 254 rows already sit above. See `modsqr_tri`.
FLAT_DIAG_CHUNK = 128
_flat_diag = functools.partial(gidney_add_chunked, chunk=FLAT_DIAG_CHUNK, cmp_bits=None)


def _tri_case(ripple: int | None, fail: float,
              diag_add_fn: AddFn | None = None) -> Case:
    """`_acc_unary_case` with the triangle's reference. No operand restriction.

    `gen` draws `x` from all of `[0, p)`, zero included. The first draft of this restricted
    it, on the argument that `z = 2(x - D_hi)` is identically zero at `x = 0` and a
    complemented zero accumulator is the `modmuladd` defect -- which is the right shape and
    the wrong conclusion, because at `x = 0` the addends vanish too and
    `modadd_lazy`'s erasure is exact on a zero addend. `modsqr_tri` records the measurement.
    """
    circ = Circuit()
    x = circ.input("x", _N)
    out = circ.input("out", _N)
    circ.rebind("out", modsqr_tri(circ, x, out, carry_window=(
        CARRY_WINDOW if ripple is not None else None),
        cmp_bits=(CMP_BITS if ripple is not None else None),
        window=(CARRY_WINDOW if ripple is not None else None), ripple=ripple,
        diag_add_fn=diag_add_fn))

    def ref(v):
        # NOT parameterised by `diag_add_fn`, and that is the assertion: an exact chunking
        # computes the same sum, so the reference is the one the unchunked version uses.
        return {"x": v["x"], "out": modsqr_tri_ref(v["x"], ripple=ripple)}

    def gen(rng: random.Random):
        return {"x": rng.randrange(PRIME), "out": 0}

    return Case(circ, ref, gen, meta=_ACC_META(fail))


@circuit_version("mul.modsqr_tri.v1")
def _build_modsqr_tri_v1() -> Case:
    """
    out <- x^2 mod p, TRIANGULAR, with the accumulator still U bits. Exact schedule.

    The arithmetic of the triangle with none of its saving: `ripple=None` runs every row
    at full width, so this costs what `mul.modsqr_acc.v1` costs and exists only to show
    that `x^2 = S + 2^U x - D` is right before the approximation is layered on top.
    Registered because a measured exact version is the only honest baseline for the
    truncated one -- comparing a truncated circuit against an exact rival is the mistake
    `mul.modsqr_acc_trunc.v1`'s docstring already records.
    """
    return _tri_case(None, TRI_OVERFLOW)


@circuit_version("mul.modsqr_tri_trunc.v1")
def _build_modsqr_tri_trunc_v1() -> Case:
    """
    The triangle with its rows actually narrowed: the attack on ecdsa.fail's 67,467.

    Row `j` adds the `j`-bit slice `x[0:j]` into `z[0 : j+ripple]` and drops the carry out
    of that window, which removes BOTH the width (`j` instead of U) and the per-row
    modular reduction (85 Toffoli of `+F` and comparator that a non-reducing row does not
    need). 209 of the 254 rows qualify; the top 45 cannot and stay `_signed_row`.

    The head-to-head is against `mul.modsqr_acc_trunc.v1` (same schedule family, same
    windows, no triangle) and `mul.modsqr.v1` (the triangle, bought with a 2U-bit product
    register). Beating the first proves the triangle; beating the second on peak is what
    the whole module is for.
    """
    return _tri_case(CARRY_WINDOW, TRI_TRUNC)


def _trisub_case(ripple: int | None, fail: float, ctrl: bool,
                 diag_add_fn: AddFn | None = None) -> Case:
    """`acc <- acc - [c] x^2`, measured on the live state -- one circuit, not a sum.

    `acc` is drawn from `[0, p)` including zero, and `x` likewise. Neither is a
    precondition here: the accumulator has an independent source, so the top row cannot
    see a deterministic zero the way `modsqr_tri`'s can.
    """
    circ = Circuit()
    c = circ.input("c", 1) if ctrl else None
    x = circ.input("x", _N)
    acc = circ.input("acc", _N)
    circ.rebind("acc", modsqrsub_inplace(
        circ, x, acc, PRIME, gidney_add,
        CARRY_WINDOW if ripple is not None else None,
        CMP_BITS if ripple is not None else None,
        CARRY_WINDOW if ripple is not None else None,
        ripple, None if c is None else c[0], diag_add_fn))

    def ref(v):
        out = {"x": v["x"],
               "acc": modsqrsub_inplace_ref(v["x"], v["acc"], ripple=ripple,
                                            ctrl=v.get("c", 1))}
        if ctrl:
            out["c"] = v["c"]
        return out

    def gen(rng: random.Random):
        v = {"x": rng.randrange(PRIME), "acc": rng.randrange(PRIME)}
        if ctrl:
            v["c"] = rng.randrange(2)
        return v

    return Case(circ, ref, gen, meta=_ACC_META(fail))


@circuit_version("mul.modsqrsub_inplace.v1")
def _build_modsqrsub_inplace_v1() -> Case:
    """
    acc <- acc - x^2 mod p, fused, exact schedule. The shape a point addition wants.

    Registered exact for the same reason `mul.modsqr_tri.v1` is: the truncated number
    below is only meaningful beside an exact one measured on the same schedule.
    """
    return _trisub_case(None, TRISUB_OVERFLOW, ctrl=False)


@circuit_version("mul.modsqrsub_inplace_trunc.v1")
def _build_modsqrsub_inplace_trunc_v1() -> Case:
    """
    The fused subtract-square at the tuned windows. Compare against the UNFUSED route.

    The alternative is `mul.modsqr_tri_trunc.v1` into a scratch register, one
    `field.modsub_direct_trunc.v2`, then the square again to uncompute -- 2x58,549 + 340
    Toffoli and 255 more qubits. Whatever this measures, that 117,438 is the number it has
    to beat, and the fusion's entire cost is the `U` pre-halvings.
    """
    return _trisub_case(CARRY_WINDOW, TRISUB_TRUNC, ctrl=False)


@circuit_version("mul.modsqrsub_inplace_ctrl_trunc.v1")
def _build_modsqrsub_inplace_ctrl_trunc_v1() -> Case:
    """
    acc <- acc - c*x^2 mod p, gated on one qubit. The Layer-3 shape, priced.

    The control is applied to the OPERAND (`t = c AND x`), never to the rows -- see
    `modsqrsub_inplace` for why gating a Clifford-signed row computes `acc + x^2` instead
    of nothing. That costs U Toffoli and U qubits of peak, and this version exists so the
    point-addition schedule can be argued from a measured number rather than an estimate.
    """
    return _trisub_case(CARRY_WINDOW, TRISUB_TRUNC, ctrl=True)


# --- the same three with the diagonal transient flattened ---------------------
# A `_flat` version differs from its parent in ONE argument and computes the SAME
# function: `FLAT_DIAG_CHUNK`-block addition with an exact boundary is the same sum. So
# these share their parents' reference and their parents' failure rate, and both of those
# are claims a test has to make rather than a comment -- see `test_mul_accumulate.py`,
# "--- the flattened diagonal ---". What moves is peak, and only because the global peak
# of the parent lives in 0.86% of its ops (`modsqr_tri`'s docstring has the trace).
#
# Registered as separate versions rather than retuned in place because the trade is not
# free -- +0.44% Toffoli -- and which side of it wins depends on whether the caller is
# peak-bound. Layer 3 decides that; this module only prices it.

@circuit_version("mul.modsqr_tri_trunc_flat.v1")
def _build_modsqr_tri_trunc_flat_v1() -> Case:
    """
    `mul.modsqr_tri_trunc.v1` with the two diagonal subtractions chunked. 58,805 @ 811.

    The parent's 894 is 510 operands + a 128-qubit diagonal pad + 255 carries + a borrow,
    held for 4,114 ops out of 958,864 while the 254 rows -- the entire rest of the circuit
    -- run at 811. Chunking the carries of those two additions alone costs 256 Toffoli and
    removes the transient, which is 8.9% off `circuit_volume`.

    EXACT: `cmp_bits=None`, so the boundary carry is killed by a real comparator and not
    vented. Same reference, same `TRI_TRUNC`. This is the only lever in the module that
    moves peak without moving a failure rate.
    """
    return _tri_case(CARRY_WINDOW, TRI_TRUNC, diag_add_fn=_flat_diag)


@circuit_version("mul.modsqrsub_inplace_trunc_flat.v1")
def _build_modsqrsub_inplace_trunc_flat_v1() -> Case:
    """
    The fused subtract-square with its two diagonal adds chunked. The Layer-3 candidate.

    Same argument as `mul.modsqr_tri_trunc_flat.v1`, applied to the version a point
    addition would actually call. The `-x` injection is left unchunked on purpose: `x` is
    dense, so it carries no pad and already sits below the rows.
    """
    return _trisub_case(CARRY_WINDOW, TRISUB_TRUNC, ctrl=False, diag_add_fn=_flat_diag)


@circuit_version("mul.modsqrsub_inplace_ctrl_trunc_flat.v1")
def _build_modsqrsub_inplace_ctrl_trunc_flat_v1() -> Case:
    """
    The gated fused subtract-square, flattened. The narrowest form of the Layer-3 shape.

    The control's `t = c AND x` adds U qubits on top, so this is the version whose peak
    the point-addition schedule has to budget for.
    """
    return _trisub_case(CARRY_WINDOW, TRISUB_TRUNC, ctrl=True, diag_add_fn=_flat_diag)


@circuit_version("mul.modsqr_tri_trunc_flat.v2")
def _build_modsqr_tri_trunc_flat_v2() -> Case:
    """`.v1` with the narrow-row ripple set to the F-independent SQUARE_RIPPLE."""
    return _tri_case(SQUARE_RIPPLE, TRI_TRUNC_R, diag_add_fn=_flat_diag)


@circuit_version("mul.modsqrsub_inplace_trunc_flat.v2")
def _build_modsqrsub_inplace_trunc_flat_v2() -> Case:
    """`.v1` with the narrow-row ripple set to the F-independent SQUARE_RIPPLE."""
    return _trisub_case(SQUARE_RIPPLE, TRISUB_TRUNC_R, ctrl=False, diag_add_fn=_flat_diag)


@circuit_version("mul.modsqrsub_inplace_ctrl_trunc_flat.v2")
def _build_modsqrsub_inplace_ctrl_trunc_flat_v2() -> Case:
    """`.v1` with the narrow-row ripple set to the F-independent SQUARE_RIPPLE."""
    return _trisub_case(SQUARE_RIPPLE, TRISUB_TRUNC_R, ctrl=True, diag_add_fn=_flat_diag)


@circuit_version("mul.modmuladd_acc.v1")
def _build_modmuladd_acc_v1() -> Case:
    """
    z <- z + x*y mod p into a LIVE accumulator: qarton's `zero_input=False` branch.

    Prices the U-2 pre-halvings that a nonzero target costs, so Layer 2 can decide
    whether accumulating in place beats a fresh multiply plus a modular add.
    """
    return _acc_muladd_case(lambda c, x, y, z: modmuladd(c, x, y, z), MULADD_OVERFLOW)


@circuit_version("mul.modmuladd_acc_trunc.v1")
def _build_modmuladd_acc_trunc_v1() -> Case:
    """
    The windowed accumulating multiply -- the only one of these three Layer 2 would call.

    The U-2 pre-halvings are not free in either budget: they are exposed to the halving
    non-closure (which the correction's restoring halvings escape) AND they each truncate.
    Nearly twice `mul.modmul_acc_trunc.v1`'s error on both counts.
    """
    return _acc_muladd_case(
        lambda c, x, y, z: modmuladd(c, x, y, z, carry_window=CARRY_WINDOW,
                                     cmp_bits=CMP_BITS, window=CARRY_WINDOW),
        MULADD_TRUNC)
