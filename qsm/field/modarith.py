"""Modular arithmetic over F_p with p = 2^255 - 19.

`modadd(circ, a, b)` computes b <- (a + b) mod p in place with a preserved, for inputs
already reduced (a, b < p).

Construction (v1 -- correct first, cheap later)
----------------------------------------------
Work on an (n+1)-bit view of b so the intermediate sum a+b < 2p fits. `a` is likewise
zero-extended by one qubit, which the adder preserves.

    1.  B += A                      B = a+b            in [0, 2p)
    2.  B -= p                      B = a+b-p          in [-p, p), two's complement
    3.  flag ^= B[n]                B[n] is the sign: set iff a+b < p
    4.  B += p  if flag             B = (a+b) mod p    in [0, p), so B[n] = 0
    5.  B -= A                      B = ((a+b) mod p) - a
    6.  flag ^= NOT B[n]            uncomputes flag (see below)
    7.  B += A                      B = (a+b) mod p    restored

Why step 6 clears the flag. After step 5:
  - no reduction happened (flag = 1): B = b_orig, and b_orig < p < 2^n, so B[n] = 0;
  - reduction happened  (flag = 0): B = b_orig - p < 0, which wraps to
      2^(n+1) + b_orig - p,
    and 0 <= b_orig < p < 2^n bounds this in [2^n, 2^(n+1)), so B[n] = 1.
    (This holds for any p < 2^n, not just pseudo-Mersenne -- the small-prime tests
    exercise p = 5, 7, 13, 29, 31, 61 to confirm it.)
So B[n] == NOT flag, and flag ^= NOT B[n] returns flag to |0>. This is the standard
"recompute the comparison from the result" trick; it costs the two extra adder passes
(5 and 7), which a dedicated comparator would avoid -- the obvious v2.

Cost: 3 register additions + 1 constant subtraction + 1 controlled constant addition.
With the register-loading constant adder from `constadd` (v1) each of the latter two is
itself a full adder pass, so this is ~5 adder passes. Litinski (arXiv:2306.08585, Fig. 8)
reports 4n Toffoli for modular addition, so v1 leaves roughly one pass on the table --
recovered by (a) an in-place constant adder that exploits p's pseudo-Mersenne form, and
(b) a comparator in place of steps 5/7.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import (
    CARRY_WINDOW,
    CMP_BITS,
    F,
    PRIME,
    U,
    carry_fail_prob,
    controlled_carry_fail_prob,
    moddouble_carry_fail_prob,
    modhalve_carry_fail_prob,
    cmp_fail_prob,
)
from .add import gidney_add, gidney_add_ctrl
from .compare import cmp_ge_const, cmp_lt, cmp_lt_phase
from .constadd import (
    _const_add,
    add_const_inplace,
    addsub_p,
    sub_const,
    sub_const_controlled,
    sub_const_inplace,
)
from .shift import controlled_lshift1, controlled_rshift1
from .sub import sub

AddFn = Callable[[Circuit, Register, Register], None]
CtrlAddFn = Callable[[Circuit, int, Register, Register], None]

# `addsub_p` (the (B ^ 2^u) -+ f decomposition) now lives in `constadd`, where it sits
# with the other constant adders and can take a truncation window. `_addsub_p` is kept
# as a thin alias so the argument order used below stays readable.
def _addsub_p(
    circ: Circuit,
    B: Register,
    n: int,
    f: int,
    adding: bool,
    ctrl: int | None,
    inplace: bool,
    add_fn: AddFn,
    window: int | None = None,
) -> None:
    addsub_p(circ, B, n, f, adding=adding, ctrl=ctrl, inplace=inplace,
             add_fn=add_fn, window=window)


def modadd(
    circ: Circuit,
    a: Register,
    b: Register,
    p: int = PRIME,
    add_fn: AddFn = gidney_add,
    inplace_const: bool = True,
    pseudo_mersenne: bool = True,
) -> None:
    """In place: b <- (a + b) mod p, a preserved. **Requires b < p**; a may be lazy.

    The real precondition is weaker than "both reduced", and its violation is worse than
    a wrong answer. Measured across four widths, the failure set is exactly `b >= p`,
    independent of `a`, and the failure is a **DIRTY ANCILLA** rather than a bad value:
    step 6 erases the reduction flag by recomputing the comparison from the result, and
    that argument assumes `b_orig < p`. With a lazy accumulator the flag does not return
    to |0>, so the freed qubit comes back |1> -- entangled garbage that a value-only test
    cannot see, because the returned sum may still be correct.

    Predicate: `qsm.params.canonical_modadd_dirties`. Frequency under a lazy schedule is
    F/2^U = 2^-250.8 per call.

    `modadd_lazy` (v4) does not have this hazard for a single lazy operand -- it is closed
    for canonical+lazy and only lazy+lazy has a hole (~2^-501.6). If the schedule cannot
    guarantee a canonical accumulator, use v4.

    `inplace_const` selects how the two constant operations (steps 2 and 4) are built:
      False -> v1: load the constant into a borrowed register and reuse the adder.
      True  -> v2: in-place constant adder, no constant register at all.

    `pseudo_mersenne` writes both +-p as (B ^ 2^n) -+ f with f = 2^n - p, so only the
    small constant f is ever materialised. Set False to compare against the naive form.

    `a` is passed to the adders at its natural width; they zero-extend a short first
    operand, so no explicit padding qubit is needed.
    """
    n = len(a)
    assert len(b) == n, "operand widths must match"
    assert p < (1 << n), "p must fit in n bits"
    if n == 0:
        return

    ovf = circ.alloc(1)      # b's (n+1)-th bit: holds the carry of a+b
    flag = circ.alloc(1)     # 1 iff a+b < p (i.e. no reduction was needed)

    B = Register(list(b.ids) + [ovf[0]])
    msb = B[n]
    f = (1 << n) - p

    def sub_p(ctrl: int | None) -> None:
        if pseudo_mersenne:
            _addsub_p(circ, B, n, f, False, ctrl, inplace_const, add_fn)
        elif inplace_const:
            sub_const_inplace(circ, B, p, ctrl)
        elif ctrl is None:
            sub_const(circ, B, p, add_fn)
        else:
            sub_const_controlled(circ, ctrl, B, p, add_fn)

    def add_p(ctrl: int | None) -> None:
        if pseudo_mersenne:
            _addsub_p(circ, B, n, f, True, ctrl, inplace_const, add_fn)
        else:
            _const_add(circ, B, p, ctrl, inplace_const, add_fn)

    add_fn(circ, a, B)                                      # 1
    sub_p(None)                                             # 2
    circ.cx(msb, flag[0])                                   # 3
    add_p(flag[0])                                          # 4
    sub(circ, a, B, add_fn)                                 # 5
    circ.x(msb)                                             # 6
    circ.cx(msb, flag[0])
    circ.x(msb)
    add_fn(circ, a, B)                                      # 7

    circ.free(flag)   # simulator asserts each of these is back to |0>
    circ.free(ovf)


def modadd_cmp(
    circ: Circuit,
    a: Register,
    b: Register,
    p: int = PRIME,
    add_fn: AddFn = gidney_add,
    inplace_const: bool = True,
    vent: bool = False,
) -> None:
    """In place: b <- (a + b) mod p via compare-then-subtract. v3.

    **Requires b < p** (a may be lazy). Step 4's `flag ^= (B < a)` is only a valid
    uncompute while `b < p`, and violating it leaves a DIRTY ANCILLA -- see
    `qsm.params.canonical_modadd_dirties`.

    `vent=True` erases the flag by MEASUREMENT plus a phase correction instead of by
    recomputing the predicate. That does not make a lazy accumulator correct, but it
    changes the failure CLASS: a wrong predicate then leaves residual phase rather than
    an entangled dirty ancilla, which is the class the rest of the layer already budgets
    for (`modadd_lazy`, `modsub_direct`). Strictly better and it costs nothing -- the top
    carry becomes a conditioned CZ+Z rather than an `_and` -- so the only reason the
    default is False is to keep the unvented circuit as the published exact baseline.

    Why not fix it properly: after step 3 the three cases are

        not reduced        B = b_orig       in [0, 2^n)
        reduced, b < p     B = b_orig - p   in [2^n, 2^(n+1))   wraps, bit n set
        reduced, b >= p    B = b_orig - p   in [0, F)           no wrap, bit n clear

    and cases 1 and 3 OVERLAP, so no single bit separates them. A "was it lazy" bit would,
    but computing it is the `b >= p` comparison the construction exists to avoid -- it can
    only be carried in from the schedule. Use `modadd_lazy` (v4) when the accumulator
    cannot be guaranteed canonical; it is closed for canonical+lazy.

        1.  B += A                B = a+b            in [0, 2p)
        2.  flag ^= (B >= p)
        3.  B -= p  if flag       B = (a+b) mod p
        4.  flag ^= (B < a)       uncomputes flag  (or is vented)

    Four passes instead of v1/v2's seven steps (five adder passes). Two things get
    better at once:

    * Steps 5 and 7 of the old construction spent two full adder passes re-deriving one
      bit; a comparator does it in one. Step 4 is valid because a reduction happened iff
      the result is below `a`: with no reduction B = a+b >= a, and with reduction
      B = a+b-p < a (since b < p).
    * The old step 4 *added* p, and p is dense (253 set bits). Here p is only ever
      *subtracted*, and -p folds into the pseudo-Mersenne form (B ^ 2^n) + f with f = 19.

    Peak is unchanged from v2 -- the comparators need their own carry ancillas, so this
    is a Toffoli/Clifford win, not a space one.
    """
    n = len(a)
    assert len(b) == n, "operand widths must match"
    assert p < (1 << n), "p must fit in n bits"
    if n == 0:
        return

    ovf = circ.alloc(1)      # b's (n+1)-th bit: holds the carry of a+b
    flag = circ.alloc(1)     # 1 iff a+b >= p (i.e. a reduction is needed)

    B = Register(list(b.ids) + [ovf[0]])
    f = (1 << n) - p

    add_fn(circ, a, B)                                            # 1
    cmp_ge_const(circ, B, p, flag[0])                             # 2
    _addsub_p(circ, B, n, f, False, flag[0], inplace_const, add_fn)  # 3
    if vent:
        m = circ.measure(flag[0])       # erases the flag with CERTAINTY, 0 Toffoli
        cmp_lt_phase(circ, a, B, m)     # ...and repairs the phase it left behind
    else:
        cmp_lt(circ, a, B, flag[0])                               # 4

    circ.free(flag)   # simulator asserts both are back to |0>
    circ.free(ovf)


def modadd_lazy(
    circ: Circuit,
    a: Register,
    b: Register,
    p: int = PRIME,
    add_fn: AddFn = gidney_add,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
    vent: bool = True,
) -> None:
    """APPROXIMATE, lazily reduced: b <- a + b (mod p), a preserved. v4.

    Schrottenloher (arXiv:2606.02235) Algorithm 10. Three steps, no separate flag qubit:

        1.  B += A                  B = (b, ovf); ovf is the carry out of bit U-1
        2.  b += F   if ovf         == subtract p, TRUNCATED to the low `carry_window`
        3.  erase ovf using (b < a) TRUNCATED to the top `cmp_bits`

    `vent` selects how step 3 erases the flag, and the default is the only sane choice:

      True  (default)  m = measure(ovf); phase ^= m * (b < a).
                       The measurement erases ovf with CERTAINTY and the comparator only
                       drives a phase, so a wrong (truncated) predicate costs a PHASE
                       error. One Toffoli CHEAPER than the alternative, because the top
                       carry becomes a conditioned CZ+Z instead of an AND.

      False            ovf ^= (b < a), written with a CX -- Algorithm 10 as published.
                       A wrong predicate fails OPEN: ovf is freed at |1>, entangled
                       garbage sitting beside a perfectly CORRECT value, so no output
                       check can see it. Kept only to demonstrate the difference; see
                       docs/decisions/03-approximate-arithmetic.md.

    Step 2 is where the pseudo-Mersenne shape pays. `ovf` already *holds* the 2^U, so
    subtracting p = 2^U - F means nothing more than adding F to the low limb and later
    dropping `ovf` -- there is not even a bit to flip. The overflow bit doubles as the
    reduction flag, so unlike v1..v3 there is no comparator to compute it and no separate
    flag qubit to allocate.

    Step 3 works because a reduction happened iff the result fell below `a`: with
    reduction b = a+b_0-p < a (as b_0 < p), without it b = a+b_0 >= a.

    CONTRACT CHANGE -- outputs are only *lazily* reduced.
    ---------------------------------------------------------------
    The test is `a+b >= 2^U`, not `a+b >= p`, so results land in [0, 2^U) rather than
    [0, p). The two differ on the F values in [p, 2^U), i.e. with probability
    F/2^U ~ 2^-250.8. That is not an error -- the result is always congruent to
    a+b mod p -- but it IS a redundant representation, and every downstream circuit must
    tolerate inputs up to 2^U-1 rather than p-1. Feeding a lazily-reduced input back in
    is safe: it only breaks step 3's erasure when b_0 lands in [p, 2^U), again ~2^-250.8.

    Failure probabilities (see qsm.params):
        truncated +F carry      F / 2^(carry_window+1)      ~ 2^-41.8
        truncated erasure       2^-(cmp_bits+1)             ~ 2^-41.0
        (lazy representation)   F / 2^U                     ~ 2^-250.8, not an error

    Note the carry term is F/2^(L+1), NOT the generic F/2^L: the `+F` is CONTROLLED on
    the overflow bit, so it only runs on the reduction branch (P -> 1/2) and half of all
    calls cannot lose a carry because they add nothing. Same P(flag) factor `moddouble`
    carries. Measured by `python -m qsm.audit rates`; it was quoted 2x high here for a
    long time, which was conservative but wrong.

    Set carry_window/cmp_bits to None for the exact version of the same algorithm --
    still lazily reduced, but with no truncation error.
    """
    n = len(a)
    assert len(b) == n, "operand widths must match"
    assert p < (1 << n), "p must fit in n bits"
    if n == 0:
        return

    u = n
    f = (1 << u) - p
    ovf = circ.alloc(1)                       # carry of a+b AND the reduction flag
    B = Register(list(b.ids) + [ovf[0]])

    add_fn(circ, a, B)                                                    # 1
    add_const_inplace(circ, b, f, ctrl=ovf[0], window=carry_window)       # 2

    if vent:                                                              # 3
        m = circ.measure(ovf[0])            # erases with CERTAINTY, 0 Toffoli
        cmp_lt_phase(circ, a, b, m, k=cmp_bits)
        # `measure` consumed ovf and recycled its id: nothing to free, and nothing
        # that *could* be left dirty.
    else:
        cmp_lt(circ, a, b, ovf[0], k=cmp_bits)
        circ.free(ovf)   # fails open if the truncated predicate was wrong


def modadd_lazy_ctrl(
    circ: Circuit,
    ctrl: int,
    a: Register,
    b: Register,
    p: int = PRIME,
    add_ctrl_fn: CtrlAddFn = gidney_add_ctrl,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> None:
    """`modadd_lazy` under a quantum control: b <- a + b (mod p) if ctrl, else b.

    Step 2 is free under the control; step 3 is not, and the difference is worth stating
    because only one of the two is the obvious answer:

        1.  B += A  if ctrl      the only step that changes shape
        2.  b += F  if ovf       ovf cannot be set unless step 1 ran, so this is already
                                 conditioned on ctrl at no cost
        3.  m = measure(ovf);    the erasure predicate must be gated too -- see below
            phase ^= m*ctrl*(b < a)

    **Step 3's control is not optional, and it is not free.** The vent works because the
    measured flag and the re-derived predicate `b < a` agree; with ctrl = 0 the flag is 0
    while the result is the untouched `b`, so `b < a` is a coin flip and they disagree on
    half of all *correct* runs. `m` is an X-basis outcome, uniformly random whatever the
    flag held, so "ctrl = 0 means m = 0" is simply false. What survives is a relative
    PHASE, which no value comparison can see; it took a phase-tracking simulation of the
    replay at p = 29 to find it. Gating `cmp_lt_phase` on `ctrl` costs one Toffoli --
    the chain never materialises the top carry, so `ctrl` only has to reach the two
    top-of-chain phase gates.

    So a controlled modular addition costs `add_ctrl_fn` minus `add_fn`, plus that one
    Toffoli -- for `gidney_add_ctrl`, one per operand bit and one for the vent. Everything
    else about the algorithm, including both truncation rates, is `modadd_lazy`'s; read
    its docstring for why step 3 erases and what "lazy" costs downstream.

    The vent itself is still mandatory. Writing the predicate into `ovf` with a CX (the
    `vent=False` branch) fails harder, not softer: it would leave `ovf` at |1> on that
    same half of correct runs, i.e. free a dirty ancilla rather than leak a phase.

    The dialog-GCD replay is the caller this exists for: `if b0: b += a mod p` is one of
    its three per-iteration steps and it runs 399 of them.
    """
    n = len(a)
    assert len(b) == n, "operand widths must match"
    assert p < (1 << n), "p must fit in n bits"
    if n == 0:
        return

    f = (1 << n) - p
    ovf = circ.alloc(1)                       # carry of a+b AND the reduction flag
    B = Register(list(b.ids) + [ovf[0]])

    add_ctrl_fn(circ, ctrl, a, B)                                         # 1
    add_const_inplace(circ, b, f, ctrl=ovf[0], window=carry_window)       # 2
    m = circ.measure(ovf[0])                                              # 3
    cmp_lt_phase(circ, a, b, m, k=cmp_bits, ctrl=ctrl)


def modsub_lazy_ctrl(
    circ: Circuit,
    ctrl: int,
    a: Register,
    b: Register,
    p: int = PRIME,
    add_ctrl_fn: CtrlAddFn = gidney_add_ctrl,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> None:
    """`modadd_lazy_ctrl` RUN BACKWARDS: b <- b - a (mod p) if ctrl, else b.

    Not "a subtraction that happens to undo an addition" -- it is `modadd_lazy_ctrl`'s
    three steps in reverse order, each replaced by its own inverse, so composing the two
    gives the identity on every input including the ones where both approximate:

        R3. ovf ^= ctrl & (b < a)   re-materialise the flag the forward MEASURED away
        R2. b -= F   if ovf         `add_const_inplace` -> `sub_const_inplace`
        R1. B -= A   if ctrl        B = (b, ovf); the borrow drives `ovf` back to |0>

    Step R1 is the controlled add with a complement layer either side: `~(~B + A)` is
    `B - A`, and with `ctrl = 0` the two X layers cancel, so it costs exactly what the
    forward's controlled add costs and needs no separate controlled subtractor.

    **This is where the forward's cheapest trick stops being free**, and it is worth being
    exact about the difference rather than calling the two directions symmetric.

    Forward, `ovf` is the TRUE carry of a+b, and the comparator `b < a` is only a proxy
    for it -- wrong when the accumulator lands in [p, 2^U) (`modadd_lazy_vent_phase_prob`)
    and wrong again when the comparator is truncated. The forward can afford both because
    it MEASURES `ovf` first: the flag is gone with certainty and the proxy only drives a
    phase, so a disagreement is a PHASE error beside a correct value.

    Backward there is nothing to measure. `ovf` has to be materialised, because R2 needs
    something to control the constant subtraction on, so the same disagreement writes the
    wrong flag and R2 subtracts F when it should not -- a VALUE error, and one that R1
    then strands as a dirty `ovf`. Same probability, louder symptom. Both terms are
    therefore VALUE terms in any `fail_analytic` that includes this circuit, where in the
    forward direction they were phase terms:

        truncated -F borrow     controlled_carry_fail_prob(carry_window)
        truncated comparator    cmp_fail_prob(cmp_bits)
        lazy accumulator        modadd_lazy_vent_phase_prob()   <- promoted to a value

    The consolation is that it cannot fail silently: a wrong flag leaves `ovf` at |1> and
    the free raises. Contrast `modsub_direct`, which is a subtraction in its own right
    with its own (better) predicate -- use that one when the value is what matters and
    nothing has to be undone. Use THIS one only to reverse a `modadd_lazy_ctrl`, which is
    what `inversion.unreplay` does 399 times.
    """
    n = len(a)
    assert len(b) == n, "operand widths must match"
    assert p < (1 << n), "p must fit in n bits"
    if n == 0:
        return

    f = (1 << n) - p
    ovf = circ.alloc(1)
    B = Register(list(b.ids) + [ovf[0]])

    cmp_lt(circ, a, b, ovf[0], k=cmp_bits, ctrl=ctrl)                    # R3
    sub_const_inplace(circ, b, f, ctrl=ovf[0], window=carry_window)      # R2

    for q in B:                                                          # R1
        circ.x(q)
    add_ctrl_fn(circ, ctrl, a, B)
    for q in B:
        circ.x(q)
    circ.free(ovf)


def modneg(
    circ: Circuit,
    a: Register,
    p: int = PRIME,
    window: int | None = None,
) -> None:
    """In place: a <- p - a. Requires a <= p; **corrupts above it**.

    Violating the precondition does not merely produce a lazy result -- it produces a
    NON-CONGRUENT one, which no downstream reduction repairs. `~(a + F - 1)` is `p - a`
    only while `a + F - 1` fits the register, so the failure set is exactly `a > p`
    (the top F-1 values). At u=11, f=19: `modneg(2030)` returns 18 where -2030 mod p is
    2028. Predicate: `qsm.params.modneg_corrupts`.

    This propagates into `modsub.v1`, which calls `modneg` on `a`. `modsub_direct.v2`
    does not call it and is immune. Frequency under a lazy schedule is
    (F-1)/2^U = 2^-250.9, so it is a scheduling constraint rather than a budget line --
    anything guaranteeing a canonical operand makes this exact.

    Two operations, and that is the whole circuit:

        1. a += F - 1        (18 for 2^255-19 -- sparse, 2 set bits, truncatable)
        2. X on every bit    (complement)

    Derivation. NOT is "all-ones minus", and p is all-ones minus F-1:

        ~x  =  (2^U - 1) - x            <- what NOT means
        p   =  (2^U - 1) - (F - 1)      <- p = 2^U - F IS all-ones minus F-1

    so  ~(a + F - 1) = (2^U - 1) - a - F + 1 = 2^U - F - a = p - a.

    Why F-1 and not F: NOT lands one BELOW 2^U - a, and that free step already pays one
    of the F. See docs/decisions/03-approximate-arithmetic.md.

    Why add-then-complement rather than complement-then-subtract: a small *subtract*
    would itself go through the complement trick (X, add, X) to keep its constant sparse,
    and those X layers cancel against the leading one. Same result, one X layer instead
    of three.

    NO flag, NO comparator, NO conditional -- so unlike modadd there is nothing to erase
    and no lazy-vs-canonical contract to choose. Exact by construction.

    Output range is [1, p]: the single input a = 0 gives p rather than 0. That is
    congruent (p == 0 mod p) and inside the lazy contract, but it IS the one value
    outside [0, p).

    `window` truncates step 1 to the low `window` bits, wrong exactly when
    (a mod 2^window) + (F-1) >= 2^window, i.e. P = (F-1) / 2^window ~ 2^-40.9. `a + F - 1`
    never overflows the register, since a < p = 2^U - F.
    """
    n = len(a)
    assert p < (1 << n), "p must fit in n bits"
    if n == 0:
        return
    f = (1 << n) - p
    add_const_inplace(circ, a, f - 1, window=window)   # no-op when f == 1
    for q in a:
        circ.x(q)


def modneg_ref(a: int, p: int) -> int:
    """Exact semantics of `modneg`: p - a, NOT (p - a) % p (they differ at a = 0)."""
    return p - a


def modsub(
    circ: Circuit,
    a: Register,
    b: Register,
    p: int = PRIME,
    add_fn: AddFn = gidney_add,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> None:
    """In place: b <- (b - a) mod p, a preserved and restored. v1, the "sandwich".

        modneg(a)              a <- p - a
        modadd_lazy(a, b)      b <- b + (p - a)  ==  b - a   (mod p)
        modneg(a)              a <- p - (p - a)  ==  a       restored

    Why this is the natural construction rather than a lucky one: `modadd_lazy`'s flag
    erasure needs the predicate `result < a'`, and here a' IS `p - a` -- the very value
    the first `modneg` had to materialise anyway. The sandwich hands the comparator its
    operand for free. (The direct construction has to earn that predicate another way;
    see `modsub_direct`.)

    Cost is 2 x modneg + 1 x modadd_lazy, against Litinski's curve-agnostic 6n = 1530.

    OUTPUT IS LAZY. Routing through `modadd_lazy` inherits its contract: results land in
    [0, 2^U), not [0, p). `modsub_direct` does NOT -- its borrow bit answers exactly the
    right question, so it is canonically reduced. Compare against `modsub_lazy_ref` here
    and `modsub_ref` there; a `% p` in the test would paper over the difference, which is
    the more interesting half of the comparison.

    LAZY INPUTS. `modneg` assumes a < p. Feeding it a lazily-reduced `a` in (p, 2^U)
    gives a result that is NOT congruent to -a -- it is off by F, a genuinely wrong
    value rather than an unreduced one. That is F-1 of the 2^U inputs, i.e.
    (F-1)/2^U ~ 2^-250.9, some 210 orders of magnitude below the truncation error this
    circuit already carries, so it is priced into the budget rather than forbidden by
    the contract. Note a = p exactly is FINE (it yields 0); only (p, 2^U) breaks.
    Predicate: `qsm.params.modneg_corrupts`. `modsub_direct` is immune -- it never calls
    `modneg`, and measures unaffected even with every operand lazy.
    """
    modneg(circ, a, p, window=carry_window)
    modadd_lazy(circ, a, b, p, add_fn,
                carry_window=carry_window, cmp_bits=cmp_bits, vent=True)
    modneg(circ, a, p, window=carry_window)


def modsub_direct(
    circ: Circuit,
    a: Register,
    b: Register,
    p: int = PRIME,
    add_fn: AddFn = gidney_add,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> None:
    """In place: b <- (b - a) mod p, a preserved. v2 -- one pass instead of three.

    **The most robust circuit in Layer 0 under a violated precondition**, though not
    immune. Its real requirement is `a - b <= p`, weaker than the usual `a < p` (which
    implies it), because it subtracts once and adds `p` back on a borrow -- so it can
    repair any difference down to `-p` and no further. Exhaustively, the failure set is
    exactly `{(a, b) : a - b > p}`, of size f(f-1)/2, i.e. **2^-502.6**.

    That is a CONJUNCTION -- a lazy `a` *and* a `b` below `a - p` -- where `modneg` needs
    only a single lazy operand (2^-250.9) and `modadd`/`modadd_cmp` dirty an ancilla on a
    single lazy accumulator. 250 orders of margin is a reason to prefer this over the
    sandwich beyond the 84-Toffoli saving. Predicate:
    `qsm.params.modsub_direct_corrupts`; table: `python -m qsm.audit preconditions`.

        1.  B -= A                  B = (b, ovf); ovf is the BORROW out of bit U-1
        2.  b -= F   if ovf         == add p, TRUNCATED to the low `carry_window`
        3.  erase ovf using (~a < b) TRUNCATED to the top `cmp_bits`

    Exactly `modadd_lazy` with every sign flipped, and it works for the mirror-image
    reason. Two's complement in U+1 bits makes the borrow land in `ovf`: b - a is
    negative iff b < a, in which case the register holds 2^(U+1) + b - a, so bit U is
    set and the low limb is 2^U + b - a. The answer we want is b - a + p =
    (2^U + b - a) - F, so subtracting F from the low limb and DROPPING ovf is the whole
    reduction -- the same "the overflow bit already holds the 2^U" identity that makes
    modadd cheap, read backwards.

    Step 3 is the only part that is not a mirror image. modadd could erase with
    `result < a`; here the EXACT separating predicate is `result >= p - a`, and `p - a`
    is not in any register -- materialising it is precisely what the `modsub` sandwich
    spends two `modneg` passes on. Instead use the pseudo-Mersenne identity

        p - a  ==  ~a - (F - 1)

    and compare against `~a`, which is free (a layer of X gates). The two predicates
    differ on a window of width F, so the erasure is wrong with probability

        F / p  ~  2^-250.75            (measured; matches to 3 significant figures at
                                        u = 6, 7, 8, 10)

    A wrong erasure costs PHASE only -- the flag is vented -- so the returned VALUE is
    correct for every input, at every width. That is worth stating plainly because it is
    what makes this version safe to use: the approximation is confined to the erasure
    path, exactly as in `modadd_lazy`.

    CAUTION FOR TESTS. F/p is only negligible when p is large. At u = 6 it is 4.8%, so
    an exhaustive small-width test WILL see phase errors in bulk; it must assert on
    values and treat phase as expected. Do not read a small-width phase rate as a defect.

    (`>= ~a` would give (F-1)/p instead of F/p -- 2^-250.83 vs 2^-250.75, i.e. nothing --
    and needs a polarity correction, since `cmp_lt_phase` computes a strict `<`. Not
    worth it.)

    Unlike the sandwich, this version is NOT lazy: the borrow bit answers exactly the
    question that matters, so canonical inputs give canonical outputs. See `modsub_ref`.

    Saves 84 Toffoli over the sandwich by never materialising p - a at all: two
    `modneg` passes collapse into one layer of X gates.
    """
    n = len(a)
    assert len(b) == n, "operand widths must match"
    assert p < (1 << n), "p must fit in n bits"
    if n == 0:
        return

    f = (1 << n) - p
    ovf = circ.alloc(1)                       # borrow of b-a AND the reduction flag
    B = Register(list(b.ids) + [ovf[0]])

    sub(circ, a, B, add_fn)                                               # 1
    sub_const_inplace(circ, b, f, ctrl=ovf[0], window=carry_window)       # 2

    m = circ.measure(ovf[0])                                             # 3
    for q in a:
        circ.x(q)                             # a -> ~a, free
    cmp_lt_phase(circ, b, a, m, k=cmp_bits)   # phase ^= m * [~a < b]
    for q in a:
        circ.x(q)                             # restore a


def moddouble(
    circ: Circuit,
    a: Register,
    p: int = PRIME,
    window: int | None = None,
) -> Register:
    """a <- 2a mod p. Returns the register holding the result (see RELABELLING below).

    The cheapest circuit in the layer, because doubling gives away two things that
    addition has to pay for.

        1.  shift left by 1     FREE -- a relabelling of wires, not a gate
        2.  b += F   if flag    flag is bit U-1 of `a`, already there
        3.  flag ^= bit 0       ONE CX. no comparator.

    **Free lunch 1: the doubling costs nothing.** `2a` is `a` with every bit moved up one
    place. In a reversible circuit that is not an operation -- it is a decision about
    which wire you call "bit 1". Allocate a fresh |0> for the new bit 0 and read the old
    top bit as the overflow. Zero gates.

    **Free lunch 2: the erasure predicate is the PARITY of the result.** `2a` is even and
    `p` is odd, so subtracting `p` flips the parity:

        no reduction:  result = 2a       EVEN
        reduction:     result = 2a - p   ODD

    So bit 0 of the result *is* the reduction flag, and erasing the flag is a single CX
    from bit 0 -- no comparator, no carry chain. Contrast `modadd`, which needs a
    full-width `result < a`, and `modsub_direct`, which needs `result > ~a` and settles
    for an approximate predicate. Here the predicate is exact AND free.

    Concretely, bit 0 of the shifted value starts at |0>; step 2 adds F, which is odd, so
    bit 0 ends up holding exactly `flag`. The CX cancels it and the flag qubit is
    recycled.

    So the ENTIRE cost is step 2: one controlled constant addition of F.

    RELABELLING. Because the shift is free, the result does not live on the qubits `a`
    came in on -- it lives on `[fresh, a_0, ..., a_{U-2}]`, and `a_{U-1}` is recycled.
    This function therefore RETURNS the result register; the caller must use the return
    value. Use `moddouble_inplace` if the value must stay on the original wires, which
    costs 2(U) CNOTs and no Toffoli.

    LAZY, like modadd: the test is `2a >= 2^U`, not `2a >= p`, so results land in
    [0, 2^U). `window` truncates step 2. Its failure rate is the doubling-specific
    `moddouble_carry_fail_prob`, not generic F/2^window: the controlled +F runs only when
    the top bit is set, and the shifted low limb can reach only even residues.

    Requires a < p. The scheduler must use `moddouble_lazy_wide` when that provenance is
    unavailable.
    """
    low, flag = _moddouble_core(circ, a, p, window)
    circ.free(Register([flag]))
    return low


def _moddouble_core(
    circ: Circuit,
    a: Register,
    p: int,
    window: int | None,
) -> tuple[Register, int]:
    """Shared body: returns (result register, the now-|0> flag qubit, NOT yet freed).

    The flag is handed back rather than freed so `moddouble_inplace` can use it as a
    move target while it is still legitimately owned. Freeing first and writing to it
    afterwards happens to work while nothing else allocates, but it is a
    use-after-free waiting for the first caller that does.
    """
    n = len(a)
    assert p < (1 << n), "p must fit in n bits"
    assert n >= 2, "need at least two bits to shift"
    f = (1 << n) - p

    fresh = circ.alloc(1)
    flag = a[n - 1]                                   # bit U-1: does 2a reach 2^U?
    low = Register([fresh[0]] + list(a.ids[: n - 1]), a.name)   # the shift, free

    add_const_inplace(circ, low, f, ctrl=flag, window=window)   # 2
    circ.cx(low[0], flag)                                       # 3: parity == flag
    return low, flag


def moddouble_ctrl_inplace(
    circ: Circuit,
    ctrl: int,
    a: Register,
    p: int = PRIME,
    window: int | None = None,
) -> Register:
    """if ctrl: a <- 2a mod p, on `a`'s own wires. `len(a)` Toffoli plus the constant add.

    **This is the one place `moddouble`'s free lunch is not on the menu**, and it is worth
    saying why rather than just charging for it. The shift is free because it is a decision
    about which wire is called "bit 1" -- and a decision is exactly what a circuit cannot
    make about data. Under a quantum control the relabelling has to become a real ladder,
    at `len(a)` Toffoli, which is a hundred times what the unconditional doubling costs.

    The construction is the mirror of `field.shift.controlled_rshift1`'s use in Phase 1:

        wide = a ++ |0>         a known zero above the top, so a rotate IS a shift
        lshift1 under ctrl      wide = 2a exactly, in n+1 bits; wide[n] is the fold flag
        += F if wide[n]         the same step 2 `moddouble` pays
        flag ^= ctrl & wide[0]  the parity erasure, GATED

    Only the last line differs from `moddouble`, and it is the subtle one. `moddouble`
    erases its flag with a bare CX because the result's parity *is* the flag -- true only
    because the low bit started at |0>. With `ctrl = 0` nothing shifted, `a[0]` is whatever
    the field element says, and a bare CX would write that into the flag and corrupt it on
    half of all inputs. The Toffoli costs one gate and makes the `ctrl = 0` branch exactly
    the identity.

    Lazy in the same sense as `moddouble` and with the same `moddouble_carry_fail_prob`
    per fired call -- but note it fires only when `ctrl` is set, so a caller compounding
    the rate over a loop should weight it by the control's probability, not the iteration
    count. `qsm.params.replay_j2_fail_prob` does that.

    Requires a < p when it fires, exactly as `moddouble` does. Returns `a` for symmetry
    with `moddouble`; the register is unchanged, since nothing was relabelled.
    """
    n = len(a)
    assert n >= 2, "need at least two bits to shift"
    assert p < (1 << n), "p must fit in n bits"
    f = (1 << n) - p
    assert f & 1, "parity erasure requires an odd pseudo-Mersenne deficit"

    hi = circ.alloc(1)
    wide = Register(list(a.ids) + [hi[0]])
    controlled_lshift1(circ, ctrl, wide)          # precondition: wide[n] is the fresh |0>
    flag = wide[n]                                # ctrl & (2a >= 2^n)
    add_const_inplace(circ, Register(wide.ids[:n]), f, ctrl=flag, window=window)
    circ.ccx(ctrl, wide[0], flag)                 # gated parity erasure -- see above
    circ.free(Register([flag]))
    return a


def modhalve_ctrl_inplace(
    circ: Circuit,
    ctrl: int,
    a: Register,
    p: int = PRIME,
    window: int | None = None,
) -> Register:
    """`moddouble_ctrl_inplace` RUN BACKWARDS: if ctrl: a <- a/2 mod p, on `a`'s own wires.

    Read the forward's four lines bottom-up, each replaced by its own inverse:

        flag ^= ctrl & a[0]     the gated parity, now CREATING the flag it there erased
        -= F if flag            `add_const_inplace` -> `sub_const_inplace`
        rshift1 under ctrl      the same Fredkin ladder walked the other way
        free flag               `wide[n]` is the zero the rotation carried back up

    **This is the one reversal in the design that costs its forward exactly**, and it is
    worth saying why, because the composite's other two reversals do not. `modhalve` is
    free-for-free against `moddouble` because the erasure predicate was a PARITY, exact at
    every width; `undo.un_to_dialog` pays a full comparator because Phase 1's predicate was
    a COMPARISON; `modsub_lazy_ctrl` pays two Toffoli because `modadd_lazy_ctrl`'s flag was
    MEASURED. Here the predicate is a parity *and* nothing was measured -- the forward
    erases with `ccx(ctrl, wide[0], flag)`, which is its own inverse -- so the reversal is
    the same gate count, the same width, and exact wherever the forward was.

    So jump 2's whole surcharge over jump 1 is unchanged by the direction: one controlled
    shift ladder per conditional halving, `len(a)` Toffoli plus the gated parity, going
    either way. `un_replay_j2_fail_prob` is `replay_j2_fail_prob` for the same reason.

    Preconditions are `modhalve`'s, inherited exactly and only when the control fires: `a`
    even, or `a >= F`, or the `-F` borrows out of the register and the result is not
    congruent (`params.modhalve_corrupts`). With `ctrl = 0` every line is the identity --
    the Toffoli, not a bare CX, is what makes that true, mirroring the forward.

    The `controlled_rshift1` precondition (`ctrl` set implies `wide[0] == 0`) is supplied by
    the line above it rather than by the caller: F is odd, so the `-F` clears bit 0 exactly
    on the branch where the flag fired, and on the other branch bit 0 was already the zero
    `moddouble_ctrl_inplace`'s shift put there.

    Returns `a` for symmetry with `moddouble_ctrl_inplace`; nothing is relabelled.
    """
    n = len(a)
    assert n >= 2, "need at least two bits to shift"
    assert p < (1 << n), "p must fit in n bits"
    f = (1 << n) - p
    assert f & 1, "parity recovery requires an odd pseudo-Mersenne deficit"

    flag = circ.alloc(1)
    wide = Register(list(a.ids) + [flag[0]])
    circ.ccx(ctrl, wide[0], flag[0])              # gated parity recovery
    sub_const_inplace(circ, a, f, ctrl=flag[0], window=window)
    controlled_rshift1(circ, ctrl, wide)          # wide[0] is |0> here -- see docstring
    circ.free(flag)
    return a


def moddouble_inplace(
    circ: Circuit,
    a: Register,
    p: int = PRIME,
    window: int | None = None,
) -> None:
    """`moddouble`, but the result ends up on the qubits `a` came in on.

    Only the bookkeeping differs. The relabelled result sits on
    `[fresh, a_0, ..., a_{U-2}]`; sliding it back onto `a` is a rotation by one place.

    Each step moves a value into a qubit that is provably |0> -- the flag was just erased,
    and each swap vacates the next one -- so it is a 2-CNOT move rather than a 3-CNOT
    SWAP. Total 2(U) CNOTs, **zero Toffoli**.

    Prefer `moddouble` where the caller can track the relabelling: in a doubling chain
    (a Montgomery ladder, say) the shifts compose into a single offset that can be
    tracked classically and never emitted at all.
    """
    n = len(a)
    res, flag = _moddouble_core(circ, a, p, window)
    # res = [fresh, a_0, ..., a_{n-2}]; we want those values on [a_0, ..., a_{n-1}].
    # That is a rotation by one place. Walk DOWNWARDS so each destination has already
    # been vacated: a_{n-1} is the just-erased flag (|0>), and every later step moves
    # into the qubit the previous one emptied.
    assert flag == a[n - 1]
    for i in range(n - 1, -1, -1):
        src, dst = res[i], a[i]
        if src == dst:
            continue
        circ.cx(src, dst)      # dst is provably |0>, so this COPIES
        circ.cx(dst, src)      # ...and this CLEARS the source. 2 CNOTs, not 3.
    circ.free(Register([res[0]]))   # the fresh qubit, now vacated and back to |0>


def modhalve(
    circ: Circuit,
    b: Register,
    p: int = PRIME,
    window: int | None = None,
) -> Register:
    """`moddouble` run backwards: b <- b/2 mod p. Returns the relabelled result.

    The cheapest inverse in the layer, and for once the reverse really is as cheap as the
    forward *and* as exact -- which the dialog GCD's Phase 1 is not (see
    `inversion.undo`). Read `moddouble`'s three steps bottom-up:

        3'. flag ^= bit 0       ONE CX -- and this time it CREATES the flag, not erases it
        2'. b -= F   if flag    the same controlled constant, subtracted
        1'. shift right by 1    FREE -- `[b_1, ..., b_(U-1), flag]`

    Step 3' is why nothing is lost. `moddouble`'s erasure predicate was the PARITY of its
    output, and parity is exact at every width -- there is no comparator here to truncate
    and no flag to vent. So a CX reads the fold decision straight back out of bit 0, and
    the only approximation in either direction is the windowed constant, at the same rate
    both ways (`moddouble_carry_fail_prob`: the forward adds F to an even limb, this
    subtracts F from an odd one, and each has floor(F/2) reachable danger residues).

    After 2', bit 0 holds |0> again -- it is the fresh qubit `moddouble` allocated for the
    shift -- so it is freed, and the flag becomes the new top bit. Net qubit change zero.

    DOMAIN -- `b` even, or `b >= F`. The dual of `moddouble`'s `a < p`, and it fails the
    same way: step 2' borrows out of the register for the floor(F/2) odd values below F,
    and the result is then not even congruent to b/2. It is a precondition rather than a
    check because `moddouble`'s IMAGE satisfies it for free -- the odd outputs are exactly
    `2a - p` for a in [2^(U-1), p), which is [F, p) -- so anything this circuit is asked
    to undo is inside the domain by construction. Predicate:
    `qsm.params.modhalve_corrupts`; frequency under a lazy schedule floor(F/2)/2^U =
    9/2^255 = 2^-251.8, the same order as every other lazy-contract term in the layer.

    So `modhalve(moddouble(a)) == a` for every `a < p`, which is `moddouble`'s own
    precondition. The reverse composition is NOT an identity on the whole lazy range: for
    b odd and large, (b+p)/2 can exceed p and `moddouble` refuses it. A halving chain is
    lazy-in, lazy-out and must not be fed back into `moddouble` without the usual
    provenance argument -- the asymmetry `moddouble_lazy_wide` exists to close in the
    other direction.

    RELABELLING, mirroring `moddouble`: the result lives on `[b_1, ..., b_(U-1), flag]`,
    not on the wires `b` came in on. Callers that track the offset (the replay loop does)
    pay nothing; `modhalve_inplace` slides it back for 2U CNOTs and no Toffoli.
    """
    low, spent = _modhalve_core(circ, b, p, window)
    circ.free(Register([spent]))
    return low


def _modhalve_core(
    circ: Circuit,
    b: Register,
    p: int,
    window: int | None,
) -> tuple[Register, int]:
    """Shared body: returns (result register, the emptied shift qubit, NOT yet freed).

    The mirror of `_moddouble_core`, and unfreed for the same reason: `modhalve_inplace`
    uses that qubit as the first destination of its rotation, and writing to a wire that
    has already gone back to the pool works only until something else allocates.
    """
    n = len(b)
    assert p < (1 << n), "p must fit in n bits"
    assert n >= 2, "need at least two bits to shift"
    f = (1 << n) - p
    assert f & 1, "parity recovery requires an odd pseudo-Mersenne deficit"

    flag = circ.alloc(1)
    circ.cx(b[0], flag[0])                                       # 3': parity == flag
    sub_const_inplace(circ, b, f, ctrl=flag[0], window=window)   # 2'

    low = Register(list(b.ids[1:]) + [flag[0]], b.name)          # 1': the shift, free
    return low, b[0]                                             # b[0] vacated by 2'


def modhalve_inplace(
    circ: Circuit,
    b: Register,
    p: int = PRIME,
    window: int | None = None,
) -> None:
    """`modhalve`, but the result ends up on the qubits `b` came in on.

    The mirror of `moddouble_inplace`, and the walk goes the other way: the result sits
    on `[b_1, ..., b_(U-1), flag]` and has to slide DOWN one place, so each destination is
    vacated by the step before it. `b_0` is |0> already (the constant subtraction emptied
    it), which is what makes every move 2 CNOTs rather than a 3-CNOT SWAP.
    """
    n = len(b)
    res, gone = _modhalve_core(circ, b, p, window)
    assert gone == b[0]
    for i in range(n):
        src, dst = res[i], b[i]
        if src == dst:
            continue
        circ.cx(src, dst)      # dst is provably |0>, so this COPIES
        circ.cx(dst, src)      # ...and this CLEARS the source. 2 CNOTs, not 3.
    circ.free(Register([res[n - 1]]))   # the flag, now vacated and back to |0>


def moddouble_lazy_wide(circ: Circuit, a: Register, p: int = PRIME) -> Register:
    """Once-fold an arbitrary lazy input and RETAIN the second carry. EXACT ONLY.

    `moddouble` requires `a < p`, because its U-bit output wraps on the top floor(F/2)
    values admitted by the wider lazy contract `a < 2^U`. This variant accepts that full
    range and returns U+1 bits:

        y = 2a                  if a < 2^(U-1)
        y = 2a - p              otherwise

    The second branch reaches `2^U + F - 2`, so the extra bit is information, not
    workspace: keeping it is what makes the map injective over all 2^U inputs, and it is
    the only Layer-0 circuit that closes the doubling non-closure rather than accepting it
    at 2^-251.8.

    The shift gives `[fresh|0>, a_0, ..., a_(U-1)] = 2a`. Copy the old top bit to a
    temporary control and clear its output position, implementing the `-2^U` part. A
    controlled `+F` over the FULL U+1-bit result then leaves any second carry in bit U.
    F is odd, so output parity still equals the fold flag and erases the control in one CX.

    **There is deliberately no `window` here, and cannot be.** The `+F` is injected at bit
    0, so a carry only reaches bit U by propagating through every bit between; truncating
    to a window of L bits is precisely the refusal to propagate past bit L-1. A windowed
    version does not merely risk losing the carry -- it can NEVER set bit U, at any window
    up to and including U itself, which makes it bit-for-bit identical to `moddouble` while
    costing an extra qubit. It was briefly registered as
    `field.moddouble_lazy_wide_trunc.v1`, and the dead end is kept measured by
    `test_a_windowed_lazy_wide_CANNOT_retain_the_carry`. Retention and truncation are
    mutually exclusive -- the same obstruction that stops a 250-bit-deep predicate
    surviving a 45-bit window (docs/decisions/03-approximate-arithmetic.md).

    So the scheduler's choice is: `moddouble*` when provenance gives a canonical input,
    this when it does not and the carry must be exact, `double_wide` when no fold is
    wanted. Whether the input is canonical is a scheduler fact; testing `a < p` here would
    reintroduce the full comparator the split exists to avoid.
    """
    n = len(a)
    assert n >= 2, "need at least two bits to shift"
    assert p < (1 << n), "p must fit in n bits"
    f = (1 << n) - p
    assert f & 1, "parity erasure requires an odd pseudo-Mersenne deficit"

    fresh = circ.alloc(1)
    wide = Register([fresh[0]] + list(a.ids), a.name)   # exact U+1-bit 2a
    flag = a[n - 1]
    saved = circ.alloc(1)

    circ.cx(flag, saved[0])                             # preserve the fold decision
    circ.cx(saved[0], flag)                             # subtract flag * 2^U
    add_const_inplace(circ, wide, f, ctrl=saved[0])     # full width: carry may reach bit U
    circ.cx(wide[0], saved[0])                          # parity == original flag
    circ.free(saved)
    return wide


def double_wide(circ: Circuit, a: Register) -> Register:
    """a <- 2a with NO reduction at all. Returns the widened register.

    **Zero gates. Not "cheap" -- literally none.** Doubling is a relabelling, and if we
    decline to reduce there is no flag to compute, no constant to add and nothing to
    erase. The register simply grows by one bit:

        [a_0, ..., a_{n-1}]   ->   [fresh|0>, a_0, ..., a_{n-1}]

    So a chain of k doublings costs 0 Toffoli, 0 Cliffords and +k qubits, against
    `moddouble`'s 45k Toffoli. That looks like a pure win, and within a run of doublings
    it is. The catch is what happens next, and it is worth stating precisely because it
    decides how far deferral pays:

    1. **The eventual reduction is not a generic circuit.** Folding U+k bits down to U is
       many-to-one on the full range, so no reversible circuit does it in isolation --
       the same obstruction as canonicalization (see
       docs/decisions/04-no-in-circuit-canonicalization.md). It IS injective on the
       *reachable* set (only p distinct values are reachable), so a circuit can exist,
       but it has to exploit that structure rather than reduce generically. In practice
       that means folding the reduction into an operation that can be reverse-computed --
       which is exactly what a multiply-then-reduce does, and why the reduction belongs
       to the multiplication layer.

    2. **Extra bits are quadratically expensive downstream.** A schoolbook U x U multiply
       costs ~U^2 partial products; at U+k it costs ~(U+k)^2, so +k bits adds ~2Uk. At
       U = 255 a SINGLE deferred bit costs ~510 in the next multiplication against the 45
       it saved -- an order of magnitude the wrong way.

    So the rule the numbers give: **defer freely between reductions, but never carry a
    deferred bit into a multiplication.** Whether that is worth anything depends on the
    point formulas -- in double-and-add, doublings are interleaved with additions rather
    than chained, so long free runs may simply not arise. That is a point-layer question
    and is not settled here.
    """
    fresh = circ.alloc(1)
    return Register([fresh[0]] + list(a.ids), a.name)


def double_wide_ref(a: int, k: int = 1) -> int:
    """Exact semantics of `double_wide`: plain 2^k * a, no reduction, no modulus."""
    return a << k


def moddouble_lazy_wide_ref(a: int, p: int, u: int) -> int:
    """Exact once-folded semantics for any lazy input `0 <= a < 2^u`.

    Unlike `moddouble_ref`, the result is intentionally allowed to use bit u rather than
    wrapping it away.
    """
    return 2 * a - p if a >= (1 << (u - 1)) else 2 * a


def moddouble_ref(a: int, p: int, u: int) -> int:
    """Exact semantics of `moddouble`: congruent to 2a, lazily reduced into [0, 2^u).

    Same lazy contract as `modadd_lazy` and for the same reason -- the free bit answers
    `2a >= 2^u`, not `2a >= p`.
    """
    return 2 * a - p if 2 * a >= (1 << u) else 2 * a


def modhalve_ref(b: int, p: int, u: int) -> int:
    """Exact semantics of `modhalve`, wrap and all: `(b + (b&1)*p) / 2` when b >= F.

    On the domain (`b` even, or `b >= F`) this is `(b + flag*p) // 2` -- congruent to
    b/2, lazily reduced -- and the exact left inverse of `moddouble_ref`, because the fold
    decision comes back as `b & 1`. Off it, the constant subtraction borrows out of the
    register and the result is not even congruent, which is what this reference reproduces
    rather than hides. Predicate: `qsm.params.modhalve_corrupts`.
    """
    f = (1 << u) - p
    flag = b & 1
    t = (b - flag * f) % (1 << u)                  # the borrow leaves the register
    return (t >> 1) | (flag << (u - 1))


def modsub_ref(a: int, b: int, p: int) -> int:
    """Exact semantics of `modsub_direct`: (b - a) mod p, CANONICALLY reduced.

    NOTE THE ASYMMETRY WITH `modadd_lazy_ref` -- there is no `u`, because the direct
    subtraction has no lazy contract to express. `modadd`'s free bit answers
    "is a+b >= 2^U?" while the question it needs is "is a+b >= p?", and that mismatch IS
    laziness. Subtraction's free borrow bit answers "is b < a?", which is exactly the
    question, with no gap. So `modsub_direct` returns a canonically reduced result for
    canonical inputs, while costing about the same as the lazy modadd -- subtraction gets
    for free what addition has to be lazy to afford.

    This does NOT describe the `modsub` sandwich; use `modsub_lazy_ref` for that.
    """
    return b - a + p if b < a else b - a


def modsub_lazy_ref(a: int, b: int, p: int, u: int) -> int:
    """Exact semantics of the `modsub` SANDWICH, which is lazy where the direct one is not.

    The sandwich routes through `modadd_lazy`, so it inherits laziness: it computes
    `modadd_lazy(p - a, b)`, whose test is `>= 2^u`, not `>= p`. Results are congruent to
    b - a but land in [0, 2^u).

    Worth keeping both references rather than comparing everything modulo p: the whole
    point of `modsub_direct` is that it is canonically reduced, and a `% p` in the test
    would hide exactly that difference.
    """
    return modadd_lazy_ref(modneg_ref(a, p), b, p, u)


def modadd_lazy_ref(a: int, b: int, p: int, u: int) -> int:
    """Exact semantics of `modadd_lazy` -- the deterministic function it approximates.

    Note this is NOT `(a+b) % p`: the result is congruent to it but may be lazily
    reduced, landing in [p, 2^u) instead of being folded down.
    """
    s = a + b
    return s - p if s >= (1 << u) else s


# ---------------------------------------------------------------------------
# Registered benchmark version.
# ---------------------------------------------------------------------------
_N = U


def _modadd_case(inplace_const: bool) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    modadd(circ, a, b, PRIME, gidney_add, inplace_const=inplace_const)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": (v["a"] + v["b"]) % PRIME}

    def gen(rng: random.Random) -> dict[str, int]:
        # inputs must already be reduced
        return {"a": rng.randrange(PRIME), "b": rng.randrange(PRIME)}

    return Case(circ, ref, gen)


@circuit_version("field.modadd.v1")
def _build_modadd_v1() -> Case:
    """
    Constant steps via a borrowed constant register (baseline).
    512 base = 255 + 255 + 2

    a+b overflow bit because a+b < 2p-2,
    1-bit flag
    (the old 1-bit zero-extension is gone: the adders zero-extend a short operand)

      5 constant register  -- holds f = 19, not p, via B -+ p = (B ^ 2^255) +- 19
    255 Gidney carries
    772
    """
    return _modadd_case(inplace_const=False)


@circuit_version("field.modadd.v2")
def _build_modadd_v2() -> Case:
    """
    Constant steps in place -- no constant register.
    512 base = 255 + 255 + 2

    a+b overflow bit because a+b < 2p-2,
    1-bit flag

    255 Gidney carries
    767
    """
    return _modadd_case(inplace_const=True)


def _modadd_lazy_case(carry_window: int | None, cmp_bits: int | None) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    modadd_lazy(circ, a, b, PRIME, gidney_add,
                carry_window=carry_window, cmp_bits=cmp_bits)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": modadd_lazy_ref(v["a"], v["b"], PRIME, _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.randrange(PRIME), "b": rng.randrange(PRIME)}

    fail = 0.0
    if carry_window is not None:
        # CONTROLLED on the overflow bit -> F/2^(L+1), not the unconditional F/2^L.
        fail += controlled_carry_fail_prob(carry_window, (1 << _N) - PRIME)
    if cmp_bits is not None:
        fail += cmp_fail_prob(cmp_bits)
    if not fail:
        return Case(circ, ref, gen)
    # `vent=True` erases the flag by measurement, so a wrong truncated predicate leaves
    # residual PHASE rather than a dirty ancilla. That is the designed failure mode, so
    # it is declared here -- the harness treats phase as fatal for everything else.
    return Case(circ, ref, gen,
                meta={"fail_analytic": fail, "tolerates_phase": True})


def _modneg_case(window: int | None) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    modneg(circ, a, PRIME, window=window)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": modneg_ref(v["a"], PRIME)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.randrange(PRIME)}

    if window is None:
        return Case(circ, ref, gen)
    return Case(circ, ref, gen,
                meta={"fail_analytic": carry_fail_prob(window, (1 << _N) - PRIME - 1)})


@circuit_version("field.modneg.v1")
def _build_modneg_v1() -> Case:
    """
    EXACT. a <- p - a as `a += 18; X all bits`. No flag, no comparator, no conditional.
    Litinski's curve-agnostic figure for modular negation is 2n = 510 Toffoli.
    """
    return _modneg_case(None)


@circuit_version("field.modneg_trunc.v1")
def _build_modneg_trunc_v1() -> Case:
    """
    APPROXIMATE. Same, with the `+18` confined to the low CARRY_WINDOW bits.
    P(wrong) = (F-1) / 2^CARRY_WINDOW = 18/2^45 ~ 2^-40.9.
    """
    return _modneg_case(CARRY_WINDOW)


@circuit_version("field.modadd_lazy.v4")
def _build_modadd_lazy_v4() -> Case:
    """
    APPROXIMATE + lazily reduced (Schrottenloher Alg. 10). The overflow bit doubles as
    the reduction flag, so there is no comparator to compute it and no flag qubit; the
    reduction is `+F` into the low CARRY_WINDOW bits; the erasure is a CMP_BITS-wide
    comparison.

    Outputs land in [0, 2^255), NOT [0, p) -- congruent, but redundantly represented.
    """
    return _modadd_lazy_case(CARRY_WINDOW, CMP_BITS)


@circuit_version("field.modadd_lazy_exact.v4")
def _build_modadd_lazy_exact_v4() -> Case:
    """
    Same algorithm with both truncations switched off: exact arithmetic, still lazily
    reduced. Isolates what the *lazy representation* buys from what the *truncation*
    buys, which the paper needs to report separately.
    """
    return _modadd_lazy_case(None, None)


@circuit_version("field.modadd.v3")
def _build_modadd_v3() -> Case:
    """
    Compare-then-subtract: 4 passes instead of 5 adder passes, and p is only ever
    *subtracted* (the cheap, sparse direction).
    512 base = 255 + 255 + 2

    255 comparator carries (`carry_out*` allocates n-1; the carry-OUT is the answer and
    goes straight to the flag rather than to an ancilla of its own)
    767
    """
    return _modadd_cmp_case(vent=False)


def _modadd_cmp_case(vent: bool) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    modadd_cmp(circ, a, b, PRIME, gidney_add, inplace_const=True, vent=vent)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": (v["a"] + v["b"]) % PRIME}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.randrange(PRIME), "b": rng.randrange(PRIME)}

    return Case(circ, ref, gen)


@circuit_version("field.modadd_vented.v3")
def _build_modadd_vented_v3() -> Case:
    """
    v3 with the flag erased by MEASUREMENT rather than by recomputing `B < a`.

    Exact and phase-clean on canonical input, one Toffoli CHEAPER than v3 (1021 vs 1022,
    the top carry becoming a conditioned CZ+Z instead of an `_and`), and it changes what
    happens when the accumulator is lazy: v3 leaves a DIRTY ANCILLA on every such input,
    this leaves residual PHASE, which is the failure class the rest of the layer already
    budgets for. Strictly better; `field.modadd.v3` stays registered as the published
    unvented baseline.

    It does NOT make a lazy accumulator correct -- nothing at this cost can, since the
    two failing cases overlap in the sign bit. Use `modadd_lazy` (v4) for that.
    """
    return _modadd_cmp_case(vent=True)


# ---------------------------------------------------------------------------
# modsub: two constructions, and the surprise is that the cheaper one has the
# better output contract.
# ---------------------------------------------------------------------------
def _modsub_case(direct: bool, carry_window: int | None, cmp_bits: int | None) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    fn = modsub_direct if direct else modsub
    fn(circ, a, b, PRIME, gidney_add, carry_window, cmp_bits)

    if direct:
        def ref(v: dict[str, int]) -> dict[str, int]:
            return {"a": v["a"], "b": modsub_ref(v["a"], v["b"], PRIME)}
    else:
        def ref(v: dict[str, int]) -> dict[str, int]:
            return {"a": v["a"], "b": modsub_lazy_ref(v["a"], v["b"], PRIME, _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.randrange(PRIME), "b": rng.randrange(PRIME)}

    fail = 0.0
    if carry_window is not None:
        # Both constructions gate the +F on a reduction flag (the sandwich through
        # modadd_lazy's overflow bit, the direct form through the borrow), so both take
        # the controlled rate F/2^(L+1). See qsm.params.controlled_carry_fail_prob.
        fail += controlled_carry_fail_prob(carry_window, (1 << _N) - PRIME)
    if cmp_bits is not None:
        fail += cmp_fail_prob(cmp_bits)
    meta: dict = {}
    if direct:
        # the `~a` erasure predicate is wrong on a window of width F -> F/(2p) phase.
        # 2^-251.75, invisible next to the truncations, but it IS a phase term.
        fail += F / (2.0 * PRIME)
        meta["tolerates_phase"] = True
    if fail:
        meta["fail_analytic"] = fail
    return Case(circ, ref, gen, meta=meta) if meta else Case(circ, ref, gen)


@circuit_version("field.modsub_trunc.v1")
def _build_modsub_v1() -> Case:
    """
    SANDWICH: modneg(a); modadd_lazy(a,b); modneg(a).

    Exact erasure -- materialising a' = p - a hands modadd's comparator exactly the
    operand it wants, so there is no phase term at all. Pays for it in Toffoli, and
    inherits modadd_lazy's LAZY output contract ([0, 2^255), not [0, p)).
    """
    return _modsub_case(False, CARRY_WINDOW, CMP_BITS)


@circuit_version("field.modsub_direct_trunc.v2")
def _build_modsub_v2() -> Case:
    """
    DIRECT: borrow flag, `-F if borrow`, vented erasure against `~a`.

    Cheaper than the sandwich AND canonically reduced -- the borrow bit answers exactly
    the question that matters, so subtraction needs no lazy contract. Costs a phase term
    of F/(2p) = 2^-251.75 because `p - a` is never materialised.
    """
    return _modsub_case(True, CARRY_WINDOW, CMP_BITS)


@circuit_version("field.modsub_direct.v2")
def _build_modsub_v2_notrunc() -> Case:
    """
    Same, truncations off. Isolates what the CONSTRUCTION buys from what the windows do.

    DELIBERATELY NOT NAMED `_exact`. Turning the windows off does not make this exact:
    the F/(2p) = 2^-251.75 phase term comes from the erasure PREDICATE (comparing against
    `~a` rather than the unavailable `p - a`), which no window setting touches. The
    `_exact` suffix on `modadd_lazy_exact.v4` is accurate because modadd's predicate
    `b < a` really is exact; carrying the suffix here would have claimed something false,
    and the benchmark table caught it -- `log2_fail = 2^-251.8` beside a name saying
    "exact".

    The genuinely exact modular subtraction is `field.modsub.v1`, the sandwich with
    windows off: materialising p - a is precisely what buys exactness, and precisely what
    costs the extra Toffoli.
    """
    return _modsub_case(True, None, None)


@circuit_version("field.modsub.v1")
def _build_modsub_v1_exact() -> Case:
    """
    The sandwich with truncations off: genuinely EXACT modular subtraction, no error
    term of any kind. Materialising a' = p - a hands modadd's comparator the exact
    operand it wants, so there is no phase term either.

    This is the honest exact baseline for the modsub line, and the price of exactness is
    visible against `modsub_direct.v2`: same construction family, 500-odd more
    Toffoli, no 2^-251.8.
    """
    return _modsub_case(False, None, None)


# ---------------------------------------------------------------------------
# moddouble: the cheapest circuit in the layer, and structurally so.
# ---------------------------------------------------------------------------
def _moddouble_case(inplace: bool, window: int | None) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    if inplace:
        moddouble_inplace(circ, a, PRIME, window)
    else:
        circ.rebind("a", moddouble(circ, a, PRIME, window))

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": moddouble_ref(v["a"], PRIME, _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.randrange(PRIME)}

    if window is None:
        return Case(circ, ref, gen)
    return Case(circ, ref, gen, meta={
        "fail_analytic": moddouble_carry_fail_prob(window, F, _N, PRIME)})


def _modhalve_case(inplace: bool, window: int | None) -> Case:
    circ = Circuit()
    b = circ.input("b", _N)
    if inplace:
        modhalve_inplace(circ, b, PRIME, window)
    else:
        circ.rebind("b", modhalve(circ, b, PRIME, window))

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"b": modhalve_ref(v["b"], PRIME, _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        # Canonical, to match the doubling cases and the rate helper. A lazy draw is the
        # more realistic distribution for the actual caller and gives P(odd) = exactly
        # 1/2 rather than (p-1)/2p, i.e. `moddouble_lazy_input_carry_fail_prob` -- the
        # same number halving and doubling share on lazy input.
        return {"b": rng.randrange(PRIME)}

    if window is None:
        return Case(circ, ref, gen)
    return Case(circ, ref, gen, meta={
        "fail_analytic": modhalve_carry_fail_prob(window, F, _N, PRIME)})


def _moddouble_lazy_wide_case() -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    circ.rebind("a", moddouble_lazy_wide(circ, a, PRIME))

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": moddouble_lazy_wide_ref(v["a"], PRIME, _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.randrange(1 << _N)}

    return Case(circ, ref, gen)


@circuit_version("field.moddouble.v1")
def _build_moddouble_v1() -> Case:
    """
    EXACT. a <- 2a mod p, lazily reduced.

    Two free lunches over modadd: the shift is a wire relabelling (zero gates), and the
    erasure predicate is the PARITY of the result -- 2a is even, p is odd, so subtracting
    p flips bit 0. One CX, no comparator. The entire Toffoli cost is the controlled +F.
    """
    return _moddouble_case(False, None)


@circuit_version("field.moddouble_trunc.v1")
def _build_moddouble_trunc_v1() -> Case:
    """
    APPROXIMATE, with the +F confined to the low CARRY_WINDOW bits.

    Note the failure rate is NOT the generic F/2^L: doubling is ~2x safer for free,
    because the +F only fires when the flag is set and the shifted value's low limb is
    always EVEN, so only floor(F/2) of the F dangerous residues are reachable.
    9/2^45 = 2^-41.8 against modadd's 19/2^45 = 2^-40.8.
    """
    return _moddouble_case(False, CARRY_WINDOW)


@circuit_version("field.moddouble_inplace.v2")
def _build_moddouble_inplace_v2() -> Case:
    """
    Same arithmetic, but the result is slid back onto the qubits `a` came in on.

    Identical Toffoli; costs 2n CNOTs of moves. Registered to make the price of the
    in-place CONVENTION visible: it is Cliffords only, and a caller that can track the
    relabelling (a doubling chain, say) should not pay it at all.
    """
    return _moddouble_case(True, None)


@circuit_version("field.moddouble_inplace_trunc.v2")
def _build_moddouble_inplace_trunc_v2() -> Case:
    """In-place convention plus the truncated +F."""
    return _moddouble_case(True, CARRY_WINDOW)


@circuit_version("field.modhalve.v1")
def _build_modhalve_v1() -> Case:
    """
    EXACT. b <- b/2 mod p, lazily reduced. `moddouble.v1` run backwards.

    The one place in the layer where reversal is genuinely free: doubling erased its flag
    with the output PARITY, which is exact at every width, so halving recreates it with
    one CX and pays nothing for the privilege. Contrast `inv.un_to_dialog_*`, where the
    erasure predicate is a comparator and the reverse pays the forward's price in full.

    Identical Toffoli to `field.moddouble.v1` -- pinned by
    `test_halving_costs_exactly_what_doubling_costs`.
    """
    return _modhalve_case(False, None)


@circuit_version("field.modhalve_trunc.v1")
def _build_modhalve_trunc_v1() -> Case:
    """
    APPROXIMATE, with the -F confined to the low CARRY_WINDOW bits.

    Same 2x-safer-for-free story as doubling, read from the other end: the -F only fires
    on an odd input, and an odd low limb can reach only floor(F/2) of the F residues that
    would borrow out. 9/2^45 = 2^-41.8. See `modhalve_carry_fail_prob`.
    """
    return _modhalve_case(False, CARRY_WINDOW)


@circuit_version("field.modhalve_inplace.v1")
def _build_modhalve_inplace_v1() -> Case:
    """The result slid back onto the wires `b` came in on. 2U CNOTs, zero Toffoli."""
    return _modhalve_case(True, None)


@circuit_version("field.modhalve_inplace_trunc.v1")
def _build_modhalve_inplace_trunc_v1() -> Case:
    """In-place convention plus the truncated -F."""
    return _modhalve_case(True, CARRY_WINDOW)


@circuit_version("field.moddouble_lazy_wide.v1")
def _build_moddouble_lazy_wide_v1() -> Case:
    """
    EXACT for every U-bit lazy input; the once-folded U+1-bit output retains carry U.

    There is no `_trunc` sibling, and the missing one is a result rather than an
    omission: truncation cannot coexist with carry retention, because the carry has to
    cross every bit from 0 to U and a window is the refusal to let it. See
    `moddouble_lazy_wide` and the doubling section of
    docs/decisions/03-approximate-arithmetic.md.
    """
    return _moddouble_lazy_wide_case()


# `double_wide` is deliberately NOT registered. A circuit version is a gate sequence the
# harness can benchmark; `double_wide` emits no gates at all, so every column except
# peak_qubits would be zero and the row would claim to measure something it does not.
# It is a change of REPRESENTATION -- U-bit `x` reinterpreted as (U+1)-bit `2x` -- which
# belongs to the scheduler's type bookkeeping, not to the benchmark table. Its zero cost
# is pinned by `test_double_wide_is_literally_free` instead, and
# `test_every_registered_version_actually_emits_gates` keeps the distinction enforced.
