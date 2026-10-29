"""Integer products: x*y and x^2 into a zeroed 2n-bit register.

No modulus here -- this is the raw product. The pseudo-Mersenne fold that turns it into
a field element lives in `mul/fold.py`, and the reason for the split is that the two have
completely different structure: the product is quadratic and dominates, the fold is
linear and is ~2% of it.

The add/subtract trick
----------------------
The naive schoolbook multiply issues `n` CONTROLLED additions, one per bit of `y`.
ecdsa.fail's own lower-bound analysis (`src/point_add/memory/03-proven-floors.md`) makes
the cost of that precise, and it is the single most useful fact for this layer:

    uncontrolled n-bit add:   floor n-1 Toffoli, TIGHT, achieved     (1.00 CCX/bit)
    controlled   n-bit add:   floor n, best known 2n                 (2.00 CCX/bit)

So a controlled add costs twice an unconditional one, and the factor-2 gap between the
proven floor and the best construction is an open problem, not an engineering oversight.
The way out is to stop asking for a controlled add. Adding `a` when `y_k = 1` and
`2^n - a` when `y_k = 0` is an UNCONDITIONAL add sandwiched between CNOT layers:

    ctrl=1:  B += a
    ctrl=0:  B += 2^n - a          (complement the accumulator, add, complement back)

Every row then costs `n` Toffoli instead of `2n`, and the rows compute a different
function whose difference from `x*y` is a closed form in `x` and `y` alone -- four extra
n-bit additions, ~1.5% of the total. That is the entire trick, and it is what
`schoolbook` and `square_symmetric` below are built on.

Squaring is half a multiply
---------------------------
    x^2 = SUM_j x_j 2^(2j)  +  2 * SUM_(i<j) x_i x_j 2^(i+j)

The diagonal is free (x_j^2 = x_j, so it is a CNOT layer onto a zeroed register, and a
zeroed register can absorb ANY XOR-linear function of x for zero Toffoli). The cross
terms appear twice, so only n(n-1)/2 of them are computed rather than n^2 -- and with the
add/subtract trick each row is a single unconditional add of a SHRINKING operand. Total
n(n-1)/2 Toffoli against the multiply's n^2: exactly half, and the halving is structural.

Correctness of both correction formulas was checked by exhaustive enumeration over all
inputs at n = 2..10 before either circuit was written, and is re-checked in-circuit by
`tests/test_mul_product.py`.
"""

from __future__ import annotations

from ..core.circuit import Circuit, Register
from ..field.add import gidney_add
from ..field.constadd import add_const_inplace, sub_const_inplace
from ..field.sub import sub


# ---------------------------------------------------------------------------
# The row primitive.
# ---------------------------------------------------------------------------
def addsub(circ: Circuit, a: Register, B: Register, ctrl: int,
           add_fn=gidney_add) -> None:
    """B += a if `ctrl` is |1>, else B += 2^n - a.  len(B) == len(a) + 1 == n + 1.

    One UNCONDITIONAL add (n Toffoli) plus 2n+4 Cliffords, against the 2n Toffoli a
    controlled add would cost. `ctrl` is preserved.

    The sign is applied to the ACCUMULATOR, not to the operand, which is what keeps the
    add narrow: complementing `a` would need a carry-in and would make the "-a" branch a
    full-width value, forcing every row to run the width of the whole product register.
    Complementing B's low n bits instead confines the arithmetic to n+1 bits.

    Identity, on the (n+1)-bit register with N(B) = B XOR (2^n - 1) (low n bits only):

        N(N(B) + a) + 2^n  ==  B - a + 2^n      (mod 2^(n+1))

    so the trailing CX onto B[n] is not bookkeeping -- it is the 2^n that makes the
    complement pair come out to a subtraction.
    """
    n = len(a)
    assert len(B) == n + 1, "the accumulator slice must be one bit wider than the operand"
    if n == 0:
        return
    circ.x(ctrl)                       # gates below fire on NOT ctrl
    for q in B[0:n]:
        circ.cx(ctrl, q)
    add_fn(circ, a, B)
    for q in B[0:n]:
        circ.cx(ctrl, q)
    circ.cx(ctrl, B[n])
    circ.x(ctrl)


def addsub_inverse(circ: Circuit, a: Register, B: Register, ctrl: int,
                   add_fn=gidney_add) -> None:
    """Inverse of `addsub`: the same CNOT layers in reverse with a subtraction inside."""
    n = len(a)
    assert len(B) == n + 1
    if n == 0:
        return
    circ.x(ctrl)
    circ.cx(ctrl, B[n])
    for q in B[0:n]:
        circ.cx(ctrl, q)
    sub(circ, a, B, add_fn)
    for q in B[0:n]:
        circ.cx(ctrl, q)
    circ.x(ctrl)


# ---------------------------------------------------------------------------
# Schoolbook multiply.
# ---------------------------------------------------------------------------
def _schoolbook_correction(circ: Circuit, x: Register, y: Register, wide: Register,
                           add_fn, inverse: bool) -> None:
    """The aggregate difference between the add/subtract rows and 2*x*y.

    Summing row k's contribution `2^k * [a*y_k + (1-y_k)(2^n - a)]` over k gives

        2xy - x*2^n + x + 2^2n - 2^n - y*2^n

    so the correction is `(x + y + 1)*2^n - x - 2^2n`. Three of those four terms are
    additions and the `-2^2n` is a single X on the top bit (mod 2^(2n+1) it is its own
    negation). ~5n Toffoli against the rows' n^2.

    The `+2^n` looks like it should be free -- an X on a still-|0> qubit before the rows
    run -- and it is not. Every row adds into a slice `wide[k : k+n+1]` whose top bit is
    exactly the top bit the running total can reach, so the register has ZERO headroom
    and anything pre-loaded makes some row's carry fall off the end of its slice. Adding
    it here instead costs n Toffoli and is the price of the narrow rows.
    """
    n = len(x)
    if inverse:
        circ.x(wide[2 * n])
        sub_const_inplace(circ, wide[n:2 * n + 1], 1)
        sub(circ, y, wide[n:2 * n + 1], add_fn)
        sub(circ, x, wide[n:2 * n + 1], add_fn)
        add_fn(circ, x, wide)
    else:
        sub(circ, x, wide, add_fn)                  # -x, full width (the borrow ripples)
        add_fn(circ, x, wide[n:2 * n + 1])          # +x * 2^n
        add_fn(circ, y, wide[n:2 * n + 1])          # +y * 2^n
        add_const_inplace(circ, wide[n:2 * n + 1], 1)   # +2^n
        circ.x(wide[2 * n])                         # -2^2n == +2^2n (mod 2^(2n+1))


def schoolbook(circ: Circuit, x: Register, y: Register, prod: Register,
               add_fn=gidney_add) -> None:
    """prod <- x * y.  len(prod) == 2n, and prod MUST arrive |0>.

    n^2 + 4n Toffoli: n rows of `addsub` at n each, plus the four-term correction.

    The rows accumulate into `wide = [low] + prod`, one bit BELOW prod, because the rows
    naturally compute `2xy` -- each contributes `a(2 y_k - 1)`, and the factor of two is
    absorbed for free by reading the answer off a register shifted up by one. `low` is
    provably |0> at the end (2xy is even) and is freed there.
    """
    n = len(x)
    assert len(y) == n, "operands must have equal width"
    assert len(prod) == 2 * n, "product register must be 2n bits"
    if n == 0:
        return

    low = circ.alloc(1)
    wide = Register([low[0]] + list(prod.ids))      # 2n+1 bits; wide == 2 * prod

    for k in range(n):
        addsub(circ, x, wide[k:k + n + 1], y[k], add_fn)
    _schoolbook_correction(circ, x, y, wide, add_fn, inverse=False)

    circ.free(low)


def schoolbook_inverse(circ: Circuit, x: Register, y: Register, prod: Register,
                       add_fn=gidney_add) -> None:
    """Inverse of `schoolbook`: returns a register holding x*y to |0>. Same cost."""
    n = len(x)
    assert len(y) == n and len(prod) == 2 * n
    if n == 0:
        return

    low = circ.alloc(1)
    wide = Register([low[0]] + list(prod.ids))

    _schoolbook_correction(circ, x, y, wide, add_fn, inverse=True)
    for k in reversed(range(n)):
        addsub_inverse(circ, x, wide[k:k + n + 1], y[k], add_fn)

    circ.free(low)


# ---------------------------------------------------------------------------
# Symmetric square.
# ---------------------------------------------------------------------------
def _square_correction(circ: Circuit, x: Register, prod: Register,
                       add_fn, inverse: bool) -> None:
    """The aggregate difference between the add/subtract rows and x^2.

    Row i adds `2^(2i+1) * [v_i x_i + (1-x_i)(2^k_i - v_i)]` with `v_i = x >> (i+1)`.
    The wanted part is `2 * SUM_(i<j) x_i x_j 2^(i+j)`; the unwanted part telescopes,
    using the identity

        SUM_i 2^(2i+1) (x >> (i+1))  ==  spread(x) - x,     spread(x) = SUM_j x_j 2^2j

    so the rows hand back a spurious `-spread(x)`, and the whole correction is

        2*spread(x)  -  x  +  2^n (x mod 2^(n-1))  +  2^n  -  2^(2n-1)

    ~6n Toffoli against the rows' n(n-1)/2.

    `2*spread(x)` -- x's bits at the ODD positions 2j+1 -- is the only awkward term: it
    is a strided view of `x`, not a contiguous register, so it is materialised as an
    interleaved operand `[z_0, x_0, z_1, x_1, ...]` over n borrowed |0> ancillas. The
    adder preserves its first operand, so those come back |0> and cost no Toffoli of
    their own. Half that operand is provably |0>, though, and our adder does not know it:
    a mask-aware adder (ecdsa.fail has one, `cuccaro_add_from_zero`) would drop one
    Toffoli per zero bit and take this term from 2n to n. Not taken yet -- it is a
    general adder improvement, and it belongs in `field/add.py` rather than here.

    It cannot be pre-loaded onto the zeroed register for free, for the same reason the
    multiply's `+2^n` cannot: the rows have no headroom, and this term reaches bit 2n-1.
    """
    n = len(x)
    c = 1 + (1 << (n - 1))              # (2^n + 2^(2n-1)) / 2^n ; -2^(2n-1) == +2^(2n-1)
    z = circ.alloc(n)
    interleaved = Register([q for j in range(n) for q in (z[j], x[j])])   # 2*spread(x)
    if inverse:
        sub_const_inplace(circ, prod[n:2 * n], c)
        sub(circ, x[0:n - 1], prod[n:2 * n], add_fn)
        add_fn(circ, x, prod)
        sub(circ, interleaved, prod, add_fn)
    else:
        add_fn(circ, interleaved, prod)                  # +2*spread(x)
        sub(circ, x, prod, add_fn)                       # -x, full width
        add_fn(circ, x[0:n - 1], prod[n:2 * n])          # +2^n * (x mod 2^(n-1))
        add_const_inplace(circ, prod[n:2 * n], c)        # +2^n - 2^(2n-1)
    circ.free(z)


def square_symmetric(circ: Circuit, x: Register, prod: Register,
                     add_fn=gidney_add) -> None:
    """prod <- x * x.  len(prod) == 2n, and prod MUST arrive |0>.

    n(n-1)/2 + 4n Toffoli -- half a general multiply, and the halving is structural
    rather than a constant-factor tuning: each unordered pair {i,j} is touched once and
    contributes twice, so the cross-term work is a triangle rather than a square.

    Row i is a single `addsub` of the SHRINKING operand `x[i+1:]` (n-1-i bits) at offset
    2i+1, so the rows cost n-1, n-2, ..., 1 rather than n each.
    """
    n = len(x)
    assert len(prod) == 2 * n, "product register must be 2n bits"
    if n == 0:
        return
    if n == 1:
        circ.cx(x[0], prod[0])          # x^2 = x for one bit
        return

    for i in range(n - 1):
        k = n - 1 - i
        addsub(circ, x[i + 1:n], prod[2 * i + 1:2 * i + k + 2], x[i], add_fn)
    _square_correction(circ, x, prod, add_fn, inverse=False)


def square_symmetric_inverse(circ: Circuit, x: Register, prod: Register,
                             add_fn=gidney_add) -> None:
    """Inverse of `square_symmetric`: returns a register holding x^2 to |0>."""
    n = len(x)
    assert len(prod) == 2 * n
    if n == 0:
        return
    if n == 1:
        circ.cx(x[0], prod[0])
        return

    _square_correction(circ, x, prod, add_fn, inverse=True)
    for i in reversed(range(n - 1)):
        k = n - 1 - i
        addsub_inverse(circ, x[i + 1:n], prod[2 * i + 1:2 * i + k + 2], x[i], add_fn)
