"""Compile-time layout of the dialog-GCD loop: how many iterations, how wide, where.

A quantum circuit cannot branch on data, so every data-dependent quantity in the binary
GCD has to be replaced by a fixed schedule chosen at build time. There are three:

  `iterations(n)`   how many times to run the loop. Data-dependent in the algorithm;
                    fixed here, with a tail failure probability (`params.GCD_ITER_VAR`).
  `width(i, n)`     how wide the `u`/`v` window is at iteration i. The values shrink,
                    so the adders can too -- this is the largest gate saving in Phase 1.
  `dialog_layout()` which qubits hold the compressed tape, and where the freed high bits
                    of `u`/`v` are, because that is where the tape lives.

Every constant these read is named in `qsm.params`; nothing is hard-coded here.

All three are *classical* functions of the position in the loop, never of the register
contents, which is exactly the "compile-time branching on algorithmic position" that
`params`' failure-predicate note calls out as legal.
"""

from __future__ import annotations

from ..params import (
    GCD_CMP_MARGIN,
    GCD_ITER_BLOCK,
    GCD_WIDTH_PAD_VAR,
    GCD_J2_WIDTH_RATE,
    GCD_WIDTH_RATE,
    GCD_WIDTH_STEP,
    U,
    gcd_compare_fail_prob,
    gcd_dialog_bits,
    gcd_iteration_fail_prob,
    gcd_iterations,
    gcd_j2_dialog_bits,
    gcd_j2_iteration_fail_prob,
    gcd_j2_iterations,
)

__all__ = [
    "iterations",
    "dialog_bits",
    "iteration_fail_prob",
    "width_pad",
    "width",
    "widths",
    "iterations_j2",
    "dialog_bits_j2",
    "iteration_fail_prob_j2",
    "width_j2",
    "widths_j2",
    "conservative_iterations_j2",
    "compare_bits",
    "compare_fail_prob",
    "block_of",
    "position_in_block",
    "dialog_slice",
]

# Re-exported so the inversion package has one import for its schedule.
iterations = gcd_iterations
dialog_bits = gcd_dialog_bits
iteration_fail_prob = gcd_iteration_fail_prob
compare_fail_prob = gcd_compare_fail_prob


def _ceil(t: float) -> int:
    i = int(t)
    return i if i == t else i + 1


def width_pad(n: int = U, var: float = GCD_WIDTH_PAD_VAR) -> int:
    """Safety margin on the shrinking window: ceil(var * sqrt(n)). 37 at n = 255."""
    return _ceil(var * (n ** 0.5))


def width(
    i: int,
    n: int = U,
    rate: float = GCD_WIDTH_RATE,
    pad: int | None = None,
    step: int = GCD_WIDTH_STEP,
) -> int:
    """Bits of `u` / `v` the circuit operates on at iteration `i`.

        clamp(ceil(n - rate*i + pad), 0, n), rounded up to a multiple of `step`

    `rate` is half `GCD_SHIFT_RATE`: one bit leaves `v` per iteration and the pair sheds
    half a bit. `pad` covers the spread; at n = 255 the default pad of 37 left ~12 bits
    of mid-loop headroom across 798,000 measured iteration-slots (`params` documents the
    measurement). Rounding to `step` costs at most one bit and keeps the window aligned
    for chunked arithmetic.

    Anything above this window is *truncated*, not carried -- and the freed bits are
    reused for the tape -- so widening `pad` is the only correct response to a clip.
    """
    if pad is None:
        pad = width_pad(n)
    w = min(max(_ceil(n - rate * i + pad), 0), n)
    return min(_ceil(w / step) * step, n)


def compare_bits(n: int = U, margin: int = GCD_CMP_MARGIN, pad: int | None = None) -> int:
    """Comparator width inside the shrinking window: `pad + margin`. 100 at n = 255.

    The width is quoted relative to the pad because that is the only part of it that buys
    anything. The window sits `slack` bits above the values by construction, so the top
    `slack` compared bits are zero in both operands and only `width - slack` are
    informative; widening the pad by a bit therefore costs a bit of comparator to stand
    still. `params.gcd_compare_fail_prob` prices the difference, and the measurement
    behind it -- including why the natural `CMP_BITS` figure of `2^-41` is 19 bits
    optimistic here -- is documented alongside it.

    `CMP_BITS` is still the right constant for a comparator on two full-width registers;
    it is only wrong inside a padded window, which is a situation unique to this loop.

    Once `width(i)` drops below this, `cmp_lt` compares everything and is exact, so the
    last ~93 of the 399 iterations pay nothing for the truncation either way.
    """
    if pad is None:
        pad = width_pad(n)
    return pad + margin


def widths(n: int = U, iters: int | None = None, **kw) -> list[int]:
    """The whole width schedule, one entry per iteration."""
    if iters is None:
        iters = iterations(n)
    return [width(i, n, **kw) for i in range(iters)]


# --- the same three, for jump 2 --------------------------------------------
#
# Everything above is per-*divstep*. A jump-2 iteration consumes one or two of them, so
# only the iteration count and the shed rate rescale; the pad and the comparator margin
# are properties of the value spread and carry over unchanged.
iterations_j2 = gcd_j2_iterations
dialog_bits_j2 = gcd_j2_dialog_bits
iteration_fail_prob_j2 = gcd_j2_iteration_fail_prob


def width_j2(i: int, n: int = U, rate: float = GCD_J2_WIDTH_RATE, **kw) -> int:
    """`width` at the jump-2 shed rate of 1.0620 bits per iteration. See `width`."""
    return width(i, n, rate=rate, **kw)


def widths_j2(n: int = U, iters: int | None = None, **kw) -> list[int]:
    """The jump-2 width schedule, one entry per iteration.

    Not `widths` resampled: an iteration eats a variable number of halvings, so the
    per-index schedule is noisier than the per-divstep one even though the *rate* is
    exactly the jump-1 rate divided by the iteration ratio (see `params`). Measured from
    `reference.dialog_widths_jump`, which exists for this reason.
    """
    if iters is None:
        iters = iterations_j2(n)
    return [width_j2(i, n, **kw) for i in range(iters)]


def conservative_iterations_j2(n: int) -> int:
    """Deterministic full-width bound for canonical 0 < x < odd prime p < 2**n.

    The boundary iteration begins with u=p odd, v=x. It halves v at most
    twice, swaps iff the resulting v is odd (then p>v), and subtracts the
    smaller odd number. Thus u stays positive odd and v becomes even and
    nonnegative; both remain <2**n and gcd(u,v)=1. The boundary cannot end
    at v=0 for x<p unless it already reached the terminal pair.

    At each steady iteration v is first halved unconditionally. A second
    halving and, if needed, ordering/subtraction cannot increase u*v.
    Whenever both old and new products are positive, the new product is
    at most half the old one. Starting below 2**(2*n), there cannot be
    2*n successive positive steady products: that would give an integer
    strictly between zero and one. Hence one boundary plus 2*n steady
    iterations reaches v=0. Gcd preservation forces u=1. Padding is stable:
    v=0 has c=1, b0=b1=0; shifts of zero and the inactive operations do
    nothing. No width clipping or probabilistic input assumption is used.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("GCD width must be an integer at least two")
    return 2 * n + 1


# --- where the compressed tape lives ---------------------------------------
def block_of(i: int, block: int = GCD_ITER_BLOCK) -> int:
    """Index of the compression block iteration `i` writes into."""
    return i // block


def position_in_block(i: int, block: int = GCD_ITER_BLOCK) -> int:
    """Which of the `block` slots inside that block iteration `i` is."""
    return i % block


def dialog_slice(
    i: int, block: int = GCD_ITER_BLOCK, bits: int = 5
) -> tuple[int, int]:
    """[lo, hi) qubit offsets of the tape word iteration `i` contributes to.

    Offsets are into the dialog register as a whole; the caller decides whether that
    register is standalone (simple, costs 665 qubits) or a view over the high bits that
    `u` and `v` have vacated (what makes Phase 1 fit in ~927).
    """
    b = block_of(i, block)
    return b * bits, (b + 1) * bits
