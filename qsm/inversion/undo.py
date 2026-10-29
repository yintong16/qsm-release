"""Phase 1 run backwards: |d> -> |x>, giving the input back and spending the tape.

`modmul_inplace` is `to_dialog(x)`, then the replay, then *this*. It has to exist as its
own circuit rather than as a generic reversal, because `qsm.core.circuit` has no inverse
operator and could not have a mechanical one: Gidney's `_unand` is a measurement, and
reversing a measurement is not a gate transformation.

Written out, the reverse loop is `gcd.gcd_loop`'s body read bottom-up:

    v <<= 1                 undo the free shift -- a rotation of the id list
    if b0: v += u           the controlled subtract, the other way round
    if s:  swap(u, v)       cswap is its own inverse
    s ^= b0 & (u > v)       ERASE s -- the same controlled comparator
    b0 ^= v & 1             erase b0 -- one CNOT

**The comparator does not go away, and that is the whole cost story of the composite.**
Reconstructing the *values* needs no comparison at all -- `reference.un_dialog` does it
with a shift, an add and a swap. But the tape cannot simply be left behind: `d` is
entangled with `x`, so it has to come back to |0>, and the only way to erase `s` is to
recompute `b0 & (u > v)` on the restored operands. Same predicate, same width, same
truncation. So the inverse costs exactly what the forward costs, and the composite is
2 x Phase 1 + Phase 2 -- which is what both baselines report (frenchie:
`218,432 * 2 + 568,214 = 1,005,078` exactly).

MEASURED at n = 255 on the matched pair, which is `inv.to_dialog_packed.v1` against
`inv.un_to_dialog_trunc.v1` -- packed both ways, truncated both ways:

    forward   217,769 Toffoli   peak 1,112   depth 545,106
    reverse   217,769 Toffoli   peak 1,111   depth 643,146

Equal on gates to the gate, one qubit *narrower*, and 98,040 DEEPER -- +18%. That last
one is the part to carry into a paper table: "2 x Phase 1" is exact for Toffoli count and
wrong for depth, so a composite depth has to be summed region by region rather than
doubled. The extra depth is the erasure comparator sitting on the critical path where the
forward had it overlapping the subtract it was feeding.

The *unmatched* pair is not equal and should not be quoted as if it were: the exact
`inv.un_to_dialog.v1` costs 243,149, because an exact comparator cannot undo a tape a
truncated one wrote. Inside `modmul_inplace` both halves truncate identically on identical
operands, so the matched pair is the only one that ever runs.

Everything the forward loop *frees* the reverse loop *allocates*, and vice versa:

    forward                              reverse
    window narrows, slots -> tape        tape retires, slots -> window
    tape allocates a top-up qubit        the loop frees a surplus qubit

and both directions get that for free from `Circuit.free`/`Circuit.alloc`, which recycle
ids through one pool and track the peak off `_live`. So the reverse needs no bookkeeping
of its own: retire a tape qubit the moment its pair is erased, allocate the moment the
window widens, and the peak comes out mirrored without a schedule.

Algorithm from [GRLGS23] PRL 131(4) 040602 and [KSGZ+25] arXiv:2510.10967; written from
the papers, not from either reference implementation (both are AGPL-3.0).
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add_ctrl, gidney_add_ctrl_threaded
from ..field.compare import cmp_lt
from ..field.shift import controlled_lshift1
from ..field.swap import cswap
from ..params import PRIME, U, gcd_width_clip_fail_prob
from . import pack, schedule
from .gcd import decode_dialog, decode_dialog_j2, encode_dialog, encode_dialog_j2
from .reference import to_dialog as to_dialog_ref
from .reference import to_dialog_jump as to_dialog_jump_ref
from .reference import un_dialog, un_dialog_jump

__all__ = [
    "FlatUnTape",
    "FlatUnTapeJ2",
    "PackedUnTape",
    "PackedUnTapeJ2",
    "un_gcd_loop",
    "un_gcd_loop_j2",
    "un_to_dialog",
    "un_to_dialog_j2",
]

# The mirror of `gcd.py`'s tape protocol. `gcd_loop` asks a layout for a pair and then
# folds it away; `un_gcd_loop` asks for the same pair and then RETIRES it, because by the
# time the reverse body is done both qubits are |0> and the loop wants them back.
#
#   pair(i) -> the two qubits iteration i wrote, unfolded if the layout packs
#   ... the reverse body, which erases both ...
#   retire(i) -> hand every qubit the layout has finished with back to the circuit
#   close() -> nothing should be left


class FlatUnTape:
    """Iteration `i` reads (d[2i], d[2i+1]) and frees the pair. Mirror of `gcd.FlatTape`.

    The dedicated-register layout. It donates 2 qubits per iteration back to the circuit's
    pool as it goes, which is where the widening window gets most of its wires from -- so
    even the "simple" layout is not simply 798 qubits held to the end.
    """

    def __init__(self, circ: Circuit, d: Register) -> None:
        self.circ = circ
        self.d = d

    def pair(self, i: int) -> tuple[int, int]:
        return self.d[2 * i], self.d[2 * i + 1]

    def retire(self, i: int) -> None:
        self.circ.free(Register(list(self.pair(i))))

    def close(self) -> None:
        """Every pair was retired as it was read; nothing is held."""


class PackedUnTape:
    """Read a `pack`-format tape backwards and consume it. Mirror of `gcd.PackedTape`.

    The reverse loop meets a group's digits in the order 2, 1, 0, which is the order
    `unfold_digit` peels them off in -- the same coincidence `replay.PackedReader` relies
    on. The difference is that this one never folds anything back: once digit `j` is out,
    the accumulator has shrunk to `DIGIT_WIDTH[j-1]` and the bits above it are |0>
    permanently, so they are freed rather than reclaimed. After digit 0 the accumulator
    *is* the pair, and erasing the pair empties the group.

    Per group of 3 iterations, in reverse order:

        digit 2   alloc 2 for the pair, free 1 accumulator bit, free the pair on retire
        digit 1   alloc 2 for the pair, free 2 accumulator bits, free the pair on retire
        digit 0   the pair is the accumulator's bottom two qubits; free them on retire

    -- 4 allocated and 9 freed, a net 5 handed back, which is the group's width. The
    allocations are satisfied out of the ids the previous digit just freed, so the reader
    carries 2 working qubits, not 2 per group. Its transient peak is higher: `cmp_ge_const`
    borrows carry ancillas to read a digit back, exactly as it does in the replay.
    """

    def __init__(self, circ: Circuit, d: Register, iters: int) -> None:
        assert len(d) == pack.packed_bits(iters), (
            f"a {iters}-iteration packed tape is {pack.packed_bits(iters)} qubits, "
            f"not {len(d)}"
        )
        self.circ = circ
        self._groups: list[Register] = []
        off = 0
        for g in range(0, iters, pack.GROUP):
            w = pack.group_bits(min(pack.GROUP, iters - g))
            self._groups.append(Register(d.ids[off:off + w]))
            off += w
        self._pair: tuple[int, int] | None = None

    def pair(self, i: int) -> tuple[int, int]:
        g, j = divmod(i, pack.GROUP)
        ids = self._groups[g].ids
        r = Register(ids[:pack.DIGIT_WIDTH[j]])
        p = (r[0], r[1]) if j == 0 else tuple(self.circ.alloc(2).ids)
        pack.unfold_digit(self.circ, r, j, *p)
        if j:
            # The digit is out, so the accumulator is DIGIT_WIDTH[j-1] wide and the bits
            # above it are |0> for good. Nothing in this direction ever folds them again.
            self.circ.free(
                Register(ids[pack.DIGIT_WIDTH[j - 1]:pack.DIGIT_WIDTH[j]])
            )
        self._pair = p
        return p

    def retire(self, i: int) -> None:
        assert self._pair is not None, "retire without a pair"
        self.circ.free(Register(sorted(self._pair)))
        self._pair = None

    def close(self) -> None:
        assert self._pair is None, "a pair was left live"


def un_gcd_loop(
    circ: Circuit,
    u: Register,
    v: Register,
    d: Register | FlatUnTape | PackedUnTape,
    iters: int,
    cmp_bits: int | None = None,
    widths: list[int] | None = None,
    add_ctrl=gidney_add_ctrl,
    n: int | None = None,
) -> tuple[Register, Register]:
    """Undo `iters` divsteps, consuming the tape. `(u, v)` in are the FINAL windows.

    Call it with the pair `gcd_loop` returned -- width `widths[iters-1]`, holding (1, 0)
    on a converged run -- and it gives back the pair `gcd_loop` was handed, width `n`,
    holding (p, x). `cmp_bits`, `widths` and `add_ctrl` must be the same values the
    forward loop ran with, because the reverse re-derives `s` with the same comparator and
    a different one would disagree with the tape and leave it dirty.

    The window GROWS: `widths` is walked from the back, and the extra wires come from
    `Circuit.alloc`, which hands back the ids the tape has just retired.
    """
    if widths is None:
        widths = [n if n is not None else len(u)] * iters
    if n is None:
        n = widths[0]
    assert len(widths) >= iters, "width schedule is shorter than the loop"
    tape = FlatUnTape(circ, d) if isinstance(d, Register) else d

    uw, vw = u, v
    assert len(uw) == len(vw) == widths[iters - 1], (
        f"the reverse loop starts on the final window ({widths[iters - 1]}), got {len(uw)}"
    )
    for i in reversed(range(iters)):
        # Undo `v >>= 1`. The forward rotated the low wire to the top of the window; the
        # top wire is |0> again by now, so rotating it back down is the same free shift.
        vw = Register(vw.ids[-1:] + vw.ids[:-1], vw.name)

        b0, s = tape.pair(i)
        add_ctrl(circ, b0, uw, vw)                    # undo `if b0: v -= u`
        cswap(circ, s, uw, vw)                        # cswap is its own inverse
        cmp_lt(circ, uw, vw, s, k=cmp_bits, ctrl=b0)  # s ^= b0 & (v < u)  ->  0
        circ.cx(vw[0], b0)                            # b0 ^= v & 1        ->  0
        tape.retire(i)

        w_next = widths[i - 1] if i else n
        assert w_next >= len(uw), "the width schedule must not narrow going backwards"
        grow = w_next - len(uw)
        if grow:
            uw = Register(uw.ids + list(circ.alloc(grow).ids), uw.name)
            vw = Register(vw.ids + list(circ.alloc(grow).ids), vw.name)

    return uw, vw


def un_to_dialog(
    circ: Circuit,
    d: Register | FlatUnTape | PackedUnTape,
    n: int,
    p: int = PRIME,
    iters: int | None = None,
    widths: list[int] | None = None,
    **kw,
) -> Register:
    """|d> -> |x>: spend the tape, give the input back. The inverse of `gcd.to_dialog`.

    The prime comes back in the scratch register and is XORed away with the same X gates
    that put it there, so `p` stays a compile-time constant in both directions. A tape
    that is not a real dialog -- or one replayed with a comparator the forward did not
    use -- leaves `u != p` or a tape qubit at |1>, and the free below turns that into a
    `DirtyAncillaError` rather than a wrong answer.
    """
    if iters is None:
        iters = schedule.iterations(n)
    w_last = widths[iters - 1] if widths is not None else n
    tape = FlatUnTape(circ, d) if isinstance(d, Register) else d

    u = circ.alloc(w_last)
    v = circ.alloc(w_last)
    circ.x(u[0])                                  # the state the forward loop ended in

    uw, vw = un_gcd_loop(circ, u, v, tape, iters, widths=widths, n=n, **kw)
    tape.close()

    for j in range(n):
        if (p >> j) & 1:
            circ.x(uw[j])
    circ.free(uw)
    return vw


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
#
# Three points, mirroring the forward's: the flat tape, the packed tape, and the packed
# tape with the comparator truncated -- which is the one `modmul_inplace` actually runs.
# The claim to check here is the parity one: each of these must cost the SAME Toffoli as
# the forward version it undoes, because it is the same gates in the other order.
# ---------------------------------------------------------------------------
_N = U
_ITERS = schedule.iterations(_N)


def _un_to_dialog_case(cmp_bits: int | None, layout: str = "packed") -> Case:
    widths = schedule.widths(_N, _ITERS)
    circ = Circuit()
    if layout == "flat":
        tape = FlatUnTape(circ, circ.input("d", 2 * _ITERS))
    else:
        tape = PackedUnTape(circ, circ.input("d", pack.packed_bits(_ITERS)), _ITERS)
    circ.input("x", 0)                            # x has no wires yet; it is the output
    x = un_to_dialog(circ, tape, _N, PRIME, iters=_ITERS, cmp_bits=cmp_bits,
                     widths=widths)
    circ.rebind("x", x)
    circ.rebind("d", Register([], "d"))           # the tape is spent, not just zero
    packed = layout != "flat"
    encode = pack.pack_dialog if packed else encode_dialog
    decode = pack.unpack_dialog if packed else decode_dialog

    def ref(v: dict[str, int]) -> dict[str, int]:
        _, x_out = un_dialog(decode(v["d"], _ITERS))
        return {"d": 0, "x": x_out}

    def gen(rng: random.Random) -> dict[str, int]:
        tape, _, _ = to_dialog_ref(PRIME, rng.randrange(1, PRIME), _ITERS)
        return {"d": encode(tape), "x": 0}

    # The iteration budget is the same floor as the forward's: a tape that did not
    # converge does not undo. The comparator term is subtler and is NOT the forward's --
    # see `_build_un_to_dialog_trunc_v1`.
    fail = schedule.iteration_fail_prob(_N, _ITERS)
    if cmp_bits is not None:
        fail += schedule.compare_fail_prob(cmp_bits - schedule.width_pad(_N))
    return Case(circ, ref, gen, meta={"fail_analytic": fail})


@circuit_version("inv.un_to_dialog_flat.v1")
def _build_un_to_dialog_flat_v1() -> Case:
    """The 798-qubit tape, exact comparator. Undoes `inv.to_dialog.v1`."""
    return _un_to_dialog_case(cmp_bits=None, layout="flat")


@circuit_version("inv.un_to_dialog.v1")
def _build_un_to_dialog_v1() -> Case:
    """The 665-qubit packed tape, exact comparator.

    NOT the partner of `inv.to_dialog_packed.v1`, which truncates -- an exact comparator
    disagrees with the truncated one that wrote the tape and leaves `s` at |1>. This is
    the exact *baseline* for the packed layout, and the gap to `_trunc` is what the
    truncation is worth on the way back: 243,149 against 217,769, so 25,380 Toffoli.
    """
    return _un_to_dialog_case(cmp_bits=None)


@circuit_version("inv.un_to_dialog_trunc.v1")
def _build_un_to_dialog_trunc_v1() -> Case:
    """Packed and truncated -- the half of `modmul_inplace` that runs last.

    The comparator term in `fail_analytic` means something different here than it does in
    the forward. Forward, a truncated compare writes a *different but self-consistent*
    tape: the loop simply takes the other branch and carries on. Backward, the tape is
    already fixed, so a comparator that disagrees with the one that wrote it leaves `s` at
    |1> and the free raises. That makes it a loud failure rather than a silent one, but
    only for a tape written by some *other* comparator -- which is what this standalone
    benchmark feeds it, since `gen` builds the tape from the exact reference. Inside
    `modmul_inplace` both halves truncate identically on identical operands, so they agree
    on every run and this term is not paid at all.
    """
    return _un_to_dialog_case(cmp_bits=schedule.compare_bits(_N))


# ===========================================================================
# The same thing at jump 2.
#
# This exists because the jump-2 composite was two measurements and an analogy: regions 1
# and 2 were built and benched, region 3 was `to_dialog_j2`'s number written twice on the
# strength of the jump-1 result above. The analogy was a good one -- the erasure argument
# does not care how many halvings an iteration ran -- but the claim it props up is the one
# a reader checks first (`inv.apply_dialog_j2_cdkm_trunc.v1` peaks at 1,169, six under
# PEAK_BUDGET), and a composite peak is the MAX over its regions, not the sum. An unrun
# region is exactly where a peak claim goes wrong, so it got run. MEASURED at n = 255:
#
#     forward   inv.to_dialog_j2_packed.v1     190,530 Toffoli   peak 1,107   depth 457,503
#     reverse   inv.un_to_dialog_j2_trunc.v1   190,530 Toffoli   peak 1,105   depth 504,003
#
# The analogy held on gates -- equal to the gate, and one qubit narrower, the same shape
# jump 1 gave. The peak the composite claim needed is 1,105, sixty-four under the replay's
# 1,169, so region 3 is not the composite's widest region and the claim stands as stated.
#
# Depth does NOT hold to the analogy, and this is the number worth carrying into a table.
# The reverse is 46,500 deeper, +10.2%, against +18% at jump 1. The penalty is the erasure
# comparator sitting alone on the critical path where the forward had it overlapping the
# subtract it fed, so it scales with the number of comparators -- 261 here against 399 --
# which accounts for the direction but not all of the size (245.7 per iteration at jump 1,
# 178.2 here). The residue is not isolated; do not quote a per-iteration constant.
#
# Two things differ from the jump-1 reverse, and both are structural rather than a matter
# of constants:
#
#   the shift is a circuit    At jump 1 `v >>= 1` is a relabel in both directions and its
#                             undo is free. Here the second halving is data-dependent, so
#                             the reverse pays a Fredkin ladder per iteration exactly as
#                             the forward does -- see `_un_rshift1`.
#   the rotate moves ends     `gcd_loop_j2` rotates at the TOP of an iteration and
#                             `gcd_loop` at the bottom, so the undo sits at the bottom
#                             here and at the top there. That is the same opposite-ends
#                             fact that makes the halving-count offset between the two
#                             tapes exactly +1.
# ===========================================================================


def _un_rshift1(circ: Circuit, ctrl: int, r: Register, *, top_zero: bool = False) -> None:
    """The exact inverse of `field.shift.controlled_rshift1`.

    `controlled_lshift1` already IS that inverse -- the same Fredkins in the opposite
    order -- and the `top_zero` case is the same call on a one-shorter slice, because a
    ladder that stops one short never touches `r[w-1]` in either direction. So this needs
    no new primitive, and the docstring in `field.shift` that declines to add a `top_zero`
    twin stays right: the width does the work.
    """
    controlled_lshift1(circ, ctrl, r[0:len(r) - 1] if top_zero else r)


class FlatUnTapeJ2(FlatUnTape):
    """Iteration `i` reads (d[3i], d[3i+1], d[3i+2]) and frees the triple."""

    def triple(self, i: int) -> tuple[int, int, int]:
        return self.d[3 * i], self.d[3 * i + 1], self.d[3 * i + 2]

    def retire(self, i: int) -> None:
        self.circ.free(Register(list(self.triple(i))))


class PackedUnTapeJ2:
    """Read a base-5 `pack` tape backwards and consume it. Mirror of `gcd.PackedTapeJ2`.

    Simpler than `PackedUnTape` in the same way the forward is, and for the same reason:
    base 5 has no free digit 0, so the accumulator never doubles as the signal qubits.
    All three signals are drawn fresh and freed on retire at every digit, including digit
    0 -- where `PackedUnTape` instead hands back the accumulator itself, because there the
    pair *is* its bottom two qubits. Here the accumulator survives digit 0 as three
    separate |0>s and has to be freed explicitly, which is the one line this class needs
    that its jump-1 sibling does not.

    Per group of 3 iterations, in reverse order:

        digit 2   alloc 3, free 2 accumulator bits, free the 3 on retire
        digit 1   alloc 3, free 2 accumulator bits, free the 3 on retire
        digit 0   alloc 3, free the 3 on retire, and free the 3 the accumulator is left as

    -- 9 allocated, 16 freed, a net 7 handed back, which is `GROUP_BITS_J2`. The allocs
    come out of the ids the previous digit just freed, so the steady-state carry is the
    3-qubit triple and not 3 per group.
    """

    def __init__(self, circ: Circuit, d: Register, iters: int) -> None:
        assert len(d) == pack.packed_bits_j2(iters), (
            f"a {iters}-iteration packed jump-2 tape is {pack.packed_bits_j2(iters)} "
            f"qubits, not {len(d)}"
        )
        self.circ = circ
        self._groups: list[Register] = []
        off = 0
        for g in range(0, iters, pack.GROUP_J2):
            w = pack.group_bits_j2(min(pack.GROUP_J2, iters - g))
            self._groups.append(Register(d.ids[off:off + w]))
            off += w
        self._trip: tuple[int, int, int] | None = None

    def triple(self, i: int) -> tuple[int, int, int]:
        g, j = divmod(i, pack.GROUP_J2)
        ids = self._groups[g].ids
        r = Register(ids[:pack.DIGIT_WIDTH_J2[j]])
        t = tuple(self.circ.alloc(3).ids)
        if i == 0:
            pack.unfold_digit_j2_boundary(self.circ, r, *t)
        else:
            pack.unfold_digit_j2(self.circ, r, j, *t)
        if j:
            # Digit out, accumulator down to DIGIT_WIDTH_J2[j-1]; the bits above are |0>
            # for good, because nothing in this direction ever folds them again.
            self.circ.free(
                Register(ids[pack.DIGIT_WIDTH_J2[j - 1]:pack.DIGIT_WIDTH_J2[j]])
            )
        self._trip = t
        return t

    def retire(self, i: int) -> None:
        assert self._trip is not None, "retire without a triple"
        self.circ.free(Register(sorted(self._trip)))
        self._trip = None
        if i % pack.GROUP_J2 == 0:
            # Digit 0 is out, so the whole accumulator is |0> -- and unlike at jump 1 it
            # is three qubits belonging to nobody, so free them here or the group leaks.
            g = i // pack.GROUP_J2
            self.circ.free(Register(self._groups[g].ids[:pack.DIGIT_WIDTH_J2[0]]))

    def close(self) -> None:
        assert self._trip is None, "a triple was left live"


def un_gcd_loop_j2(
    circ: Circuit,
    u: Register,
    v: Register,
    d: Register | FlatUnTapeJ2 | PackedUnTapeJ2,
    iters: int,
    cmp_bits: int | None = None,
    widths: list[int] | None = None,
    add_ctrl=gidney_add_ctrl,
    n: int | None = None,
) -> tuple[Register, Register]:
    """Undo `iters` jump-2 iterations, consuming the tape. `gcd_loop_j2` read bottom-up.

    Same contract as `un_gcd_loop`: `(u, v)` in are the FINAL windows, out are the ones
    the forward was handed. `cmp_bits`, `widths` and `add_ctrl` must match the forward
    run, because `b1` is erased by recomputing the same comparison.
    """
    if widths is None:
        widths = [n if n is not None else len(u)] * iters
    if n is None:
        n = widths[0]
    assert len(widths) >= iters, "width schedule is shorter than the loop"
    tape = FlatUnTapeJ2(circ, d) if isinstance(d, Register) else d

    uw, vw = u, v
    assert len(uw) == len(vw) == widths[iters - 1], (
        f"the reverse loop starts on the final window ({widths[iters - 1]}), got {len(uw)}"
    )
    for i in reversed(range(iters)):
        a, b, c = tape.triple(i)                       # the unfold, mirroring `fold`

        # `b0` MOVES between the two iteration kinds and this is the easiest thing in the
        # file to get wrong: the boundary triple is `(s0, s1, b0)`, so `b0` is the third
        # wire there and the second everywhere else. The swap control needs no such case
        # -- it is `b1` past the boundary and `b0` at it, and both are `c`, because
        # iteration 0 stores no `b1` at all (`u = p > v` makes it equal to `b0`).
        b0 = c if i == 0 else b
        add_ctrl(circ, b0, uw, vw)                     # undo `if b0: v -= u`
        cswap(circ, c, uw, vw)                         # cswap is its own inverse

        if i == 0:
            s0, s1 = a, b
            circ.cx(vw[0], b0)                         # erase b0
            _un_rshift1(circ, s1, vw, top_zero=True)
            circ.x(s1)
            circ.cx(vw[0], s1)                         # erase s1
            _un_rshift1(circ, s0, vw)                  # no top_zero: x may have a top bit
            circ.x(s0)
            circ.cx(vw[0], s0)                         # erase s0
        else:
            c_, b1 = a, c
            cmp_lt(circ, uw, vw, b1, k=cmp_bits, ctrl=b0)   # erase b1
            circ.cx(vw[0], b0)                         # erase b0
            _un_rshift1(circ, c_, vw, top_zero=True)
            circ.x(c_)
            circ.cx(vw[0], c_)                         # erase c
        tape.retire(i)

        if i:
            # Undo the unconditional halving's relabel. Forward rotated left at the TOP of
            # the iteration, so its undo is the last thing the reverse body does.
            vw = Register(vw.ids[-1:] + vw.ids[:-1], vw.name)

        w_next = widths[i - 1] if i else n
        assert w_next >= len(uw), "the width schedule must not narrow going backwards"
        grow = w_next - len(uw)
        if grow:
            uw = Register(uw.ids + list(circ.alloc(grow).ids), uw.name)
            vw = Register(vw.ids + list(circ.alloc(grow).ids), vw.name)

    return uw, vw


def un_to_dialog_j2(
    circ: Circuit,
    d: Register | FlatUnTapeJ2 | PackedUnTapeJ2,
    n: int,
    p: int = PRIME,
    iters: int | None = None,
    widths: list[int] | None = None,
    **kw,
) -> Register:
    """|d> -> |x>: spend a jump-2 tape, give the input back. Inverse of `gcd.to_dialog_j2`."""
    if iters is None:
        iters = schedule.iterations_j2(n)
    w_last = widths[iters - 1] if widths is not None else n
    tape = FlatUnTapeJ2(circ, d) if isinstance(d, Register) else d

    u = circ.alloc(w_last)
    v = circ.alloc(w_last)
    circ.x(u[0])                                  # the state the forward loop ended in

    uw, vw = un_gcd_loop_j2(circ, u, v, tape, iters, widths=widths, n=n, **kw)
    tape.close()

    for j in range(n):
        if (p >> j) & 1:
            circ.x(uw[j])
    circ.free(uw)
    return vw


_ITERS_J2 = schedule.iterations_j2(_N)


def _un_to_dialog_j2_case(cmp_bits: int | None, layout: str = "packed",
                          add_ctrl=gidney_add_ctrl) -> Case:
    widths = schedule.widths_j2(_N, _ITERS_J2)
    circ = Circuit()
    if layout == "flat":
        tape = FlatUnTapeJ2(circ, circ.input("d", 3 * _ITERS_J2))
    else:
        tape = PackedUnTapeJ2(
            circ, circ.input("d", pack.packed_bits_j2(_ITERS_J2)), _ITERS_J2
        )
    circ.input("x", 0)
    x = un_to_dialog_j2(circ, tape, _N, PRIME, iters=_ITERS_J2, cmp_bits=cmp_bits,
                        widths=widths, add_ctrl=add_ctrl)
    circ.rebind("x", x)
    circ.rebind("d", Register([], "d"))           # the tape is spent, not just zero
    packed = layout != "flat"

    def ref(v: dict[str, int]) -> dict[str, int]:
        t = (pack.unpack_dialog_j2(v["d"], _ITERS_J2, boundary=True) if packed
             else decode_dialog_j2(v["d"], _ITERS_J2))
        _, x_out = un_dialog_jump(t)
        return {"d": 0, "x": x_out}

    def gen(rng: random.Random) -> dict[str, int]:
        t, _, _ = to_dialog_jump_ref(PRIME, rng.randrange(1, PRIME), _ITERS_J2)
        dv = pack.pack_dialog_j2(t, boundary=True) if packed else encode_dialog_j2(t)
        return {"d": dv, "x": 0}

    # The forward's three terms, and all three apply: the reverse runs the same iteration
    # budget, the same narrowing window and the same comparator. The width-clip term is
    # NOT carried by `_un_to_dialog_case` above -- an omission in the jump-1 reverse
    # rather than a difference between the jumps, flagged here rather than copied.
    fail = schedule.iteration_fail_prob_j2(_N, _ITERS_J2)
    fail += gcd_width_clip_fail_prob(n=_N, jump=2)
    if cmp_bits is not None:
        fail += schedule.compare_fail_prob(cmp_bits - schedule.width_pad(_N))
    return Case(circ, ref, gen, meta={"fail_analytic": fail})


@circuit_version("inv.un_to_dialog_j2_flat.v1")
def _build_un_to_dialog_j2_flat_v1() -> Case:
    """The 783-qubit tape, exact comparator. Undoes `inv.to_dialog_j2.v1`.

    The second matched pair, and the one that shows the parity claim is about *matching*
    rather than about truncation: 198,436 both ways, exact against exact, as against
    190,530 both ways truncated against truncated. Peaks at 1,129, which is 24 over the
    packed reverse and NOT the 174 the tape widths differ by: the peak falls where `u` and
    `v` are widest and most of the tape is already spent, so a wider tape costs far less at
    the peak than it does at the start. Still 46 under PEAK_BUDGET either way.
    """
    return _un_to_dialog_j2_case(cmp_bits=None, layout="flat")


@circuit_version("inv.un_to_dialog_j2.v1")
def _build_un_to_dialog_j2_v1() -> Case:
    """The 609-qubit packed tape, exact comparator: the exact baseline for base 5.

    207,309 @ 1,105. Its own forward partner is not registered, so read the gap to `_trunc`
    rather than to anything above: 16,779 Toffoli is what the truncation is worth on the
    way back at jump 2, against 25,380 at jump 1 -- fewer comparators, less to save.
    """
    return _un_to_dialog_j2_case(cmp_bits=None)


@circuit_version("inv.un_to_dialog_j2_trunc.v1")
def _build_un_to_dialog_j2_trunc_v1() -> Case:
    """Packed and truncated -- region 3 of the jump-2 `modmul_inplace`.

    The one this half of the file was written for. Its partner is
    `inv.to_dialog_j2_packed.v1` (190,530 @ 1,107), and the jump-2 composite's peak claim
    rests on this circuit not exceeding the replay's 1,169. See `_un_to_dialog_trunc_v1`
    for what the comparator term means in the reverse direction; it means the same here.

    MEASURED 190,530 @ 1,105, depth 504,003 -- equal to its partner on gates, one qubit
    narrower, 64 under the replay. So the jump-2 composite is three measurements now:

        Phase 1 x 2   381,060 (190,530 + 190,530, both measured)
        Phase 2       464,312
        total         845,372 Toffoli, peak 1,169 -- set by Phase 2, not by this.

    **That composite was never real, and the error was in the peak, not the gates.** A
    Phase-1 peak is only the composite's peak if nothing else is live during it, and
    something always is: the forward half runs with `y` alongside (1,107 + 255 = 1,362) and
    this half runs with `xy` alongside (1,105 + 255 = 1,360), both far over PEAK_BUDGET =
    1,175. The replay is the one region whose measured peak already counts everything,
    because it holds `d + a + b` by construction -- which is exactly what made it easy to
    read the other two as if they did too. See `_trunc_threaded_v1` below, and doc 10.
    """
    return _un_to_dialog_j2_case(cmp_bits=schedule.compare_bits(_N))


@circuit_version("inv.un_to_dialog_j2_trunc_threaded.v1")
def _build_un_to_dialog_j2_trunc_threaded_v1() -> Case:
    """`_trunc.v1` with the mask taken out of the controlled subtract. The reverse partner
    of `inv.to_dialog_j2_packed_threaded.v1`, and the region that makes the composite fit.

    `gidney_add_ctrl` materialises `ctrl & a_i` into a full-width mask register; threading
    the control into the sum write does the same job without it, at the same Toffoli count
    (see `field.add.gidney_add_ctrl_threaded`, and doc 11 for why the predicted extra
    Toffoli is not there). Here that mask is 254 qubits of the 1,105.

    MEASURED 190,530 @ 852, depth 583,571 -- identical gates, 253 qubits back, and one
    narrower than the forward's 853 (the masked pair sat two apart, 1,105 vs 1,107; the
    mask that goes away is 254 wide forward and 253 here). With `xy` live that is 1,107,
    under PEAK_BUDGET and under the replay's 1,169, so for the first time the peak claim
    survives the operand the composite actually holds.

    Depth rises 504,003 -> 583,571 for the same reason it does forward: the mask's ANDs
    were mutually independent and the threaded sums sit on the carry chain.
    """
    return _un_to_dialog_j2_case(cmp_bits=schedule.compare_bits(_N),
                                 add_ctrl=gidney_add_ctrl_threaded)
