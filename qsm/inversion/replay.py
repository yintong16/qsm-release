"""Phase 2 of the dialog-GCD: replay the tape |d> against a pair of field elements.

The historical lazy replay below is experimental and distribution-conditioned.
Its measurement-vented arithmetic reverse is not its physical quantum inverse:
independent measurement outcomes do not cancel approximate phase corrections.
`apply_dialog_canonical_inplace` instead uses exact matched canonical primitives;
each primitive's temporary measurements have exact corrections on its domain.
The lazy zero below means p on the nonzero-product branch, not a free |0>.

The loop is `reference.apply_dialog`, run in reverse tape order:

    b = 2b mod p            moddouble -- a relabelling plus one controlled +F
    if b0: b += a mod p     modadd_lazy_ctrl
    if s:  swap(a, b)       cswap

and the whole tape applied to (y, 0) gives (0, x*y mod p), which is the multiply. Phase 1
is prime-agnostic integer arithmetic; **every modular reduction in the construction is
here**, which is why this is the half that cares about the shape of the prime, and why
frenchie's `ApplyBitVector` carries a `special_prime` flag while their `ToBitVector` does
not. 43% of their composite is immune to the prime; this is the other 57%.

Three things set the cost.

**The control costs the adder plus one Toffoli.** `if b0: b += a mod p` gates the register
addition, and the `+F` fold comes along free -- it is conditioned on an overflow bit that
cannot be set unless the addition ran. What is *not* free is the vent: with b0 = 0 the flag
is 0 while the re-derived predicate `b < a` is a coin flip, so the erasure has to be gated
on b0 as well, at one Toffoli per iteration. See `field.modadd_lazy_ctrl`; that term is
phase-only and was found by simulating the replay's measurements at p = 29, not by any
value comparison.

**The doubling is nearly free, and stays lazy.** `moddouble` is a wire relabelling plus a
controlled `+F`, with the reduction flag read off the result's parity rather than a
comparator. It wants `a < p` and we do not have that: the loop runs 399 doublings back
into their own output, and doc 05 records that no fold in the design supplies a canonical
value. Rather than pay ~763 Toffoli x 399 to canonicalise -- over half the replay's whole
budget -- the loop runs in the lazy `[0, 2^U)` domain throughout and accepts the
non-closure at `params.moddouble_lazy_overflow_prob` per step. `replay_fail_prob`
compounds that over the loop; at 255 bits it is 2^-243.2, which is 28 bits below the
iteration budget Phase 1 is already carrying.

**The tape has to survive.** `modmul_inplace` needs `d` intact after the replay, because
`to_dialog` has to be run backwards to give `x` back. So a packed group is unfolded one
digit at a time on the way down (digits 2, 1, 0 -- which is already the reverse order the
replay wants) and folded back up on the way out. `PackedReader` carries 3 qubits above the
tape to do it, because each unfold hands back the accumulator bits the digit it removed was
using -- but it *peaks* at 7, because `cmp_ge_const` borrows four carry ancillas to read a
digit back. Those four are transient and do not overlap the adder; the three are held for
the whole loop.

Algorithm from [GRLGS23] PRL 131(4) 040602 and [KSGZ+25] arXiv:2510.10967; written from
the papers, not from either reference implementation (both are AGPL-3.0).
"""

from __future__ import annotations

import functools
import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import cdkm_add_ctrl, gidney_add_ctrl
from ..field.add_chunked import (
    gidney_add_ctrl_chunked,
    gidney_add_ctrl_chunked_threaded,
)
from ..field.modarith import modadd_lazy_ctrl, moddouble, moddouble_ctrl_inplace
from ..field.swap import cswap
from ..params import (
    CARRY_WINDOW,
    CMP_BITS,
    PRIME,
    U,
    replay_fail_prob,
    replay_j2_fail_prob,
)
from . import pack, schedule
from .gcd import decode_dialog_j2, encode_dialog_j2
from .reference import apply_dialog_jump_lazy, apply_dialog_lazy
from .reference import to_dialog as to_dialog_ref
from .reference import to_dialog_jump as to_dialog_jump_ref

__all__ = [
    "FlatReader",
    "PackedReader",
    "apply_dialog",
    "replay_loop",
    "FlatReaderJ2",
    "PackedReaderJ2",
    "apply_dialog_j2",
    "replay_loop_j2",
    "apply_dialog_canonical_inplace",
]


class FlatReader:
    """Iteration `i`'s pair is (d[2i], d[2i+1]). The mirror of `gcd.FlatTape`.

    Nothing to unfold and nothing to give back: the tape is already two qubits per
    iteration, and the replay only ever reads it. Kept because it is the layout the
    packed reader is checked against.
    """

    def __init__(self, d: Register) -> None:
        self.d = d

    def pair(self, i: int) -> tuple[int, int]:
        return self.d[2 * i], self.d[2 * i + 1]

    def done(self, i: int) -> None:
        """End of iteration `i`. The pair stays where it is."""

    def close(self) -> None:
        """End of the replay. Nothing was borrowed."""


class PackedReader:
    """Read a `pack`-format tape one iteration at a time, and put it back.

    The replay walks the tape backwards, so it meets a group's digits in the order
    2, 1, 0 -- which is exactly the order `unfold_digit` can peel them off in. Each unfold
    also *shrinks* the accumulator (after digit 2 comes out the group holds a value below
    9, so the top qubit is |0>), and those bits pay for most of the next unfold's pair.
    Over a full group the borrowing works out to three qubits:

        digit 2   take 2 fresh          -> 5 + 2 = 7 live, 1 accumulator bit released
        digit 1   take 1 fresh + that 1 -> 5 + 3 = 8 live, 2 accumulator bits released
        digit 0   the pair *is* the accumulator's bottom two qubits; take nothing

    and the three are recycled across all 133 groups, so the reader *carries* 3 qubits on
    top of the tape, not 3 per group. `done()` folds the group back up in the reverse order
    (0, 1, 2) once its last digit has been used, which is what leaves `d` intact for
    `to_dialog`-inverse to consume.

    Its *peak* is 7, not 3: `cmp_ge_const` reads a digit back through a carry chain and
    borrows four ancillas doing it. They are transient and they never coexist with the
    adder's, so they do not add to the arithmetic's working set -- but at 255 bits
    `d + a + b` is `PEAK_BUDGET` exactly, so the reader alone is 7 over. Measured in
    `test_inversion_replay.py`, both numbers, at eight tape lengths.

    The fold arithmetic is paid twice over the composite -- once packing in Phase 1, once
    here unpacking and repacking -- for 133 qubits. `pack` prices it.
    """

    def __init__(self, circ: Circuit, d: Register, iters: int) -> None:
        assert len(d) == pack.packed_bits(iters), (
            f"a {iters}-iteration packed tape is {pack.packed_bits(iters)} qubits, "
            f"not {len(d)}"
        )
        self.circ = circ
        self.iters = iters
        self._groups: list[Register] = []
        off = 0
        for g in range(0, iters, pack.GROUP):
            w = pack.group_bits(min(pack.GROUP, iters - g))
            self._groups.append(Register(d.ids[off:off + w]))
            off += w
        self._spare: list[int] = []          # |0> qubits available to hand out
        self._fresh: list[int] = []          # ...of which these were allocated by us
        self._pairs: dict[int, tuple[int, int]] = {}

    def _take(self) -> int:
        if self._spare:
            return self._spare.pop()
        q = self.circ.alloc(1)[0]
        self._fresh.append(q)
        return q

    def _reclaim(self, ids: list[int]) -> None:
        """Take specific |0> qubits back out of the pool, to fold a digit into them."""
        for q in ids:
            self._spare.remove(q)

    def pair(self, i: int) -> tuple[int, int]:
        g, j = divmod(i, pack.GROUP)
        ids = self._groups[g].ids
        r = Register(ids[:pack.DIGIT_WIDTH[j]])
        p = (r[0], r[1]) if j == 0 else (self._take(), self._take())
        pack.unfold_digit(self.circ, r, j, *p)
        if j:
            # The digit is out, so the accumulator has shrunk to DIGIT_WIDTH[j-1] and the
            # bits above it are |0> again. They pay for the next digit's pair.
            self._spare.extend(ids[pack.DIGIT_WIDTH[j - 1]:pack.DIGIT_WIDTH[j]])
        self._pairs[j] = p
        return p

    def done(self, i: int) -> None:
        g, j = divmod(i, pack.GROUP)
        if j:
            return                                    # mid-group; the pairs stay out
        ids = self._groups[g].ids
        for k in range(len(self._pairs)):             # fold back low digit first
            if k:
                self._reclaim(ids[pack.DIGIT_WIDTH[k - 1]:pack.DIGIT_WIDTH[k]])
            pack.fold_digit(self.circ, Register(ids[:pack.DIGIT_WIDTH[k]]), k,
                            *self._pairs[k])
            if k:
                self._spare.extend(self._pairs[k])    # folded away; back to |0>
        self._pairs.clear()

    def close(self) -> None:
        """Give back the working qubits. Every group has been folded, so all are |0>."""
        assert not self._pairs, "a group was left unfolded"
        self.circ.free(Register(sorted(self._fresh)))
        self._spare = [q for q in self._spare if q not in set(self._fresh)]
        self._fresh = []


def replay_loop(
    circ: Circuit,
    reader: FlatReader | PackedReader,
    a: Register,
    b: Register,
    iters: int,
    p: int = PRIME,
    add_ctrl_fn=gidney_add_ctrl,
    double_window: int | None = None,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> tuple[Register, Register]:
    """Replay `iters` tape entries in reverse against (a, b). Returns the two registers.

    `b` is RELABELLED every iteration -- `moddouble`'s shift is a rename of wires, not a
    gate -- so the return value is the only correct handle on the result. `a` keeps its
    wires; `cswap` moves values, not ids.

    All three windows default to None, which is the exact circuit. The caller has to ask
    for each truncation, and `params.replay_fail_prob` prices the set it asked for.
    """
    for i in reversed(range(iters)):
        b0, s = reader.pair(i)
        b = moddouble(circ, b, p, window=double_window)
        modadd_lazy_ctrl(circ, b0, a, b, p, add_ctrl_fn,
                         carry_window=carry_window, cmp_bits=cmp_bits)
        cswap(circ, s, a, b)
        reader.done(i)
    return a, b


def apply_dialog(
    circ: Circuit,
    d: Register,
    a: Register,
    b: Register,
    iters: int | None = None,
    p: int = PRIME,
    packed: bool = True,
    **kw,
) -> tuple[Register, Register]:
    """|d, a, b> -> |d, a', b'>: the Bezout replay, with `d` returned unchanged.

    With `packed`, `d` is a `pack`-format tape of `pack.packed_bits(iters)` qubits -- what
    `gcd.PackedTape` produces; otherwise it is the flat two-per-iteration layout.
    """
    if iters is None:
        iters = schedule.iterations(len(a))
    reader = PackedReader(circ, d, iters) if packed else FlatReader(d)
    a, b = replay_loop(circ, reader, a, b, iters, p, **kw)
    reader.close()
    return a, b


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
#
# The replay's peak is the binding constraint of the whole design: d + a + b is
# 665 + 255 + 255 = 1175 = PEAK_BUDGET with nothing left over, so the adder's ancillas
# decide whether it fits. These versions price that choice -- Gidney's carry ladder
# against CDKM's single borrowed qubit -- against ecdsa.fail's 615,830 @ 1,151 and
# frenchie's ApplyBitVector at 568,214 @ 1,184.
# ---------------------------------------------------------------------------
_N = U
_ITERS = schedule.iterations(_N)


def _apply_dialog_case(add_ctrl_fn, trunc: bool, chunk: int | None = None) -> Case:
    windows = dict(
        double_window=CARRY_WINDOW if trunc else None,
        carry_window=CARRY_WINDOW if trunc else None,
        cmp_bits=CMP_BITS if trunc else None,
    )
    # The chunked adder adds a sixth per-iteration failure mode; `chunk` is passed to the
    # rate helper so the declared number covers the adder that was actually built.
    rate_kw = dict(windows, chunk=chunk, chunk_cmp_bits=chunk)
    circ = Circuit()
    d = circ.input("d", pack.packed_bits(_ITERS))
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    a, b = apply_dialog(circ, d, a, b, _ITERS, PRIME,
                        add_ctrl_fn=add_ctrl_fn, **windows)
    circ.rebind("a", a)
    circ.rebind("b", b)      # moddouble relabels b once per iteration

    def ref(v: dict[str, int]) -> dict[str, int]:
        # The LAZY reference, not `apply_dialog`: every intermediate lands in [0, 2^U)
        # rather than [0, p). The two diverge only where the lazy domain fails to close,
        # which `replay_fail_prob` prices at 2^-243.2 over the whole loop.
        tape = pack.unpack_dialog(v["d"], _ITERS)
        a_out, b_out = apply_dialog_lazy(tape, v["a"], v["b"], PRIME, _N)
        return {"d": v["d"], "a": a_out, "b": b_out}

    def gen(rng: random.Random) -> dict[str, int]:
        # `d` has to be a real tape: an arbitrary 5-bit group could exceed 26 and break
        # `unfold_digit`'s precondition that the accumulator is a base-3 numeral.
        tape, _, _ = to_dialog_ref(PRIME, rng.randrange(1, PRIME), _ITERS)
        return {"d": pack.pack_dialog(tape), "a": rng.randrange(PRIME), "b": 0}

    # Phase is tolerated even in the EXACT version, and that is not a truncation artefact:
    # the modadd's erasure is vented by measurement, and the replay hands it a lazy
    # accumulator 399 times over, so `modadd_lazy_vent_phase_prob` fires whether or not any
    # window is narrowed. A value comparison cannot see it; `replay_fail_prob` counts it.
    return Case(circ, ref, gen,
                meta={"fail_analytic": replay_fail_prob(_ITERS, **rate_kw),
                      "tolerates_phase": True})


# ---------------------------------------------------------------------------
# Jump 2.
#
# The dual of `gcd.gcd_loop_j2`, and it pays for the jump in the same coin: one controlled
# shift ladder per iteration. Phase 1 needed one because a conditional HALVING cannot be a
# relabel; Phase 2 needs one because a conditional DOUBLING cannot either. That symmetry is
# not a coincidence -- the replay is the forward loop reversed step by step -- and it means
# the honest way to price jump 2 is per construction, never per phase.
#
# The doubling count does not change. A jump-2 iteration runs `s` doublings and the `s`
# sum to the divstep count, which is what jump 1 does one at a time. What jump 2 removes
# is 138 additions, 138 swaps and 138 comparators; what it adds is 261 shift ladders at
# `U + 1` Toffoli each -- `U` for the ladder, which runs over `moddouble`'s widened view,
# and one for the gated parity erasure. 66,816 Toffoli, then, against 138 of everything.
#
# MEASURED, on matched pairs differing only in the jump: 0.890 on the exact Gidney adder,
# 0.904 on the truncated one, 0.814 on truncated CDKM. It trades well everywhere, and it
# trades best where an addition is dearest, which is the sanity check on the account above.
# The peak is the larger prize: 1,223 -> 1,169 on CDKM, six under PEAK_BUDGET, and the only
# circuit in the design where the shorter tape has anywhere to show up.
#
# Iteration 0 is again the exception: both its doublings are conditional (its `s` may be
# 0), so it pays two ladders, and its `b1` is `b0` -- the same CNOT-not-a-wire that saved
# Phase 1 a comparator saves the replay a cswap control.
# ---------------------------------------------------------------------------
class FlatReaderJ2:
    """Iteration `i`'s triple is (d[3i], d[3i+1], d[3i+2]). The mirror of `FlatTapeJ2`."""

    def __init__(self, d: Register) -> None:
        self.d = d

    def triple(self, i: int) -> tuple[int, int, int]:
        return self.d[3 * i], self.d[3 * i + 1], self.d[3 * i + 2]

    def done(self, i: int) -> None:
        """End of iteration `i`. The triple stays where it is."""

    def close(self) -> None:
        """End of the replay. Nothing was borrowed."""


class PackedReaderJ2:
    """`PackedReader` for the base-5 tape: three signals a digit, widths (3, 5, 7).

    The same borrowing argument one digit wider, and it works out slightly worse:

        digit 2   take 3 fresh                 -> 2 accumulator bits released
        digit 1   take 1 fresh + those 2       -> 2 more released
        digit 0   the triple IS the accumulator's three qubits; take nothing

    so the reader carries 4 qubits against `PackedReader`'s 3. Base 5 has no free digit --
    `pack` explains why -- and this is where that shows up on the replay side: at jump 1
    digit 0's pair is two of the accumulator's own bits with one to spare, and here the
    triple consumes the accumulator exactly.

    One qubit of carry against 56 of tape is not a trade worth thinking about, but it is
    worth *stating*, because the 56 is the entire case for jump 2 and every qubit spent
    getting at the tape comes off it.
    """

    def __init__(self, circ: Circuit, d: Register, iters: int) -> None:
        assert len(d) == pack.packed_bits_j2(iters), (
            f"a {iters}-iteration packed jump-2 tape is {pack.packed_bits_j2(iters)} "
            f"qubits, not {len(d)}"
        )
        self.circ = circ
        self.iters = iters
        self._groups: list[Register] = []
        off = 0
        for g in range(0, iters, pack.GROUP_J2):
            w = pack.group_bits_j2(min(pack.GROUP_J2, iters - g))
            self._groups.append(Register(d.ids[off:off + w]))
            off += w
        self._spare: list[int] = []
        self._fresh: list[int] = []
        self._trips: dict[int, tuple[int, int, int]] = {}

    def _take(self) -> int:
        if self._spare:
            return self._spare.pop()
        q = self.circ.alloc(1)[0]
        self._fresh.append(q)
        return q

    def _reclaim(self, ids: list[int]) -> None:
        for q in ids:
            self._spare.remove(q)

    def triple(self, i: int) -> tuple[int, int, int]:
        g, j = divmod(i, pack.GROUP_J2)
        ids = self._groups[g].ids
        r = Register(ids[:pack.DIGIT_WIDTH_J2[j]])
        t = (self._take(), self._take(), self._take())
        if g == 0 and j == 0:
            pack.unfold_digit_j2_boundary(self.circ, r, *t)
        else:
            pack.unfold_digit_j2(self.circ, r, j, *t)
        # The digit is out, so the accumulator has shrunk and the bits above it are |0>
        # again. At j = 0 the accumulator empties COMPLETELY -- base 5 has no free digit,
        # so digit 0 lives in three qubits of its own rather than being the signals'
        # binary form, and all three come back. That is the one place this differs
        # structurally from `PackedReader`, and it is why the loop below has no `if k`
        # guard on its reclaim where the base-3 one does.
        lo = 0 if j == 0 else pack.DIGIT_WIDTH_J2[j - 1]
        self._spare.extend(ids[lo:pack.DIGIT_WIDTH_J2[j]])
        self._trips[j] = t
        return t

    def done(self, i: int) -> None:
        g, j = divmod(i, pack.GROUP_J2)
        if j:
            return                                    # mid-group; the triples stay out
        ids = self._groups[g].ids
        for k in range(len(self._trips)):             # fold back low digit first
            lo = 0 if k == 0 else pack.DIGIT_WIDTH_J2[k - 1]
            self._reclaim(ids[lo:pack.DIGIT_WIDTH_J2[k]])
            r = Register(ids[:pack.DIGIT_WIDTH_J2[k]])
            if g == 0 and k == 0:
                pack.fold_digit_j2_boundary(self.circ, r, *self._trips[k])
            else:
                pack.fold_digit_j2(self.circ, r, k, *self._trips[k])
            self._spare.extend(self._trips[k])        # folded away; back to |0>
        self._trips.clear()

    def close(self) -> None:
        assert not self._trips, "a group was left unfolded"
        self.circ.free(Register(sorted(self._fresh)))
        self._spare = [q for q in self._spare if q not in set(self._fresh)]
        self._fresh = []


def replay_loop_j2(
    circ: Circuit,
    reader: FlatReaderJ2 | PackedReaderJ2,
    a: Register,
    b: Register,
    iters: int,
    p: int = PRIME,
    add_ctrl_fn=gidney_add_ctrl,
    double_window: int | None = None,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> tuple[Register, Register]:
    """Replay `iters` jump-2 entries in reverse against (a, b). Returns the two registers.

    Per `reference.apply_dialog_jump`, a forward iteration is `shift^s; swap^b1; sub^b0`,
    so the replay is `add^b0; swap^b1; double^s` -- the reverse order with each step
    dualised. `double^s` is where the jump costs something: the first doubling is the free
    relabel `moddouble` always was, and the second is `moddouble_ctrl_inplace`.

    `b` is still relabelled by the unconditional doubling, so the returned handle is again
    the only correct one. The conditional doubling does NOT relabel -- it cannot -- which
    is why the two are not interchangeable and why the loop cannot simply run `moddouble`
    twice under a control.
    """
    for i in reversed(range(iters)):
        x, y, z = reader.triple(i)
        if i:
            c, b0, b1 = x, y, z
            modadd_lazy_ctrl(circ, b0, a, b, p, add_ctrl_fn,
                             carry_window=carry_window, cmp_bits=cmp_bits)
            cswap(circ, b1, a, b)
            b = moddouble(circ, b, p, window=double_window)          # s >= 1 always
            moddouble_ctrl_inplace(circ, c, b, p, window=double_window)
        else:
            s0, s1, b0 = x, y, z
            modadd_lazy_ctrl(circ, b0, a, b, p, add_ctrl_fn,
                             carry_window=carry_window, cmp_bits=cmp_bits)
            cswap(circ, b0, a, b)                                    # b1 IS b0 here
            moddouble_ctrl_inplace(circ, s0, b, p, window=double_window)
            moddouble_ctrl_inplace(circ, s1, b, p, window=double_window)
        reader.done(i)
    return a, b


def apply_dialog_j2(
    circ: Circuit,
    d: Register,
    a: Register,
    b: Register,
    iters: int | None = None,
    p: int = PRIME,
    packed: bool = True,
    **kw,
) -> tuple[Register, Register]:
    """|d, a, b> -> |d, a', b'> for a jump-2 tape, with `d` returned unchanged."""
    if iters is None:
        iters = schedule.iterations_j2(len(a))
    reader = PackedReaderJ2(circ, d, iters) if packed else FlatReaderJ2(d)
    a, b = replay_loop_j2(circ, reader, a, b, iters, p, **kw)
    reader.close()
    return a, b


def apply_dialog_canonical_inplace(
    circ: Circuit, d: Register, a: Register, b: Register,
    iters: int, p: int = PRIME, packed: bool = True,
) -> tuple[Register, Register]:
    """Matched canonical replay; (y,0) -> (0,x*y mod p), tape preserved.

    Inputs a,b are canonical. Every arithmetic output remains canonical,
    including the emptied register. Tape must encode a terminating GCD of
    canonical nonzero x with p. Explicit arithmetic, not reversed measured IR.
    """
    from ..field import canonical

    reader = PackedReaderJ2(circ, d, iters) if packed else FlatReaderJ2(d)
    for i in reversed(range(iters)):
        x, y, z = reader.triple(i)
        if i:
            extra, add_control, swap_control = x, y, z
            canonical.add_inplace(circ, a, b, p, ctrl=add_control)
            cswap(circ, swap_control, a, b)
            b = canonical.double_inplace(circ, b, p)
            b = canonical.double_inplace(circ, b, p, ctrl=extra)
        else:
            s0, s1, add_control = x, y, z
            canonical.add_inplace(circ, a, b, p, ctrl=add_control)
            cswap(circ, add_control, a, b)
            b = canonical.double_inplace(circ, b, p, ctrl=s0)
            b = canonical.double_inplace(circ, b, p, ctrl=s1)
        reader.done(i)
    reader.close()
    return a, b


@circuit_version("inv.apply_dialog.v1")
def _build_apply_dialog_v1() -> Case:
    """
    EXACT arithmetic (no truncation anywhere), Gidney's controlled adder. The correctness
    baseline: everything it gets wrong is the lazy domain's non-closure, nothing else.
    """
    return _apply_dialog_case(gidney_add_ctrl, trunc=False)


@circuit_version("inv.apply_dialog_trunc.v1")
def _build_apply_dialog_trunc_v1() -> Case:
    """
    Gidney, with all three constant-addition/comparison windows truncated. The cheap end
    on Toffoli; the expensive end on peak, because `gidney_add_ctrl` holds 509 ancillas.
    """
    return _apply_dialog_case(gidney_add_ctrl, trunc=True)


@circuit_version("inv.apply_dialog_cdkm_trunc.v1")
def _build_apply_dialog_cdkm_trunc_v1() -> Case:
    """
    The same circuit on CDKM's in-place carry: one ancilla instead of 509, at roughly
    twice the Toffoli in the adder. This is the version that has a chance of fitting the
    peak budget, and doc 10 quotes the pair against each other.
    """
    return _apply_dialog_case(cdkm_add_ctrl, trunc=True)


@circuit_version("inv.apply_dialog_chunked_trunc.v1")
def _build_apply_dialog_chunked_trunc_v1() -> Case:
    """
    The middle of the adder Pareto: Gidney's ladder run 64 bits at a time, so only one
    block's carries are live. 130 ancillas instead of 509 or 1, for Toffoli between the
    two.

    64 is not a free parameter here. The vented boundary drops the carry-in and so is
    wrong at 2^-(chunk+1) however wide the comparison is, and 399 iterations amplify that
    by 8.6 bits -- chunk = 64 lands at 2^-54.4 and vanishes under the replay's existing
    truncations, chunk = 32 lands at 2^-22.4 and would become the largest term in the
    table. The peak saving that a narrower block would buy is not available at any
    acceptable rate.
    """
    return _apply_dialog_case(
        functools.partial(gidney_add_ctrl_chunked, chunk=64, cmp_bits=64),
        trunc=True, chunk=64)


# --- jump 2, at the real field width ---------------------------------------
_ITERS_J2 = schedule.iterations_j2(_N)


def _apply_dialog_j2_case(
    add_ctrl_fn, trunc: bool, chunk: int | None = None, packed: bool = True,
    chunk_cmp_bits: int | None = None,
) -> Case:
    """`chunk_cmp_bits` defaults to `chunk`, which is what every version before the tuned
    one used -- but the two are independent knobs and tying them was hiding a lever.
    `chunk` sets the block count and therefore the peak; `chunk_cmp_bits` sets how many
    bits the boundary vent compares, and so buys failure rate with Toffoli at a flat
    ~1,305 per bit. Pass them apart to sit anywhere on that line."""
    if chunk_cmp_bits is None:
        chunk_cmp_bits = chunk
    windows = dict(
        double_window=CARRY_WINDOW if trunc else None,
        carry_window=CARRY_WINDOW if trunc else None,
        cmp_bits=CMP_BITS if trunc else None,
    )
    circ = Circuit()
    dw = pack.packed_bits_j2(_ITERS_J2) if packed else 3 * _ITERS_J2
    d = circ.input("d", dw)
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    a, b = apply_dialog_j2(circ, d, a, b, _ITERS_J2, PRIME,
                           packed=packed, add_ctrl_fn=add_ctrl_fn, **windows)
    circ.rebind("a", a)
    circ.rebind("b", b)

    def ref(v: dict[str, int]) -> dict[str, int]:
        tape = (pack.unpack_dialog_j2(v["d"], _ITERS_J2, boundary=True) if packed
                else decode_dialog_j2(v["d"], _ITERS_J2))
        a_out, b_out = apply_dialog_jump_lazy(tape, v["a"], v["b"], PRIME, _N)
        return {"d": v["d"], "a": a_out, "b": b_out}

    def gen(rng: random.Random) -> dict[str, int]:
        tape, _, _ = to_dialog_jump_ref(PRIME, rng.randrange(1, PRIME), _ITERS_J2)
        dv = (pack.pack_dialog_j2(tape, boundary=True) if packed
              else encode_dialog_j2(tape))
        return {"d": dv, "a": rng.randrange(PRIME), "b": 0}

    return Case(circ, ref, gen,
                meta={"fail_analytic": replay_j2_fail_prob(
                          _ITERS_J2, chunk=chunk, chunk_cmp_bits=chunk_cmp_bits,
                          **windows),
                      "tolerates_phase": True})


@circuit_version("inv.apply_dialog_j2.v1")
def _build_apply_dialog_j2_v1() -> Case:
    """
    EXACT arithmetic, Gidney's controlled adder, jump-2 tape. The correctness baseline
    for Phase 2 at jump 2, and the matched partner of inv.apply_dialog.v1 -- same adder,
    same windows, same everything but the jump, which is the only way the two are
    comparable. MEASURED 550,703 against 618,450, a ratio of 0.890 with every truncation
    switched off -- so the win is not an artefact of the windows.
    """
    return _apply_dialog_j2_case(gidney_add_ctrl, trunc=False)


@circuit_version("inv.apply_dialog_j2_trunc.v1")
def _build_apply_dialog_j2_trunc_v1() -> Case:
    """
    Gidney with all three windows truncated, jump-2 tape. Against
    inv.apply_dialog_trunc.v1 this was the measurement that decided whether jump 2 is a
    win on the whole construction or only on Phase 1: it drops 138 additions, 138 swaps
    and 138 comparators, and pays 261 controlled shift ladders for them.

    MEASURED 330,158 against 365,085 -- a win, at 0.904, and the NARROWEST of the three
    pairs, because the truncated Gidney adder is the cheapest thing the jump has to buy
    back. The lowest Toffoli count anywhere in Phase 2, and unschedulable: its 509 live
    carries put it at 1,635, which the 56 qubits the tape frees does not begin to cover.
    """
    return _apply_dialog_j2_case(gidney_add_ctrl, trunc=True)


@circuit_version("inv.apply_dialog_j2_cdkm_trunc.v1")
def _build_apply_dialog_j2_cdkm_trunc_v1() -> Case:
    """
    CDKM's in-place carry, jump-2 tape: the version that has to fit the peak budget,
    since d + a + b is 609 + 255 + 255 = 1119 here rather than exactly 1175. This is the
    only circuit in the design where the tape saving has anywhere to show up.

    MEASURED 1,169 -- six under the budget, and the first replay in the design that fits
    one. Its jump-1 partner is 1,223. The 54 is the 56-qubit tape saving less the two
    extra qubits the base-5 reader carries.
    """
    return _apply_dialog_j2_case(cdkm_add_ctrl, trunc=True)


@circuit_version("inv.apply_dialog_j2_chunked_trunc.v1")
def _build_apply_dialog_j2_chunked_trunc_v1() -> Case:
    """
    The middle of the adder Pareto at jump 2, so that all four corners are matched pairs.
    Same 64-bit blocks and same reasoning as inv.apply_dialog_chunked_trunc.v1; the
    boundary term is if anything cheaper here, since 261 iterations amplify it by 8.0
    bits instead of 8.6.

    Still over budget -- 130 live carries do not fit in the 56 the shorter tape frees --
    so it is a volume instrument, not a schedulable version.
    """
    return _apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked, chunk=64, cmp_bits=64),
        trunc=True, chunk=64)


@circuit_version("inv.apply_dialog_j2_threaded_trunc.v1")
def _build_apply_dialog_j2_threaded_trunc_v1() -> Case:
    """
    The chunked adder with the control threaded instead of masked (doc 11), sized to land
    on cdkm's peak exactly. **MEASURED 391,493 at 1,169 -- the same 1,169 the CDKM version
    holds, for 15.7% fewer Toffoli.**

    Why 38-bit blocks. The masked chunked adder holds `2*chunk + blocks`; threading drops
    that to `chunk + blocks`, which is what buys the room. Sweeping the chunk, Phase 2's
    peak is FLAT at 1,169 for every chunk up to 38 and only then starts climbing -- so
    below 38 the adder is not what sets the peak at all, and the widest block that stays
    free is the one to take. (44 also fits, at 389,144 and 2^-31.4, but spends 5 of the 6
    qubits the budget has left; if anything downstream needs them back, 38 is the version
    that has them.)

    The failure rate is 2^-28.2 against the CDKM version's 2^-31.5, both immaterial next
    to Phase 1's 2^-16.0, which is what the composite rate is actually made of.
    """
    return _apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked_threaded, chunk=38, cmp_bits=38),
        trunc=True, chunk=38)


@circuit_version("inv.apply_dialog_j2_threaded_wide_trunc.v1")
def _build_apply_dialog_j2_threaded_wide_trunc_v1() -> Case:
    """
    `_threaded_trunc.v1` with the block widened to 44, which is the largest that still
    fits `PEAK_BUDGET`. MEASURED 389,144 at 1,174 and 2^-31.4 -- the lowest Toffoli count
    of any schedulable Phase 2, at the CDKM version's failure rate to within a tenth of a
    bit, for five of the six spare qubits.

    Kept as the volume corner and as the measurement behind the sentence above: the
    difference between this and `_threaded_trunc.v1` is 2,349 Toffoli and 3.2 bits, which
    is what a qubit of peak is worth here.
    """
    return _apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked_threaded, chunk=44, cmp_bits=44),
        trunc=True, chunk=44)


# --- what a chunk sweep says, and why the two above are not the answer ------------
#
# Both versions above set `cmp_bits = chunk`, which looked like one knob and is two.
# Sweeping `chunk` from 26 to 49 with them tied gives a peak that is FLAT at 1,169 all
# the way to 38, then rises by exactly one qubit per chunk (1,170 at 39 ... 1,175 at 45,
# over at 46), and a Toffoli count that saws: inside a fixed block count it climbs with
# `chunk` -- because the tied `cmp_bits` is climbing -- and drops each time the block
# count falls. So the minima sit at the SMALLEST chunk of each block count, 32 (8 blocks,
# 390,710), 37 (7 blocks, 389,927), 43 (6 blocks, 387,839), and both registered chunks,
# 38 and 44, are one step past a minimum rather than at one.
#
# The flat 1,169 below chunk 38 is worth stating separately, because it says the adder is
# NOT what sets the peak there -- something else in the replay is, and it has not been
# identified. Anything spent making the adder narrower than chunk 38 is spent for nothing.
#
# Untying the knobs is the bigger result. `chunk_cmp_bits` buys failure rate at a flat
# ~1,305 Toffoli per bit and nothing else -- peak does not move at all:
#
#     chunk 43:  cmp 43 -> 387,839 @ 2^-31.2      cmp 28 -> 368,264 @ 2^-18.6
#                cmp 36 -> 378,704 @ 2^-26.6      cmp 24 -> 363,044 @ 2^-14.7
#                cmp 32 -> 373,484 @ 2^-22.6
#
# and the composite cannot use precision that fine. Phase 1 runs twice at 2^-16.0 each,
# so the composite floor is 2^-15.0 before Phase 2 contributes anything, and a Phase 2 at
# 2^-31 is sixteen orders below a term that is already there. Costing it out: 2^-22.6
# raises the composite by 0.007 bits, 2^-18.6 by 0.114. The first is free in any sense
# that matters and the second is arguably still cheap.
#
# `_threaded_tuned_trunc.v1` takes the free one. Going further is a real decision about
# the error budget rather than a free lunch, so it is left as a lever, not taken here.


@circuit_version("inv.apply_dialog_j2_threaded_tuned_trunc.v1")
def _build_apply_dialog_j2_threaded_tuned_trunc_v1() -> Case:
    """Chunk 43 for the block count, vent at 32 bits for the error budget. The cheapest
    schedulable Phase 2 in the design.

    MEASURED 373,484 at 1,173 and 2^-22.6, against `_threaded_wide_trunc.v1`'s 389,144 at
    1,174 and 2^-31.4: **15,660 fewer Toffoli, one qubit narrower, and 8.8 bits of a
    failure rate the composite cannot spend.** The composite it sits in is 2^-15.0 with or
    without those 8.8 bits -- Phase 1 runs twice at 2^-16.0 and dominates by sixteen orders
    -- so the exact cost of this version over the widest-vent one is 0.007 bits.

    Chunk 43 rather than 44 because 43 is the smallest chunk that still cuts 255 into 6
    blocks, and inside a block count the Toffoli only rises. Peak 1,173, two under
    PEAK_BUDGET; 1,169 is available at chunk 37 for 8,613 more Toffoli, and that trade is
    `_threaded_tuned_narrow_trunc.v1` below.
    """
    return _apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked_threaded, chunk=43, cmp_bits=32),
        trunc=True, chunk=43, chunk_cmp_bits=32)


@circuit_version("inv.apply_dialog_j2_threaded_tuned_narrow_trunc.v1")
def _build_apply_dialog_j2_threaded_tuned_narrow_trunc_v1() -> Case:
    """The same tuning at chunk 37, which is where the peak's flat floor of 1,169 ends.

    MEASURED 382,097 at 1,169 and 2^-22.4. Kept because 1,169 is the number every earlier
    Phase 2 was measured against, and because it prices the last four qubits: 8,613
    Toffoli, or 2,153 per qubit, which is far dearer than the ~1,305 per bit the vent
    charges. Take it only if something else in the design needs the headroom.
    """
    return _apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked_threaded, chunk=37, cmp_bits=32),
        trunc=True, chunk=37, chunk_cmp_bits=32)
