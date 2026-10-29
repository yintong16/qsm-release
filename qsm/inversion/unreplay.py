"""Phase 2 arithmetic reversed: |d, a, b> -> |d, a', b'>, tape intact.

Scope: the historical lazy reverse below is experimental. "Inverse" here means
an arithmetic reversal on its valid representation domain, not a physical inverse
of measurement-vented IR. Approximate phase corrections cannot be canceled by a
separately built reverse with independent outcomes; full-width lazy arithmetic
still has representation holes. `un_apply_dialog_canonical_inplace` is the new
matched exact canonical implementation, including zero/structured numerators.

This is the division. `apply_dialog` applied to `(y, 0)` gives `(0, x*y mod p)`; run the
same tape the other way and `(0, z)` comes back as `(z/x mod p, 0)`. Both baselines get
their division exactly this way -- frenchie by calling `IPModMul.inverse()`, ecdsa.fail by
flipping a `Direction` enum -- and neither pays anything extra for it.

That `0` is the lazy one, and Step 4c has to design around it: the emptied register comes
back holding `p`, not `|0>`, for every nonzero product, because the loop never
canonicalises. So the composite must consume it with the reverse rather than release it --
freeing it raises. Measured exhaustively at p = 31 and p = 127 in
`test_the_emptied_register_holds_the_lazy_zero_not_a_free_able_one`.

We do pay something extra, and it is small and exactly locatable. `replay.replay_loop`'s
body read bottom-up is

    if s:  swap(a, b)       cswap is its own inverse -- FREE
    if b0: b -= a mod p     modsub_lazy_ctrl -- the forward's adder, plus TWO Toffoli
    b = b/2 mod p           modhalve -- the forward's doubling, gate for gate

so the whole surcharge is `2 * iters` Toffoli, 798 at 255 bits against the replay's
~440,000. Where those two go is the interesting part, because the two steps that could
have been expensive are not:

**Halving is free because doubling erased its flag with the PARITY.** `moddouble` reads
its reduction flag off bit 0 of the result -- exact at every width, one CX -- so the
reverse writes the same bit back with one CX. No comparator appears. Contrast Phase 1,
where `undo.un_to_dialog` has to re-derive `b0 & (u > v)` and therefore costs the forward's
full truncated comparator (see that module's header). Same construction, opposite outcome,
and the difference is entirely whether the erasure predicate was a parity or a comparison.

**The addition is not free, because its flag was MEASURED.** `modadd_lazy_ctrl` erases its
overflow by measuring it and venting the disagreement into a phase; a measurement has no
inverse, so `modsub_lazy_ctrl` has to materialise the predicate with a real `cmp_lt`. That
costs two Toffoli more than the phase form at every width -- the top of the carry chain,
plus the controlled write -- and the constant is where the 798 comes from. It also moves
two of the loop's five error terms from phase to value; `params.un_replay_fail_prob` says
which, and the total is unchanged.

The overflow flag is recomputed rather than vented in the reverse. An incorrect
predicate can strand it and produce `DirtyAncillaError` in simulation; that is a
diagnostic, not an executable quantum branch. Temporary AND measurements inside
exact adders/comparators still exist, with exact corrections. Experiments on the
forward image at p=29 and p=251 observed no residual phase; they do not establish
that independently measurement-vented approximate circuits are quantum inverses.

Width is unchanged, with one located exception: `cmp_lt` is one qubit wider at its widest
than `cmp_lt_phase`, which is invisible behind Gidney's `n-1` carry ladder but not behind
CDKM's single ancilla. Exact CDKM at 255 bits peaks at 1434 against the forward's 1433;
truncating the comparator to `CMP_BITS` puts the peak back into the register state and the
qubit comes back. No registered version is exact-and-CDKM, so nothing in the design pays.

**The tape is read forwards, which is the wrong way round for the packing.** The forward
replay walks the tape in reverse and so meets a group's digits in the order 2, 1, 0, which
is the only order `unfold_digit` can peel them off in. Walking forwards it needs digit 0
first, and digit 0 is at the bottom of a base-3 numeral -- unreachable until the two above
it are out. `GroupReader` therefore unfolds a whole group at once and serves the three
pairs in order, which costs no more qubits than `PackedReader` (both hold all three pairs
across the group anyway) and exactly the same gates.

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
from ..field.modarith import modhalve, modhalve_ctrl_inplace, modsub_lazy_ctrl
from ..field.swap import cswap
from ..params import (
    CARRY_WINDOW,
    CMP_BITS,
    PRIME,
    U,
    un_replay_fail_prob,
    un_replay_j2_fail_prob,
)
from . import pack, schedule
from .gcd import decode_dialog_j2, encode_dialog_j2
from .reference import to_dialog as to_dialog_ref
from .reference import to_dialog_jump as to_dialog_jump_ref
from .reference import (
    apply_dialog_circuit,
    apply_dialog_jump_circuit,
    un_apply_dialog_circuit,
    un_apply_dialog_jump_circuit,
)
from .replay import FlatReader, FlatReaderJ2

__all__ = [
    "GroupReader",
    "un_apply_dialog",
    "un_replay_loop",
    "GroupReaderJ2",
    "un_apply_dialog_j2",
    "un_replay_loop_j2",
    "un_apply_dialog_canonical_inplace",
]


class GroupReader:
    """Read a `pack`-format tape FORWARDS, one group at a time, and put it back.

    `replay.PackedReader` gets a coincidence for free: it wants iteration 2 before
    iteration 1, and 2 is the digit `unfold_digit` can reach. Going forwards there is no
    such luck -- digit 0 is the low digit of `t0 + 3*t1 + 9*t2` and nothing can be read out
    of the accumulator until the digits above it have been removed. So this reader unfolds
    the group's digits 2, 1, 0 the moment iteration `3g` asks for its pair, holds all three
    pairs, serves them in the order 0, 1, 2, and folds the group back up after the last.

    That costs nothing over the forward reader. `PackedReader` also holds every pair it has
    unfolded until the group ends -- it folds them all back in one go too -- so the live
    counts coincide: three carried qubits, and a transient peak of seven while
    `cmp_ge_const` borrows its four carry ancillas to read a digit back. The only thing
    that changed is the order the pairs are handed out in, which costs no gates at all.

    Per group:

        unfold digit 2   take 2 fresh          -> 1 accumulator bit released
        unfold digit 1   take 1 fresh + that 1 -> 2 accumulator bits released
        unfold digit 0   the pair IS the accumulator's bottom two qubits; take nothing
        ... serve 0, 1, 2 ...
        fold 0, 1, 2 back, reclaiming the released bits in the order they were released

    and the three fresh qubits are recycled across all 133 groups. `close()` gives them
    back; `d` is left exactly as it arrived, which is what lets `undo.un_to_dialog` consume
    it afterwards.
    """

    def __init__(self, circ: Circuit, d: Register, iters: int) -> None:
        assert len(d) == pack.packed_bits(iters), (
            f"a {iters}-iteration packed tape is {pack.packed_bits(iters)} qubits, "
            f"not {len(d)}"
        )
        self.circ = circ
        self.iters = iters
        self._groups: list[Register] = []
        self._sizes: list[int] = []
        off = 0
        for g in range(0, iters, pack.GROUP):
            m = min(pack.GROUP, iters - g)
            w = pack.group_bits(m)
            self._groups.append(Register(d.ids[off:off + w]))
            self._sizes.append(m)
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

    def _unfold(self, g: int) -> None:
        """Peel the whole group apart, top digit first -- the only order that works."""
        ids = self._groups[g].ids
        for j in reversed(range(self._sizes[g])):
            r = Register(ids[:pack.DIGIT_WIDTH[j]])
            p = (r[0], r[1]) if j == 0 else (self._take(), self._take())
            pack.unfold_digit(self.circ, r, j, *p)
            if j:
                # The digit is out, so the accumulator has shrunk to DIGIT_WIDTH[j-1] and
                # the bits above it are |0> again. They pay for the next digit's pair.
                self._spare.extend(ids[pack.DIGIT_WIDTH[j - 1]:pack.DIGIT_WIDTH[j]])
            self._pairs[j] = p

    def pair(self, i: int) -> tuple[int, int]:
        g, j = divmod(i, pack.GROUP)
        if j == 0:
            self._unfold(g)
        return self._pairs[j]

    def done(self, i: int) -> None:
        g, j = divmod(i, pack.GROUP)
        if j != self._sizes[g] - 1:
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


def un_replay_loop(
    circ: Circuit,
    reader: FlatReader | GroupReader,
    a: Register,
    b: Register,
    iters: int,
    p: int = PRIME,
    add_ctrl_fn=gidney_add_ctrl,
    double_window: int | None = None,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> tuple[Register, Register]:
    """Undo `iters` tape entries, in FORWARD tape order. Returns the two registers.

    `b` is relabelled every iteration, as it is in `replay.replay_loop` -- `modhalve`'s
    shift is a rename of wires -- so the return value is the only correct handle on it.
    The ids are not the ones the forward left behind: the forward's doubling freed the top
    qubit and allocated a fresh bottom one, and the reverse does the mirror. Values match;
    wire identities churn through the pool, which is exactly what `Circuit.alloc`'s
    recycling is for.

    The windows must MATCH the forward's. A reverse built with a different `cmp_bits`
    re-derives a different predicate and strands the overflow qubit; that is a loud
    failure, not a silent one, but it is still a failure -- see `un_apply_dialog`.
    """
    for i in range(iters):
        b0, s = reader.pair(i)
        cswap(circ, s, a, b)
        modsub_lazy_ctrl(circ, b0, a, b, p, add_ctrl_fn,
                         carry_window=carry_window, cmp_bits=cmp_bits)
        b = modhalve(circ, b, p, window=double_window)
        reader.done(i)
    return a, b


def un_apply_dialog(
    circ: Circuit,
    d: Register,
    a: Register,
    b: Register,
    iters: int | None = None,
    p: int = PRIME,
    packed: bool = True,
    **kw,
) -> tuple[Register, Register]:
    """|d, a, b> -> |d, a', b'>: the Bezout replay undone, with `d` returned unchanged.

    Arithmetic reversal of `replay.apply_dialog` on its valid lazy image, with
    matched windows. This is not a physical inverse of approximate measurement
    corrections. `d` survives for Phase 1 undo to recover `x`.
    """
    if iters is None:
        iters = schedule.iterations(len(a))
    reader = GroupReader(circ, d, iters) if packed else FlatReader(d)
    a, b = un_replay_loop(circ, reader, a, b, iters, p, **kw)
    reader.close()
    return a, b


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
#
# One per adder, matching `replay.py`'s versions so the pair can be differenced. The
# claim doc 10 quotes is that each of these is its forward exactly plus 2 * iters = 798
# Toffoli, and that its peak is unchanged -- i.e. the division is free but for a constant
# that does not scale with the field.
# ---------------------------------------------------------------------------
_N = U
_ITERS = schedule.iterations(_N)


def _un_apply_dialog_case(add_ctrl_fn, trunc: bool) -> Case:
    windows = dict(
        double_window=CARRY_WINDOW if trunc else None,
        carry_window=CARRY_WINDOW if trunc else None,
        cmp_bits=CMP_BITS if trunc else None,
    )
    circ = Circuit()
    d = circ.input("d", pack.packed_bits(_ITERS))
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    a, b = un_apply_dialog(circ, d, a, b, _ITERS, PRIME,
                           add_ctrl_fn=add_ctrl_fn, **windows)
    circ.rebind("a", a)
    circ.rebind("b", b)      # modhalve relabels b once per iteration

    def ref(v: dict[str, int]) -> dict[str, int]:
        tape = pack.unpack_dialog(v["d"], _ITERS)
        a_out, b_out = un_apply_dialog_circuit(tape, v["a"], v["b"], PRIME, _N)
        return {"d": v["d"], "a": a_out, "b": b_out}

    def gen(rng: random.Random) -> dict[str, int]:
        # The reverse is only total on the forward's IMAGE -- a `b` below F with the flag
        # set borrows out of the register and the circuit frees a dirty ancilla. So the
        # generator does not draw (a, b) directly; it draws the forward's input and hands
        # over what the forward would have produced. That is the contract this circuit is
        # built to, and `test_inversion_unreplay.py` pins both halves of it.
        tape, _, _ = to_dialog_ref(PRIME, rng.randrange(1, PRIME), _ITERS)
        a_in, b_in = apply_dialog_circuit(tape, rng.randrange(PRIME), 0, PRIME, _N)
        return {"d": pack.pack_dialog(tape), "a": a_in, "b": b_in}

    # No `tolerates_phase`, and that is the substantive difference from the forward's Case
    # rather than an oversight. The forward earns the tolerance by ERASING its overflow
    # flag with a measurement, which turns a wrong predicate into a relative phase. Nothing
    # here measures: `modsub_lazy_ctrl` re-derives the flag with a value comparator and
    # then frees it, so every one of the forward's phase terms comes back as a raise. The
    # reverse is phase-clean at every window setting -- measured exhaustively at p = 29 and
    # p = 251, exact and truncated, 0 leaks in all four (`test_inversion_unreplay.py::
    # test_the_reverse_never_leaks_phase_it_raises_instead`).
    return Case(circ, ref, gen,
                meta={"fail_analytic": un_replay_fail_prob(_ITERS, **windows)})


@circuit_version("inv.un_apply_dialog.v1")
def _build_un_apply_dialog_v1() -> Case:
    """
    EXACT arithmetic, Gidney's controlled adder. The inverse of `inv.apply_dialog.v1`,
    and the version that pins the +2-per-iteration claim without any truncation in the way.
    """
    return _un_apply_dialog_case(gidney_add_ctrl, trunc=False)


@circuit_version("inv.un_apply_dialog_trunc.v1")
def _build_un_apply_dialog_trunc_v1() -> Case:
    """
    Gidney with all three windows truncated -- the inverse of `inv.apply_dialog_trunc.v1`,
    and the version where two of the five error terms are values rather than phases.
    """
    return _un_apply_dialog_case(gidney_add_ctrl, trunc=True)


@circuit_version("inv.un_apply_dialog_cdkm_trunc.v1")
def _build_un_apply_dialog_cdkm_trunc_v1() -> Case:
    """
    CDKM's in-place carry: the schedulable end, and the one doc 10 quotes for the
    division. Inverse of `inv.apply_dialog_cdkm_trunc.v1`, at the same peak.
    """
    return _un_apply_dialog_case(cdkm_add_ctrl, trunc=True)


# ---------------------------------------------------------------------------
# Jump 2 -- the division at the tuning the design actually uses.
#
# Everything above is the reverse of a jump-1 replay, and the design does not run one. The
# settled Phase 2 is `inv.apply_dialog_j2_threaded_tuned_trunc.v1`, so the settled DIVISION
# is this file's mirror of it, and until it existed there was no division circuit at the
# operating point at all.
#
# The surcharge is the same constant and it is now smaller, because it rides on the thing
# jump 2 removed. `modsub_lazy_ctrl` costs its forward +2 Toffoli per ADDITION, and jump 2
# runs 261 additions rather than 399: **522 Toffoli, not 798.** Nothing else moves. The
# doublings -- which jump 2 did not reduce, and which are the whole of its extra structure
# -- reverse for free in both their forms:
#
#     moddouble              -> modhalve                 flag off the parity, one CX
#     moddouble_ctrl_inplace -> modhalve_ctrl_inplace     flag off the parity, one CCX
#
# so the conditional shift ladder that jump 2 pays for costs exactly the same backwards as
# forwards, gate for gate and qubit for qubit (`test_field_modhalve.py::
# test_the_gated_halving_costs_exactly_what_the_gated_doubling_costs`). Had the gated
# forward erased its flag by measurement, as the addition does, the reverse would have
# needed a comparator per ladder and jump 2 would have been a worse trade backwards than
# forwards. It does not, and it is not.
#
# Iteration 0 is the exception in the same place it is forwards: both its halvings are
# conditional, and its `b1` is `b0`, so the cswap and the subtraction share a control.
# ---------------------------------------------------------------------------
class GroupReaderJ2:
    """`GroupReader` for the base-5 tape. Same reason, one digit wider.

    The forward reader gets the order it needs for free -- `replay_loop_j2` walks the tape
    backwards, so it meets a group's digits as 2, 1, 0, which is the only order
    `unfold_digit_j2` can peel them off in. Walking forwards, digit 0 is the low digit of
    `t0 + 5*t1 + 25*t2` and nothing can be read out of the accumulator until the two above
    it are gone. So this unfolds the whole group when iteration `3g` asks for its triple,
    holds all three, serves them 0, 1, 2, and folds the group back after the last.

    That is free against `PackedReaderJ2` for the same reason it was free at jump 1, and
    here the argument is even tighter: the forward reader ALSO holds all three triples
    across a group (it folds only at `j == 0`, which is the end of its backwards pass), so
    the two readers issue the identical sequence of takes, unfolds, releases and folds.
    Only the order the triples are handed out in differs, and that costs no gates at all.
    Measured equal in both Toffoli and peak by
    `test_the_group_reader_costs_exactly_what_the_packed_reader_costs`.

    Per group, with widths (3, 5, 7):

        unfold digit 2   take 3 fresh                 -> 2 accumulator bits released
        unfold digit 1   take 3 (2 spare + 1 fresh)   -> 2 more released
        unfold digit 0   take 3 (2 spare + 1 fresh)   -> all 3 released; base 5 has no
                                                        free digit, so the triple does not
                                                        sit in the accumulator's own bits
        ... serve 0, 1, 2 ...
        fold 0, 1, 2 back, reclaiming the released bits as they are needed

    which carries 5 working qubits against the base-3 reader's 3. That is the same
    arithmetic `PackedReaderJ2` does and it comes off the 56 qubits the shorter tape saves,
    which is the entire case for jump 2 -- so it is worth stating even though it does not
    change the answer.
    """

    def __init__(self, circ: Circuit, d: Register, iters: int) -> None:
        assert len(d) == pack.packed_bits_j2(iters), (
            f"a {iters}-iteration packed jump-2 tape is {pack.packed_bits_j2(iters)} "
            f"qubits, not {len(d)}"
        )
        self.circ = circ
        self.iters = iters
        self._groups: list[Register] = []
        self._sizes: list[int] = []
        off = 0
        for g in range(0, iters, pack.GROUP_J2):
            m = min(pack.GROUP_J2, iters - g)
            w = pack.group_bits_j2(m)
            self._groups.append(Register(d.ids[off:off + w]))
            self._sizes.append(m)
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

    def _unfold(self, g: int) -> None:
        """Peel the whole group apart, top digit first -- the only order that works."""
        ids = self._groups[g].ids
        for j in reversed(range(self._sizes[g])):
            r = Register(ids[:pack.DIGIT_WIDTH_J2[j]])
            t = (self._take(), self._take(), self._take())
            if g == 0 and j == 0:
                pack.unfold_digit_j2_boundary(self.circ, r, *t)
            else:
                pack.unfold_digit_j2(self.circ, r, j, *t)
            lo = 0 if j == 0 else pack.DIGIT_WIDTH_J2[j - 1]
            self._spare.extend(ids[lo:pack.DIGIT_WIDTH_J2[j]])
            self._trips[j] = t

    def triple(self, i: int) -> tuple[int, int, int]:
        g, j = divmod(i, pack.GROUP_J2)
        if j == 0:
            self._unfold(g)
        return self._trips[j]

    def done(self, i: int) -> None:
        g, j = divmod(i, pack.GROUP_J2)
        if j != self._sizes[g] - 1:
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


def un_replay_loop_j2(
    circ: Circuit,
    reader: FlatReaderJ2 | GroupReaderJ2,
    a: Register,
    b: Register,
    iters: int,
    p: int = PRIME,
    add_ctrl_fn=gidney_add_ctrl,
    double_window: int | None = None,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> tuple[Register, Register]:
    """Undo `iters` jump-2 entries, in FORWARD tape order. Returns the two registers.

    `replay_loop_j2`'s body run bottom-up, each step dualised:

        halve^s                 the conditional ladders first, then the free relabel
        if b1: swap(a, b)       cswap is its own inverse
        if b0: b -= a mod p     the +2-Toffoli comparator

    The order inside `halve^s` matters and is not symmetric with the forward's. Forwards
    the free `moddouble` runs FIRST and the gated ladder second; backwards the gated ladder
    must run first, because it is the outermost of the two. Getting that round the wrong
    way still produces a circuit -- both are halvings -- and it computes the wrong function
    on every iteration whose control is set.

    `b` is relabelled once per iteration by `modhalve`, so the returned handle is the only
    correct one; `modhalve_ctrl_inplace` does not relabel, which is exactly why it exists.
    The windows must MATCH the forward's, as at jump 1.
    """
    for i in range(iters):
        x, y, z = reader.triple(i)
        if i:
            c, b0, b1 = x, y, z
            modhalve_ctrl_inplace(circ, c, b, p, window=double_window)
            b = modhalve(circ, b, p, window=double_window)           # s >= 1 always
            cswap(circ, b1, a, b)
        else:
            s0, s1, b0 = x, y, z
            modhalve_ctrl_inplace(circ, s1, b, p, window=double_window)
            modhalve_ctrl_inplace(circ, s0, b, p, window=double_window)
            cswap(circ, b0, a, b)                                    # b1 IS b0 here
        modsub_lazy_ctrl(circ, b0, a, b, p, add_ctrl_fn,
                         carry_window=carry_window, cmp_bits=cmp_bits)
        reader.done(i)
    return a, b


def un_apply_dialog_j2(
    circ: Circuit,
    d: Register,
    a: Register,
    b: Register,
    iters: int | None = None,
    p: int = PRIME,
    packed: bool = True,
    **kw,
) -> tuple[Register, Register]:
    """|d, a, b> -> |d, a', b'>: the jump-2 replay undone, with `d` returned unchanged.

    Arithmetic reversal of `replay.apply_dialog_j2`, with matched windows and
    the same representation caveats, not a physical inverse of vented phase.
    On the experimental nonzero domain the forward maps `(y,0)` to `(p,x*y)`;
    this maps `(p,z)` to `(z/x,0)`. Canonical zero instead of p is supplied only
    by the separately named canonical reverse below.
    """
    if iters is None:
        iters = schedule.iterations_j2(len(a))
    reader = GroupReaderJ2(circ, d, iters) if packed else FlatReaderJ2(d)
    a, b = un_replay_loop_j2(circ, reader, a, b, iters, p, **kw)
    reader.close()
    return a, b


def un_apply_dialog_canonical_inplace(
    circ: Circuit, d: Register, a: Register, b: Register,
    iters: int, p: int = PRIME, packed: bool = True,
) -> tuple[Register, Register]:
    """Matched canonical reverse; (0,z) -> (z/x mod p,0), tape preserved.

    Unlike lazy unreplay, this accepts zero and every structured canonical
    numerator. Every step is the exact arithmetic inverse on canonical inputs.
    """
    from ..field import canonical

    reader = GroupReaderJ2(circ, d, iters) if packed else FlatReaderJ2(d)
    for i in range(iters):
        x, y, z = reader.triple(i)
        if i:
            extra, sub_control, swap_control = x, y, z
            b = canonical.halve_inplace(circ, b, p, ctrl=extra)
            b = canonical.halve_inplace(circ, b, p)
            cswap(circ, swap_control, a, b)
        else:
            s0, s1, sub_control = x, y, z
            b = canonical.halve_inplace(circ, b, p, ctrl=s1)
            b = canonical.halve_inplace(circ, b, p, ctrl=s0)
            cswap(circ, sub_control, a, b)
        canonical.sub_inplace(circ, a, b, p, ctrl=sub_control)
        reader.done(i)
    reader.close()
    return a, b


# --- registered versions, matched one-for-one against `replay.py`'s jump-2 set ---
_ITERS_J2 = schedule.iterations_j2(_N)


def _un_apply_dialog_j2_case(
    add_ctrl_fn, trunc: bool, chunk: int | None = None, packed: bool = True,
    chunk_cmp_bits: int | None = None,
) -> Case:
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
    a, b = un_apply_dialog_j2(circ, d, a, b, _ITERS_J2, PRIME,
                              packed=packed, add_ctrl_fn=add_ctrl_fn, **windows)
    circ.rebind("a", a)
    circ.rebind("b", b)      # modhalve relabels b once per iteration

    def ref(v: dict[str, int]) -> dict[str, int]:
        tape = (pack.unpack_dialog_j2(v["d"], _ITERS_J2, boundary=True) if packed
                else decode_dialog_j2(v["d"], _ITERS_J2))
        a_out, b_out = un_apply_dialog_jump_circuit(tape, v["a"], v["b"], PRIME, _N)
        return {"d": v["d"], "a": a_out, "b": b_out}

    def gen(rng: random.Random) -> dict[str, int]:
        # As at jump 1: the reverse is only total on the forward's IMAGE, so the generator
        # draws the forward's INPUT and hands over what the forward would have produced.
        tape, _, _ = to_dialog_jump_ref(PRIME, rng.randrange(1, PRIME), _ITERS_J2)
        dv = (pack.pack_dialog_j2(tape, boundary=True) if packed
              else encode_dialog_j2(tape))
        a_in, b_in = apply_dialog_jump_circuit(tape, rng.randrange(PRIME), 0, PRIME, _N)
        return {"d": dv, "a": a_in, "b": b_in}

    # No `tolerates_phase`, for the reason the jump-1 Case gives: nothing here measures, so
    # the forward's vent term arrives as a stranded flag and a raise rather than a phase.
    return Case(circ, ref, gen,
                meta={"fail_analytic": un_replay_j2_fail_prob(
                          _ITERS_J2, chunk=chunk, chunk_cmp_bits=chunk_cmp_bits,
                          **windows)})


@circuit_version("inv.un_apply_dialog_j2.v1")
def _build_un_apply_dialog_j2_v1() -> Case:
    """
    EXACT arithmetic, Gidney's controlled adder, jump-2 tape. The correctness baseline for
    the division, and the version that pins the +2-per-ADDITION claim with no truncation in
    the way: it should be `inv.apply_dialog_j2.v1` plus exactly 522.
    """
    return _un_apply_dialog_j2_case(gidney_add_ctrl, trunc=False)


@circuit_version("inv.un_apply_dialog_j2_trunc.v1")
def _build_un_apply_dialog_j2_trunc_v1() -> Case:
    """
    Gidney with all three windows truncated. The Toffoli floor of the division, and
    unschedulable for the same reason its forward is: 509 live carries.
    """
    return _un_apply_dialog_j2_case(gidney_add_ctrl, trunc=True)


@circuit_version("inv.un_apply_dialog_j2_cdkm_trunc.v1")
def _build_un_apply_dialog_j2_cdkm_trunc_v1() -> Case:
    """
    CDKM's in-place carry: the inverse of the first Phase 2 in the design that fit a peak
    budget. Kept as the matched partner of `inv.apply_dialog_j2_cdkm_trunc.v1`, so the pair
    can be differenced at the peak every earlier version was measured against.
    """
    return _un_apply_dialog_j2_case(cdkm_add_ctrl, trunc=True)


@circuit_version("inv.un_apply_dialog_j2_chunked_trunc.v1")
def _build_un_apply_dialog_j2_chunked_trunc_v1() -> Case:
    """
    64-bit masked blocks, matching `inv.apply_dialog_j2_chunked_trunc.v1`. A volume
    instrument rather than a schedulable version, on both sides of the pair.
    """
    return _un_apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked, chunk=64, cmp_bits=64),
        trunc=True, chunk=64)


@circuit_version("inv.un_apply_dialog_j2_threaded_tuned_trunc.v1")
def _build_un_apply_dialog_j2_threaded_tuned_trunc_v1() -> Case:
    """Experimental arithmetic reverse of settled Phase 2 -- chunk 43 for the block
    count, vent at 32 bits for the error budget -- and therefore the version every
    composite that divides is built from.

    Its forward is 373,484 at 1,173 and 2^-22.6. This is that plus 522 Toffoli at the same
    peak and the same rate, which is the claim `test_inversion_unreplay_j2.py` pins.
    """
    return _un_apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked_threaded, chunk=43, cmp_bits=32),
        trunc=True, chunk=43, chunk_cmp_bits=32)


@circuit_version("inv.un_apply_dialog_j2_threaded_tuned_narrow_trunc.v1")
def _build_un_apply_dialog_j2_threaded_tuned_narrow_trunc_v1() -> Case:
    """The same tuning at chunk 37, mirroring
    `inv.apply_dialog_j2_threaded_tuned_narrow_trunc.v1`. Kept for the same reason: it
    prices the last four qubits of headroom, and the price has to be known in both
    directions before a composite spends them.
    """
    return _un_apply_dialog_j2_case(
        functools.partial(gidney_add_ctrl_chunked_threaded, chunk=37, cmp_bits=32),
        trunc=True, chunk=37, chunk_cmp_bits=32)
