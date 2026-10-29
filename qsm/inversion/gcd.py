"""Phase 1 of the dialog-GCD: turn |x> into the branch tape |d>.

The loop is the one in `reference.to_dialog`, gate for gate:

    b0 = v & 1              one CNOT
    s  = b0 & (u > v)       one *controlled* comparator into the tape qubit
    if s:  swap(u, v)       one cswap
    if b0: v -= u           one controlled subtract
    v >>= 1                 free

Three things about it are worth stating, because they are what make the cost what it is.

**The shift is free, and it frees a qubit.** After the conditional subtract,

    v0' = v0 ^ b0*u0 = v0 ^ v0 = 0

because `b0` *is* `v0` and `u` is odd throughout (u starts at the prime, and the only
thing that ever writes u is the swap, which fires only when `b0 = 1`, i.e. when `v` is
odd). So the bottom qubit of `v` is guaranteed |0> at the end of every iteration and the
`>>= 1` is a rotation of the register's id list -- no gates, and the vacated top slot is a
clean qubit. That is the mechanism that later lets the tape live inside `u` and `v`.

**The comparator's control is nearly free.** `s = b0 & (u > v)` is one call to
`cmp_lt(..., ctrl=b0)`, which costs two Toffoli more than the uncontrolled compare, not
2x: the carry ladder is erased whatever `ctrl` does, so only the single write into the
flag has to be gated (`qsm/field/compare.py:carry_out`). Computing `(u > v)` into scratch
and ANDing afterwards would pay for the ladder twice.

**The window shrinks.** `u` and `v` lose ~0.7075 bits per iteration, so `widths=` narrows
every operation to `schedule.widths(n)`. Anything above the window is truncated rather
than carried, which is why the padding is a failure knob and not an optimisation -- see
`schedule.width`.

The loop is pure integer arithmetic. There is no modular reduction anywhere in it, which
is why Phase 1 is prime-agnostic and only Phase 2 pays for the shape of the prime.

Algorithm from [GRLGS23] PRL 131(4) 040602 and [KSGZ+25] arXiv:2510.10967; written from
the papers, not from either reference implementation (both are AGPL-3.0).
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add_ctrl, gidney_add_ctrl_threaded
from ..field.compare import cmp_lt
from ..field.shift import controlled_rshift1
from ..field.swap import cswap
from ..params import PRIME, U, gcd_width_clip_fail_prob
from . import pack, schedule
from .reference import to_dialog as to_dialog_ref
from .reference import to_dialog_jump as to_dialog_jump_ref

__all__ = [
    "FlatTape",
    "FlatTapeJ2",
    "PackedTape",
    "PackedTapeJ2",
    "PooledTape",
    "PooledTapeJ2",
    "dialog_extra_bits",
    "decode_dialog",
    "decode_dialog_j2",
    "encode_dialog",
    "encode_dialog_j2",
    "gcd_loop",
    "gcd_loop_j2",
    "to_dialog",
    "to_dialog_j2",
]

# A tape layout is four methods, and `gcd_loop` calls them in this order per iteration:
#
#   pair(i)  -> the two qubits iteration i writes (b0, s) to
#   ... the divstep's own gates, which read both ...
#   fold(i)  -> the layout's chance to compress and hand qubits back
#   release(ids) is called separately, whenever the window narrows
#   owned()  -> every qubit the layout ever held, so `to_dialog` knows what is not tape
#
# Three layouts implement it: a dedicated register, the same thing hosted in the slots the
# window releases, and that plus the 5-per-3 packing.


class FlatTape:
    """Iteration `i` owns qubits (2i, 2i+1) of a dedicated register. 798 at n = 255.

    The simple layout: a separate 2-per-iteration register that nothing else touches.
    Correct, and the thing every other layout is checked against, but it pays 798 qubits
    for information the loop is simultaneously freeing.
    """

    def __init__(self, d: Register) -> None:
        self.d = d

    def pair(self, i: int) -> tuple[int, int]:
        return self.d[2 * i], self.d[2 * i + 1]

    def fold(self, i: int) -> None:
        """End of iteration `i`. Nothing to do: the pair is already where it lives."""

    def release(self, ids: list[int]) -> None:
        """Slots `u` and `v` no longer need. This layout has no use for them."""

    def owned(self) -> set[int]:
        return set(self.d.ids)

    def taken(self) -> Register:
        return self.d


class PooledTape:
    """The tape in the qubits `u` and `v` give back, topped up on demand.

    Every iteration shrinks the pair by ~0.7075 bits, so the window releases ~1.415
    qubits per iteration while the tape consumes 2. The pool never quite catches up --
    over the whole loop it is 486 released against 798 needed -- and a fresh qubit is
    allocated only for the shortfall, at the moment the shortfall appears.

    *When* the top-up is allocated is the whole point, not a detail. The shortfall is
    small early (the window has released nothing, but the tape has consumed almost
    nothing either) and large late (312 by the end, but by then the window is 12 bits and
    the adder's carry ladder has gone). The two curves peak at opposite ends of the loop,
    so allocating on demand costs 1129 qubits where reserving all 312 up front costs
    1331 -- the difference between fitting under `PEAK_BUDGET` with a Gidney adder and
    having to fall back to CDKM at roughly 1.5x the Toffoli.

    Slots are handed out in release order; the order is arbitrary but fixed, and
    `taken()` reports it, so the replay reads the tape off exactly the wires Phase 1 wrote
    it to. `extra` pre-supplies slots for a caller that would rather reserve them (it
    makes the peak worse, and exists to measure that).

    A released slot is |0> because the width schedule says the values fit inside the
    narrowed window -- so a schedule that clips does not merely truncate a value, it
    scribbles on the tape. That is the reason `schedule.width`'s padding is a correctness
    knob and not a performance one.
    """

    def __init__(self, circ: Circuit, extra: Register | None = None) -> None:
        self.circ = circ
        self._avail: list[int] = list(extra.ids) if extra is not None else []
        self._owned: set[int] = set(self._avail)
        self._order: list[int] = []
        self.allocated = 0
        """Fresh qubits taken because the window had not released one yet.

        `dialog_extra_bits` predicts this, and the prediction is what sizes the tape
        register the replay has to be handed."""

    def release(self, ids: list[int]) -> None:
        self._avail.extend(ids)
        self._owned.update(ids)

    def take(self) -> int:
        """One |0> qubit: a released slot if the window has handed one back, else fresh."""
        if not self._avail:
            q = self.circ.alloc(1)[0]
            self.allocated += 1
            self._owned.add(q)
            return q
        return self._avail.pop(0)

    def give_back(self, ids: list[int]) -> None:
        """Return |0> qubits for immediate reuse; `PackedTape` returns every folded pair."""
        self._avail[0:0] = list(ids)

    def owned(self) -> set[int]:
        return set(self._owned)

    def pair(self, i: int) -> tuple[int, int]:
        q = (self.take(), self.take())
        self._order.extend(q)
        return q

    def fold(self, i: int) -> None:
        """End of iteration `i`. This layout keeps the pair exactly where it landed."""

    def taken(self) -> Register:
        return Register(self._order, "d")


class PackedTape(PooledTape):
    """`PooledTape` plus the 5-per-3 packing: 665 qubits of tape instead of 798.

    Each iteration still takes two qubits from the pool, but at the end of the iteration
    they are folded into the group's base-3 accumulator and handed straight back -- so the
    pair is transient and only the accumulators persist. See `pack` for why 27 states fit
    in 5 qubits and what the fold costs.

    `taken()` is the concatenation of the accumulators, low group first, which is the
    layout `pack.pack_dialog` describes and `pack.unfold_digit` reads back.
    """

    def __init__(self, circ: Circuit, extra: Register | None = None) -> None:
        super().__init__(circ, extra)
        self._groups: list[Register] = []
        self._pair: tuple[int, int] | None = None

    def pair(self, i: int) -> tuple[int, int]:
        j = i % pack.GROUP
        if j == 0:
            # Digit 0's two qubits *become* the bottom of a new accumulator.
            self._groups.append(Register([self.take(), self.take()]))
        else:
            # Widen the accumulator to hold one more digit, then draw the pair on top.
            grow = pack.DIGIT_WIDTH[j] - len(self._groups[-1])
            self._groups[-1] = Register(
                self._groups[-1].ids + [self.take() for _ in range(grow)]
            )
        self._pair = (self.take(), self.take()) if j else tuple(self._groups[-1].ids)
        return self._pair

    def fold(self, i: int) -> None:
        j = i % pack.GROUP
        pack.fold_digit(self.circ, self._groups[-1], j, *self._pair)
        if j:
            self.give_back(list(self._pair))     # folded away; back to |0>
        self._pair = None

    def taken(self) -> Register:
        return Register([q for g in self._groups for q in g.ids], "d")


def dialog_extra_bits(
    n: int = U,
    iters: int | None = None,
    widths: list[int] | None = None,
    bits_per_iter: float = 2.0,
) -> int:
    """Qubits `PooledTape` needs on top of what `u` and `v` release. 312 at n = 255.

    The pool holds `2*(n - widths[i])` slots while iteration `i` runs and the tape has
    consumed `ceil(bits_per_iter * (i+1))` of them by the end of it, so the top-up has to
    cover the worst shortfall over the whole loop. With no narrowing there are no released
    slots and this returns the whole tape.

    `bits_per_iter` is 2 for `FlatTape`'s layout and 5/3 for the packed one; at 5/3 the
    answer is 179, which is what makes the replay's 665 + 255 + 255 budget reachable.
    """
    if iters is None:
        iters = schedule.iterations(n)
    if widths is None:
        widths = [n] * iters
    need = 0
    for i in range(iters):
        used = int(-(-(bits_per_iter * (i + 1)) // 1))    # ceil
        need = max(need, used - 2 * (n - widths[i]))
    return max(need, 0)


def encode_dialog(tape: list[tuple[int, int]]) -> int:
    """The integer a `d` register holds for a given classical tape, two bits per iteration.

    This is the value of the register `to_dialog` *returns*, which for a pooled tape is a
    scattered view -- bit 2i is still iteration i's `b0` wherever that wire physically is.
    """
    val = 0
    for i, (b0, s) in enumerate(tape):
        val |= b0 << (2 * i)
        val |= s << (2 * i + 1)
    return val


def decode_dialog(d: int, iters: int) -> list[tuple[int, int]]:
    """Inverse of `encode_dialog`."""
    return [((d >> (2 * i)) & 1, (d >> (2 * i + 1)) & 1) for i in range(iters)]


def _csub(circ: Circuit, ctrl: int, a: Register, b: Register, add_ctrl) -> None:
    """In place: b -= a (mod 2^n) if ctrl.

    `~(~b + a)`. The NOTs are unconditional and self-cancelling, so a controlled subtract
    costs exactly a controlled add plus 2n NOTs -- no Toffoli. Local to this module
    because the dialog loop is the only caller that wants a non-modular one.
    """
    for q in b:
        circ.x(q)
    add_ctrl(circ, ctrl, a, b)
    for q in b:
        circ.x(q)


def gcd_loop(
    circ: Circuit,
    u: Register,
    v: Register,
    d: Register | FlatTape | PooledTape,
    iters: int | None = None,
    cmp_bits: int | None = None,
    widths: list[int] | None = None,
    add_ctrl=gidney_add_ctrl,
) -> tuple[Register, Register]:
    """Run `iters` divsteps on (u, v), logging (b0, b0&b1) into `d`. Returns the windows.

    `u` and `v` must be the same width. `d` is either a plain register -- wrapped in
    `FlatTape`, the 2-per-iteration layout -- or a tape object, which additionally gets
    handed the slots the narrowing window releases. On a run that converges, the two
    returned registers are the *final windows*, holding (1, 0); everything above them in
    the original registers is |0> or tape.

    `widths[i]` narrows iteration `i` to the low bits of both registers and must be
    non-increasing; `cmp_bits` truncates the comparator to its top k bits. Both are
    approximations with their own failure rates -- `schedule.width` and
    `schedule.compare_fail_prob` -- and both are off by default so the caller has to ask.

    The returned `v` is a *rotated view*: `v >>= 1` moves id 0 to the top of the window
    each iteration, which is legal only because that qubit is provably |0> by then (see
    the module docstring). Nothing else in the loop relabels.
    """
    n = len(u)
    assert len(v) == n, "u and v must be the same width"
    if iters is None:
        iters = schedule.iterations(n)
    if widths is None:
        widths = [n] * iters
    assert len(widths) >= iters, "width schedule is shorter than the loop"
    tape = FlatTape(d) if isinstance(d, Register) else d

    uw, vw = u, v
    for i in range(iters):
        w = widths[i]
        assert 1 <= w <= len(uw), f"iteration {i}: window {w} outside [1, {len(uw)}]"
        if w < len(uw):
            # The values fit in w bits, so the slots above are |0> and can hold tape.
            tape.release(uw.ids[w:] + vw.ids[w:])
            uw, vw = uw[0:w], vw[0:w]
        b0, s = tape.pair(i)

        circ.cx(vw[0], b0)                       # b0 = v & 1
        cmp_lt(circ, uw, vw, s, k=cmp_bits, ctrl=b0)   # s ^= b0 & (v < u)
        cswap(circ, s, uw, vw)                   # if s: swap(u, v)
        _csub(circ, b0, uw, vw, add_ctrl)        # if b0: v -= u

        tape.fold(i)                             # b0 and s are dead to the loop now

        # v >>= 1. vw.ids[0] is |0> now, so rotating it to the top of the window is a
        # shift. Rotating within the window, not the whole register, is what keeps a slot
        # handed to the tape from drifting back down into the arithmetic.
        vw = Register(vw.ids[1:] + vw.ids[:1], vw.name)

    return uw, vw


def to_dialog(
    circ: Circuit,
    x: Register,
    d: Register | FlatTape | PooledTape,
    p: int = PRIME,
    **kw,
) -> Register:
    """|x, 0> -> |d>: consume `x` into the tape. Returns the tape in iteration order.

    The prime goes into a scratch register as X gates -- it is a compile-time constant --
    and comes back as 1, so every qubit of `u` and `x` that is not holding tape is
    returned to the pool. A run that did not converge frees a dirty qubit, which the
    simulator reports as a `DirtyAncillaError` rather than a wrong answer: the iteration
    budget's failure mode is loud by construction.

    With a `PooledTape` the returned register is scattered across `u`, `x` and the top-up
    register, so it is the return value and not the argument that describes where the
    tape actually is.
    """
    n = len(x)
    u = circ.alloc(n)
    for j in range(n):
        if (p >> j) & 1:
            circ.x(u[j])

    tape = FlatTape(d) if isinstance(d, Register) else d
    uw, vw = gcd_loop(circ, u, x, tape, **kw)

    circ.x(uw[0])                                # u == 1
    # Everything the loop touched is |0> except the tape itself: the window's own slots
    # (u == 1 was just flipped away, v == 0), the slots it released, and any pool qubit
    # the tape borrowed and gave back. Free all of it in one pass so the layout does not
    # have to know which register a qubit came from.
    d = tape.taken()
    held = set(d.ids)
    pool = set(u.ids) | set(x.ids) | tape.owned()
    circ.free(Register(sorted(pool - held)))
    if x.name is not None and circ.inputs.get(x.name) is not None:
        circ.rebind(x.name, Register([], x.name))   # x is gone, not just zero
    return d


# ---------------------------------------------------------------------------
# Jump 2: two halvings per iteration, one comparator.
#
# Worth writing the arithmetic down, because the iteration ratio invites a much larger
# claim than the truth and a mis-costed adder invites a much smaller one. Per steady
# iteration at window `w` and comparator `k`, measured with no residual:
#
#     J=1   3w + k + 1        cswap (w) + controlled subtract (2w) + compare (k)
#     J=2   4w + k - 1        the same, plus the controlled shift ladder at `w - 2`
#
# Jump 2 runs 2/3 as many, so the ratio is (8w + 2k - 2)/(9w + 3k + 3) -- 0.833 at k = w,
# 8/9 as k falls, below 1 always. Measured end to end at n = 255: 239,159 -> 198,436.
# See `reference.py`, which also records why an earlier draft of this had it at 4-8%.
#
# The bigger prize is *qubits*: 261 iterations at 7 bits per 3 (base 5) is a 609-qubit
# tape against 665, and the tape is what pins the REPLAY against PEAK_BUDGET. It does not
# show up here -- Phase 1 peaks early, before the tape is full (1107 packed against 1112).
#
# Two structural differences from `gcd_loop`, both in `gcd_loop_j2`:
#
#   * the first halving stays free but has to happen at the TOP of the iteration, because
#     that is where `v` is known even. Rotating there is also what puts a known |0> in the
#     top window slot, which is what lets the conditional ladder drop its top Fredkin.
#   * iteration 0 is different in kind, not degree: `x` may be even, so both of its
#     halvings are conditional and it logs the two shift controls rather than `c`. It pays
#     one extra ladder and saves a comparator -- `u = p > v` outright, so `b1 = b0` is a
#     CNOT -- and its four states still fit one base-5 digit. See `pack`.
# ---------------------------------------------------------------------------


class FlatTapeJ2(FlatTape):
    """Iteration `i` owns qubits (3i, 3i+1, 3i+2) of a dedicated register. 783 at 261."""

    def triple(self, i: int) -> tuple[int, int, int]:
        return self.d[3 * i], self.d[3 * i + 1], self.d[3 * i + 2]


class PooledTapeJ2(PooledTape):
    """`PooledTape` handing out three qubits an iteration instead of two."""

    def triple(self, i: int) -> tuple[int, int, int]:
        q = (self.take(), self.take(), self.take())
        self._order.extend(q)
        return q


class PackedTapeJ2(PooledTape):
    """`PooledTape` plus the 7-per-3 base-5 packing: 609 qubits of tape.

    Structurally simpler than `PackedTape` and slightly dearer per group, for the same
    reason: base 5 has no free digit 0, so the accumulator is always its own register and
    all three signal qubits are transient. They are drawn from the pool, folded, and handed
    straight back, so the layout's steady-state cost is the accumulators alone.
    """

    def __init__(self, circ: Circuit, extra: Register | None = None) -> None:
        super().__init__(circ, extra)
        self._groups: list[Register] = []
        self._trip: tuple[int, int, int] | None = None

    def triple(self, i: int) -> tuple[int, int, int]:
        j = i % pack.GROUP_J2
        if j == 0:
            self._groups.append(
                Register([self.take() for _ in range(pack.DIGIT_WIDTH_J2[0])])
            )
        else:
            grow = pack.DIGIT_WIDTH_J2[j] - len(self._groups[-1])
            self._groups[-1] = Register(
                self._groups[-1].ids + [self.take() for _ in range(grow)]
            )
        self._trip = (self.take(), self.take(), self.take())
        return self._trip

    def fold(self, i: int) -> None:
        j = i % pack.GROUP_J2
        if i == 0:
            pack.fold_digit_j2_boundary(self.circ, self._groups[-1], *self._trip)
        else:
            pack.fold_digit_j2(self.circ, self._groups[-1], j, *self._trip)
        self.give_back(list(self._trip))         # folded away; back to |0>
        self._trip = None

    def taken(self) -> Register:
        return Register([q for g in self._groups for q in g.ids], "d")


def encode_dialog_j2(tape: list[tuple[int, int, int]]) -> int:
    """The integer a flat jump-2 `d` register holds, three qubits per iteration.

    The three are the loop's *raw controls*, not the tape triple: `(c, b0, b1)` at every
    iteration past the first and `(s0, s1, b0)` at iteration 0, whose `b1` is not a wire.
    """
    val = 0
    for i, (s, b0, b1) in enumerate(tape):
        a, b, c = (
            (1 if s >= 1 else 0, 1 if s == 2 else 0, b0) if i == 0
            else (1 if s == 2 else 0, b0, b1)
        )
        val |= a << (3 * i) | b << (3 * i + 1) | c << (3 * i + 2)
    return val


def decode_dialog_j2(d: int, iters: int) -> list[tuple[int, int, int]]:
    """Inverse of `encode_dialog_j2`, back to the `(shifts, b0, b1)` tape.

    Both directions are exact rather than lossy, and neither is obvious. Past iteration 0
    the shift count is recovered as `1 + c` because `s >= 1` is an invariant of the loop
    -- odd minus odd is even, so the first halving always fires. At iteration 0 `s` may be
    0, which is why it spends two qubits there, and `b1` is not stored at all: `u = p > v`
    outright, so `b1 = b0` and the replay's cswap reads the same wire the add did.
    """
    tape = []
    for i in range(iters):
        a = (d >> (3 * i)) & 1
        b = (d >> (3 * i + 1)) & 1
        c = (d >> (3 * i + 2)) & 1
        tape.append((a + b, c, c) if i == 0 else (1 + a, b, c))
    return tape


def gcd_loop_j2(
    circ: Circuit,
    u: Register,
    v: Register,
    d: Register | FlatTapeJ2 | PooledTapeJ2,
    iters: int | None = None,
    cmp_bits: int | None = None,
    widths: list[int] | None = None,
    add_ctrl=gidney_add_ctrl,
) -> tuple[Register, Register]:
    """`gcd_loop` with two halvings per iteration. Same contract, same return.

    `widths[i]` is the window at the *top* of iteration `i`, before either halving --
    which is what `reference.dialog_widths_jump` reports and what `schedule.widths_j2`
    models. Narrowing happens before the free rotate so that the rotate lands the known
    |0> inside the new window and not in a slot just handed to the tape.
    """
    n = len(u)
    assert len(v) == n, "u and v must be the same width"
    if iters is None:
        iters = schedule.iterations_j2(n)
    if widths is None:
        widths = [n] * iters
    assert len(widths) >= iters, "width schedule is shorter than the loop"
    tape = FlatTapeJ2(d) if isinstance(d, Register) else d

    uw, vw = u, v
    for i in range(iters):
        w = widths[i]
        assert 1 <= w <= len(uw), f"iteration {i}: window {w} outside [1, {len(uw)}]"
        if w < len(uw):
            tape.release(uw.ids[w:] + vw.ids[w:])
            uw, vw = uw[0:w], vw[0:w]

        if i:
            # Halving one is unconditional -- `v` is even because odd minus odd is -- so it
            # is a relabel, and it rotates vw[0]'s guaranteed |0> into the top slot.
            vw = Register(vw.ids[1:] + vw.ids[:1], vw.name)

        a, b, c = tape.triple(i)
        if i == 0:
            s0, s1, b0 = a, b, c
            circ.cx(vw[0], s0)                   # s0 = !v[0]
            circ.x(s0)
            controlled_rshift1(circ, s0, vw)     # x may have a live top bit: no top_zero
            circ.cx(vw[0], s1)                   # s1 = !v[0], and s0 = 0 forces it to 0
            circ.x(s1)
            controlled_rshift1(circ, s1, vw, top_zero=True)
            circ.cx(vw[0], b0)                   # b0 = v & 1
            # b1 = b0 & (u > v), and u = p > v holds for every x < p, so b1 IS b0. One
            # comparator saved, and the state it would have distinguished never occurs.
            swap_ctrl = b0
        else:
            c_, b0, b1 = a, b, c
            circ.cx(vw[0], c_)                   # c = !v[0]: halving two fires
            circ.x(c_)
            controlled_rshift1(circ, c_, vw, top_zero=True)
            circ.cx(vw[0], b0)                   # b0 = v & 1
            cmp_lt(circ, uw, vw, b1, k=cmp_bits, ctrl=b0)   # b1 ^= b0 & (v < u)
            swap_ctrl = b1

        cswap(circ, swap_ctrl, uw, vw)           # if b1: swap(u, v)
        _csub(circ, b0, uw, vw, add_ctrl)        # if b0: v -= u
        tape.fold(i)

    return uw, vw


def to_dialog_j2(
    circ: Circuit,
    x: Register,
    d: Register | FlatTapeJ2 | PooledTapeJ2,
    p: int = PRIME,
    **kw,
) -> Register:
    """|x, 0> -> |d>, the jump-2 tape. Same contract as `to_dialog`; see it."""
    n = len(x)
    u = circ.alloc(n)
    for j in range(n):
        if (p >> j) & 1:
            circ.x(u[j])

    tape = FlatTapeJ2(d) if isinstance(d, Register) else d
    uw, _ = gcd_loop_j2(circ, u, x, tape, **kw)

    circ.x(uw[0])                                # u == 1
    d = tape.taken()
    held = set(d.ids)
    pool = set(u.ids) | set(x.ids) | tape.owned()
    circ.free(Register(sorted(pool - held)))
    if x.name is not None and circ.inputs.get(x.name) is not None:
        circ.rebind(x.name, Register([], x.name))
    return d


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
#
# Five points on the same construction, so each mechanism can be priced on its own: full
# width + exact compare, then the shrinking window, then the truncated comparator, then
# the tape moved into the slots the window releases, then the 5-per-3 packing on top. The
# first three carry a dedicated 798-qubit tape and are over PEAK_BUDGET; the last two are
# the same divsteps on 486 and 619 fewer qubits.
# ---------------------------------------------------------------------------
_N = U
_ITERS = schedule.iterations(_N)


def _to_dialog_case(cmp_bits: int | None, narrow: bool, layout: str = "flat") -> Case:
    widths = schedule.widths(_N, _ITERS) if narrow else None
    circ = Circuit()
    x = circ.input("x", _N)
    if layout == "flat":
        tape = FlatTape(circ.input("d", 2 * _ITERS))
    else:
        circ.input("d", 0)   # the tape has no wires of its own; rebound once it exists
        tape = (PooledTape if layout == "pooled" else PackedTape)(circ)
    d = to_dialog(circ, x, tape, PRIME, iters=_ITERS, cmp_bits=cmp_bits, widths=widths)
    if layout != "flat":
        circ.rebind("d", d)   # the tape is scattered over u, x and the top-up qubits
    encode = pack.pack_dialog if layout == "packed" else encode_dialog

    def ref(v: dict[str, int]) -> dict[str, int]:
        tape, _, _ = to_dialog_ref(PRIME, v["x"], _ITERS)
        return {"x": 0, "d": encode(tape)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"x": rng.randrange(1, PRIME), "d": 0}

    # The iteration budget dominates whatever the comparator does, and it is the only
    # failure a *wide* build has at all -- so it is the floor for every one of these.
    # Narrowing adds a second, much larger term than the comparator: the window can clip,
    # at a measured 2^-18.6, and that is a wrong answer and not a truncation. See
    # `params.gcd_width_clip_fail_prob` for why the tail is exponential rather than
    # Gaussian, which is worth 21 bits.
    fail = schedule.iteration_fail_prob(_N, _ITERS)
    if narrow:
        fail += gcd_width_clip_fail_prob(n=_N, jump=1)
    if cmp_bits is not None:
        fail += schedule.compare_fail_prob(cmp_bits - schedule.width_pad(_N))
    return Case(circ, ref, gen, meta={"fail_analytic": fail})


@circuit_version("inv.to_dialog_wide.v1")
def _build_to_dialog_wide_v1() -> Case:
    return _to_dialog_case(cmp_bits=None, narrow=False)


@circuit_version("inv.to_dialog.v1")
def _build_to_dialog_v1() -> Case:
    return _to_dialog_case(cmp_bits=None, narrow=True)


@circuit_version("inv.to_dialog_trunc.v1")
def _build_to_dialog_trunc_v1() -> Case:
    # `compare_bits`, not CMP_BITS: inside the window the top CMP_BITS bits are padding,
    # and comparing padding against padding is a coin flip. See schedule.compare_bits.
    return _to_dialog_case(cmp_bits=schedule.compare_bits(_N), narrow=True)


@circuit_version("inv.to_dialog_pooled.v1")
def _build_to_dialog_pooled_v1() -> Case:
    # Same gates as _trunc, same tape, 486 fewer qubits: the tape sits in the slots the
    # narrowing window hands back. Relabelling only, so a Toffoli difference here is a bug.
    return _to_dialog_case(schedule.compare_bits(_N), narrow=True, layout="pooled")


@circuit_version("inv.to_dialog_packed.v1")
def _build_to_dialog_packed_v1() -> Case:
    # ... and the 5-per-3 packing on top: 665 qubits of tape, for real gates this time.
    # This is the Phase 1 the replay has to be handed, because 798 + 255 + 255 does not fit.
    return _to_dialog_case(schedule.compare_bits(_N), narrow=True, layout="packed")


# --- the same five, at jump 2 ----------------------------------------------
_ITERS_J2 = schedule.iterations_j2(_N)


def _to_dialog_j2_case(
    cmp_bits: int | None,
    narrow: bool,
    layout: str = "flat",
    add_ctrl=gidney_add_ctrl,
) -> Case:
    widths = schedule.widths_j2(_N, _ITERS_J2) if narrow else None
    circ = Circuit()
    x = circ.input("x", _N)
    if layout == "flat":
        tape = FlatTapeJ2(circ.input("d", 3 * _ITERS_J2))
    else:
        circ.input("d", 0)
        tape = (PooledTapeJ2 if layout == "pooled" else PackedTapeJ2)(circ)
    d = to_dialog_j2(
        circ, x, tape, PRIME, iters=_ITERS_J2, cmp_bits=cmp_bits, widths=widths,
        add_ctrl=add_ctrl,
    )
    if layout != "flat":
        circ.rebind("d", d)

    def ref(v: dict[str, int]) -> dict[str, int]:
        tape, _, _ = to_dialog_jump_ref(PRIME, v["x"], _ITERS_J2)
        enc = (
            pack.pack_dialog_j2(tape, boundary=True) if layout == "packed"
            else encode_dialog_j2(tape)
        )
        return {"x": 0, "d": enc}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"x": rng.randrange(1, PRIME), "d": 0}

    fail = schedule.iteration_fail_prob_j2(_N, _ITERS_J2)
    if narrow:
        fail += gcd_width_clip_fail_prob(n=_N, jump=2)
    if cmp_bits is not None:
        # `compare_fail_prob` is a whole-run figure measured on the 399-iteration jump-1
        # loop, and jump 2 makes fewer calls than that -- 260, one per iteration except
        # the boundary's, whose comparison is provably 1 and is not emitted. Reusing the
        # jump-1 number is therefore conservative by ~0.6 bits, and at 2^-45 against a
        # 2^-16 iteration budget it changes no digit of the total. Re-measure it here
        # before quoting a jump-2 comparator rate on its own.
        fail += schedule.compare_fail_prob(cmp_bits - schedule.width_pad(_N))
    return Case(circ, ref, gen, meta={"fail_analytic": fail})


@circuit_version("inv.to_dialog_j2_wide.v1")
def _build_to_dialog_j2_wide_v1() -> Case:
    return _to_dialog_j2_case(cmp_bits=None, narrow=False)


@circuit_version("inv.to_dialog_j2.v1")
def _build_to_dialog_j2_v1() -> Case:
    return _to_dialog_j2_case(cmp_bits=None, narrow=True)


@circuit_version("inv.to_dialog_j2_trunc.v1")
def _build_to_dialog_j2_trunc_v1() -> Case:
    return _to_dialog_j2_case(cmp_bits=schedule.compare_bits(_N), narrow=True)


@circuit_version("inv.to_dialog_j2_pooled.v1")
def _build_to_dialog_j2_pooled_v1() -> Case:
    return _to_dialog_j2_case(schedule.compare_bits(_N), narrow=True, layout="pooled")


@circuit_version("inv.to_dialog_j2_packed.v1")
def _build_to_dialog_j2_packed_v1() -> Case:
    # The one the replay gets: 609 qubits of tape against jump 1's 665, which is the whole
    # point of jump 2. The gate saving is a third of one comparator per iteration.
    return _to_dialog_j2_case(schedule.compare_bits(_N), narrow=True, layout="packed")


@circuit_version("inv.to_dialog_j2_packed_threaded.v1")
def _build_to_dialog_j2_packed_threaded_v1() -> Case:
    """`_packed.v1` with the mask taken out of the controlled subtract. Free 254 qubits.

    Doc 11 tracked our 1,107 peak to four registers, and 254 of them were the mask that
    `gidney_add_ctrl` materialises to gate its operand. `gidney_add_ctrl_threaded` writes
    the gated sum before restoring the operand instead, so the mask never exists: measured
    853 at an *identical* 190,530 Toffoli, and no new failure term -- the tape, the
    widths, the comparator and the reference are all unchanged from `_packed.v1`. Depth
    rises 457,503 -> 571,586, because the mask's ANDs were independent of each other and
    the threaded sums sit on the carry chain.

    This supersedes `_packed.v1` on every axis the budget cares about; that one is kept as
    the reference point the doc-11 measurement is quoted against.
    """
    return _to_dialog_j2_case(
        schedule.compare_bits(_N), narrow=True, layout="packed",
        add_ctrl=gidney_add_ctrl_threaded,
    )
