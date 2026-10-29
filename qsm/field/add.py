"""Integer ripple-carry addition — CDKM and Gidney variants.

Two in-place adders computing ``b += a  (mod 2^n)`` (sum in b, a preserved). They sit
at opposite corners of the Toffoli<->space tradeoff, which is exactly what the
benchmark harness is here to make visible:

- `cdkm_add`   (Cuccaro-Draper-Kutin-Moulton, 2004): 2n Toffoli, 1 carry ancilla.
- `gidney_add` (Gidney, 2017): n-1 Toffoli, n-1 carry ancillas. Each carry is an AND
  computed with one CCX and later *erased for free* by an X-basis measurement plus a
  measurement-conditioned CZ (0 Toffoli). This halves the Toffoli count at the cost of
  holding all n-1 carries simultaneously (peak ~ 3n-1 qubits vs CDKM's 2n+1).

References:
  Cuccaro, Draper, Kutin, Moulton, "A new quantum ripple-carry addition circuit",
    arXiv:quant-ph/0410184.
  Gidney, "Halving the cost of quantum addition", arXiv:1709.06648.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import U


def _maj(c: Circuit, cin: int, b: int, a: int) -> None:
    """Majority: after this, `a` holds maj(cin,b,a) = carry-out; b,cin transformed."""
    c.cx(a, b)
    c.cx(a, cin)
    c.ccx(cin, b, a)


def _uma(c: Circuit, cin: int, b: int, a: int) -> None:
    """UnMajority-and-Add: inverse of _maj that leaves the sum bit in b, restores a,cin."""
    c.ccx(cin, b, a)
    c.cx(a, cin)
    c.cx(cin, b)


def cdkm_add(circ: Circuit, a: Register, b: Register) -> None:
    """
    In place: b += a (mod 2^n), a preserved.
    len(a) may be < len(b); missing high bits of `a` are treated as |0>.
    2 x n in-place plus 1 carry bit.

    A narrow `a` saves nothing here: CDKM parks each carry *inside* the `a` register,
    so the padding has to be materialised. (Contrast `gidney_add`, whose carries live
    in their own ancillas, making a narrow operand genuinely free.)
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    if n == 0 or na == 0:
        return
    if na < n:
        pad = circ.alloc(n - na)
        _cdkm_add_equal(circ, Register(list(a.ids) + list(pad.ids)), b)
        circ.free(pad)
        return
    _cdkm_add_equal(circ, a, b)


def _cdkm_add_equal(circ: Circuit, a: Register, b: Register) -> None:
    """CDKM proper; requires len(a) == len(b)."""
    n = len(a)
    assert len(b) == n, "operand widths must match"
    if n == 0:
        return
    carry = circ.alloc(1)
    c0 = carry[0]

    _maj(circ, c0, b[0], a[0])
    for i in range(1, n):
        _maj(circ, a[i - 1], b[i], a[i])
    # carry-out currently sits in a[n-1]; discarded for the mod-2^n adder.
    for i in range(n - 1, 0, -1):
        _uma(circ, a[i - 1], b[i], a[i])
    _uma(circ, c0, b[0], a[0])

    circ.free(carry)


# ---------------------------------------------------------------------------
# Gidney adder: temporary-AND carries, measurement-based erasure.
# ---------------------------------------------------------------------------
def _and(c: Circuit, x: int, y: int, tgt: int) -> None:
    """Compute tgt |= x & y into a fresh |0> ancilla. One Toffoli."""
    c.ccx(x, y, tgt)


def _unand(c: Circuit, x: int, y: int, tgt: int) -> None:
    """Erase tgt (== x & y) for free: X-basis measure + conditioned CZ. Zero Toffoli.

    The measurement of tgt=x&y leaves phase (-1)^((x&y)*m); the m-conditioned CZ(x,y)
    adds (-1)^(m*(x&y)). They cancel for every outcome m, so the erasure is phase-clean
    and returns tgt to |0>. `measure` consumes tgt (recycles its qubit id).
    """
    m = c.measure(tgt)
    with c.condition(m):
        c.cz(x, y)


def gidney_add(circ: Circuit, a: Register, b: Register) -> None:
    """In place: b += a (mod 2^n), a preserved. n-1 carry ancillas, n-1 Toffoli.

    Carry recurrence c_{i+1} = MAJ(a_i,b_i,c_i) = c_i ^ (a_i^c_i)(b_i^c_i), with c_0 = 0
    and the top carry c_n discarded (mod 2^n). g[k-1] holds carry c_k for k = 1..n-1.

    `len(a)` may be < `len(b)`: the missing high bits of `a` are implicitly |0>, and
    cost *no qubits at all*. Above na the recurrence degenerates to

        c_{i+1} = MAJ(0, b_i, c_i) = b_i AND c_i

    and the sum bit is just s_i = b_i ^ c_i, which the forward pass has already written
    in place -- so that region needs neither an `a` qubit nor a restore step. This is
    what lets a k-bit constant be added to an n-bit register with only k qubits of
    operand rather than n.

    This is `_gidney_add_block` with no carry in and the carry out discarded; the gate
    stream is identical to the one this function emitted before that generalisation.
    """
    _gidney_add_block(circ, a, b)


def _gidney_add_block(
    circ: Circuit,
    a: Register,
    b: Register,
    cin: int | None = None,
    cout: int | None = None,
) -> None:
    """In place: b += a + cin (mod 2^n); `cout` ^= the carry out of that same sum.

    The `gidney_add` recurrence with the two ends opened up. `a` and `cin` are preserved;
    `cout` is XORed into, not overwritten, and must be distinct from every other qubit
    here. `cin=None` means a carry-in known to be |0>, `cout=None` discards the carry-out
    (mod 2^n) -- with both defaulted this *is* `gidney_add`, gate for gate.

    Opening the ends is what makes a wide addition splittable: block j takes block j-1's
    carry-out as its carry-in, so only one block's n-1 carries are ever live. See
    `add_chunked.gidney_add_chunked`.

    n-1 ancillas and n-1 Toffoli, plus one more Toffoli for `cout` when it is asked for.
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    if n == 0 or (na == 0 and cin is None):
        return                                   # b += 0
    if n == 1:
        _block_top(circ, a, b, 0, na, cin, cout)
        return

    g = circ.alloc(n - 1)  # g[k-1] == carry c_k, for k = 1..n-1

    # --- forward: fill carries g[0..n-2] ---
    # `prev` holds c_i, or None while c_i is known to be |0>. Since c_0 = cin and a
    # starts at index 0, prev can only be None at i == 0, and only when cin is None.
    prev = cin
    for i in range(n - 1):
        if prev is None:                         # c_i == 0, so no MAJ, just an AND
            assert i < na, "a zero carry above a spent operand cannot reach here"
            _and(circ, a[i], b[i], g[i])         # c_1 = a_0 & b_0
        elif i < na:
            circ.cx(prev, a[i])                  # a_i ^= c_i
            circ.cx(prev, b[i])                  # b_i ^= c_i
            _and(circ, a[i], b[i], g[i])         # (a_i^c_i)(b_i^c_i)
            circ.cx(prev, g[i])                  # ^= c_i  ->  c_{i+1}
        else:                                    # a_i == 0
            circ.cx(prev, b[i])                  # b_i ^= c_i  (this IS s_i already)
            _and(circ, prev, b[i], g[i])         # c_i & (b_i^c_i)
            circ.cx(prev, g[i])                  # ^= c_i  ->  b_i AND c_i
        prev = g[i]

    # --- top sum bit, and the carry out if the caller wants it ---
    _block_top(circ, a, b, n - 1, na, prev, cout)

    # --- backward: write sums, erase carries (high bit to low) ---
    for i in range(n - 2, -1, -1):
        prev = g[i - 1] if i >= 1 else cin
        if prev is None:                         # c_i == 0: g[i] is already a pure AND
            _unand(circ, a[i], b[i], g[i])       # erase c_1, 0 Toffoli
            circ.cx(a[i], b[i])                  # s_0
            continue
        circ.cx(prev, g[i])                      # g[i] back to pure AND
        if i < na:
            _unand(circ, a[i], b[i], g[i])       # erase c_{i+1}, 0 Toffoli
            circ.cx(prev, a[i])                  # restore a_i
            circ.cx(a[i], b[i])                  # b_i = a_i ^ b_i ^ c_i = s_i
        else:
            _unand(circ, prev, b[i], g[i])       # operands are c_i and (b_i^c_i)
            # b_i already holds s_i = b_i ^ c_i; nothing to restore.
    # every g qubit was consumed by _unand's measurement; no free() needed.


def _block_top(
    circ: Circuit, a: Register, b: Register, i: int, na: int,
    prev: int | None, cout: int | None,
) -> None:
    """Write s_i into b[i], and c_{i+1} into `cout` if requested. `prev` holds c_i.

    Assumes the forward pass has left a_i and b_i untouched (this is the top position,
    which no later step reads), so the restores here are local.
    """
    if cout is None:                             # mod 2^n: the carry is thrown away
        if prev is not None:
            circ.cx(prev, b[i])                  # b_i ^= c_i
        if i < na:
            circ.cx(a[i], b[i])                  # b_i ^= a_i  ->  s_i
        return
    if prev is None:                             # c_i == 0
        if i < na:
            _and(circ, a[i], b[i], cout)         # c_{i+1} = a_i & b_i
            circ.cx(a[i], b[i])                  # s_i
        return
    if i < na:
        circ.cx(prev, a[i])
        circ.cx(prev, b[i])
        _and(circ, a[i], b[i], cout)             # (a_i^c_i)(b_i^c_i)
        circ.cx(prev, cout)                      # ^= c_i  ->  c_{i+1}
        circ.cx(prev, a[i])                      # restore a_i
        circ.cx(a[i], b[i])                      # b_i = a_i ^ b_i ^ c_i = s_i
    else:                                        # a_i == 0
        circ.cx(prev, b[i])                      # b_i ^= c_i  ->  s_i
        _and(circ, prev, b[i], cout)
        circ.cx(prev, cout)                      # ^= c_i  ->  b_i AND c_i


def _carry_out_block(
    circ: Circuit, a: Register, B: Register, cin: int | None, flag: int
) -> None:
    """flag ^= carry-out of (B + a + cin). B, a and cin all restored exactly.

    `compare.carry_out` with a carry-in -- the same forward ladder as
    `_gidney_add_block`, but the top carry goes to `flag` (never uncomputed, it *is* the
    answer) and the backward pass restores B instead of writing sums. n Toffoli when cin
    is given, n-1 ancillas.

    It lives here rather than in `compare` because `compare` imports this module; and
    because it is the exact inverse of the block adder's forward pass, which is the only
    reason it is correct.
    """
    n = len(B)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(B)"
    if n == 0 or (na == 0 and cin is None):
        return                                   # B + 0 never carries out
    if n == 1:
        _carry_top(circ, a, B, 0, na, cin, flag)
        return

    g = circ.alloc(n - 1)
    prev = cin
    for i in range(n - 1):
        if prev is None:
            assert i < na, "a zero carry above a spent operand cannot reach here"
            _and(circ, a[i], B[i], g[i])
        elif i < na:
            circ.cx(prev, a[i])
            circ.cx(prev, B[i])
            _and(circ, a[i], B[i], g[i])
            circ.cx(prev, g[i])
        else:
            circ.cx(prev, B[i])
            _and(circ, prev, B[i], g[i])
            circ.cx(prev, g[i])
        prev = g[i]

    _carry_top(circ, a, B, n - 1, na, prev, flag)

    for i in range(n - 2, -1, -1):               # unwind: 0 Toffoli, restores a and B
        prev = g[i - 1] if i >= 1 else cin
        if prev is None:
            _unand(circ, a[i], B[i], g[i])
            continue
        circ.cx(prev, g[i])
        if i < na:
            _unand(circ, a[i], B[i], g[i])
            circ.cx(prev, a[i])
            circ.cx(prev, B[i])
        else:
            _unand(circ, prev, B[i], g[i])
            circ.cx(prev, B[i])


def _carry_top(
    circ: Circuit, a: Register, B: Register, i: int, na: int,
    prev: int | None, flag: int,
) -> None:
    """flag ^= c_{i+1}, leaving a_i and B_i exactly as they were. `prev` holds c_i."""
    if prev is None:                             # c_i == 0
        if i < na:
            _and(circ, a[i], B[i], flag)
        return
    if i < na:
        circ.cx(prev, a[i])
        circ.cx(prev, B[i])
        _and(circ, a[i], B[i], flag)
        circ.cx(prev, flag)                      # ^= c_i  ->  c_{i+1}
        circ.cx(prev, a[i])
        circ.cx(prev, B[i])
    else:
        circ.cx(prev, B[i])
        _and(circ, prev, B[i], flag)
        circ.cx(prev, flag)
        circ.cx(prev, B[i])


# ---------------------------------------------------------------------------
# Controlled addition: b += a if ctrl, b unchanged otherwise. `a` and `ctrl` preserved.
#
# The dialog-GCD needs add-or-*nothing*, which `mul/product.py:addsub` does not provide
# (that one is add-or-subtract, and gets its second operand for free precisely because
# it never has to do nothing). So this is a separate primitive, and it comes in the same
# two flavours as the uncontrolled adders -- and for the same reason, because the
# inversion needs both corners: Phase 1 has ancillas to spare, Phase 2 has none at all.
#
#   gidney_add_ctrl:           na + n-1 Toffoli, na + n-1 ancillas
#   gidney_add_ctrl_threaded:  2n-1   Toffoli, n-1     ancillas
#   cdkm_add_ctrl:             4n     Toffoli, 1       ancilla
# ---------------------------------------------------------------------------


def gidney_add_ctrl_threaded(circ: Circuit, ctrl: int, a: Register, b: Register) -> None:
    """In place: b += a (mod 2^n) if ctrl. 2n-1 Toffoli, n-1 ancillas. Requires len(a)==n.

    `gidney_add_ctrl` with the mask removed, and it is not a tradeoff: measured at n =
    2/8/64/255 this is the *same* `2n-1` Toffoli for exactly `n` fewer qubits. (Doc 11
    predicted one extra Toffoli for the threading; there isn't one, because the top bit
    has no carry out and so never pays for a MAJ.) That is the whole reason this exists:
    the dialog loop's peak is set by its controlled subtract, and a mask is a second
    full-width register held live for the entire addition. `gidney_add_ctrl` survives only
    because it accepts a narrow `a`; this one cannot, see below.

    The mask is avoidable because of *when* the sum is written. `_gidney_add_block` clears
    the carry offset off `a_i` and then writes the sum with a plain `cx(a[i], b[i])` --
    two-qubit, so gating it needs a second control, so it needs `ctrl & a_i` to exist as a
    wire. Here the sum is written while `a_i` still carries its offset:

        cx(c_i, b[i])            # b_i ^ c_i  ->  b_i
        ccx(ctrl, a[i], b[i])    # b_i        ->  b_i ^ (ctrl & (a_i ^ c_i))
        cx(c_i, a[i])            # a_i ^ c_i  ->  a_i    (restore AFTER the write)

    One Toffoli delivers `s_i = a_i ^ b_i ^ c_i` when ctrl=1 and leaves `b_i` alone when
    ctrl=0, because the carry term rides in on the operand instead of costing its own
    gate. The carry recurrence is untouched Gidney and never sees `ctrl` at all: with
    ctrl=0 the carries are still computed from the unmodified inputs and then erased, so
    `b` is untouched and every ancilla comes back clean.

    Credit: the ordering is ecdsa.fail's (`trailmix_ludicrous/arith.rs:546`,
    `clean_add_threaded_opt`); see `docs/decisions/11-ecdsafail-controlled-adder.md`. The
    carry recurrence and the measurement-based erasure are Gidney, arXiv:1709.06648.

    A short `a` is zero-extended, as everywhere else, but it is not free here the way it
    is for `gidney_add`: above `na` the operand bit is |0>, so there is no offset operand
    for the sum to ride in on and each of those positions pays its own
    `ccx(ctrl, c_i, b_i)` -- `2n - 1` Toffoli whatever `na` is, against `gidney_add_ctrl`'s
    `na + n - 1`. So the mask is still the cheaper circuit for a genuinely narrow operand;
    this one wins at `na = n`, which is what the dialog loop passes, and at `na = n - 1`,
    which is what `modarith.modadd_lazy_ctrl` passes when it widens `b` by an overflow bit.
    """
    _gidney_add_block_ctrl_threaded(circ, ctrl, a, b)


def _gidney_add_block_ctrl_threaded(
    circ: Circuit,
    ctrl: int,
    a: Register,
    b: Register,
    cin: int | None = None,
    cout: int | None = None,
) -> None:
    """`_gidney_add_block` with the control threaded: b += a + cin if ctrl, cout ^= carry.

    `_gidney_add_block` is to `gidney_add` what this is to `gidney_add_ctrl_threaded` --
    the same recurrence with both ends opened so a wide addition can be cut into blocks
    (`add_chunked`). With `cin` and `cout` both defaulted this IS the threaded adder, gate
    for gate.

    **The carry out has to be gated, and that is not free.** The sums here are gated but
    the carry chain is not: it is computed from the unmodified `a`, `b`, `cin` whatever
    `ctrl` says, and erased again on the way down. That is fine while the carries stay
    inside the block, which is why the closed form never pays for it -- but a `cout` leaves
    the block, and an ungated one would carry a true `a + b` carry into the next block on
    a `ctrl = 0` run and corrupt it. So `cout` is written through `ccx(ctrl, c_n, cout)`
    against a scratch carry that is then erased: **2 extra Toffoli at the top**, once per
    block boundary, not once per bit. The masked `gidney_add_ctrl_chunked` gets the same
    property for free because a zero mask zeroes the whole chain -- and pays `chunk`
    ancillas for the privilege.

    Given a gated `cout`, `cin` is |0> on every `ctrl = 0` run, so the invariant holds
    inductively across a chain of blocks.

    n-1 ancillas; 2n-1 Toffoli, plus 2 when `cout` is asked for. Requires len(a)==len(b),
    for the reason in `gidney_add_ctrl_threaded`.
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    if n == 0 or (na == 0 and cin is None):
        return                                   # b += 0

    g = circ.alloc(n - 1) if n > 1 else Register([])   # g[k-1] == carry c_k

    # --- forward: fill carries, offsetting a_i and b_i by c_i as we go ---
    prev = cin                                   # c_0; None while it is known to be |0>
    for i in range(n - 1):
        if prev is None:                         # c_i == 0, so the MAJ is a bare AND
            assert i < na, "a zero carry above a spent operand cannot reach here"
            _and(circ, a[i], b[i], g[i])
        elif i < na:
            circ.cx(prev, a[i])                  # a_i ^= c_i
            circ.cx(prev, b[i])                  # b_i ^= c_i
            _and(circ, a[i], b[i], g[i])         # (a_i^c_i)(b_i^c_i)
            circ.cx(prev, g[i])                  # ^= c_i  ->  c_{i+1}
        else:                                    # a_i == 0: c_{i+1} = b_i & c_i
            circ.cx(prev, b[i])                  # b_i ^= c_i
            _and(circ, prev, b[i], g[i])         # c_i & (b_i^c_i)
            circ.cx(prev, g[i])                  # ^= c_i  ->  b_i AND c_i
        prev = g[i]

    # --- top bit: the sum always, the carry out only if the caller wants it ---
    top = n - 1
    if cout is None:
        _top_sum_ctrl(circ, ctrl, a, b, top, na, prev)
    else:
        gt = circ.alloc(1)[0]
        _top_carry_ctrl(circ, a, b, top, na, prev, gt)   # gt = c_n, ungated
        circ.ccx(ctrl, gt, cout)                 # the gate the chain does not have
        _top_carry_ctrl(circ, a, b, top, na, prev, gt, undo=True)
        _top_sum_ctrl(circ, ctrl, a, b, top, na, prev)

    # --- backward: erase carries, write sums (high to low) ---
    for i in range(n - 2, -1, -1):
        prev = g[i - 1] if i >= 1 else cin
        if prev is not None:
            circ.cx(prev, g[i])                  # c_{i+1} back to the pure AND
        if i < na:
            _unand(circ, a[i], b[i], g[i])       # erase against the offset operands
            if prev is not None:
                circ.cx(prev, b[i])              # b_i ^ c_i  ->  b_i
            circ.ccx(ctrl, a[i], b[i])           # the gated sum, a_i still offset
            if prev is not None:
                circ.cx(prev, a[i])              # restore a_i
        else:
            _unand(circ, prev, b[i], g[i])       # operands are c_i and (b_i^c_i)
            circ.cx(prev, b[i])                  # b_i ^ c_i  ->  b_i
            circ.ccx(ctrl, prev, b[i])           # s_i = b_i ^ (ctrl & c_i)
    # every g qubit was consumed by _unand's measurement; no free() needed.


def _top_sum_ctrl(
    circ: Circuit, ctrl: int, a: Register, b: Register, top: int, na: int,
    prev: int | None,
) -> None:
    """The gated top sum bit. Nothing up here was offset by the forward pass."""
    if top >= na:                                # a_top == 0, so s = b_top ^ c_top
        if prev is not None:
            circ.ccx(ctrl, prev, b[top])
        return
    if prev is None:                             # no carry either: s = b_top ^ a_top
        circ.ccx(ctrl, a[top], b[top])
        return
    circ.cx(prev, a[top])                        # borrow the offset just for the write
    circ.ccx(ctrl, a[top], b[top])               # b ^= ctrl & (a_top ^ c_top)
    circ.cx(prev, a[top])                        # and give it straight back


def _top_carry_ctrl(
    circ: Circuit, a: Register, b: Register, top: int, na: int, prev: int | None,
    gt: int, undo: bool = False,
) -> None:
    """Land the block's true (ungated) carry out on `gt`, or take it back off again.

    Split out because the gated `cout` needs the carry to exist as a wire for exactly one
    Toffoli and then go away -- there is no mask here whose zeroing would do it for free.
    """
    if top >= na:                                # a_top == 0: c_n = b_top & c_top
        if prev is None:
            return                               # c_n == 0
        if not undo:
            circ.cx(prev, b[top])
            _and(circ, prev, b[top], gt)
            circ.cx(prev, gt)
        else:
            circ.cx(prev, gt)
            _unand(circ, prev, b[top], gt)
            circ.cx(prev, b[top])
        return
    if prev is None:                             # c_n = a_top & b_top
        if not undo:
            _and(circ, a[top], b[top], gt)
        else:
            _unand(circ, a[top], b[top], gt)
        return
    if not undo:
        circ.cx(prev, a[top])
        circ.cx(prev, b[top])
        _and(circ, a[top], b[top], gt)           # (a^c)(b^c)
        circ.cx(prev, gt)                        # ^= c  ->  c_n
    else:
        circ.cx(prev, gt)
        _unand(circ, a[top], b[top], gt)
        circ.cx(prev, b[top])
        circ.cx(prev, a[top])


def gidney_add_ctrl(circ: Circuit, ctrl: int, a: Register, b: Register) -> None:
    """In place: b += a (mod 2^n) if ctrl. na + n-1 Toffoli, na + n-1 ancillas.

    Masks `a` into an ancilla register first -- `ctrl & a` costs one Toffoli per bit --
    then runs the ordinary adder on the mask. The *unmask* is free: each mask bit is a
    genuine two-qubit AND, so `_unand` erases it by measurement at zero Toffoli, and
    `_gidney_add_block` has restored the mask exactly by then.

    That asymmetry is why controlling a Gidney adder costs one extra Toffoli per operand
    bit rather than two: only the masking direction is paid for.
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    if n == 0 or na == 0:
        return

    t = circ.alloc(na)
    for i in range(na):
        _and(circ, ctrl, a[i], t[i])             # t_i = ctrl & a_i
    _gidney_add_block(circ, t, b)                # b += t; t restored
    for i in range(na):
        _unand(circ, ctrl, a[i], t[i])           # free; consumes t


def _cmaj(c: Circuit, ctrl: int, cin: int, b: int, a: int) -> None:
    """`_maj` with its one write into `b` gated on ctrl."""
    c.ccx(ctrl, a, b)
    c.cx(a, cin)
    c.ccx(cin, b, a)


def _cuma(c: Circuit, ctrl: int, cin: int, b: int, a: int) -> None:
    """`_uma` with its one write into `b` gated on ctrl. Inverse of `_cmaj` when ctrl=0."""
    c.ccx(cin, b, a)
    c.cx(a, cin)
    c.ccx(ctrl, cin, b)


def cdkm_add_ctrl(circ: Circuit, ctrl: int, a: Register, b: Register) -> None:
    """In place: b += a (mod 2^n) if ctrl. 4n Toffoli, 1 carry ancilla.

    CDKM touches `b` in exactly two places -- the `cx(a,b)` inside MAJ and the
    `cx(cin,b)` inside UMA -- so gating just those two turns the whole adder into a
    controlled one. Every other gate runs unconditionally, and with ctrl=0 each `_cuma`
    is the exact inverse of its `_cmaj`, so the carry chain scribbles on `a` and the
    carry bit and then unscribbles, leaving `b` untouched.

    Twice the Toffoli of plain CDKM, but still one ancilla. That is the trade the
    dialog replay has to take: it holds the tape plus both operands and has no room for
    a Gidney carry chain, let alone a mask.
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    if n == 0 or na == 0:
        return
    if na < n:                                   # carries live inside `a`; pad it out
        pad = circ.alloc(n - na)
        _cdkm_add_ctrl_equal(circ, ctrl, Register(list(a.ids) + list(pad.ids)), b)
        circ.free(pad)
        return
    _cdkm_add_ctrl_equal(circ, ctrl, a, b)


def _cdkm_add_ctrl_equal(circ: Circuit, ctrl: int, a: Register, b: Register) -> None:
    """Controlled CDKM proper; requires len(a) == len(b)."""
    n = len(a)
    assert len(b) == n, "operand widths must match"
    if n == 0:
        return
    carry = circ.alloc(1)
    c0 = carry[0]

    _cmaj(circ, ctrl, c0, b[0], a[0])
    for i in range(1, n):
        _cmaj(circ, ctrl, a[i - 1], b[i], a[i])
    # carry-out sits in a[n-1]; discarded for the mod-2^n adder.
    for i in range(n - 1, 0, -1):
        _cuma(circ, ctrl, a[i - 1], b[i], a[i])
    _cuma(circ, ctrl, c0, b[0], a[0])

    circ.free(carry)


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
# ---------------------------------------------------------------------------
_N = U


def _add_case(add_fn) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    add_fn(circ, a, b)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": (v["a"] + v["b"]) % (1 << _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.getrandbits(_N), "b": rng.getrandbits(_N)}

    return Case(circ, ref, gen)


@circuit_version("field.add_cdkm.v1")
def _build_add_cdkm_v1() -> Case:
    return _add_case(cdkm_add)


@circuit_version("field.add_gidney.v1")
def _build_add_gidney_v1() -> Case:
    return _add_case(gidney_add)


def _add_ctrl_case(add_fn) -> Case:
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    add_fn(circ, ctrl[0], a, b)

    def ref(v: dict[str, int]) -> dict[str, int]:
        b = (v["a"] + v["b"]) % (1 << _N) if v["ctrl"] else v["b"]
        return {"ctrl": v["ctrl"], "a": v["a"], "b": b}

    def gen(rng: random.Random) -> dict[str, int]:
        return {
            "ctrl": rng.getrandbits(1),
            "a": rng.getrandbits(_N),
            "b": rng.getrandbits(_N),
        }

    return Case(circ, ref, gen)


@circuit_version("field.add_cdkm_ctrl.v1")
def _build_add_cdkm_ctrl_v1() -> Case:
    return _add_ctrl_case(cdkm_add_ctrl)


@circuit_version("field.add_gidney_ctrl.v1")
def _build_add_gidney_ctrl_v1() -> Case:
    return _add_ctrl_case(gidney_add_ctrl)
