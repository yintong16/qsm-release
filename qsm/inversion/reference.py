"""Classical reference for the dialog-GCD inversion — no circuits, no qubits.

Everything the quantum circuits in this package have to reproduce, written plainly so
the tests have something to compare against and so the *algorithm* can be argued about
without the register plumbing in the way.

The construction is the binary-GCD "dialog" with Bezout reconstruction: run a binary GCD
on (p, x), record the two branch bits taken at each step, then replay that tape backwards
against a second pair of registers doing modular arithmetic. It is due to
[GRLGS23] Gouzien, Ruiz-Perez, Guilbaud, Sangouard, Phys. Rev. Lett. 131(4) 040602, and
is used in the form here by [KSGZ+25] arXiv:2510.10967. This file is written from those
papers; no code was taken from the AGPL reference implementations.

Two phases
----------
Phase 1 (`to_dialog`) is a binary GCD with the invariant `u` odd. Per iteration:

    b0 = v & 1                    # is v odd?
    b1 = u > v
    log (b0, b0 & b1)
    if b0 & b1: swap(u, v)        # keep the smaller in u
    if b0:      v -= u            # v now even
    v >>= 1

It terminates at (u, v) = (1, 0). **There is no modular reduction anywhere in it** --
it is pure integer arithmetic on p and x -- which is why this half of the circuit is
prime-agnostic, and why frenchie's `ToBitVector` has no `special_prime` flag while
`ApplyBitVector` does.

Phase 2 (`apply_dialog`) replays the tape in reverse, on a pair (a, b) mod p:

    b = 2b mod p
    if b0:      b += a mod p
    if b0 & b1: swap(a, b)

**All the modular arithmetic lives here**, which is why only this half is prime-sensitive.

Direction
---------
The replay is linear in (a, b), so it is a 2x2 matrix R over Z/p, and running the *whole*
tape reconstructs the Bezout identity of Phase 1: R applied to (1, 0) gives (p, x) over
the integers, and p == 0 mod p. So

    apply_dialog(tape, y, 0)  ==  (0, y * x mod p)

-- the composed circuit is the **multiply**, and its reverse is the division. This is
settled by `test_inversion_reference.py`, not assumed: frenchie's `ApplyBitVector`
docstring and `IPModMul.dummy_classical_function` state opposite conventions, so one of
the two is written against the reversed circuit.
"""

from __future__ import annotations

from ..params import gcd_iterations

# The three states an iteration can log. (0, 1) is impossible: b0 = 0 forces b0 & b1 = 0.
DIALOG_STATES = ((0, 0), (1, 0), (1, 1))


def to_dialog(
    p: int, x: int, iters: int | None = None
) -> tuple[list[tuple[int, int]], int, int]:
    """Phase 1. Returns (tape, u, v) after exactly `iters` iterations.

    `p` must be odd (it is a prime > 2), which is the `u` odd invariant's base case.
    A successful run ends with (u, v) == (1, 0); a run that has not converged is the
    iteration-budget failure `params.gcd_iteration_fail_prob` estimates, and shows up
    here as a non-(1, 0) return rather than as an exception.

    Running past convergence is harmless and is what the fixed circuit actually does:
    with v == 0 every later iteration logs (0, 0) and changes nothing.
    """
    assert p & 1, "p must be odd"
    if iters is None:
        iters = gcd_iterations(p.bit_length())

    u, v = p, x
    tape: list[tuple[int, int]] = []
    for _ in range(iters):
        b0 = v & 1
        s = b0 & (1 if u > v else 0)
        tape.append((b0, s))
        if s:
            u, v = v, u
        if b0:
            v -= u
        v >>= 1
    return tape, u, v


def apply_dialog(
    tape: list[tuple[int, int]], a: int, b: int, p: int
) -> tuple[int, int]:
    """Phase 2. Replays `tape` in reverse against (a, b) mod p."""
    for b0, s in reversed(tape):
        b = (2 * b) % p
        if b0:
            b = (b + a) % p
        if s:
            a, b = b, a
    return a, b


def apply_dialog_lazy(
    tape: list[tuple[int, int]], a: int, b: int, p: int, u: int
) -> tuple[int, int]:
    """`apply_dialog` in the LAZY [0, 2^u) domain -- congruent, but not reduced to [0, p).

    The circuit's `moddouble` and `modadd_lazy` both answer "did that reach 2^u?" rather
    than "did that reach p?", because the first question is a free carry bit and the
    second is a full-width comparison. So every intermediate, and the result, is a lazy
    representative: congruent to the mathematical answer, possibly sitting in [p, 2^u).
    This is the function `apply_dialog`'s circuit is *designed* to compute.

    It is not quite the function the gates compute, which is `apply_dialog_circuit`: at
    the top of the register the lazy domain is not closed, and this reference stays
    congruent where the circuit wraps. That difference is the design's smallest error
    term (`params.replay_fail_prob`, 2^-243.2 over the whole loop at 255 bits) and it is
    unreachable at real width -- but at test widths it happens constantly, so small-prime
    tests must compare against `apply_dialog_circuit` and not this.
    """
    m = 1 << u
    for b0, s in reversed(tape):
        b = 2 * b - p if 2 * b >= m else 2 * b          # moddouble
        if b0:
            t = a + b
            b = t - p if t >= m else t                  # modadd_lazy
        if s:
            a, b = b, a
    return a, b


def apply_dialog_circuit(
    tape: list[tuple[int, int]], a: int, b: int, p: int, u: int
) -> tuple[int, int]:
    """Exactly what the replay's gates compute, wraps and all.

    `apply_dialog_lazy` with both reductions taken mod 2^u, which is the whole difference:
    a u-bit register cannot hold `2b - p` or `a + b - p` when the lazy input was near the
    top of its range, and what it holds instead is that value minus 2^u -- **not**
    congruent, because 2^u is not 0 mod p. Two places it bites:

        2b - p  >= 2^u      needs b >= 2^u - f/2      `moddouble_overflows`
        a+b - p >= 2^u      needs a + b >= 2^u + p    both operands deep in [p, 2^u)

    At u = 255, f = 19 those are 9/2^255 and ~2^-503 per iteration. At u = 5, p = 29 the
    first is 9/32, which is why this function exists: the small-prime tests would
    otherwise be testing the non-closure rather than the circuit.
    """
    m = 1 << u
    for b0, s in reversed(tape):
        b = (2 * b - p) % m if 2 * b >= m else 2 * b
        if b0:
            t = a + b
            b = (t - p) % m if t >= m else t
        if s:
            a, b = b, a
    return a, b


def apply_dialog_faults(
    tape: list[tuple[int, int]], a: int, b: int, p: int, u: int
) -> dict[str, int]:
    """Count the iterations of one classical replay where the lazy domain does not close.

    The auditing counterpart of `apply_dialog_circuit`: same arithmetic, but it reports
    *where* the two references diverge instead of returning the result. Three kinds, and
    they are not the same kind of wrong:

      `double_overflow`  `2b - p` outgrew the register. Non-congruent value.
      `add_overflow`     `a + b - p` outgrew the register. Non-congruent value.
      `erasure`          the reduction happened but the result did NOT fall below `a`,
                         because `b` was a lazy representative in [p, 2^u). The value is
                         perfectly correct; what fails is `modadd_lazy`'s step-3 flag
                         erasure, and under the measurement vent that costs a PHASE.

    The third is why a small-prime replay test cannot assert a clean phase unconditionally
    -- at p = 29 the lazy band is 3 of 32 values, so it fires on most runs. At 255 bits
    it is `lazy_unreduced_prob` = 2^-250.8 per call. Auditing, never circuits: this reads
    one concrete value sequence, and a register holds a superposition of them.
    """
    m = 1 << u
    out = {"double_overflow": 0, "add_overflow": 0, "erasure": 0}
    for b0, s in reversed(tape):
        if 2 * b >= m:
            out["double_overflow"] += 2 * b - p >= m
            b = (2 * b - p) % m
        else:
            b = 2 * b
        if b0:
            t, prev = a + b, b
            if t >= m:
                if t - p >= m:
                    out["add_overflow"] += 1
                elif prev >= p:
                    out["erasure"] += 1       # result >= a, so the flag is not erased
                b = (t - p) % m
            else:
                b = t
        if s:
            a, b = b, a
    return out


def dialog_matrix(tape: list[tuple[int, int]], p: int | None = None) -> tuple[int, ...]:
    """The 2x2 matrix the replay implements, as (m00, m01, m10, m11).

    The replay is linear, so this characterises it completely; over the integers
    (`p=None`) the whole tape gives ((p, .), (x, .)) applied to (1, 0), which is where
    the Bezout identity and hence the direction of the circuit comes from.
    """
    m = [1, 0, 0, 1]                              # rows: (a, b) <- m . (a0, b0)
    red = (lambda t: t % p) if p is not None else (lambda t: t)
    for b0, s in reversed(tape):
        m[2], m[3] = red(2 * m[2]), red(2 * m[3])
        if b0:
            m[2], m[3] = red(m[2] + m[0]), red(m[3] + m[1])
        if s:
            m[0], m[1], m[2], m[3] = m[2], m[3], m[0], m[1]
    return tuple(m)


def modmul_inplace_ref(p: int, x: int, y: int) -> int:
    """y -> x*y mod p, via the full tape. What `inv.modmul_inplace` must compute."""
    tape, _, _ = to_dialog(p, x)
    _, out = apply_dialog(tape, y, 0, p)
    return out


def moddiv_inplace_ref(p: int, x: int, y: int) -> int:
    """y -> y/x mod p. The same tape run the other way."""
    tape, _, _ = to_dialog(p, x)
    a, _ = apply_dialog_inverse(tape, 0, y, p)
    return a


def apply_dialog_inverse(
    tape: list[tuple[int, int]], a: int, b: int, p: int
) -> tuple[int, int]:
    """The reverse of `apply_dialog`, gate-for-gate in the circuit sense."""
    inv2 = (p + 1) // 2                           # 2^-1 mod p, p odd
    for b0, s in tape:
        if s:
            a, b = b, a
        if b0:
            b = (b - a) % p
        b = (b * inv2) % p
    return a, b


def un_apply_dialog_circuit(
    tape: list[tuple[int, int]], a: int, b: int, p: int, u: int
) -> tuple[int, int]:
    """Exactly what the REVERSE replay's gates compute. Mirror of `apply_dialog_circuit`.

    Three gate steps per iteration, in forward tape order, each written with the wrap it
    actually performs rather than the congruence it is meant to preserve:

        if s:  swap(a, b)       cswap, its own inverse, exact
        if b0: b -= a mod p     modsub_lazy_ctrl -- re-derives `b < a` as a VALUE
        b = b/2 mod p           modhalve -- reads the flag off the parity, exact

    The middle step is where the reverse differs in kind from the forward. Forward, the
    overflow flag is erased by measurement and a disagreeing comparison costs a phase;
    backward, the flag has to be materialised to drive the `-F`, so the same disagreement
    is arithmetic. It also cannot be silent: a flag that fires on `b < F` leaves the
    subtraction's borrow stranded and the circuit frees a dirty ancilla. This reference
    raises `ValueError` at exactly those inputs, which is what that exception means.

    The forward's image never contains them (`test_field_modhalve.py` pins both halves of
    that argument), so on any (a, b) that came out of `apply_dialog_circuit` this is its
    exact inverse -- non-closure included, since the two wrap the same way.
    """
    m = 1 << u
    f = m - p
    for b0, s in tape:
        if s:
            a, b = b, a
        if b0:
            if b < a and b < f:
                raise ValueError(
                    f"b = {b} is below F = {f} with the flag set: the -F borrows out of "
                    f"the register and the circuit frees a dirty ancilla here"
                )
            b = b - a if b >= a else b + p - a
        flag = b & 1
        b = (((b - flag * f) % m) >> 1) | (flag << (u - 1))
    return a, b


def un_dialog(tape: list[tuple[int, int]]) -> tuple[int, int]:
    """|d> -> |p, x>: the tape run backward over the INTEGERS. Inverse of `to_dialog`.

    The composite needs `x` back after the replay has spent it, and this is how it comes
    back. What makes it cheap is that it is the replay's own loop body with the modular
    arithmetic taken out:

        replay      b = 2b mod p;  if b0: b = a + b mod p;  if s: swap
        un_dialog   b = 2b;        if b0: b = a + b;        if s: swap

    Starting from (u, v) = (1, 0) -- which is exactly the state `to_dialog` leaves behind
    -- this reconstructs (p, x). `p` is a compile-time constant, so in the circuit it is
    XORed away and only `x` survives; the reference returns both so a test can check that
    the p half really did come back, which is the strongest available signal that the
    tape was replayed against the right state.

    No comparator appears here, and it is worth being precise about what that does and
    does not buy. Phase 1 spends a truncated `u > v` every iteration to *derive* `b1`;
    this pass reads `b0 & b1` off the tape and branches on it, so reconstructing the
    VALUES needs no comparison. But the composite needs `|d> -> |x>`, not
    `|d> -> |d, x>`: the tape has to come back to |0> or it stays entangled with `x`.
    Erasing `s` means re-deriving `b0 & (u > v)` on the restored `(u, v)` -- the same
    controlled comparator, on the same operands, at the same width. So `to_dialog` and
    its inverse cost the *same*, gate for gate, and the composite really is
    2 x Phase 1 + Phase 2. That is exactly what both baselines report:
    frenchie's `IPModMul` is `218,432 * 2 + 568,214 = 1,005,078`.

    What this function is for, then, is the state trajectory: it says what `(u, v)` the
    reverse pass holds at every step, which is what the circuit is checked against, and
    it shows that the reverse loop body is the replay's own body with the modulus
    deleted -- so `undo.un_gcd_loop` reuses `gcd_loop`'s primitives rather than needing
    new ones.

    The registers GROW here rather than shrink -- `v` doubles every iteration -- so the
    width schedule runs in reverse too: `schedule.width(iters - 1 - i, n)`.
    """
    u, v = 1, 0
    for b0, s in reversed(tape):
        v <<= 1
        if b0:
            v += u
        if s:
            u, v = v, u
    return u, v


# ---------------------------------------------------------------------------
# Jump-J: the same GCD, with the loop boundary moved and J halvings per iteration.
# ---------------------------------------------------------------------------
#
# The loop above ends each iteration with `v >>= 1`. Move that shift to the *top* of the
# next iteration and nothing changes -- `v` is even there, because the subtraction that
# preceded it made an odd number odd. Having moved it, a second halving can be attached
# to it, taken only when `v` is still even:
#
#     v >>= 1                        # unconditional: v is even at the top of the loop
#     repeat J-1 times:
#         if v even: v >>= 1         # a CONTROLLED rotation, the only new gate
#     b0 = v & 1
#     if b0:  b1 = u > v;  if b1: swap(u, v);  v -= u
#
# So an iteration now consumes up to J halvings while still paying for exactly one
# comparison, one cswap and one controlled subtraction. Trailing zeros after a subtraction
# are Geom(1/2) with mean 2, so at J = 2 the iteration count falls by E[ceil(tz/2)]/E[tz]
# = (4/3)/2 = 2/3 -- measured 0.668 over 20,000 random x at 255 bits.
#
# That is NOT a 3/2 speedup, and the size of the shortfall decides J. Each iteration gains
# a controlled rotation of the live window, so the per-iteration cost goes from
# `cswap(w) + ctrl_sub(w) + cmp(k)` to `+ ctrl_rot(w)`. The measured Toffoli counts fit
# both lines exactly -- at every n in {32, 64, 128, 255} and every k in {5, 20, w}, no
# residual (`tests/test_inversion_gcd_j2.py` pins the fit):
#
#     J=1   H * (3w + k + 1)
#     J=2   (2/3)H * (4w + k - 1)
#
# The cswap is `w` and the controlled subtract is `2w` -- a controlled Gidney add is two
# Toffoli per bit, not one, which is the whole reason the `w` terms do not cancel. Their
# ratio is (8w + 2k - 2) / (9w + 3k + 3): 0.833 at k = w, rising to 8/9 as k falls. So
#
#   * jump 2 wins on gates UNCONDITIONALLY -- the inequality reduces to `0 < w + k + 5` --
#     by 11% to 17%, and the comparator only modulates a win it does not cause. Dropping
#     one iteration in three saves `3w + k + 1`; the ladder that buys it costs `w - 2`.
#   * it is still nowhere near 1.53x, and quoting the iteration ratio as a gate ratio
#     overstates the win by a factor of about two.
#
# An earlier draft of this note had the subtract at `w`, concluded the `w` terms cancelled
# exactly, and put the whole win at one third of a comparator (4-8%). That was wrong in
# the arithmetic, not the measurement, and it was wrong in the *pessimistic* direction --
# worth recording because it nearly cost the jump-2 branch on a mis-costed adder.
#
# The win that decides it is width, not gates. `s2 = 0` forces `b0 = 1` (an unshifted `v`
# is odd by definition), so an iteration has 2(J-1) + 3 = 2J+1 states, not 4J:
#
#     J=1  3 states, 3^3 = 27 <= 2^5    5 bits / 3 iters = 1.667/iter
#     J=2  5 states, 5^3 = 125 <= 2^7   7 bits / 3 iters = 2.333/iter
#     J=3  7 states, 7^5 <= 2^14       14 bits / 5 iters = 2.800/iter
#
# Against 2/3 as many iterations, J=2's tape is 0.933x J=1's and J=3's is 0.963x, so J=2
# is the tape optimum -- J=3 buys fewer iterations but spends them on a base that packs
# badly. At a 4-sigma budget that is 609 tape qubits against 665, and since the replay
# holds d + a + b, it is 56 qubits of headroom on a design that currently sits at
# PEAK_BUDGET exactly. Measured at the replay: 1,223 -> 1,169 on CDKM, so 54 of the 56
# after the base-5 reader's two extra carried qubits, and six under the budget. That is
# the first replay in the design that fits, and it is the result the jump was built for.
#
# Phase 2 pays for the same thing Phase 1 does, and this is worth stating plainly because
# it is easy to price the jump once and forget: a conditional shift can never be a relabel
# in EITHER direction, so Phase 1 buys its shorter loop with a `controlled_rshift1` ladder
# per iteration and Phase 2 buys it again with a `controlled_lshift1` ladder per iteration.
# Phase 2's is `n + 1` Toffoli (the ladder runs over the widened register, plus one for the
# gated parity erasure), 66,816 over 261 iterations, against 138 dropped additions, swaps
# and comparators. Measured on matched pairs at n = 255: 0.890 exact-Gidney, 0.904
# truncated-Gidney, 0.814 truncated-CDKM. The spread is not noise -- the win is widest
# where an addition is dearest -- and none of it is near 261/399 = 0.654.
#
# Iteration 0 is the one place the shape breaks: `x` may be even, so its first halving is
# conditional too and it has its own 6-state alphabet. It is one iteration, encoded once,
# and is kept out of the uniform groups rather than widening all of them.

DIALOG_STATES_J1 = DIALOG_STATES


def jump_states(jump: int) -> tuple[tuple[int, int, int], ...]:
    """The `2*jump + 1` states an iteration can log, as (shifts, b0, b1).

    `shifts` < `jump` means the halving chain stopped early, which happens only because
    `v` came up odd -- so `b0` is forced to 1 there and those states carry only `b1`. Only
    a full chain of `jump` halvings can leave `v` even and log `b0 = 0`.
    """
    assert jump >= 1
    out = [(s, 1, b1) for s in range(1, jump) for b1 in (0, 1)]
    out += [(jump, 0, 0), (jump, 1, 0), (jump, 1, 1)]
    return tuple(sorted(out))


# Iteration 0 is the only one whose first halving is conditional, so it is the only one
# that can log `shifts == 0`. It logs nothing else the others cannot: `b1 = u > v` with
# u = p and v = x < p is always 1 there.
JUMP_BOUNDARY_STATE = (0, 1, 1)


def jump_boundary_states(jump: int) -> tuple[tuple[int, int, int], ...]:
    """The alphabet of tape entry 0: `jump_states` plus the one boundary state."""
    return tuple(sorted(jump_states(jump) + (JUMP_BOUNDARY_STATE,)))


def to_dialog_jump(
    p: int, x: int, iters: int, jump: int = 2
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Phase 1 with `jump` halvings per iteration. Returns (tape, u, v).

    Entry 0 of the tape is the boundary iteration and may carry `shifts == 0`; every later
    entry is one of `jump_states(jump)`.

    Running past convergence is harmless exactly as it is at jump 1, but only because the
    halving chain is NOT stopped at `v == 0`. The circuit's controlled rotation is
    conditioned on `!v[0]`, which is set when `v` is zero, so it fires and rotates zeros;
    guarding on `v` here instead would log `(1, 0, 0)` for every converged iteration and
    add a state to the alphabet that the gates never produce.
    """
    assert p & 1, "p must be odd"
    assert jump >= 1
    u, v = p, x
    tape: list[tuple[int, int, int]] = []
    for i in range(iters):
        s = 0
        for j in range(jump):
            if j == 0 and i > 0:
                v >>= 1                       # v is even here by the loop's invariant
                s += 1
            elif v % 2 == 0:
                v >>= 1
                s += 1
            else:
                break
        b0 = v & 1
        b1 = (1 if u > v else 0) if b0 else 0
        tape.append((s, b0, b1))
        if b1:
            u, v = v, u
        if b0:
            v -= u
    return tape, u, v


def apply_dialog_jump(
    tape: list[tuple[int, int, int]], a: int, b: int, p: int
) -> tuple[int, int]:
    """Phase 2 for a jump-J tape. Replays in reverse, dualising each op.

    A forward iteration runs `shift^s`, then `swap^b1`, then `sub^b0`. Reversing a
    composition reverses the order and dualises each step, so the replay runs `add^b0`,
    then `swap^b1`, then `double^s` -- the same three primitives the jump-1 replay uses,
    with the doubling repeated instead of the loop.
    """
    for s, b0, b1 in reversed(tape):
        if b0:
            b = (b + a) % p
        if b1:
            a, b = b, a
        for _ in range(s):
            b = (2 * b) % p
    return a, b


def apply_dialog_jump_lazy(
    tape: list[tuple[int, int, int]], a: int, b: int, p: int, u: int
) -> tuple[int, int]:
    """`apply_dialog_jump` in the LAZY [0, 2^u) domain. See `apply_dialog_lazy`."""
    m = 1 << u
    for s, b0, b1 in reversed(tape):
        if b0:
            t = a + b
            b = t - p if t >= m else t                  # modadd_lazy
        if b1:
            a, b = b, a
        for _ in range(s):
            b = 2 * b - p if 2 * b >= m else 2 * b      # moddouble
    return a, b


def apply_dialog_jump_circuit(
    tape: list[tuple[int, int, int]], a: int, b: int, p: int, u: int
) -> tuple[int, int]:
    """Exactly what the jump replay's gates compute, wraps and all.

    `apply_dialog_jump_lazy` with both reductions taken mod 2^u -- see
    `apply_dialog_circuit` for why the difference exists and why small-prime tests have to
    compare against this and not the congruent version.
    """
    m = 1 << u
    for s, b0, b1 in reversed(tape):
        if b0:
            t = a + b
            b = (t - p) % m if t >= m else t
        if b1:
            a, b = b, a
        for _ in range(s):
            b = (2 * b - p) % m if 2 * b >= m else 2 * b
    return a, b


def apply_dialog_jump_inverse(
    tape: list[tuple[int, int, int]], a: int, b: int, p: int
) -> tuple[int, int]:
    """The reverse of `apply_dialog_jump`, over the congruence classes. The DIVISION.

    `apply_dialog_jump` computes `b -> b * x mod p` from a tape for `x`; run backwards on
    the same tape it computes `b -> b / x mod p`, which is the only division circuit the
    composite needs. Reversing a composition reverses the order and dualises each step, so
    the loop walks the tape FORWARDS and runs `halve^s`, then `swap^b1`, then `sub^b0`.
    """
    inv2 = (p + 1) // 2                           # 2^-1 mod p, p odd
    for s, b0, b1 in tape:
        for _ in range(s):
            b = (b * inv2) % p
        if b1:
            a, b = b, a
        if b0:
            b = (b - a) % p
    return a, b


def un_apply_dialog_jump_circuit(
    tape: list[tuple[int, int, int]], a: int, b: int, p: int, u: int
) -> tuple[int, int]:
    """Exactly what the REVERSE jump replay's gates compute. See `un_apply_dialog_circuit`.

    Three gate steps per tape entry, in forward tape order, each written with the wrap it
    performs rather than the congruence it preserves:

        halve^s                 `modhalve` once and `modhalve_ctrl_inplace` for the rest
        if b1: swap(a, b)       cswap, its own inverse, exact
        if b0: b -= a mod p     `modsub_lazy_ctrl` -- re-derives `b < a` as a VALUE

    The two reversals cost differently and the difference is structural, not incidental.
    The halvings are FREE: their forward erased its reduction flag with the output's
    PARITY, exact at every width, so the reverse reads it straight back -- and it stays
    free when gated, because the gated forward erases with a Toffoli rather than a
    measurement (`test_field_modhalve.py` pins the gate counts in both directions). The
    subtraction is not: `modadd_lazy_ctrl` MEASURED its flag and vented the disagreement
    into a phase, and a measurement has no inverse, so the reverse materialises the
    predicate with a real comparator at +2 Toffoli per entry.

    That asymmetry is why jump 2's reverse costs jump 2's forward plus `2 * 261` and not
    plus `2 * 399`: the surcharge rides on ADDITIONS, which the jump cut, and not on
    doublings, which it did not.

    `ValueError` marks the inputs where the materialised flag leaves the `-F` borrow
    stranded and the circuit frees a dirty ancilla -- the same non-total corner
    `un_apply_dialog_circuit` has, unreachable through the forward for the same reason.
    """
    m = 1 << u
    f = m - p
    for s, b0, b1 in tape:
        for _ in range(s):
            flag = b & 1
            b = (((b - flag * f) % m) >> 1) | (flag << (u - 1))
        if b1:
            a, b = b, a
        if b0:
            if b < a and b < f:
                raise ValueError(
                    f"b = {b} is below F = {f} with the flag set: the -F borrows out of "
                    f"the register and the circuit frees a dirty ancilla here"
                )
            b = b - a if b >= a else b + p - a
    return a, b


def un_apply_dialog_jump_trunc_faults(
    tape: list[tuple[int, int, int]],
    a: int,
    b: int,
    p: int,
    u: int,
    window: int | None = None,
) -> dict[str, int]:
    """Count the TRUNCATION faults a reverse jump replay would take on this trajectory.

    `un_replay_j2_fail_prob` prices the `-F` borrow at `F / 2^window` per firing, which is
    the rate for a UNIFORM operand. The operands here are not drawn, they are produced --
    each is the previous one halved or reduced -- and for a structured numerator the
    sequence stays in the danger set instead of visiting it at random. This walks the
    actual sequence and counts, which is the only way to tell the two apart.

    Three kinds, and they are not interchangeable:

      `halve`   the gated `-F` inside `modhalve_ctrl_inplace` borrows out of the window
      `sub`     the `-F` inside `modsub_lazy_ctrl` (step R2) borrows out of the window
      `band`    `b < a` with `b < F`: the EXACT circuit's own hole, which no window width
                closes -- `un_apply_dialog_jump_circuit` raises here

    `window=None` counts only `band`, so the difference between the two calls separates
    "the truncation did this" from "the algorithm does this". Totals over six tapes at
    n = 255, `window = 45`:

        numerator     halve   sub    band
        random          0       0      0
        p + 1           0      15      0
        1               7      17      1
        19              0     315      0
        2^128           0     125      0

    and the first two columns are the point. A uniform operand takes `19/2^45` per firing,
    so `2^-31.6` of a fault across a whole replay; a numerator of 19 takes fifty-two. The
    `band` column says the exact circuit has a hole too, and that `1` is in it -- widening
    the window to the full register leaves that column unchanged.

    This is the measurement `qsm/inversion/inplace.py` declines to build an inverse on.
    """
    m = 1 << u
    f = m - p
    if window is not None and window >= u:
        window = None                 # a window as wide as the register IS the exact one
    msk = (1 << window) - 1 if window is not None else 0
    out = {"halve": 0, "sub": 0, "band": 0}
    for s, b0, b1 in tape:
        for _ in range(s):
            flag = b & 1
            if flag and window is not None and (b & msk) < f:
                out["halve"] += 1
            b = (((b - flag * f) % m) >> 1) | (flag << (u - 1))
        if b1:
            a, b = b, a
        if b0:
            if b < a:
                if b < f:
                    out["band"] += 1
                elif window is not None and (b & msk) < f:
                    out["sub"] += 1
            b = b - a if b >= a else b + p - a
    return out


def apply_dialog_jump_faults(
    tape: list[tuple[int, int, int]], a: int, b: int, p: int, u: int
) -> dict[str, int]:
    """`apply_dialog_faults` for a jump-2 tape. Same three kinds, differently distributed.

    The counts move even though the arithmetic does not, and the split is the whole story
    of what jump 2 costs and what it does not:

      `add_overflow`, `erasure`   proportional to the number of ADDITIONS, which jump 2
                                  cuts from 399 to 261. Strictly fewer chances to fail.
      `double_overflow`           proportional to the number of DOUBLINGS, which is
                                  invariant under the jump -- the `s` sum to the divstep
                                  count either way. Identical, not merely similar.

    That invariance is not a coincidence to be checked per prime: it is why
    `params.replay_j2_fail_prob` keeps the doubling term at `2 * iters` and scales only the
    per-iteration terms by 261.

    The doubling here is `moddouble_ctrl_inplace` for the conditional ones, whose parity
    erasure is a Toffoli rather than a measurement vent and is therefore exact -- so unlike
    the addition it contributes no `erasure`, only the value fault.
    """
    m = 1 << u
    out = {"double_overflow": 0, "add_overflow": 0, "erasure": 0}
    for s, b0, b1 in reversed(tape):
        if b0:
            t, prev = a + b, b
            if t >= m:
                if t - p >= m:
                    out["add_overflow"] += 1
                elif prev >= p:
                    out["erasure"] += 1       # result >= a, so the flag is not erased
                b = (t - p) % m
            else:
                b = t
        if b1:
            a, b = b, a
        for _ in range(s):
            if 2 * b >= m:
                out["double_overflow"] += 2 * b - p >= m
                b = (2 * b - p) % m
            else:
                b = 2 * b
    return out


def un_dialog_jump(tape: list[tuple[int, int, int]]) -> tuple[int, int]:
    """|d> -> |p, x>: the jump tape run backward over the integers. See `un_dialog`."""
    u, v = 1, 0
    for s, b0, b1 in reversed(tape):
        if b0:
            v += u
        if b1:
            u, v = v, u
        for _ in range(s):
            v <<= 1
    return u, v


def dialog_widths_jump(p: int, x: int, iters: int, jump: int = 2) -> list[int]:
    """Bit-length of max(u, v) before each jump-J iteration. See `dialog_widths`.

    The window schedule cannot simply be `dialog_widths` resampled: an iteration consumes
    a variable number of halvings, so the values shrink per *iteration* at a different and
    noisier rate than they do per divstep. This is what that rate has to be measured from.
    """
    assert p & 1, "p must be odd"
    u, v = p, x
    out: list[int] = []
    for i in range(iters):
        out.append(max(u, v).bit_length())
        for j in range(jump):
            if j == 0 and i > 0:
                v >>= 1
            elif v % 2 == 0:
                v >>= 1
            else:
                break
        b0 = v & 1
        b1 = (1 if u > v else 0) if b0 else 0
        if b1:
            u, v = v, u
        if b0:
            v -= u
    return out


def dialog_widths(p: int, x: int, iters: int | None = None) -> list[int]:
    """Bit-length of max(u, v) before each iteration -- what the width schedule bounds.

    The circuit narrows its `u`/`v` registers as the loop runs (that shrinkage is the
    largest single gate saving in Phase 1) and any value that outgrows its window is
    silently truncated, so this is the measurement the schedule's padding has to cover.
    """
    assert p & 1, "p must be odd"
    if iters is None:
        iters = gcd_iterations(p.bit_length())

    u, v = p, x
    out: list[int] = []
    for _ in range(iters):
        out.append(max(u, v).bit_length())
        b0 = v & 1
        s = b0 & (1 if u > v else 0)
        if s:
            u, v = v, u
        if b0:
            v -= u
        v >>= 1
    return out
