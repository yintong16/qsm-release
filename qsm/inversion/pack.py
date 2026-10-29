"""Packing the dialog tape at 5 qubits per 3 iterations instead of 6.

An iteration logs `(b0, s)` with `s = b0 & b1`, so `(0, 1)` cannot occur and each
iteration carries **three** states, not four. Three iterations are 27 states, which fit in
5 qubits: 399 iterations become 665 qubits rather than 798. That 133 is what decides
whether the replay -- which has to hold the whole tape alongside two 255-bit operands --
can exist at all: 665 + 255 + 255 = 1175 is `PEAK_BUDGET` exactly, and 798 + 510 = 1308
is not close.

The packing here is derived from the 3-states-per-iteration constraint rather than taken
from either reference implementation (both AGPL-3.0; see docs/decisions/10).

**`(b0, s)` is already a thermometer code.** With `t = b0 + s`, the three legal patterns
are exactly `t = 0, 1, 2` -- `b0` says `t >= 1` and `s` says `t = 2`. So a group is a
3-digit base-3 numeral and the pack is arithmetic, not a lookup table:

    V = t0 + 3*t1 + 9*t2      in [0, 27)

and it can be accumulated *as the loop runs*, one digit at a time, which is the only form
that helps -- a group's worth of raw qubits held and then compressed would peak at 6.

Each digit costs two controlled constant additions to fold in and two constant
comparisons to clear the two qubits it arrived on:

    R += 3^j  if b0        R holds sum_{i<j} 3^i t_i < 3^j beforehand,
    R += 3^j  if s         so afterwards t_j = R div 3^j exactly, and
    b0 ^= (R >= 3^j)       b0 = [t_j >= 1], s = [t_j = 2] can be read back off R
    s  ^= (R >= 2*3^j)     and returned to |0>.

The first digit of a group is free: `R = (b0 ^ s, s)` is the binary form of `t0`, so one
CNOT converts the thermometer in place and the two qubits it arrived on simply *become*
the bottom of `R`. Only digits 1 and 2 pay, and only on registers 4 and 5 qubits wide, so
the whole scheme costs ~30 Toffoli per group against the ~1,600 the group's three
divsteps already cost.

`unfold_digit` is the same steps reversed, and is what the replay uses to read the tape
back one iteration at a time.

**Jump-2 packs the same way in base 5**, in the second half of this module. A jump-2
iteration logs `(c, b0, b1)` -- did the second halving fire, is `v` odd now, is `u > v` --
and those carry five states, not eight: `c = 0` means the halving chain stopped early,
which it only does on an odd `v`, so `b0 = 1`; and `b0 = 0` forces `b1 = 0`. Five per
iteration and `5**3 = 125 <= 128`, so three iterations fit 7 qubits: 2.3333 bits per
iteration against an entropy bound of 2.3219, and **609 qubits at 261 iterations against
665 at 399**. That is the whole reason jump-2 is worth building; the gate count barely
moves (see `params.GCD_J2_SHIFT_RATE`).

The digit is again affine in the raw signals, so again no lookup table:

    t = 3*c + b1 + (b0 - 1)      in [0, 5)

which is three controlled constant additions. The `(b0 - 1)` term is the one that has to
be conditioned on `NOT b0`, and the ORDER is load-bearing: folding `+3*5^j` and `+5^j`
first and taking `-5^j` last keeps the accumulator inside `5^(j+1)`, whereas adding all
three positive terms and subtracting afterwards transiently reaches `5^(j+1) + 5^j - 1`
and needs an eighth qubit at j = 2.
"""

from __future__ import annotations

from ..core.circuit import Circuit, Register
from ..field.compare import cmp_ge_const
from ..field.constadd import add_const_inplace, sub_const_inplace

__all__ = [
    "DIGIT_WIDTH",
    "DIGIT_WIDTH_J2",
    "GROUP",
    "GROUP_BITS",
    "GROUP_BITS_J2",
    "GROUP_J2",
    "digit_j2",
    "fold_digit",
    "fold_digit_j2",
    "fold_digit_j2_boundary",
    "group_bits",
    "group_bits_j2",
    "pack_dialog",
    "pack_dialog_j2",
    "packed_bits",
    "packed_bits_j2",
    "unfold_digit",
    "unfold_digit_j2",
    "unfold_digit_j2_boundary",
    "unpack_dialog",
    "unpack_dialog_j2",
]

RADIX = 3
GROUP = 3           # iterations per packed group
GROUP_BITS = 5      # qubits a full group occupies: 3**3 = 27 <= 32

#: Width of the accumulator after folding digit j -- ceil(log2(3**(j+1))).
DIGIT_WIDTH = (2, 4, 5)

#: (b0, s) -> digit. (0, 1) is unreachable: s = b0 & b1.
_DIGIT = {(0, 0): 0, (1, 0): 1, (1, 1): 2}
_PATTERN = {v: k for k, v in _DIGIT.items()}


def group_bits(m: int) -> int:
    """Qubits a group of `m` iterations occupies. 5 for a full group, less for a stub."""
    assert 0 <= m <= GROUP
    return (0, *DIGIT_WIDTH)[m]


def packed_bits(iters: int) -> int:
    """Total tape width. 665 at 399 iterations, against 798 unpacked."""
    full, rem = divmod(iters, GROUP)
    return full * GROUP_BITS + group_bits(rem)


def pack_dialog(tape: list[tuple[int, int]]) -> int:
    """The integer a packed tape register holds for a given classical tape."""
    val, off = 0, 0
    for g in range(0, len(tape), GROUP):
        digits = tape[g:g + GROUP]
        v = sum(_DIGIT[d] * RADIX ** j for j, d in enumerate(digits))
        val |= v << off
        off += group_bits(len(digits))
    return val


def unpack_dialog(d: int, iters: int) -> list[tuple[int, int]]:
    """Inverse of `pack_dialog`."""
    tape, off = [], 0
    for g in range(0, iters, GROUP):
        m = min(GROUP, iters - g)
        w = group_bits(m)
        v = (d >> off) & ((1 << w) - 1)
        off += w
        for _ in range(m):
            tape.append(_PATTERN[v % RADIX])
            v //= RADIX
    return tape


def fold_digit(circ: Circuit, r: Register, j: int, b0: int, s: int) -> None:
    """Fold iteration `j` of a group into `r`, returning `b0` and `s` to |0>.

    `r` must be `DIGIT_WIDTH[j]` qubits holding `sum_{i<j} 3^i t_i`, with the two new
    qubits already among them for j = 0 and above the old value for j > 0. `b0` and `s`
    are left |0> for j > 0; for j = 0 they *are* `r`, and come back as its binary form.
    """
    assert 0 <= j < GROUP
    assert len(r) == DIGIT_WIDTH[j], f"digit {j} wants a {DIGIT_WIDTH[j]}-qubit accumulator"
    if j == 0:
        assert list(r.ids) == [b0, s], "digit 0's accumulator is the pair itself"
        circ.cx(s, b0)                       # (b0, s) -> (b0 ^ s, s) = binary t0
        return
    k = RADIX ** j
    add_const_inplace(circ, r, k, ctrl=b0)
    add_const_inplace(circ, r, k, ctrl=s)
    cmp_ge_const(circ, r, k, b0)             # b0 ^= [t_j >= 1] = b0
    cmp_ge_const(circ, r, 2 * k, s)          # s  ^= [t_j = 2]  = s


def unfold_digit(circ: Circuit, r: Register, j: int, b0: int, s: int) -> None:
    """Inverse of `fold_digit`: recover `(b0, s)` from `r` and take the digit back out."""
    assert 0 <= j < GROUP
    assert len(r) == DIGIT_WIDTH[j], f"digit {j} wants a {DIGIT_WIDTH[j]}-qubit accumulator"
    if j == 0:
        assert list(r.ids) == [b0, s]
        circ.cx(s, b0)
        return
    k = RADIX ** j
    cmp_ge_const(circ, r, 2 * k, s)
    cmp_ge_const(circ, r, k, b0)
    sub_const_inplace(circ, r, k, ctrl=s)
    sub_const_inplace(circ, r, k, ctrl=b0)


# --- jump-2: the same scheme in base 5 --------------------------------------------
RADIX_J2 = 5
GROUP_J2 = 3          # iterations per packed group
GROUP_BITS_J2 = 7     # qubits a full group occupies: 5**3 = 125 <= 128

#: Width of the accumulator after folding digit j -- ceil(log2(5**(j+1))).
DIGIT_WIDTH_J2 = (3, 5, 7)

#: (c, b0, b1) -> digit, via t = 3c + b1 + (b0 - 1). The three missing patterns are the
#: ones the loop cannot produce: c = 0 forces b0 = 1, and b0 = 0 forces b1 = 0.
_DIGIT_J2 = {(0, 1, 0): 0, (0, 1, 1): 1, (1, 0, 0): 2, (1, 1, 0): 3, (1, 1, 1): 4}
_PATTERN_J2 = {v: k for k, v in _DIGIT_J2.items()}

#: Iteration 0's alphabet, keyed on the tape's own `(shifts, b0, b1)`.
#:
#: The boundary iteration is the only one whose FIRST halving is conditional, so `shifts`
#: there ranges over three values and `c = [shifts == 2]` cannot separate them. It looks
#: like it needs a sixth state and therefore a wider group -- and it does not, because it
#: gains a constraint at the same time as it loses one: `u = p` and `v <= x < p`, so
#: `u > v` holds outright and `b1 = b0`. Four states, which still fits a base-5 digit, so
#: the boundary costs zero extra qubits and `packed_bits_j2` needs no special case.
#:
#: `reference.jump_boundary_states` lists six because it over-approximates on purpose:
#: `(1, 1, 0)` and `(2, 1, 0)` need `u < v`, which iteration 0 cannot produce. An
#: over-approximate alphabet is safe for sizing and this map is the exact one.
_DIGIT_J2_BOUNDARY = {(0, 1, 1): 0, (1, 1, 1): 1, (2, 0, 0): 2, (2, 1, 1): 3}
_PATTERN_J2_BOUNDARY = {v: k for k, v in _DIGIT_J2_BOUNDARY.items()}


def group_bits_j2(m: int) -> int:
    """Qubits a group of `m` jump-2 iterations occupies. 7 for a full group."""
    assert 0 <= m <= GROUP_J2
    return (0, *DIGIT_WIDTH_J2)[m]


def packed_bits_j2(iters: int) -> int:
    """Total jump-2 tape width. **609** at 261 iterations, against 665 for jump-1 at 399."""
    full, rem = divmod(iters, GROUP_J2)
    return full * GROUP_BITS_J2 + group_bits_j2(rem)


def digit_j2(entry: tuple[int, int, int], boundary: bool = False) -> int:
    """One tape entry as its base-5 digit. `boundary` selects iteration 0's alphabet."""
    if boundary:
        return _DIGIT_J2_BOUNDARY[entry]
    s, b0, b1 = entry
    return _DIGIT_J2[(1 if s == 2 else 0, b0, b1)]


def pack_dialog_j2(tape: list[tuple[int, int, int]], boundary: bool = False) -> int:
    """The integer a packed jump-2 tape register holds, given a `(shifts, b0, b1)` tape.

    `shifts` is converted to the raw control `c = [shifts == 2]` on the way in, because
    that -- not the count -- is what the circuit actually has a qubit for.

    `boundary=True` reads entry 0 with `_DIGIT_J2_BOUNDARY` instead, which is what a tape
    from `reference.to_dialog_jump` needs: its first iteration has a conditional first
    halving. Off by default so a synthetic uniform tape round-trips unchanged.
    """
    val, off = 0, 0
    for g in range(0, len(tape), GROUP_J2):
        digits = tape[g:g + GROUP_J2]
        v = sum(
            digit_j2(e, boundary and g == 0 and j == 0) * RADIX_J2 ** j
            for j, e in enumerate(digits)
        )
        val |= v << off
        off += group_bits_j2(len(digits))
    return val


def unpack_dialog_j2(d: int, iters: int, boundary: bool = False) -> list[tuple[int, int, int]]:
    """Inverse of `pack_dialog_j2`, back to `(shifts, b0, b1)`."""
    tape, off = [], 0
    for g in range(0, iters, GROUP_J2):
        m = min(GROUP_J2, iters - g)
        w = group_bits_j2(m)
        v = (d >> off) & ((1 << w) - 1)
        off += w
        for j in range(m):
            t = v % RADIX_J2
            if boundary and g == 0 and j == 0:
                tape.append(_PATTERN_J2_BOUNDARY[t])
            else:
                c, b0, b1 = _PATTERN_J2[t]
                tape.append((2 if c else 1, b0, b1))
            v //= RADIX_J2
    return tape


def fold_digit_j2(circ: Circuit, r: Register, j: int, c: int, b0: int, b1: int) -> None:
    """Fold jump-2 iteration `j` of a group into `r`, returning `c`, `b0`, `b1` to |0>.

    `r` must be `DIGIT_WIDTH_J2[j]` qubits holding `sum_{i<j} 5^i t_i`. Unlike the base-3
    scheme there is no free digit 0: the three signal qubits do not happen to be the
    binary form of any legal digit assignment, so digit 0 folds arithmetically like the
    rest, into three qubits of `r` that are |0> on entry.

    The subtraction is conditioned on NOT `b0` and comes last, which is what keeps `r`
    inside `5^(j+1)`; see the module docstring.
    """
    assert 0 <= j < GROUP_J2
    assert len(r) == DIGIT_WIDTH_J2[j], (
        f"digit {j} wants a {DIGIT_WIDTH_J2[j]}-qubit accumulator"
    )
    k = RADIX_J2 ** j
    add_const_inplace(circ, r, 3 * k, ctrl=c)
    add_const_inplace(circ, r, k, ctrl=b1)
    circ.x(b0)
    sub_const_inplace(circ, r, k, ctrl=b0)
    circ.x(b0)
    # Read the three signals back off the digit and return them to |0>. t_j = r div 5^j,
    # so each is an XOR of threshold predicates: c = [t>=2], b0 = 1 ^ [t>=2] ^ [t>=3],
    # b1 = [t>=1] ^ [t>=2] ^ [t>=4].
    cmp_ge_const(circ, r, 2 * k, c)
    circ.x(b0)
    cmp_ge_const(circ, r, 2 * k, b0)
    cmp_ge_const(circ, r, 3 * k, b0)
    cmp_ge_const(circ, r, 1 * k, b1)
    cmp_ge_const(circ, r, 2 * k, b1)
    cmp_ge_const(circ, r, 4 * k, b1)


def unfold_digit_j2(circ: Circuit, r: Register, j: int, c: int, b0: int, b1: int) -> None:
    """Inverse of `fold_digit_j2`: recover `(c, b0, b1)` and take the digit back out."""
    assert 0 <= j < GROUP_J2
    assert len(r) == DIGIT_WIDTH_J2[j], (
        f"digit {j} wants a {DIGIT_WIDTH_J2[j]}-qubit accumulator"
    )
    k = RADIX_J2 ** j
    cmp_ge_const(circ, r, 4 * k, b1)
    cmp_ge_const(circ, r, 2 * k, b1)
    cmp_ge_const(circ, r, 1 * k, b1)
    cmp_ge_const(circ, r, 3 * k, b0)
    cmp_ge_const(circ, r, 2 * k, b0)
    circ.x(b0)
    cmp_ge_const(circ, r, 2 * k, c)
    circ.x(b0)
    add_const_inplace(circ, r, k, ctrl=b0)
    circ.x(b0)
    sub_const_inplace(circ, r, k, ctrl=b1)
    sub_const_inplace(circ, r, 3 * k, ctrl=c)


def fold_digit_j2_boundary(circ: Circuit, r: Register, s0: int, s1: int, b0: int) -> None:
    """Fold iteration 0 into digit 0 of the first group, clearing `s0`, `s1`, `b0`.

    The boundary iteration's signals are not `(c, b0, b1)`: both of its halvings are
    conditional, so it carries the two shift controls themselves, and `b1` is not a wire
    at all because `u = p > v` makes it equal to `b0`. The digit is

        t = s0 + 2*s1 + b0 - 1

    on the four reachable `(s0, s1, b0)` -- `(0,0,1) (1,0,1) (1,1,0) (1,1,1)` -> 0..3.
    Affine again, so still no lookup, and still a proper subset of base 5.

    Ordering carries the same obligation as `fold_digit_j2` and a different reason: the
    negative term cannot go first because `r` starts at 0. `s0 = 0` implies `b0 = 1` (an
    unshifted `v` is odd), so `s0 + b0 >= 1` always and taking both positive units before
    the `-1` keeps every transient in [0, 3].
    """
    assert len(r) == DIGIT_WIDTH_J2[0], (
        f"the boundary digit wants a {DIGIT_WIDTH_J2[0]}-qubit accumulator"
    )
    add_const_inplace(circ, r, 1, ctrl=s0)
    add_const_inplace(circ, r, 1, ctrl=b0)
    sub_const_inplace(circ, r, 1)
    add_const_inplace(circ, r, 2, ctrl=s1)
    # t = r here, so: s0 = [t>=1], s1 = [t>=2], b0 = 1 ^ [t>=2] ^ [t>=3].
    cmp_ge_const(circ, r, 1, s0)
    cmp_ge_const(circ, r, 2, s1)
    circ.x(b0)
    cmp_ge_const(circ, r, 2, b0)
    cmp_ge_const(circ, r, 3, b0)


def unfold_digit_j2_boundary(circ: Circuit, r: Register, s0: int, s1: int, b0: int) -> None:
    """Inverse of `fold_digit_j2_boundary`."""
    assert len(r) == DIGIT_WIDTH_J2[0], (
        f"the boundary digit wants a {DIGIT_WIDTH_J2[0]}-qubit accumulator"
    )
    cmp_ge_const(circ, r, 3, b0)
    cmp_ge_const(circ, r, 2, b0)
    circ.x(b0)
    cmp_ge_const(circ, r, 2, s1)
    cmp_ge_const(circ, r, 1, s0)
    sub_const_inplace(circ, r, 2, ctrl=s1)
    add_const_inplace(circ, r, 1)
    sub_const_inplace(circ, r, 1, ctrl=b0)
    sub_const_inplace(circ, r, 1, ctrl=s0)
