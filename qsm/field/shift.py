"""Right-shift a register by one bit, under a quantum control.

At jump 1 the dialog-GCD's `v >>= 1` costs *nothing*: it happens on every iteration, so
the compiler can relabel the register's id list and move on (`gcd.gcd_loop`, and the note
in `qsm/inversion/gcd.py`'s docstring). Jump 2 is what makes a shift cost gates. Its first
halving is still unconditional -- `v` is even at the top of every iteration past the first,
because odd minus odd is even -- and stays a relabel. Its *second* halving fires only when
`v` came up even again, which is data, so it has to be a circuit.

**This is the entire price of jump 2.** Measured per steady iteration, exactly, at every
`n` in {32, 64, 128, 255} and every `k` in {5, 20, w}:

    J=1   3w + k + 1        cswap (w), controlled subtract (2w), k-bit compare
    J=2   4w + k - 1        the same, plus this ladder at `w - 2`

so the ladder is the whole difference. `qsm/inversion/reference.py` derives what that is
worth; it is repeated there and not here because this module only has to spend the `w`.

## The ladder, and why a rotate is a shift here

    for i in 0 .. w-2:  CSWAP(ctrl, r[i], r[i+1])

sends `(a0, a1, ..., a_{w-1})` to `(a1, ..., a_{w-1}, a0)` when `ctrl` is set: a *cyclic*
rotation, which is a shift only if `a0` is zero. Every caller in the GCD satisfies that by
construction rather than by luck -- the control IS `NOT r[0]`, so the bit that wraps is the
bit whose being zero enabled the gate. `CTRL_IMPLIES_LOW_ZERO` names the obligation and
`controlled_rotate1` is available for callers that genuinely want the rotate.

## The dead top Fredkin

`top_zero=True` drops the last cswap of the ladder, and it is a real Toffoli, not a
rounding error: 254 per call at the full width, over 261 iterations. Dropping it is legal
only when `r[w-1]` is also zero, and then it is exactly legal rather than approximately:
with `a0 = a_{w-1} = 0` the omitted swap exchanges two zeros. The GCD gets that for free at
every iteration past the first, because the *unconditional* halving that precedes this one
is the relabel that rotates the known-zero `r[0]` into the top slot -- so the guarantee
comes from the arithmetic, not from the width schedule. Iteration 0 is the exception: `x`
may be even, both halvings there are conditional, and the top of the window is a genuine
bit of `x`, so iteration 0 must pass `top_zero=False`.

A violated precondition is silent and total, not graceful: with `ctrl` set and `r[0] = 1`
the caller gets `(r >> 1) + 2^(w-1)` instead of `r >> 1`, and with `top_zero=True` and a
nonzero top bit it gets that bit stranded one slot below where the shift should have put
it. Neither is a small numeric error, so both are asserted in the reference and tested
directly rather than left to the composite failure rate.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import U
from .swap import cswap_bit

__all__ = [
    "controlled_rotate1",
    "controlled_rshift1",
    "controlled_lshift1",
    "rshift1_toffoli",
    "lshift1_toffoli",
]

#: The obligation every `controlled_rshift1` caller carries: `ctrl = 1` implies `r[0] = 0`.
#: Stated as a constant so the tests and the GCD can both name the same thing.
CTRL_IMPLIES_LOW_ZERO = "ctrl set requires r[0] == 0, or the wrapped bit is not a zero"


def controlled_rotate1(circ: Circuit, ctrl: int, r: Register) -> None:
    """In place: if ctrl then `r = (r >> 1) | (r[0] << (w-1))`. `w - 1` Toffoli.

    The honest primitive -- a cyclic rotation, correct for any input. `ctrl` is preserved.
    Callers that know `r[0]` is zero want `controlled_rshift1`, which is this with a name
    that records the precondition and an optional Toffoli off the top.
    """
    for i in range(len(r) - 1):
        cswap_bit(circ, ctrl, r[i], r[i + 1])


def controlled_rshift1(
    circ: Circuit, ctrl: int, r: Register, *, top_zero: bool = False
) -> None:
    """In place: if ctrl then `r >>= 1`. Requires `ctrl = 1` to imply `r[0] = 0`.

    `w - 1` Toffoli, or `w - 2` with `top_zero`, which additionally requires `r[w-1] = 0`
    and is what the dialog-GCD passes at every iteration past the first. Neither
    precondition is checkable in the circuit; see the module docstring for what a violation
    actually does.
    """
    stop = len(r) - (2 if top_zero else 1)
    for i in range(max(stop, 0)):
        cswap_bit(circ, ctrl, r[i], r[i + 1])


def rshift1_toffoli(w: int, *, top_zero: bool = False) -> int:
    """What `controlled_rshift1` costs at width `w`, without building it."""
    return max(w - (2 if top_zero else 1), 0)


def controlled_lshift1(circ: Circuit, ctrl: int, r: Register) -> None:
    """In place: if ctrl then `r <<= 1`. Requires `ctrl = 1` to imply `r[w-1] = 0`.

    The same ladder walked the other way, and the mirror image of the precondition: this
    one wraps the TOP bit into `r[0]`, so it is a shift exactly when the top bit is zero.

    There is no `top_zero` twin. Dropping a Fredkin needs a zero at *both* ends, and the
    caller that wants this -- the jump-2 replay's conditional doubling -- has a genuine
    low bit of a field element at `r[0]`. It pays the full `w - 1`.

    Phase 2 needs this for the same reason Phase 1 needs `controlled_rshift1`, which is
    worth noticing: jump 2 buys its shorter loop by paying one controlled shift ladder per
    iteration in *both* halves of the construction, never just one.
    """
    for i in range(len(r) - 1, 0, -1):
        cswap_bit(circ, ctrl, r[i], r[i - 1])


def lshift1_toffoli(w: int) -> int:
    """What `controlled_lshift1` costs at width `w`, without building it."""
    return max(w - 1, 0)


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
#
# Two points, differing by exactly one Toffoli: the general ladder and the one that drops
# its dead top. Registering both is what keeps "the top Fredkin is provably dead" a
# measured claim -- the bench shows 254 against 253, and the harness checks that both
# compute the same function on inputs that satisfy the stronger precondition.
# ---------------------------------------------------------------------------
_N = U


def _shift_case(top_zero: bool) -> Case:
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    r = circ.input("r", _N)
    controlled_rshift1(circ, ctrl[0], r, top_zero=top_zero)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"ctrl": v["ctrl"], "r": v["r"] >> 1 if v["ctrl"] else v["r"]}

    def gen(rng: random.Random) -> dict[str, int]:
        c = rng.getrandbits(1)
        # The preconditions, honoured by construction: the low bit is zero whenever the
        # gate fires, and the top bit is zero whenever the top Fredkin has been dropped.
        bits = _N - 1 if top_zero else _N
        x = rng.getrandbits(bits)
        if c:
            x &= ~1
        return {"ctrl": c, "r": x}

    return Case(circ, ref, gen)


@circuit_version("field.rshift1_ctrl.v1")
def _build_rshift1_ctrl_v1() -> Case:
    return _shift_case(top_zero=False)


@circuit_version("field.rshift1_ctrl_topzero.v1")
def _build_rshift1_ctrl_topzero_v1() -> Case:
    return _shift_case(top_zero=True)


@circuit_version("field.rotate1_ctrl.v1")
def _build_rotate1_ctrl_v1() -> Case:
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    r = circ.input("r", _N)
    controlled_rotate1(circ, ctrl[0], r)

    def ref(v: dict[str, int]) -> dict[str, int]:
        if not v["ctrl"]:
            return {"ctrl": 0, "r": v["r"]}
        return {"ctrl": 1, "r": (v["r"] >> 1) | ((v["r"] & 1) << (_N - 1))}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"ctrl": rng.getrandbits(1), "r": rng.getrandbits(_N)}

    return Case(circ, ref, gen)
