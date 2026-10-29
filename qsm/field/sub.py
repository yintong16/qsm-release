"""Integer subtraction, built from any adder by the complement identity.

    ~(~b + a) = b - a   (mod 2^n)

so `sub(circ, a, b)` is one adder pass sandwiched between two layers of X gates on b.
That costs the *same* Toffoli count as the adder it wraps and only 2n extra Cliffords,
so subtraction inherits whichever Toffoli<->space corner the chosen adder sits at --
no separate construction, and no separate correctness argument beyond the identity.

Proof of the identity (n-bit registers, arithmetic mod 2^n):
    ~x = (2^n - 1) - x
    ~(~b + a) = (2^n - 1) - ((2^n - 1) - b + a) = b - a
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import U
from .add import cdkm_add, gidney_add

AddFn = Callable[[Circuit, Register, Register], None]


def sub(circ: Circuit, a: Register, b: Register, add_fn: AddFn = gidney_add) -> None:
    """In place: b -= a (mod 2^n), a preserved. Cost = one `add_fn` + 2n X gates.

    `len(a)` may be < `len(b)`; the identity is unchanged (the complement layers run the
    full width of b, and the adder zero-extends a), so a narrow subtrahend is as cheap
    here as it is in the adder.
    """
    n = len(b)
    assert 0 <= len(a) <= n, "len(a) must not exceed len(b)"
    if n == 0 or len(a) == 0:
        return
    for q in b:
        circ.x(q)
    add_fn(circ, a, b)
    for q in b:
        circ.x(q)


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
# ---------------------------------------------------------------------------
_N = U


def _sub_case(add_fn: AddFn) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    sub(circ, a, b, add_fn)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": (v["b"] - v["a"]) % (1 << _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.getrandbits(_N), "b": rng.getrandbits(_N)}

    return Case(circ, ref, gen)


@circuit_version("field.sub_cdkm.v1")
def _build_sub_cdkm_v1() -> Case:
    return _sub_case(cdkm_add)


@circuit_version("field.sub_gidney.v1")
def _build_sub_gidney_v1() -> Case:
    return _sub_case(gidney_add)
