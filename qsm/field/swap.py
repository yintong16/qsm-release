"""Controlled swap of two registers — Fredkin, one bit at a time.

`if ctrl: a, b = b, a`. Nothing clever is possible here: a Fredkin gate is a Toffoli
plus two CNOTs, and n of them is the whole circuit. It gets its own module because in
the dialog-GCD loop it is *not* a footnote -- Phase 1 runs one cswap per iteration
alongside one controlled subtract, so at 255 bits the swaps are a comparable share of
the Toffoli count to the arithmetic they gate.

The identity used per bit is the standard three-CNOT Fredkin decomposition:

    CX(b,a) . CCX(ctrl,a,b) . CX(b,a)   sends   (a,b) -> (a ^ ctrl(a^b), b ^ ctrl(a^b))

which is (a,b) for ctrl=0 and (b,a) for ctrl=1. `ctrl` is preserved.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import U


def cswap(circ: Circuit, ctrl: int, a: Register, b: Register) -> None:
    """In place: if ctrl then swap(a, b). n Toffoli and 2n CNOT, no ancillas.

    Widths must match. `ctrl` is left exactly as it was.
    """
    assert len(a) == len(b), "cswap operands must have equal width"
    for x, y in zip(a.ids, b.ids):
        circ.cx(y, x)
        circ.ccx(ctrl, x, y)
        circ.cx(y, x)


def cswap_bit(circ: Circuit, ctrl: int, x: int, y: int) -> None:
    """The single-qubit case, for swapping loose flags. One Toffoli."""
    circ.cx(y, x)
    circ.ccx(ctrl, x, y)
    circ.cx(y, x)


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
# ---------------------------------------------------------------------------
_N = U


@circuit_version("field.cswap.v1")
def _build_cswap_v1() -> Case:
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    cswap(circ, ctrl[0], a, b)

    def ref(v: dict[str, int]) -> dict[str, int]:
        if v["ctrl"]:
            return {"ctrl": 1, "a": v["b"], "b": v["a"]}
        return {"ctrl": 0, "a": v["a"], "b": v["b"]}

    def gen(rng: random.Random) -> dict[str, int]:
        return {
            "ctrl": rng.getrandbits(1),
            "a": rng.getrandbits(_N),
            "b": rng.getrandbits(_N),
        }

    return Case(circ, ref, gen)
