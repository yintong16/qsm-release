"""Trivial demo circuits, registered so the benchmark loop can be exercised end-to-end
before any real arithmetic exists. These are NOT real circuits — the actual field / mul
/ inversion / point / scalarmul versions will register themselves in their own modules.
"""

from __future__ import annotations

import random

from ..core.circuit import Circuit
from .registry import Case, circuit_version

_N = 8


@circuit_version("demo.xor.v1")
def _xor() -> Case:
    c = Circuit()
    x = c.input("x", _N)
    y = c.input("y", _N)
    for i in range(_N):
        c.cx(x[i], y[i])  # y ^= x

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"x": v["x"], "y": v["y"] ^ v["x"]}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"x": rng.getrandbits(_N), "y": rng.getrandbits(_N)}

    return Case(c, ref, gen)


@circuit_version("demo.and.v1")
def _and() -> Case:
    c = Circuit()
    x = c.input("x", _N)
    y = c.input("y", _N)
    z = c.input("z", _N)
    for i in range(_N):
        c.ccx(x[i], y[i], z[i])  # z ^= x & y

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"x": v["x"], "y": v["y"], "z": v["z"] ^ (v["x"] & v["y"])}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"x": rng.getrandbits(_N), "y": rng.getrandbits(_N), "z": 0}

    return Case(c, ref, gen)
