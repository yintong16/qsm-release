"""Fast bitvector reversible simulator over the computational basis.

Executes an op stream on classical bits (each qubit is a 0/1 in a bytearray). This is
exact for reversible arithmetic built from X/CX/CCX, and additionally models H-basis
measurement + CZ so that measurement-based uncomputation can be *checked*:

- Every freed qubit must be |0> at its FREE point (`DirtyAncillaError` otherwise).
- A measurement of a qubit holding value v yields a random outcome m and leaves a phase
  (-1)^(v·m); CZ contributes (-1)^(a·b). A correct (phase-clean) uncomputation cancels
  all of it, so `phase == 0` must hold for every input and every measurement outcome.
  A dirty uncomputation shows up as `phase == 1` on some shot.

The harness re-runs each input under several random measurement vectors to expose that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import ops
from .circuit import Circuit, DirtyAncillaError


@dataclass
class SimResult:
    outputs: dict[str, int]
    phase: int              # 0 = phase-clean; 1 = leftover Z kickback (dirty)
    measurements: list[int]


def simulate(
    circ: Circuit,
    inputs: dict[str, int],
    measurements: list[int] | None = None,
    rng: random.Random | None = None,
) -> SimResult:
    for name in circ.inputs:
        if name not in inputs:
            raise KeyError(f"missing value for input register {name!r}")
    if rng is None:
        rng = random.Random(0)

    state = bytearray(circ._next_qid)
    cbits: dict[int, int] = dict(circ._cbit_const)
    phase = 0
    meas_out: list[int] = []
    mi = 0

    for op in circ.ops:
        k = op.kind

        if k == ops.ALLOC:
            if op.is_input:
                val = int(inputs[op.name])
                for i, q in enumerate(op.qubits):
                    state[q] = (val >> i) & 1
            else:
                for q in op.qubits:
                    state[q] = 0
            continue

        if k == ops.FREE:
            for q in op.qubits:
                if state[q] != 0:
                    raise DirtyAncillaError(f"qubit {q} freed while holding |1>")
            continue

        if op.cond:
            fired = all(cbits.get(cb, 0) == val for cb, val in op.cond)
            if not fired:
                if k == ops.MEASURE:
                    cbits[op.cbit] = 0
                continue

        if k == ops.CPARITY:
            packed = 0
            for i, cb in enumerate(op.sources):
                packed |= cbits[cb] << i
            for j, word in enumerate(op.values):
                cbits[op.base + j] = (packed & word).bit_count() & 1
        elif k == ops.XOR_WORD:
            if op.ctrl is not None and not state[op.ctrl]:
                continue
            mask = op.mask
            while mask:
                bit = mask & -mask
                state[op.qubits[bit.bit_length() - 1]] ^= 1
                mask ^= bit
        elif k == ops.X:
            state[op.qubits[0]] ^= 1
        elif k == ops.CX:
            c, t = op.qubits
            state[t] ^= state[c]
        elif k == ops.CCX:
            a, b, t = op.qubits
            state[t] ^= state[a] & state[b]
        elif k == ops.CZ:
            a, b = op.qubits
            phase ^= state[a] & state[b]
        elif k == ops.Z:
            phase ^= state[op.qubits[0]]
        elif k == ops.MEASURE:
            t = op.qubits[0]
            if measurements is not None:
                m = measurements[mi] & 1
                mi += 1
            else:
                m = rng.getrandbits(1)
            meas_out.append(m)
            cbits[op.cbit] = m
            phase ^= state[t] & m
            state[t] = 0
        else:  # pragma: no cover
            raise ValueError(f"unknown op kind {k!r}")

    outputs: dict[str, int] = {}
    for name, reg in circ.inputs.items():
        v = 0
        for i, q in enumerate(reg.ids):
            v |= (state[q] & 1) << i
        outputs[name] = v

    return SimResult(outputs, phase & 1, meas_out)
