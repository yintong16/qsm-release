"""Exact resource counting over a circuit's op stream.

All metrics are computed statically from the emitted op stream, independent of
simulation outcomes. Compact word fanout is charged as its ordered physical gate
expansion; classical parity-table work is excluded. Metric display is `bench.report`'s
job. Classical-controller latency and classical-dependency timing are not modelled.

Primitive counts
----------------
- toffoli:      CCX gates. The expensive resource: each consumes a distilled magic
                state (Eastin-Knill forbids a transversal non-Clifford gate).
- clifford:     X + CX + CZ + Z gates (lattice surgery / Pauli frame).
- measurements: H-basis measurements.
- peak_qubits:  high-water mark of simultaneously-live qubits.
- depth:        ASAP scheduling depth over all gate ops.
- toffoli_depth: ASAP critical path counting *only* CCX layers — the runtime axis when
                magic-state consumption is the bottleneck.
- n_gates:      physical gate ops (excludes bookkeeping and classical-only ops).

Cost models (three architectures rank circuits differently, so we report all three
and let the paper pick; see docs/decisions/)
---------------------------------------------------------------------------------
- circuit_volume = peak_qubits * toffoli.
    The textbook cost for a *conventional 2D-local surface code* machine: "the cost of
    a quantum computation is determined by the circuit volume, i.e. the number of
    qubits multiplied by the number of non-Clifford gates" (Litinski & Nickerson,
    arXiv:2211.15465). This is also exactly the ecdsa.fail challenge score, so it is
    our comparability anchor.

- toffoli_volume = sum over each CCX of the qubits live at that moment.
    A strict refinement of circuit_volume: it replaces the global peak with the
    *actual* live width at each magic-state consumption. Always <= circuit_volume;
    the gap is large precisely when a design's peak is a brief transient (e.g. an
    adder that borrows n carry ancillas only for the duration of the add). This is
    the honest version of "space x time" for a 2D machine.

- volume = sum over every gate of the qubits live at that moment.
    Full spacetime volume in gate-time units (idle qubits included), i.e. the discrete
    integral of live-width over the serial gate schedule.

Not modelled here: magic-state factory footprint (driven by Toffoli *rate*), routing
overhead, and the code distance d (which grows with total volume). Under an
active-volume architecture (arXiv:2211.15465) cost stops scaling with qubit count
altogether and tracks non-Clifford operations instead -- for that regime `toffoli` and
`toffoli_depth` are the relevant numbers, not the volumes.

Toffoli is counted as *emitted* (static) -- conditioned gates still count. An
executed/average figure (per-shot, respecting the condition stack) can be added later
if the paper needs it; emitted is the standard resource-estimate quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ops
from .circuit import Circuit


@dataclass(frozen=True)
class Resources:
    toffoli: int
    clifford: int
    measurements: int
    peak_qubits: int
    depth: int
    n_gates: int
    toffoli_depth: int
    volume: int
    toffoli_volume: int
    circuit_volume: int


def count(circ: Circuit) -> Resources:
    toffoli = clifford = measurements = n_gates = 0
    live = 0
    peak = 0
    volume = 0
    toffoli_volume = 0
    # ASAP depth: last "time" each qubit was touched. qtime counts every gate;
    # qtime_tof advances only on CCX, giving the magic-state critical path.
    qtime: dict[int, int] = {}
    qtime_tof: dict[int, int] = {}

    for op in circ.ops:
        k = op.kind
        if k == ops.ALLOC:
            live += len(op.qubits)
            peak = max(peak, live)
            continue
        if k == ops.FREE:
            live -= len(op.qubits)
            continue

        if k == ops.CPARITY:
            # Explicit classical workload, but no quantum gate/latency/volume.
            continue
        if k == ops.XOR_WORD:
            n = op.mask.bit_count()
            clifford += n
            n_gates += n
            volume += live * n
            # Schedule the exact ordered physical expansion, without creating Ops
            # or a flat gate list. In particular CX shares a serial control wire,
            # and propagates existing Toffoli dependencies without adding a layer.
            mask = op.mask
            ctrl = op.ctrl
            while mask:
                bit = mask & -mask
                target = op.qubits[bit.bit_length() - 1]
                if ctrl is None:
                    qtime[target] = qtime.get(target, 0) + 1
                else:
                    t = max(qtime.get(ctrl, 0), qtime.get(target, 0)) + 1
                    qtime[ctrl] = qtime[target] = t
                    t_tof = max(qtime_tof.get(ctrl, 0), qtime_tof.get(target, 0))
                    qtime_tof[ctrl] = qtime_tof[target] = t_tof
                mask ^= bit
            continue

        n_gates += 1
        if k == ops.CCX:
            toffoli += 1
        elif k in (ops.X, ops.CX, ops.CZ, ops.Z):
            clifford += 1
        elif k == ops.MEASURE:
            measurements += 1

        # Spacetime accrual. A measured qubit is still live *during* its own
        # measurement, so charge the width first and consume it afterwards.
        volume += live
        if k == ops.CCX:
            toffoli_volume += live
        if k == ops.MEASURE:
            live -= 1

        touched = op.qubits
        t = max((qtime.get(q, 0) for q in touched), default=0) + 1
        for q in touched:
            qtime[q] = t
        t_tof = (max((qtime_tof.get(q, 0) for q in touched), default=0)
                 + (1 if k == ops.CCX else 0))
        for q in touched:
            qtime_tof[q] = t_tof

    return Resources(
        toffoli=toffoli,
        clifford=clifford,
        measurements=measurements,
        peak_qubits=peak,
        depth=max(qtime.values(), default=0),
        n_gates=n_gates,
        toffoli_depth=max(qtime_tof.values(), default=0),
        volume=volume,
        toffoli_volume=toffoli_volume,
        circuit_volume=peak * toffoli,
    )
