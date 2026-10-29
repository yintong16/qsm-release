"""Incremental, actual-op-stream audits for a persistent Circuit allocator.

The caller owns emission and flushing: consume ``circ.ops`` before clearing it,
``circ.exact_blocks`` and ``circ._exact_block_seals``. Neither accumulator changes
that circuit or executes semantic sidecars. Chunk boundaries have no semantics:
wire scheduling history, live width, basis state, phase and RNG all persist.

Resources use precisely core.resources.count's conventions, including retained
ASAP dependencies when a physical wire ID is recycled and exclusion of classical
controller latency. H is one Clifford. FEEDBACK_PHASE is one *unsynthesized*
arbitrary rotation, not a Toffoli; its synthesis cost is deliberately unpriced.

BasisAccumulator is only an arithmetic computational-basis simulator. Explicit
load/discard operations support labelled deferred-oracle diagnostics, NOT a
simulation of a recycled-control quantum Fourier transform or of Shor sampling.
It rejects H and FEEDBACK_PHASE, even under a false classical condition.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import random

from qsm.core import ops
from qsm.core.circuit import DirtyAncillaError, Register
from qsm.core.resources import Resources
from qsm.core.simulator import SimResult

H = "H"
FEEDBACK_PHASE = "FEEDBACK_PHASE"


@dataclass(frozen=True, slots=True)
class FeedbackPhaseOp:
    """One feed-forward phase gate; sources are ordered classical history IDs.

    The scheduler specifies the rotation convention through its ordered sources
    and denominator power. This record counts one gate without attempting a
    Clifford+T decomposition or interpreting that angle on a basis trajectory.
    Classical dependencies add no latency in the existing resource model.
    """

    qubits: tuple[int, ...]
    sources: tuple[int, ...]
    denominator_power: int
    kind: str = field(default=FEEDBACK_PHASE, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, tuple) or not isinstance(self.sources, tuple):
            raise TypeError("qubits and sources must be immutable tuples")
        if len(self.qubits) != 1:
            raise ValueError("a feedback phase gate has exactly one target")
        for value in self.qubits + self.sources:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("wire and classical IDs must be nonnegative integers")
        if (not isinstance(self.denominator_power, int)
                or isinstance(self.denominator_power, bool)
                or self.denominator_power < 1):
            raise ValueError("denominator_power must be a positive integer")


StreamOp = ops.Op | ops.XorWordOp | ops.ClassicalParityOp | FeedbackPhaseOp
View = Register | Iterable[int]


class _StreamDigest:
    """Versioned representation hash, not an expanded-gate equivalence hash.

    Every dataclass field is covered, including bookkeeping, conditions and
    compact payloads. A bounded identity cache hashes shared immutable tuples
    only once while resident, without retaining the full operation stream.
    Python hash/randomized repr of objects is never used.
    """

    def __init__(self) -> None:
        self._hash = hashlib.sha256(b"qsm-streamed-ir-v1\n")
        self._tuples: OrderedDict[int, tuple[tuple, str]] = OrderedDict()
        self._fields: dict[type, tuple[str, ...]] = {}

    def update(self, op: StreamOp) -> None:
        cls = type(op)
        names = self._fields.get(cls)
        if names is None:
            names = tuple(f.name for f in fields(op))
            self._fields[cls] = names
        values = []
        for name in names:
            value = getattr(op, name)
            if isinstance(value, tuple):
                key = id(value)
                cached = self._tuples.get(key)
                if cached is not None and cached[0] is value:
                    digest = cached[1]
                    self._tuples.move_to_end(key)
                else:
                    # IR tuple fields contain only integers and nested tuples of
                    # integers, whose repr is deterministic across processes.
                    digest = hashlib.sha256(repr(value).encode("ascii")).hexdigest()
                    self._tuples[key] = (value, digest)
                    if len(self._tuples) > 128:
                        self._tuples.popitem(last=False)
                value = {"tuple_sha256": digest}
            values.append((name, value))
        self._hash.update(json.dumps(
            [cls.__name__, values], ensure_ascii=True, separators=(",", ":"),
        ).encode("ascii") + b"\n")

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


class ResourceAccumulator:
    """Exact persistent physical counts; consume each emitted node once.

    ``resources`` returns the unchanged core Resources type; ``snapshot()`` is
    JSON-ready and additionally reports H/rotation/node counts and live width.
    Optional digesting covers actual IR nodes (including ALLOC/FREE/CPARITY),
    not semantic blocks, chunk boundaries or the selected measurement shot.
    ``track_depth=False`` skips ASAP scheduling (in particular per-target compact
    fanout expansion), preserving exact gate/lifetime/volume counts. It reports
    depth=None, toffoli_depth=None and depth_computed=False rather than zero;
    the core Resources property is then unavailable. Choose this once at init.
    """

    def __init__(self, *, digest: bool = False, track_depth: bool = True) -> None:
        self.track_depth = track_depth
        self.toffoli = self.clifford = self.measurements = self.n_gates = 0
        self.live_qubits = self.peak_qubits = 0
        self.volume = self.toffoli_volume = 0
        self.hadamards = self.phase_rotations = self.op_nodes = 0
        self._qtime: dict[int, int] = {}
        self._qtime_tof: dict[int, int] = {}
        self._digest = _StreamDigest() if digest else None

    def consume(self, operations: Iterable[StreamOp]) -> ResourceAccumulator:
        for op in operations:
            k = op.kind
            if k not in (ops.ALLOC, ops.FREE, ops.CPARITY, ops.XOR_WORD,
                         ops.X, ops.CX, ops.CCX, ops.CZ, ops.Z, ops.MEASURE,
                         H, FEEDBACK_PHASE):
                raise ValueError(f"unknown op kind {k!r}")
            if self._digest is not None:
                self._digest.update(op)
            self.op_nodes += 1
            if k == ops.ALLOC:
                self.live_qubits += len(op.qubits)
                self.peak_qubits = max(self.peak_qubits, self.live_qubits)
                continue
            if k == ops.FREE:
                self.live_qubits -= len(op.qubits)
                continue
            if k == ops.CPARITY:
                continue
            if k == ops.XOR_WORD:
                n = op.mask.bit_count()
                self.clifford += n
                self.n_gates += n
                self.volume += self.live_qubits * n
                if not self.track_depth:
                    continue
                mask, ctrl = op.mask, op.ctrl
                while mask:
                    bit = mask & -mask
                    target = op.qubits[bit.bit_length() - 1]
                    if ctrl is None:
                        self._qtime[target] = self._qtime.get(target, 0) + 1
                    else:
                        t = max(self._qtime.get(ctrl, 0),
                                self._qtime.get(target, 0)) + 1
                        self._qtime[ctrl] = self._qtime[target] = t
                        t_tof = max(self._qtime_tof.get(ctrl, 0),
                                    self._qtime_tof.get(target, 0))
                        self._qtime_tof[ctrl] = self._qtime_tof[target] = t_tof
                    mask ^= bit
                continue

            self.n_gates += 1
            if k == ops.CCX:
                self.toffoli += 1
                self.toffoli_volume += self.live_qubits
            elif k in (ops.X, ops.CX, ops.CZ, ops.Z, H):
                self.clifford += 1
                self.hadamards += int(k == H)
            elif k == ops.MEASURE:
                self.measurements += 1
            elif k == FEEDBACK_PHASE:
                self.phase_rotations += 1

            self.volume += self.live_qubits
            if k == ops.MEASURE:
                self.live_qubits -= 1
            if not self.track_depth:
                continue
            t = max((self._qtime.get(q, 0) for q in op.qubits), default=0) + 1
            t_tof = (max((self._qtime_tof.get(q, 0) for q in op.qubits), default=0)
                     + int(k == ops.CCX))
            for q in op.qubits:
                self._qtime[q] = t
                self._qtime_tof[q] = t_tof
        return self

    @property
    def resources(self) -> Resources:
        if not self.track_depth:
            raise ValueError("depth was not computed; use snapshot(), not core Resources")
        return Resources(
            toffoli=self.toffoli, clifford=self.clifford,
            measurements=self.measurements, peak_qubits=self.peak_qubits,
            depth=max(self._qtime.values(), default=0), n_gates=self.n_gates,
            toffoli_depth=max(self._qtime_tof.values(), default=0),
            volume=self.volume, toffoli_volume=self.toffoli_volume,
            circuit_volume=self.peak_qubits * self.toffoli,
        )

    @property
    def digest(self) -> str | None:
        return None if self._digest is None else self._digest.hexdigest()

    def snapshot(self) -> dict[str, int | str | None]:
        metrics = asdict(self.resources) if self.track_depth else {
            "toffoli": self.toffoli, "clifford": self.clifford,
            "measurements": self.measurements, "peak_qubits": self.peak_qubits,
            "depth": None, "n_gates": self.n_gates, "toffoli_depth": None,
            "volume": self.volume, "toffoli_volume": self.toffoli_volume,
            "circuit_volume": self.peak_qubits * self.toffoli,
        }
        return {
            **metrics, "depth_computed": self.track_depth, "hadamards": self.hadamards,
            "phase_rotations": self.phase_rotations,
            "phase_gates": self.phase_rotations, "live_qubits": self.live_qubits,
            "op_nodes": self.op_nodes, "digest": self.digest,
            "stream_digest": self.digest,
        }


class BasisAccumulator:
    """Persistent arithmetic-only basis execution, matching core.simulator.

    Input values are loaded when their named ALLOC arrives. State grows with new
    physical IDs, but ALLOC always zeroes/reloads recycled IDs. Supply updated
    constants per consume when a persistent Circuit creates more classical bits.
    Snapshot/output mappings can use ``circ.inputs`` to respect register rebinds.

    ``record_measurements=False`` bounds measurement-history memory; cbits remain
    available until explicitly cleared at a completed stage. In that mode the
    SimResult measurement list is intentionally empty, not a complete transcript.
    """

    def __init__(
        self, inputs: Mapping[str, int] | None = None,
        measurements: Iterable[int] | None = None,
        rng: random.Random | None = None, *,
        constants: Mapping[int, int] | None = None,
        record_measurements: bool = True,
    ) -> None:
        self.inputs = dict(inputs or {})
        self.state = bytearray()
        self.cbits: dict[int, int] = {}
        self.phase = 0
        self.measurements: list[int] = []
        self.executed_measurements = 0
        self.rng = random.Random(0) if rng is None else rng
        self._measurement_input = None if measurements is None else iter(measurements)
        self._record_measurements = record_measurements
        self._live: set[int] = set()
        self._input_views: dict[str, tuple[int, ...]] = {}
        self.consume_constants(constants or {})

    def consume_constants(self, constants: Mapping[int, int]) -> None:
        self.cbits.update((cb, int(value) & 1) for cb, value in constants.items())

    def consume(
        self, operations: Iterable[StreamOp],
        constants: Mapping[int, int] | None = None, *,
        inputs: Mapping[str, int] | None = None,
    ) -> BasisAccumulator:
        if constants is not None:
            self.consume_constants(constants)
        if inputs is not None:
            self.inputs.update(inputs)
        for op in operations:
            k = op.kind
            if k not in (ops.ALLOC, ops.FREE, ops.CPARITY, ops.XOR_WORD,
                         ops.X, ops.CX, ops.CCX, ops.CZ, ops.Z, ops.MEASURE):
                raise ValueError(f"arithmetic basis simulator does not support {k!r}")
            if k == ops.ALLOC:
                value = int(self.inputs[op.name]) if op.is_input else 0
                needed = max(op.qubits, default=-1) + 1
                if needed > len(self.state):
                    self.state.extend(bytes(needed - len(self.state)))
                for i, q in enumerate(op.qubits):
                    self.state[q] = (value >> i) & 1
                self._live.update(op.qubits)
                if op.is_input:
                    self._input_views[op.name] = op.qubits
                continue
            if k == ops.FREE:
                for q in op.qubits:
                    if self.state[q] != 0:
                        raise DirtyAncillaError(f"qubit {q} freed while holding |1>")
                self._live.difference_update(op.qubits)
                continue
            # Like Circuit.measure, retirement is unconditional even when a
            # classically disabled measurement does not execute on this shot.
            if k == ops.MEASURE:
                self._live.difference_update(op.qubits)
            if op.cond and not all(self.cbits.get(cb, 0) == val for cb, val in op.cond):
                if k == ops.MEASURE:
                    self.cbits[op.cbit] = 0
                continue

            if k == ops.CPARITY:
                packed = 0
                for i, cb in enumerate(op.sources):
                    packed |= self.cbits[cb] << i
                for j, word in enumerate(op.values):
                    self.cbits[op.base + j] = (packed & word).bit_count() & 1
            elif k == ops.XOR_WORD:
                if op.ctrl is not None and not self.state[op.ctrl]:
                    continue
                mask = op.mask
                while mask:
                    bit = mask & -mask
                    self.state[op.qubits[bit.bit_length() - 1]] ^= 1
                    mask ^= bit
            elif k == ops.X:
                self.state[op.qubits[0]] ^= 1
            elif k == ops.CX:
                c, t = op.qubits
                self.state[t] ^= self.state[c]
            elif k == ops.CCX:
                a, b, t = op.qubits
                self.state[t] ^= self.state[a] & self.state[b]
            elif k == ops.CZ:
                a, b = op.qubits
                self.phase ^= self.state[a] & self.state[b]
            elif k == ops.Z:
                self.phase ^= self.state[op.qubits[0]]
            elif k == ops.MEASURE:
                if self._measurement_input is None:
                    m = self.rng.getrandbits(1)
                else:
                    try:
                        m = next(self._measurement_input) & 1
                    except StopIteration:
                        raise IndexError("measurement outcomes exhausted") from None
                if self._record_measurements:
                    self.measurements.append(m)
                self.executed_measurements += 1
                self.cbits[op.cbit] = m
                self.phase ^= self.state[op.qubits[0]] & m
                self.state[op.qubits[0]] = 0
        return self

    def _live_view(self, view: View) -> tuple[int, ...]:
        ids = tuple(view)
        if any(q not in self._live for q in ids):
            raise ValueError("diagnostic view contains a qubit that is not live")
        return ids

    def read_view(self, view: View) -> int:
        return sum(self.state[q] << i for i, q in enumerate(self._live_view(view)))

    def write_view(self, view: View, value: int) -> None:
        """Explicit external diagnostic preparation, never an emitted gate."""
        ids = self._live_view(view)
        if not isinstance(value, int) or value < 0 or value >> len(ids):
            raise ValueError("diagnostic value must be nonnegative and fit the view")
        for i, q in enumerate(ids):
            self.state[q] = (value >> i) & 1

    def discard_view(self, view: View) -> None:
        """Retire a skipped-QFT diagnostic view without claiming quantum erasure.

        This deliberately permits nonzero bits. It changes no phase, measurement
        cursor, RNG or resource count, and is NOT a physical uncomputation.
        """
        ids = self._live_view(view)
        for q in ids:
            self.state[q] = 0
        self._live.difference_update(ids)

    def outputs(self, registers: Mapping[str, View] | None = None) -> dict[str, int]:
        # Match simulate's final named-wire readout, including consumed inputs.
        views = self._input_views if registers is None else registers
        return {name: sum(self.state[q] << i for i, q in enumerate(view))
                for name, view in views.items()}

    def snapshot(self, registers: Mapping[str, View] | None = None) -> SimResult:
        return SimResult(self.outputs(registers), self.phase & 1, list(self.measurements))

    def clear_classical(self, constants: Mapping[int, int] | None = None) -> None:
        """Drop completed-stage cbits, preserving only explicitly supplied constants.

        Call only once no future operation needs those classical results. This
        does not reset phase, measurement history/cursor, RNG or quantum state.
        """
        keep = {} if constants is None else dict(constants)
        self.cbits.clear()
        self.consume_constants(keep)

    # Short spellings used by the stage-at-a-time scalar scheduler.
    read = read_view
    load = write_view
    discard = discard_view
    reset_classical = clear_classical
