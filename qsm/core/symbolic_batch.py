"""Input-batched, all-measurement-branches hazard evaluation of emitted IR.

Each Python integer stores one Boolean value across input lanes. Measurement
outcomes are *symbols*, not samples: the residual phase is c XOR sum(h_m * m).
For the accepted value-independent, affine-in-measurements IR, an input is phase
bad iff c or any h_m is nonzero. A symbol is retired only at its last actual use.

Only validated exact lookup summaries are marginalized. An unlookup whose
restored-target precondition fails is conservatively bad, never certified by an
ideal arithmetic shadow. All masks/counts concern INPUTS, not measurement shots.

Compilation snapshots a tuple of references to immutable raw operations, without
copying millions of operations into execution records. Later source-circuit edits
do not change an existing plan; compile again to analyze the edited circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np

from . import ops
from .circuit import Circuit

if TYPE_CHECKING:
    from .semantics import ExactLookupBlock


@dataclass(frozen=True, slots=True)
class SymbolicPlan:
    """Validated execution plan; ``stats`` contains small integer diagnostics.

    ``input_registers`` describes original ALLOC records; ``output_registers``
    describes final named views, which may differ after wire relabelling.
    ``raw_ops`` is a tuple of original immutable Op references, not copied Ops.
    ``ir_digest`` hashes the versioned analyzed IR (exact summaries and raw
    outside-block operations), constants and named register views.
    """

    raw_ops: tuple
    blocks: tuple[ExactLookupBlock, ...]
    input_registers: tuple[tuple[str, tuple[int, ...]], ...]
    output_registers: tuple[tuple[str, tuple[int, ...]], ...]
    classical_constants: Mapping[int, int]
    classical_last_use: Mapping[int, int]
    final_ancillas: tuple[int, ...]
    stats: Mapping[str, int]
    ir_digest: str


@dataclass(frozen=True, slots=True)
class BatchHazards:
    """Lane i is bit i of each mask; masks may overlap, bad_mask is their union.

    ``phase_mask`` includes any symbolic coefficient AND the final constant
    phase. ``constant_phase_mask`` isolates the latter. ``dirty_mask`` includes
    dirty FREEs and nonzero final live wires outside named output views.
    ``summary_mismatch_mask`` marks failed measured-unlookup preconditions.
    ``outputs`` always contains raw (not reduced modulo p) named output words.
    """

    samples: int
    bad_mask: int
    value_mask: int
    noncanonical_mask: int
    dirty_mask: int
    phase_mask: int
    constant_phase_mask: int
    summary_mismatch_mask: int
    outputs: list[dict[str, int]]

    @property
    def bad_count(self) -> int:
        return self.bad_mask.bit_count()

    @property
    def good_count(self) -> int:
        return self.samples - self.bad_count

    @property
    def value_count(self) -> int:
        return self.value_mask.bit_count()

    @property
    def noncanonical_count(self) -> int:
        return self.noncanonical_mask.bit_count()

    @property
    def dirty_count(self) -> int:
        return self.dirty_mask.bit_count()

    @property
    def phase_count(self) -> int:
        return self.phase_mask.bit_count()

    @property
    def constant_phase_count(self) -> int:
        return self.constant_phase_mask.bit_count()

    @property
    def summary_mismatch_count(self) -> int:
        return self.summary_mismatch_mask.bit_count()


_TRUE = (-1, 1)
_FALSE = (-1, 0)
_ARITY = {ops.X: 1, ops.CX: 2, ops.CCX: 3, ops.Z: 1, ops.CZ: 2, ops.MEASURE: 1}


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _condition(cond, constants: Mapping[int, int]) -> tuple[int, int]:
    """Return (-1, truth) or (measurement symbol, desired value).

    Unknown producers are checked by compilation, before this reduction. Even a
    contradictory condition must not smuggle in an unknown classical producer.
    """
    if not cond:
        return _TRUE
    if len(cond) == 1:
        cb, wanted = cond[0]
        if cb in constants:
            return _TRUE if constants[cb] == wanted else _FALSE
        return cb, wanted
    symbols = {}
    false = False
    for cb, wanted in cond:
        if cb in constants:
            false |= constants[cb] != wanted
        elif cb in symbols:
            false |= symbols[cb] != wanted
        else:
            symbols[cb] = wanted
    if false:
        return _FALSE
    if len(symbols) > 1:
        raise ValueError("nonlinear phase condition: multiple unresolved measurement symbols")
    return next(iter(symbols.items())) if symbols else _TRUE


def _wires(qubits, limit: int) -> None:
    if not isinstance(qubits, tuple):
        raise ValueError("qubit words must be immutable tuples")
    if any(not _integer(q) or not 0 <= q < limit for q in qubits):
        raise ValueError("invalid qubit id")
    if len(set(qubits)) != len(qubits):
        raise ValueError("qubit ids must be distinct")


def compile_symbolic(circ: Circuit) -> SymbolicPlan:
    """Preflight an emitted circuit, rejecting unsupported or malformed IR.

    Supported external primitives are ALLOC/FREE, X/CX/CCX, XOR_WORD, Z/CZ and
    unconditional H-basis MEASURE. Classical constants are solved. Remaining
    conditions may contain one measurement symbol and may only control phases.
    CPARITY is supported only *inside* a validated exact lookup block.
    """
    from .semantics import validate_exact_blocks

    blocks = validate_exact_blocks(circ)
    limit = circ._next_qid
    if not _integer(limit) or limit < 0:
        raise ValueError("invalid qubit-id bound")
    cbit_limit = circ._next_cbit
    if not _integer(cbit_limit) or cbit_limit < 0:
        raise ValueError("invalid classical-id bound")
    constants = dict(circ._cbit_const)
    if any(not _integer(cb) or not 0 <= cb < cbit_limit
           or not _integer(v) or v not in (0, 1) for cb, v in constants.items()):
        raise ValueError("invalid classical constant")
    live: set[int] = set()
    inputs: dict[str, tuple[int, ...]] = {}
    last_use: dict[int, int] = {}
    i = bi = 0
    raw = tuple(circ.ops)
    digest = sha256(b"qsm-symbolic-ir-v1\n")
    digest.update(repr(tuple(sorted(constants.items()))).encode("utf-8") + b"\n")
    while i <= len(raw):
        if bi < len(blocks) and blocks[bi].start == i:
            block = blocks[bi]
            if live != block.entry_live:
                raise ValueError("exact block entry lifetime mismatch")
            table_digest = sha256(repr(block.values).encode("ascii")).hexdigest()
            record = ("exact", block.kind, block.semantic_version, block.start, block.stop,
                      block.address, block.target, table_digest, block.cbit_start,
                      block.cbit_stop, tuple(sorted(block.entry_live)), tuple(sorted(block.exit_live)))
            digest.update(repr(record).encode("ascii") + b"\n")
            live = set(block.exit_live)
            i = block.stop
            bi += 1
            continue
        if i == len(raw):
            break
        op = raw[i]
        k = getattr(op, "kind", None)
        if type(op) not in (ops.Op, ops.XorWordOp):
            raise ValueError(f"unsupported op outside exact block: {k!r}")
        if k not in _ARITY and k not in (ops.ALLOC, ops.FREE, ops.XOR_WORD):
            raise ValueError(f"unsupported op: {k!r}")
        if (k == ops.XOR_WORD) != (type(op) is ops.XorWordOp):
            raise ValueError("op record does not match its kind")
        _wires(op.qubits, limit)
        if k in _ARITY and len(op.qubits) != _ARITY[k]:
            raise ValueError(f"invalid arity for {k}")
        if not isinstance(op.cond, tuple):
            raise ValueError("conditions must be immutable tuples")
        if k in (ops.ALLOC, ops.FREE, ops.MEASURE) and op.cond:
            raise ValueError(f"conditional {k} is unsupported")
        if type(op) is ops.Op and k != ops.MEASURE and op.cbit != -1:
            raise ValueError("unexpected classical producer on non-measurement op")
        digest.update(repr(op).encode("utf-8") + b"\n")
        if k == ops.ALLOC:
            if live.intersection(op.qubits):
                raise ValueError("allocating an already-live qubit")
            live.update(op.qubits)
            if op.is_input:
                if not isinstance(op.name, str) or op.name in inputs:
                    raise ValueError("invalid or duplicate input ALLOC name")
                inputs[op.name] = op.qubits
            i += 1
            continue
        if not set(op.qubits).issubset(live):
            raise ValueError(f"{k} references a non-live qubit")
        if k == ops.FREE:
            live.difference_update(op.qubits)
            i += 1
            continue
        if k == ops.XOR_WORD:
            if not _integer(op.mask) or op.mask < 0 or op.mask >> len(op.qubits):
                raise ValueError("XOR_WORD mask must fit its target word")
            if op.ctrl is not None:
                if not _integer(op.ctrl) or op.ctrl not in live or op.ctrl in op.qubits:
                    raise ValueError("invalid XOR_WORD control")
        if not isinstance(op.cond, tuple):
            raise ValueError("conditions must be immutable tuples")
        for term in op.cond:
            if not isinstance(term, tuple) or len(term) != 2:
                raise ValueError("invalid classical condition")
            cb, wanted = term
            if not _integer(cb) or not _integer(wanted) or wanted not in (0, 1):
                raise ValueError("invalid classical condition")
            if cb not in constants and cb not in last_use:
                raise ValueError(f"unknown classical producer {cb}")
        symbol, wanted = _condition(op.cond, constants)
        if symbol >= 0:
            if k not in (ops.Z, ops.CZ):
                raise ValueError("measurement-controlled value gate is unsupported")
            last_use[symbol] = i
        if k == ops.MEASURE:
            cb = op.cbit
            if not _integer(cb) or not 0 <= cb < cbit_limit:
                raise ValueError("invalid measurement classical id")
            if cb in constants or cb in last_use:
                raise ValueError("duplicate classical producer")
            last_use[cb] = i
            live.remove(op.qubits[0])
        i += 1
    if bi != len(blocks):
        raise ValueError("unreachable exact block")
    if set(inputs) != set(circ.inputs):
        raise ValueError("final named output keys must match original input ALLOC names")
    outputs = []
    output_ids: set[int] = set()
    for name, reg in circ.inputs.items():
        ids = tuple(reg.ids)
        _wires(ids, limit)
        if not set(ids).issubset(live):
            raise ValueError(f"final output {name!r} references non-live qubits")
        if output_ids.intersection(ids):
            raise ValueError("named output register views must be disjoint")
        outputs.append((name, ids))
        output_ids.update(ids)
    digest.update(repr((tuple(inputs.items()), tuple(outputs))).encode("utf-8") + b"\n")
    return SymbolicPlan(
        raw, blocks, tuple(inputs.items()), tuple(outputs), MappingProxyType(constants),
        MappingProxyType(last_use), tuple(sorted(live - output_ids)),
        MappingProxyType({"raw_ops": len(raw), "exact_blocks": len(blocks),
                          "symbolic_measurements": len(last_use), "qubits": limit,
                          "classical_bits": cbit_limit}),
        digest.hexdigest(),
    )


def validate_plan_binding(plan: SymbolicPlan, circ: Circuit) -> None:
    """Require the current circuit to be exactly the already-analyzed program.

    Use before AND after benchmarking/calibration when attaching a plan's digest
    to a result. This checks raw operation identities in one allocation-free
    scan, not just their equality, and checks the small mutable sidecars/views.
    Original input ALLOC metadata is covered by the immutable Op identities;
    final named views are checked separately because rebinding emits no Op.

    No operations are recopied, no circuit is recompiled, and no case metadata
    is trusted. A source circuit may be edited after compilation for independent
    work, but it can no longer be bound to that plan. This is an integrity check,
    not protection against concurrent edits that are reverted between checks.
    """
    if len(circ.ops) != len(plan.raw_ops):
        raise ValueError("symbolic plan binding mismatch: raw operation count changed")
    if any(current is not saved for current, saved in zip(circ.ops, plan.raw_ops)):
        raise ValueError("symbolic plan binding mismatch: raw operation identity changed")
    if (not _integer(circ._next_qid) or circ._next_qid != plan.stats["qubits"]
            or not _integer(circ._next_cbit)
            or circ._next_cbit != plan.stats["classical_bits"]):
        raise ValueError("symbolic plan binding mismatch: qubit/classical id bounds changed")
    if (circ._cbit_const != plan.classical_constants
            or any(not _integer(cb) or not _integer(value)
                   for cb, value in circ._cbit_const.items())):
        raise ValueError("symbolic plan binding mismatch: classical constants changed")
    views = tuple((name, tuple(reg.ids)) for name, reg in circ.inputs.items())
    if views != plan.output_registers:
        raise ValueError("symbolic plan binding mismatch: final output views changed")
    for _, ids in views:
        _wires(ids, circ._next_qid)
    if (len(circ.exact_blocks) != len(plan.blocks)
            or {id(block) for block in circ.exact_blocks} != {id(block) for block in plan.blocks}):
        raise ValueError("symbolic plan binding mismatch: exact block identities changed")


def _pack_words(words: list[int], width: int) -> list[int]:
    """Transpose little-endian words to Python-int lanes (padding stays zero)."""
    if not width:
        return []
    if not words:
        return [0] * width
    nbytes = (width + 7) // 8
    data = np.frombuffer(b"".join(v.to_bytes(nbytes, "little") for v in words), dtype=np.uint8)
    bits = np.unpackbits(data.reshape(len(words), nbytes), axis=1, bitorder="little")[:, :width]
    lanes = np.packbits(bits, axis=0, bitorder="little")
    return [int.from_bytes(lanes[:, bit].tobytes(), "little") for bit in range(width)]


def _unpack_words(state: list[int], ids: tuple[int, ...], samples: int) -> list[int]:
    if not ids or not samples:
        return [0] * samples
    nbytes = (samples + 7) // 8
    data = np.frombuffer(b"".join(state[q].to_bytes(nbytes, "little") for q in ids), dtype=np.uint8)
    bits = np.unpackbits(data.reshape(len(ids), nbytes), axis=1, count=samples, bitorder="little")
    words = np.packbits(bits.T, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in words]


def _validate_samples(plan, inputs, expected, moduli) -> None:
    if not isinstance(inputs, list) or not isinstance(expected, list):
        raise TypeError("inputs and expected must be lists of dictionaries")
    if len(inputs) != len(expected):
        raise ValueError("inputs and expected must have equal sample counts")
    in_widths = {name: len(ids) for name, ids in plan.input_registers}
    out_widths = {name: len(ids) for name, ids in plan.output_registers}
    if not isinstance(moduli, dict) or not set(moduli).issubset(out_widths):
        raise ValueError("output_moduli must name existing output registers")
    for name, p in moduli.items():
        if not _integer(p) or not 1 <= p <= 1 << out_widths[name]:
            raise ValueError(f"invalid modulus for output {name!r}")
    for rows, widths, label in ((inputs, in_widths, "input"), (expected, out_widths, "expected")):
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(widths):
                raise ValueError(f"{label} keys must exactly match named registers")
            for name, width in widths.items():
                value = row[name]
                bound = moduli.get(name, 1 << width) if label == "expected" else 1 << width
                if not _integer(value) or not 0 <= value < bound:
                    raise ValueError(f"{label} {name!r} must be an unsigned canonical word < {bound}")


def simulate_batch(
    plan: SymbolicPlan,
    inputs: list[dict[str, int]],
    expected: list[dict[str, int]],
    *,
    output_moduli: dict[str, int] | None = None,
) -> BatchHazards:
    """Evaluate concrete input lanes with symbolic, RNG-free phase tracking.

    Expected values must be canonical. With a modulus, value_mask compares
    ``raw % p`` to expected, while noncanonical_mask independently marks raw>=p.
    Without a modulus, exact raw-word equality is required. An early dirty FREE
    permanently marks its input but does not abort other lanes.
    """
    moduli = {} if output_moduli is None else output_moduli
    _validate_samples(plan, inputs, expected, moduli)
    samples = len(inputs)
    lane_mask = (1 << samples) - 1
    state = [0] * plan.stats["qubits"]
    loaded = {name: _pack_words([row[name] for row in inputs], len(ids))
              for name, ids in plan.input_registers}
    coeff: dict[int, int] = {}
    dirty = phase = constant = mismatch = 0
    constants = plan.classical_constants
    last_use = plan.classical_last_use
    raw = plan.raw_ops
    blocks = plan.blocks
    i = bi = 0
    while i <= len(raw):
        if bi < len(blocks) and blocks[bi].start == i:
            block = blocks[bi]
            addresses = _unpack_words(state, block.address, samples)
            table_words = [block.values[address] for address in addresses]
            table_lanes = _pack_words(table_words, len(block.target))
            if block.kind == "xor":
                for q, value in zip(block.target, table_lanes):
                    state[q] ^= value
            else:  # validated measured-unlookup, not an arbitrary inverse
                for q, value in zip(block.target, table_lanes):
                    mismatch |= state[q] ^ value
                    state[q] = 0
            i = block.stop
            bi += 1
            continue
        if i == len(raw):
            break
        op = raw[i]
        k = op.kind
        qs = op.qubits
        if k == ops.ALLOC:
            if op.is_input:
                for q, value in zip(qs, loaded[op.name]):
                    state[q] = value
            else:
                for q in qs:
                    state[q] = 0
        elif k == ops.FREE:
            for q in qs:
                dirty |= state[q]
        else:
            symbol, wanted = _condition(op.cond, constants)
            if symbol == -1 and not wanted:
                i += 1
                continue
            if k == ops.X:
                state[qs[0]] ^= lane_mask
            elif k == ops.CX:
                state[qs[1]] ^= state[qs[0]]
            elif k == ops.CCX:
                state[qs[2]] ^= state[qs[0]] & state[qs[1]]
            elif k == ops.XOR_WORD:
                value = lane_mask if op.ctrl is None else state[op.ctrl]
                mask = op.mask
                while mask:
                    low = mask & -mask
                    state[qs[low.bit_length() - 1]] ^= value
                    mask ^= low
            elif k == ops.MEASURE:
                value = state[qs[0]]
                state[qs[0]] = 0
                if last_use[op.cbit] == i:
                    phase |= value
                else:
                    coeff[op.cbit] = value
            else:  # Z/CZ; preflight excludes every other kind
                value = state[qs[0]]
                if k == ops.CZ:
                    value &= state[qs[1]]
                if symbol == -1:
                    constant ^= value
                else:
                    coeff[symbol] ^= value
                    if wanted == 0:
                        constant ^= value
                    if last_use[symbol] == i:
                        phase |= coeff.pop(symbol)
        i += 1
    for q in plan.final_ancillas:
        dirty |= state[q]
    phase |= constant
    outputs: list[dict[str, int]] = [{} for _ in inputs]
    value_mask = noncanonical = 0
    for name, ids in plan.output_registers:
        words = _unpack_words(state, ids, samples)
        p = moduli.get(name)
        for lane, word in enumerate(words):
            outputs[lane][name] = word
            if p is not None:
                if word >= p:
                    noncanonical |= 1 << lane
                word %= p
            if word != expected[lane][name]:
                value_mask |= 1 << lane
    bad = value_mask | noncanonical | dirty | phase | mismatch
    return BatchHazards(samples, bad & lane_mask, value_mask & lane_mask,
                        noncanonical & lane_mask, dirty & lane_mask, phase & lane_mask,
                        constant & lane_mask, mismatch & lane_mask, outputs)
