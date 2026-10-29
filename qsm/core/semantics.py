"""Trusted exact lookup summaries, kept entirely outside the physical circuit IR.

The lookup builders' algebra and exhaustive measurement-branch regressions justify
these summaries; this is not a proof assistant or a hostile-Python security
boundary. A circuit-local seal binds immutable metadata and operation *identities*
to the builder that captured them. Replacing a descriptor or an emitted operation
invalidates it, even if the replacement compares equal. Operation snapshots hold
only tuple references, never expanded gates, serialized payloads, or a Circuit.

Validation scans the full emitted stream: a private classical result must neither
escape nor be overwritten elsewhere. Qubit closure follows allocation lifetimes,
not numeric IDs (measurement and subsequent allocation routinely reuse IDs).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from weakref import ref

from . import ops

if TYPE_CHECKING:
    from .circuit import Circuit

EXACT_LOOKUP_VERSION = "exact-lookup-v1"
Operation = ops.Op | ops.XorWordOp | ops.ClassicalParityOp


@dataclass(frozen=True, slots=True)
class ExactLookupBlock:
    """Half-open physical-op and private-cbit ranges of an exact lookup channel.

    ``xor`` maps target to target XOR values[address], for any initial target.
    ``unlookup`` consumes target and is phase-clean for every outcome only when
    target equals values[address]. Evaluators must check that precondition rather
    than silently clearing a dirty word. The latter's target IDs can subsequently
    be reused, even inside the same block, for distinct temporary lifetimes.
    """

    kind: Literal["xor", "unlookup"]
    start: int
    stop: int
    address: tuple[int, ...]
    target: tuple[int, ...]
    values: tuple[int, ...] = field(repr=False)
    semantic_version: str
    cbit_start: int
    cbit_stop: int
    entry_live: frozenset[int]
    exit_live: frozenset[int]
    op_refs: tuple[Operation, ...] = field(repr=False)
    _seal: object | None = field(default=None, repr=False, compare=False)


def _metadata(block: ExactLookupBlock) -> tuple:
    return (
        block.kind, block.start, block.stop, block.address, block.target,
        block.values, block.semantic_version, block.cbit_start, block.cbit_stop,
        block.entry_live, block.exit_live, block.op_refs,
    )


def _record_exact_lookup(
    circ: Circuit, *, kind: Literal["xor", "unlookup"], start: int,
    cbit_start: int, entry_live: frozenset[int], address: tuple[int, ...],
    target: tuple[int, ...], values: tuple[int, ...],
) -> None:
    """Internal trusted-builder hook; call only after helper validation/emission."""
    block = ExactLookupBlock(
        kind=kind, start=start, stop=len(circ.ops), address=address, target=target,
        values=values, semantic_version=EXACT_LOOKUP_VERSION,
        cbit_start=cbit_start, cbit_stop=circ._next_cbit,
        entry_live=entry_live, exit_live=frozenset(circ._live_qids),
        op_refs=tuple(circ.ops[i] for i in range(start, len(circ.ops))), _seal=object(),
    )
    circ._exact_block_seals[block._seal] = (ref(circ), block, _metadata(block))
    circ.exact_blocks.append(block)


def validate_exact_blocks(circ: Circuit) -> tuple[ExactLookupBlock, ...]:
    """Return sorted, sealed, closed blocks, or raise ``ValueError``.

    This includes full-IR preflight of all references/producers intersecting the
    blocks' private classical intervals, including CPARITY's reserved outputs.
    A compiler can therefore discard those intervals with the validated blocks.
    Other unsupported IR outside blocks is the compiler's responsibility. No
    final ``circ.inputs`` view is consulted: those output views may be rebound.
    """
    candidates = tuple(circ.exact_blocks)
    if not candidates:
        return ()
    seals = circ._exact_block_seals
    for block in candidates:
        if not isinstance(block, ExactLookupBlock):
            raise ValueError("invalid exact-block descriptor")
        if type(block._seal) is not object:
            raise ValueError("unsealed exact-block descriptor")
        sealed = seals.get(block._seal)
        if (sealed is None or sealed[0]() is not circ or sealed[1] is not block
                or sealed[2] != _metadata(block)):
            raise ValueError("unsealed, foreign, or stale exact-block metadata")
        if block.semantic_version != EXACT_LOOKUP_VERSION:
            raise ValueError("unsupported exact-block semantic version")
        if block.kind not in ("xor", "unlookup"):
            raise ValueError("unsupported exact-block kind")
        if not 0 <= block.start < block.stop <= len(circ.ops):
            raise ValueError("invalid exact-block operation range")
        if (len(block.op_refs) != block.stop - block.start
                or any(circ.ops[i] is not saved
                       for i, saved in enumerate(block.op_refs, block.start))):
            raise ValueError("stale exact-block operation identities")
        if not 0 <= block.cbit_start <= block.cbit_stop <= circ._next_cbit:
            raise ValueError("invalid exact-block classical range")
        wires = block.address + block.target
        if (not block.address or not block.target or len(set(wires)) != len(wires)
                or not set(wires) <= block.entry_live
                or len(block.values) != 1 << len(block.address)
                or any(type(v) is not int or not 0 <= v < 1 << len(block.target)
                       for v in block.values)):
            raise ValueError("invalid exact-block lookup metadata")
        expected_exit = (block.entry_live if block.kind == "xor" else
                         block.entry_live.difference(block.target))
        if block.exit_live != expected_exit:
            raise ValueError("invalid exact-block exit lifetime")

    blocks = tuple(sorted(candidates, key=lambda block: block.start))
    if any(a.stop > b.start for a, b in zip(blocks, blocks[1:])):
        raise ValueError("overlapping exact-block operation ranges")
    private = sorted((b for b in blocks if b.cbit_start < b.cbit_stop),
                     key=lambda b: b.cbit_start)
    if any(a.cbit_stop > b.cbit_start for a, b in zip(private, private[1:])):
        raise ValueError("overlapping exact-block classical ranges")
    starts = [b.cbit_start for b in private]
    stops = [b.cbit_stop for b in private]

    def private_owner(cb: int) -> ExactLookupBlock | None:
        index = bisect_right(starts, cb) - 1
        if index >= 0 and cb < stops[index]:
            return private[index]
        return None

    for cb in circ._cbit_const:
        if private_owner(cb) is not None:
            raise ValueError("global classical constant overlaps private exact-block range")

    live: set[int] = set()
    local: set[int] = set()
    consumed_target: set[int] = set()
    allowed: set[int] = set()
    block_index = 0
    active = None
    next_cbit = 0
    for position, op in enumerate(circ.ops):
        if block_index < len(blocks) and position == blocks[block_index].start:
            active = blocks[block_index]
            if live != active.entry_live:
                raise ValueError("exact-block entry live set disagrees with emitted IR")
            local = set()
            consumed_target = set()
            allowed = set(active.address) | set(active.target)
            next_cbit = active.cbit_start

        # Conditions and parity sources are classical, not quantum controls.
        reads = (cb for cb, _ in op.cond)
        for cb in reads:
            if active is not None:
                if not active.cbit_start <= cb < next_cbit:
                    raise ValueError("exact block reads an external or unproduced classical bit")
            elif private_owner(cb) is not None:
                raise ValueError("private exact-block classical bit escapes in condition")
        if op.kind == ops.CPARITY:
            for cb in op.sources:
                if active is not None:
                    if not active.cbit_start <= cb < next_cbit:
                        raise ValueError("exact block reads external or unproduced parity source")
                elif private_owner(cb) is not None:
                    raise ValueError("private exact-block classical bit escapes in parity source")

        produced = None
        if op.kind == ops.MEASURE:
            produced = (op.cbit, op.cbit + 1)
        elif op.kind == ops.CPARITY:
            produced = (op.base, op.base + len(op.values))
        if produced is not None:
            lo, hi = produced
            if active is not None:
                if op.cond or lo != next_cbit or hi > active.cbit_stop:
                    raise ValueError("exact-block classical producers must be local and unconditional")
                next_cbit = hi
            elif lo < hi:
                index = bisect_right(stops, lo)
                if index < len(private) and starts[index] < hi:
                    raise ValueError("external classical producer overlaps private exact-block range")

        touched = op.qubits
        if op.kind == ops.XOR_WORD and op.ctrl is not None:
            touched = touched + (op.ctrl,)
        if any(type(q) is not int or q < 0 for q in touched):
            raise ValueError("invalid qubit id in emitted IR")
        if op.kind == ops.ALLOC:
            if len(set(touched)) != len(touched) or any(q in live for q in touched):
                raise ValueError("allocation overlaps a live qubit")
            if active is not None:
                if op.is_input:
                    raise ValueError("exact block may not allocate an input")
                local.update(touched)
                allowed.update(touched)
            live.update(touched)
        else:
            if any(q not in live for q in touched):
                raise ValueError("emitted operation touches a dead qubit lifetime")
            if active is not None and any(q not in allowed for q in touched):
                raise ValueError("exact block touches an undeclared bystander qubit")
            if op.kind in (ops.FREE, ops.MEASURE):
                for q in touched:
                    if active is not None:
                        allowed.remove(q)
                        if q in local:
                            local.remove(q)
                        elif (active.kind == "unlookup" and q in active.target
                              and op.kind == ops.MEASURE):
                            consumed_target.add(q)
                        else:
                            raise ValueError("exact block consumes a preserved entry qubit")
                    live.remove(q)

        if active is not None and position + 1 == active.stop:
            if live != active.exit_live or local:
                raise ValueError("exact-block exit live set disagrees with emitted IR")
            if active.kind == "unlookup" and consumed_target != set(active.target):
                raise ValueError("unlookup did not consume its original target lifetimes")
            if next_cbit != active.cbit_stop:
                raise ValueError("exact-block classical range contains unproduced bits")
            active = None
            block_index += 1
    return blocks
