"""Exact unitary and measurement-assisted selection by a quantum address.

Registers are little endian. The address is never measured or inspected in Python.
The original lookup_xor/xor_nonzero use only X/CX/CCX; lookup_xor_unary additionally
uses phase-clean temporary-AND measurements. unlookup_measure consumes a restored
scratch word and corrects its measurement phase with a runtime classical parity
table and a coherent address phase oracle. Its proof covers every outcome, not
only the output permutation.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.circuit import Circuit, Register
from ..core.semantics import _record_exact_lookup


def _validate_registers(circ: Circuit, address: Register, target: Register) -> None:
    """Reject malformed wire views before emitting any operations."""
    if not isinstance(address, Register) or not isinstance(target, Register):
        raise TypeError("address and target must be Registers")
    if len(address) < 1 or len(target) < 1:
        raise ValueError("address and target widths must be positive")
    wires = list(address) + list(target)
    if any(not isinstance(q, int) or isinstance(q, bool) for q in wires):
        raise TypeError("qubit ids must be integers")
    if len(set(wires)) != len(wires):
        raise ValueError("address and target must be disjoint, without duplicate wires")
    freed = set(circ._free_qids)
    if any(q < 0 or q >= circ._next_qid or q in freed for q in wires):
        raise ValueError("all address and target wires must be live in the circuit")


def _validate_lookup(
    circ: Circuit, address: Register, target: Register, values: Sequence[int]
) -> tuple[int, ...]:
    _validate_registers(circ, address, target)
    if not isinstance(values, Sequence):
        raise TypeError("values must be a finite sequence of integers")
    if len(values) != 1 << len(address):
        raise ValueError("table length must equal 2**len(address)")
    rows = tuple(values)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in rows):
        raise TypeError("table entries must be integers")
    if any(v < 0 or v >= 1 << len(target) for v in rows):
        raise ValueError("table entries must be unsigned words fitting the target")
    return rows


def lookup_xor(
    circ: Circuit,
    address: Register,
    target: Register,
    values: Sequence[int],
) -> None:
    """Apply ``|j,z> -> |j,z XOR values[j]>`` with clean prefix scratch.

    ``values`` contains exactly ``2**len(address)`` unsigned target-width words;
    both registers must be nonempty and disjoint. The target need not start at
    zero. Calling this function twice restores every input, including scratch.

    Traverse an MSB-first binary prefix tree. A prefix's common one-columns are
    emitted once and removed from all descendants; all-zero residual subtrees
    are omitted. The root uses X, first-level prefixes borrow an address wire,
    and deeper prefixes use a computed AND followed by its exact reverse. Thus
    at most ``len(address)-1`` clean scratch qubits coexist, independent of the
    table's word width. Uniform tables require no decoder at all.
    """
    rows = _validate_lookup(circ, address, target, values)

    def visit(words: tuple[int, ...], bit: int, prefix: int | None) -> None:
        common = words[0]
        for word in words[1:]:
            common &= word
        mask = common
        while mask:
            low = mask & -mask
            q = target[low.bit_length() - 1]
            if prefix is None:
                circ.x(q)
            else:
                circ.cx(prefix, q)
            mask ^= low
        residual = tuple(word ^ common for word in words)
        if not any(residual):
            return

        half = len(residual) // 2
        for digit, child in enumerate((residual[:half], residual[half:])):
            if not any(child):
                continue
            control = address[bit]
            if digit == 0:
                circ.x(control)
            if prefix is None:
                visit(child, bit - 1, control)
            else:
                scratch = circ.alloc(1)
                circ.ccx(prefix, control, scratch[0])
                visit(child, bit - 1, scratch[0])
                circ.ccx(prefix, control, scratch[0])
                circ.free(scratch)
            if digit == 0:
                circ.x(control)

    visit(rows, len(address) - 1, None)


def xor_nonzero(circ: Circuit, address: Register, target: int) -> None:
    """Toggle ``target`` iff the quantum address is nonzero; restore the address.

    This is a fully unitary XOR predicate, not a measurement-based flag erasure.
    It accepts either initial target bit and uses at most ``w-2`` clean ancillas.
    """
    _validate_registers(circ, address, Register([target]))
    w = len(address)
    if w == 1:
        circ.cx(address[0], target)
        return

    # nonzero = 1 XOR AND(not address[i]). Negative controls are borrowed only
    # during this ladder, then restored before returning to the caller.
    circ.x(target)
    for q in address:
        circ.x(q)
    if w == 2:
        circ.ccx(address[0], address[1], target)
    else:
        scratch = circ.alloc(w - 2)
        circ.ccx(address[0], address[1], scratch[0])
        for i in range(1, w - 2):
            circ.ccx(scratch[i - 1], address[i + 1], scratch[i])
        circ.ccx(scratch[-1], address[-1], target)
        for i in range(w - 3, 0, -1):
            circ.ccx(scratch[i - 1], address[i + 1], scratch[i])
        circ.ccx(address[0], address[1], scratch[0])
        circ.free(scratch)
    for q in reversed(address.ids):
        circ.x(q)


# Measurement-assisted APIs are deliberately separate from the exact v1 stream
# above: callers must opt into their measurement/feed-forward requirements.
from collections.abc import Callable

from ..field.add import _unand


def _unary_visit(
    circ: Circuit,
    address: Sequence[int],
    callback: Callable[[int | None, int], None],
) -> None:
    """Visit every address with a clean unary control; allow an empty address.

    The root borrows the MSB, while each deeper node shares one temporary AND
    between its zero and one children. Measurement erases that AND only after
    restoring its positive-control Boolean relation. No table pruning is done.
    """
    if not address:
        callback(None, 0)
        return

    def visit(bit: int, prefix: int, row: int) -> None:
        if bit < 0:
            callback(prefix, row)
            return
        control = address[bit]
        scratch = circ.alloc(1)[0]
        circ.ccx(prefix, control, scratch)
        circ.cx(prefix, scratch)  # prefix AND NOT control
        visit(bit - 1, scratch, row)
        circ.cx(prefix, scratch)  # prefix AND control again
        visit(bit - 1, scratch, row | (1 << bit))
        _unand(circ, prefix, control, scratch)  # consumes scratch

    msb = len(address) - 1
    root = address[msb]
    circ.x(root)
    visit(msb - 1, root, 0)
    circ.x(root)
    visit(msb - 1, root, 1 << msb)


def lookup_xor_unary(
    circ: Circuit,
    address: Register,
    target: Register,
    values: Sequence[int],
) -> None:
    """Apply exact ``|j,z> -> |j,z XOR values[j]>`` using unary iteration.

    ``target`` may be a named input and may initially contain any word. Temporary
    AND measurements have their phases corrected for *every* outcome. Neither
    address nor target is measured. An enclosing classical condition is rejected
    because the decoder allocates and consumes scratch unconditionally.

    For ``N = 2**len(address)``, this unpruned construction uses exactly ``N-2``
    CCX and temporary-AND measurements, including for zero/constant tables, and
    at most ``len(address)-1`` simultaneous scratch qubits. Word operations share
    one immutable target-wire tuple rather than expanding every table bit here.
    """
    rows = _validate_lookup(circ, address, target, values)
    if circ._cond_stack:
        raise ValueError("unary lookup cannot run inside a classical condition")
    target_ids = tuple(target)
    address_ids = tuple(address)
    start, cbit_start = len(circ.ops), circ._next_cbit
    entry_live = frozenset(circ._live_qids)

    def emit(prefix: int | None, row: int) -> None:
        circ.xor_word(target_ids, rows[row], ctrl=prefix)

    _unary_visit(circ, address.ids, emit)
    _record_exact_lookup(
        circ, kind="xor", start=start, cbit_start=cbit_start,
        entry_live=entry_live, address=address_ids, target=target_ids, values=rows,
    )


def unlookup_measure(
    circ: Circuit,
    address: Register,
    target: Register,
    values: Sequence[int],
    *,
    low_bits: int | None = None,
) -> None:
    """Consume scratch ``target == values[address]`` with exact phase correction.

    This is NOT a general XOR inverse: the target must have been restored to the
    stated table word, with no other data in it. That semantic precondition is the
    caller's responsibility. Named-input target wires (even through aliases) and
    enclosing classical conditions are rejected before any circuit mutation.
    Measurement consumes target wires; callers must neither read nor free them.

    Measuring target gives phase ``dot(m, values[j])``. Write ``j = hi*L + lo``
    with ``L = 2**low_bits``. A low-address one-hot bank and a unary high-address
    traversal apply exactly that phase using classical row-parity conditions.
    CZ does not change the bank's Boolean values, so each temporary AND can still
    be measurement-erased with its usual phase correction. Only the bank root
    needs an explicit X and free after all other bank wires have been consumed.

    The default split is ``len(address)//2``; all splits from zero through the
    full address width are supported. This particular unpruned construction uses
    ``(L-1) + max(N//L-2, 0)`` CCX, not a universal optimality claim. It measures
    ``len(target)`` output bits plus one bit per CCX; CPARITY is classical work,
    not another quantum measurement. The validated immutable rows are reused by
    the classical parity operation.
    """
    rows = _validate_lookup(circ, address, target, values)
    if circ._cond_stack:
        raise ValueError("measured unlookup cannot run inside a classical condition")
    input_ids = {q for reg in circ.inputs.values() for q in reg}
    if input_ids.intersection(target):
        raise ValueError("measured unlookup target must be scratch, not named input wires")
    if low_bits is None:
        low_bits = len(address) // 2
    if not isinstance(low_bits, int) or isinstance(low_bits, bool):
        raise TypeError("low_bits must be an integer or None")
    if not 0 <= low_bits <= len(address):
        raise ValueError("low_bits must be between zero and the address width")

    start, cbit_start = len(circ.ops), circ._next_cbit
    entry_live = frozenset(circ._live_qids)
    address_ids, target_ids = tuple(address), tuple(target)

    measurements = tuple(circ.measure(q) for q in target)
    base = circ.classical_parities(measurements, rows)
    size = 1 << low_bits
    bank = circ.alloc(size)
    circ.x(bank[0])
    for bit in range(low_bits):
        half = 1 << bit
        for i in range(half):
            circ.ccx(bank[i], address[bit], bank[i + half])
            circ.cx(bank[i + half], bank[i])

    def correct(prefix: int | None, high: int) -> None:
        for low in range(size):
            row = high * size + low
            with circ.condition(base + row):
                if prefix is None:
                    circ.z(bank[low])
                else:
                    circ.cz(prefix, bank[low])

    _unary_visit(circ, address.ids[low_bits:], correct)
    for bit in reversed(range(low_bits)):
        half = 1 << bit
        for i in reversed(range(half)):
            circ.cx(bank[i + half], bank[i])
            _unand(circ, bank[i], address[bit], bank[i + half])
    circ.x(bank[0])
    circ.free(Register([bank[0]]))
    _record_exact_lookup(
        circ, kind="unlookup", start=start, cbit_start=cbit_start,
        entry_live=entry_live, address=address_ids, target=target_ids, values=rows,
    )
