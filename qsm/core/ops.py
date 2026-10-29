"""Op model for the reversible-circuit IR.

Gates are the reversible primitives we cost: X, CX, CCX (Toffoli), CZ, and a
measurement (H-basis, used for measurement-based uncomputation). ALLOC/FREE are
bookkeeping markers in the op stream so the simulator can zero freshly allocated
qubits and enforce that freed ancillas return to |0>.

Qubit conventions (all ids are ints):
    X       qubits = (target,)
    CX      qubits = (control, target)
    CCX     qubits = (c1, c2, target)
    CZ      qubits = (a, b)                 # symmetric; contributes only phase
    Z       qubits = (t,)                   # single-qubit phase flip
    MEASURE qubits = (target,)  cbit = produced classical-bit id
    ALLOC   qubits = (ids...)   name/ is_input describe an input register
    FREE    qubits = (ids...)

XOR_WORD stores an ordered X/CX fanout compactly; all physical gates still count.
CPARITY is runtime classical feed-forward, with no quantum gate or qubit cost.
These use separate immutable records rather than enlarging every primitive Op.

`cond` is a tuple of (classical_bit_id, expected_value) pairs; the op executes on a
shot only if every listed classical bit equals its expected value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

X = "X"
CX = "CX"
CCX = "CCX"
CZ = "CZ"
Z = "Z"
MEASURE = "MEASURE"
ALLOC = "ALLOC"
FREE = "FREE"
XOR_WORD = "XOR_WORD"
CPARITY = "CPARITY"

GATE_KINDS = frozenset({X, CX, CCX, CZ, Z, MEASURE})


@dataclass(frozen=True, slots=True)
class Op:
    kind: str
    qubits: tuple[int, ...]
    cbit: int = -1
    cond: tuple[tuple[int, int], ...] = ()
    name: str | None = None
    is_input: bool = False


@dataclass(frozen=True, slots=True)
class XorWordOp:
    """Exactly the LSB-first X/CX gates selected by ``mask`` in ``qubits``.

    ``qubits`` contains targets only; ``ctrl`` is the optional CX control.
    Unselected targets are not touched. Conditions apply to every physical gate.
    Tuple payloads are shared unchanged, not copied into per-bit operations.
    """

    qubits: tuple[int, ...]
    mask: int
    ctrl: int | None = None
    cond: tuple[tuple[int, int], ...] = ()
    kind: str = field(default=XOR_WORD, init=False)


@dataclass(frozen=True, slots=True)
class ClassicalParityOp:
    """Runtime parity table with no quantum gates or quantum latency.

    Pack classical ``sources[i]`` into bit i, then store parity(packed & values[j])
    at classical id ``base + j``. Classical work is len(values) parity results and
    sum(word.bit_count() for word in values) selected source-bit incidences.
    """

    sources: tuple[int, ...]
    values: tuple[int, ...]
    base: int
    kind: str = field(default=CPARITY, init=False)
    cond: tuple[tuple[int, int], ...] = field(default=(), init=False)
    qubits: tuple[int, ...] = field(default=(), init=False)


def physical_gate_counts(op: Op | XorWordOp | ClassicalParityOp) -> dict[str, int]:
    """Physical primitive multiplicities; bookkeeping/classical work yields {}.

    XOR_WORD contributes mask.bit_count() X gates (ctrl=None), otherwise CX gates.
    It never contributes a new unit-cost quantum primitive to GATE_KINDS.
    """
    if op.kind == XOR_WORD:
        n = op.mask.bit_count()
        return {X if op.ctrl is None else CX: n} if n else {}
    if op.kind in GATE_KINDS:
        return {op.kind: 1}
    return {}
