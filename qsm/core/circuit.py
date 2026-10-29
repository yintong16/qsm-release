"""Circuit IR: qubit/ancilla allocation, gate emission, condition stack, op log.

The Circuit only *records* an op stream and qubit lifetimes. It does not simulate
(see core.simulator) and does not count resources (see core.resources).

Qubit ids are reused after free (a free-list), so the peak live-qubit count reflects
real recycling. Registers are thin ordered views over qubit ids.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from . import ops
from .ops import Op


class Register:
    """An ordered list of qubit ids. `reg[i]` is an int id; `reg[a:b]` is a Register."""

    __slots__ = ("ids", "name")

    def __init__(self, ids: Iterable[int], name: str | None = None) -> None:
        self.ids: list[int] = list(ids)
        self.name = name

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.ids)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Register(self.ids[i], self.name)
        return self.ids[i]

    def __repr__(self) -> str:
        return f"Register({self.name!r}, {self.ids})"


class DirtyAncillaError(AssertionError):
    """Raised (during simulation) when a freed qubit is not |0>."""


class Circuit:
    def __init__(self) -> None:
        self.ops: list[Op | ops.XorWordOp | ops.ClassicalParityOp] = []
        # Trusted semantic summaries are a sidecar, never physical IR/resources.
        self.exact_blocks: list = []
        self._exact_block_seals: dict = {}
        self.inputs: dict[str, Register] = {}   # name -> input register (ordered)
        self._next_qid = 0
        self._free_qids: list[int] = []
        self._live_qids: set[int] = set()
        self._live = 0
        self._peak = 0
        self._next_cbit = 0
        self._cbit_const: dict[int, int] = {}
        self._cond_stack: list[tuple[int, int]] = []

    # ----- qubit allocation -------------------------------------------------
    def _take(self, n: int) -> list[int]:
        out: list[int] = []
        for _ in range(n):
            if self._free_qids:
                out.append(self._free_qids.pop())
            else:
                out.append(self._next_qid)
                self._next_qid += 1
        self._live_qids.update(out)
        self._live += n
        self._peak = max(self._peak, self._live)
        return out

    def input(self, name: str, n: int) -> Register:
        """Allocate an input register (loaded with the test value by the simulator)."""
        if name in self.inputs:
            raise ValueError(f"duplicate input register name {name!r}")
        ids = self._take(n)
        self.ops.append(Op(ops.ALLOC, tuple(ids), name=name, is_input=True))
        reg = Register(ids, name)
        self.inputs[name] = reg
        return reg

    def rebind(self, name: str, reg: Register) -> None:
        """Point a named input register at a DIFFERENT set of qubits.

        A shift is a relabelling of wires, not a gate. Doubling moves bit i to bit i+1,
        which costs nothing physically -- you simply call a different wire "bit 1". Any
        real compiler does this; it is bookkeeping, not computation.

        But the simulator reads a named register's value from the ids recorded at
        `input()` time, so a circuit that relabels must say so or its result is read off
        the wrong wires. This is that declaration.

        Use it only for genuine relabelling. Moving a VALUE between qubits costs gates
        (see `moddouble_inplace`); this records that no value moved and the name did.
        """
        if name not in self.inputs:
            raise KeyError(f"unknown input register {name!r}")
        self.inputs[name] = reg

    def alloc(self, n: int) -> Register:
        """Allocate an ancilla register, initialized to |0> by the simulator."""
        ids = self._take(n)
        self.ops.append(Op(ops.ALLOC, tuple(ids)))
        return Register(ids)

    def free(self, reg: Register) -> None:
        """Free a register; the simulator asserts every qubit is |0> at this point."""
        ids = list(reg.ids)
        self.ops.append(Op(ops.FREE, tuple(ids)))
        self._live -= len(ids)
        self._free_qids.extend(ids)
        self._live_qids.difference_update(ids)

    # ----- classical bits & conditions -------------------------------------
    def classical_bit(self, value: int) -> int:
        cb = self._next_cbit
        self._next_cbit += 1
        self._cbit_const[cb] = int(value) & 1
        return cb

    def classical_parities(self, sources: tuple[int, ...], values: tuple[int, ...]) -> int:
        """Reserve and compute runtime parity bits; return their first classical id.

        Each result ``base+j`` is parity(packed sources & values[j]). Tuple payloads
        are retained by identity. This is classical work, not quantum measurement;
        sources can be any previously created classical bits, including parities.
        Outer classical conditions are rejected before any state changes.
        """
        if self._cond_stack:
            raise ValueError("classical_parities cannot run inside a classical condition")
        if not isinstance(sources, tuple) or not isinstance(values, tuple):
            raise TypeError("sources and values must be immutable tuples")
        for cb in sources:
            if not isinstance(cb, int) or isinstance(cb, bool):
                raise TypeError("source classical ids must be integers, not bools")
            if not 0 <= cb < self._next_cbit:
                raise ValueError(f"source classical id {cb} has not been created")
        for word in values:
            if not isinstance(word, int) or isinstance(word, bool):
                raise TypeError("parity words must be integers, not bools")
            if word < 0 or word >> len(sources):
                raise ValueError("parity word must be nonnegative and fit sources")
        base = self._next_cbit
        self.ops.append(ops.ClassicalParityOp(sources, values, base))
        self._next_cbit += len(values)
        return base

    def _cond(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._cond_stack)

    @contextmanager
    def condition(self, cbit: int, value: int = 1):
        self._cond_stack.append((cbit, int(value) & 1))
        try:
            yield
        finally:
            self._cond_stack.pop()

    # ----- gates ------------------------------------------------------------
    def x(self, t: int) -> None:
        self.ops.append(Op(ops.X, (t,), cond=self._cond()))

    def cx(self, c: int, t: int) -> None:
        self.ops.append(Op(ops.CX, (c, t), cond=self._cond()))

    def xor_word(
        self, target: Register | tuple[int, ...], mask: int, ctrl: int | None = None,
    ) -> None:
        """Record LSB-first physical X/CX fanout without per-bit Op objects.

        Validate every wire, including unselected targets, before emission. A zero
        mask emits nothing after validation. Supplied target tuples are shared by
        identity; Register views are snapshotted into immutable tuples.
        """
        if isinstance(target, Register):
            targets = tuple(target.ids)
        elif isinstance(target, tuple):
            targets = target
        else:
            raise TypeError("target must be a Register or immutable tuple")
        if not isinstance(mask, int) or isinstance(mask, bool):
            raise TypeError("mask must be an integer, not bool")
        if mask < 0 or mask >> len(targets):
            raise ValueError("mask must be nonnegative and fit targets")
        seen: set[int] = set()
        for q in targets:
            if not isinstance(q, int) or isinstance(q, bool):
                raise TypeError("target ids must be integers, not bools")
            if q not in self._live_qids:
                raise ValueError(f"target qubit {q} is not live")
            if q in seen:
                raise ValueError("target qubit ids must be distinct")
            seen.add(q)
        if ctrl is not None:
            if not isinstance(ctrl, int) or isinstance(ctrl, bool):
                raise TypeError("control id must be an integer, not bool")
            if ctrl not in self._live_qids:
                raise ValueError(f"control qubit {ctrl} is not live")
            if ctrl in seen:
                raise ValueError("control must be outside targets")
        if mask:
            self.ops.append(ops.XorWordOp(targets, mask, ctrl, self._cond()))

    def ccx(self, a: int, b: int, t: int) -> None:
        self.ops.append(Op(ops.CCX, (a, b, t), cond=self._cond()))

    def cz(self, a: int, b: int) -> None:
        self.ops.append(Op(ops.CZ, (a, b), cond=self._cond()))

    def z(self, t: int) -> None:
        """Single-qubit phase flip. Needed for measurement-based *flag* erasure, where
        the correcting predicate is a lone qubit rather than an AND of two."""
        self.ops.append(Op(ops.Z, (t,), cond=self._cond()))

    def measure(self, t: int) -> int:
        """H-basis measurement that consumes qubit `t`; returns its classical-bit id."""
        cb = self._next_cbit
        self._next_cbit += 1
        self.ops.append(Op(ops.MEASURE, (t,), cbit=cb, cond=self._cond()))
        self._live -= 1
        self._free_qids.append(t)
        self._live_qids.discard(t)
        return cb

    # ----- introspection ----------------------------------------------------
    @property
    def peak_qubits(self) -> int:
        return self._peak
