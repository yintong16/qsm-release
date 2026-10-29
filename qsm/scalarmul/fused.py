"""Streamed table operands for integer addition and direct modular subtraction.

The original integer experiment is preserved unchanged. The separate modular
backend uses unary lookup and measured unlookup, including all repeated loads,
exact boundary cleanup, constant reduction, and the direct subtraction's existing
approximate borrow-phase correction. Neither backend implies a whole-window gain.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from ..core.circuit import Circuit, Register
from ..field.add import _gidney_add_block
from ..field.add_chunked import _kill_boundary_carry
from ..field.compare import cmp_lt_phase
from ..field.constadd import sub_const_inplace
from .lookup import _validate_lookup, lookup_xor, lookup_xor_unary, unlookup_measure


def add_lookup_inplace(
    circ: Circuit,
    address: Register,
    target: Register,
    values: Sequence[int],
    *,
    block: int = 8,
) -> None:
    """Add ``values[address]`` to target modulo ``2**len(target)``.

    Preserve the address and return all scratch clean. Lookup itself is unitary;
    the existing Gidney block adder/comparator use exact measurement/feed-forward
    erasure of internal ANDs. No arithmetic approximation or modular reduction
    is used. Validation follows :func:`lookup_xor`; ``block`` is positive.

    The adder temporarily changes operand wires but restores them before each
    unlookup. Store one carry per block boundary. After the forward sum, reload
    each lower operand block A in descending order and erase its carry using
    ``cout = [S < A + cin]``. The exact existing boundary helper evaluates this
    as the carry of ``~S + A + cin``. Descending order keeps cin alive until the
    last predicate that needs it. Reloaded operands are again restored before
    unloading. Merely erasing cout against the original add inputs would be
    wrong: those inputs have already become the sum.

    For m blocks this emits 2*m-1 block lookup/unlookup pairs, versus one pair
    for whole-word load/add/unload. Scratch is O(block + n/block + address_width),
    including retained boundary carries. With block >= n this reduces to the
    whole-word Gidney baseline. This prototype is not used by the point circuit.
    """
    if not isinstance(block, int) or isinstance(block, bool):
        raise TypeError("block width must be an integer")
    if block < 1:
        raise ValueError("block width must be positive")
    rows = _validate_lookup(circ, address, target, values)
    n = len(target)

    def load(lo: int, hi: int) -> tuple[Register, tuple[int, ...]]:
        mask = (1 << (hi - lo)) - 1
        words = tuple((value >> lo) & mask for value in rows)
        operand = circ.alloc(hi - lo)
        lookup_xor(circ, address, operand, words)
        return operand, words

    def unload(operand: Register, words: tuple[int, ...]) -> None:
        lookup_xor(circ, address, operand, words)
        circ.free(operand)

    boundaries: list[tuple[int, int, int | None, int]] = []
    cin = None
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        cout = circ.alloc(1)[0] if hi < n else None
        operand, words = load(lo, hi)
        _gidney_add_block(circ, operand, target[lo:hi], cin=cin, cout=cout)
        unload(operand, words)
        if cout is not None:
            boundaries.append((lo, hi, cin, cout))
            cin = cout

    for lo, hi, cin, cout in reversed(boundaries):
        operand, words = load(lo, hi)
        _kill_boundary_carry(circ, operand, target[lo:hi], cin, cout, cmp_bits=None)
        unload(operand, words)


@contextmanager
def _region(circ: Circuit, regions: list | None, label: str) -> Iterator[None]:
    """Record disjoint IR intervals without importing the window caller."""
    start = len(circ.ops)
    yield
    if regions is not None:
        regions.append((label, start, len(circ.ops)))


def _load_slice(circ, address, rows, lo, hi, regions):
    words = tuple((value >> lo) & ((1 << (hi - lo)) - 1) for value in rows)
    with _region(circ, regions, "lookup"):
        operand = circ.alloc(hi - lo)
        lookup_xor_unary(circ, address, operand, words)
    return operand, words


def _unload_slice(circ, address, operand, words, low_bits, regions):
    # PRECONDITION: operand equals words[address], not an intermediate addend or
    # its complement. Measurement destroys it even if this precondition is false;
    # a dirty-FREE check cannot certify restoration here.
    with _region(circ, regions, "unlookup"):
        unlookup_measure(circ, address, operand, words, low_bits=low_bits)


def _add_lookup_measured(circ, address, target, rows, block, low_bits, regions):
    """Exact integer addition with 2*m-1 loaded/restored operand blocks."""
    n = len(target)
    boundaries = []
    cin = None
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        with _region(circ, regions, "arithmetic"):
            cout = circ.alloc(1)[0] if hi < n else None
        operand, words = _load_slice(circ, address, rows, lo, hi, regions)
        with _region(circ, regions, "arithmetic"):
            _gidney_add_block(circ, operand, target[lo:hi], cin=cin, cout=cout)
        # _gidney_add_block restores A before returning, for every carry-in.
        _unload_slice(circ, address, operand, words, low_bits, regions)
        if cout is not None:
            boundaries.append((lo, hi, cin, cout))
            cin = cout

    # Still in the integer SUM representation. Descending order retains the
    # carry-in required by each exact [sum < A + cin] predicate.
    for lo, hi, cin, cout in reversed(boundaries):
        operand, words = _load_slice(circ, address, rows, lo, hi, regions)
        with _region(circ, regions, "arithmetic"):
            _kill_boundary_carry(circ, operand, target[lo:hi], cin, cout, cmp_bits=None)
        # The exact carry helper restores BOTH A and the sum slice.
        _unload_slice(circ, address, operand, words, low_bits, regions)


def modsub_lookup_inplace(
    circ: Circuit,
    address: Register,
    acc: Register,
    values: Sequence[int],
    p: int,
    *,
    block: int = 128,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
    low_bits: int | None = None,
    regions: list | None = None,
) -> None:
    """Subtract a selected canonical table word, preserving the quantum address.

    Match :func:`qsm.field.modarith.modsub_direct`, including its intentional
    approximate borrow-erasure phase. With ``carry_window=None``, the value is
    ``acc - values[j] + (p if acc < values[j] else 0)``. All n-bit accumulators
    are admitted, including lazy inputs; only canonical inputs guarantee canonical
    outputs. A finite carry window inherits the direct circuit's truncated value
    contract, not ideal arithmetic. ``cmp_bits`` truncates only the final phase
    comparison; integer block boundaries are always erased exactly.

    On the n+1-bit complemented accumulator, add the zero-extended selected word
    and erase boundary carries BEFORE uncomplementing. Reduction subtracts
    F=2**n-p under the borrow. Measure that borrow, reload only the top k operand
    bits, and apply the same phase predicate as direct subtraction. Restore every
    operand slice before measured unlookup, which consumes its wires.

    For m=ceil((n+1)/block), emit 2*m-1 integer load/unlookup pairs plus one final
    k-bit comparison pair. In particular, n+1 (not n) determines block count.
    ``block`` is an exact caller choice; no whole-window space gain is promised.
    ``low_bits`` selects the measured-unlookup split in [0, len(address)].
    Optional region records use disjoint lookup/unlookup/arithmetic IR intervals.
    All input/configuration checks precede any allocation or gate emission.
    """
    rows = _validate_lookup(circ, address, acc, values)
    n = len(acc)
    if not isinstance(p, int) or isinstance(p, bool):
        raise TypeError("p must be an integer")
    if not 0 < p < 1 << n:
        raise ValueError("p must be positive and fit in the accumulator")
    if any(value >= p for value in rows):
        raise ValueError("table entries must be canonical: 0 <= value < p")
    for name, value in (("block", block), ("carry_window", carry_window),
                        ("cmp_bits", cmp_bits)):
        if value is None and name != "block":
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer" + (" or None" if name != "block" else ""))
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if low_bits is not None:
        if not isinstance(low_bits, int) or isinstance(low_bits, bool):
            raise TypeError("low_bits must be an integer or None")
        if not 0 <= low_bits <= len(address):
            raise ValueError("low_bits must be between zero and the address width")
    if regions is not None and not isinstance(regions, list):
        raise TypeError("regions must be a list or None")
    if circ._cond_stack:
        raise ValueError("streamed modular lookup cannot run inside a classical condition")

    with _region(circ, regions, "arithmetic"):
        ovf = circ.alloc(1)[0]
        wide = Register(list(acc.ids) + [ovf])
        for q in wide:
            circ.x(q)
    _add_lookup_measured(circ, address, wide, rows, block, low_bits, regions)
    with _region(circ, regions, "arithmetic"):
        for q in wide:
            circ.x(q)
        sub_const_inplace(circ, acc, (1 << n) - p, ctrl=ovf, window=carry_window)
        m = circ.measure(ovf)  # consumes the borrow; never free its recycled id

    k = n if cmp_bits is None else min(n, cmp_bits)
    operand, words = _load_slice(circ, address, rows, n - k, n, regions)
    with _region(circ, regions, "arithmetic"):
        for q in operand:
            circ.x(q)
        # Both operands already refer to the aligned top k bits. Do not apply the
        # original cmp_bits again to a full-width accumulator and a short operand.
        cmp_lt_phase(circ, acc[n - k:], operand, m, k=None)
        for q in operand:
            circ.x(q)
    _unload_slice(circ, address, operand, words, low_bits, regions)
