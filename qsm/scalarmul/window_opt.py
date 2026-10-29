"""Versioned measured-QROM windows, with materialized and streamed operands.

Lookup/measurement cleanup is exact. The reused affine arithmetic is still
approximate and incomplete; its empirical diagnostic contract is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add
from ..field.add_chunked import gidney_add_chunked
from ..field.modarith import modsub_direct
from ..inversion.config import InversionConfig
from ..inversion.inplace import moddiv_inplace, modmul_inplace
from ..mul.accumulate import modsqrsub_inplace
from ..params import PRIME, U
from ..point import active as curve
from ..point.affine import sub_const_mod, tuned_point_kwargs
from .lookup import _validate_registers, lookup_xor_unary, unlookup_measure
from .window import (
    WindowTable, _enabled, _region, _window_case, add_window_affine_inplace,
    make_window_table,
)


@dataclass(frozen=True)
class OptimizedWindowConfig:
    n: int
    p: int
    window_bits: int
    operand_strategy: str
    adder_chunk: int | None
    fused_block: int
    unlookup_low_bits: int
    inversion: InversionConfig

    def __post_init__(self):
        if not isinstance(self.inversion, InversionConfig):
            raise ValueError("window inversion must be an InversionConfig")
        self.inversion.resolved(self.n, self.p)
        if self.inversion.arithmetic_family != "tuned":
            raise ValueError("optimized windows use tuned inversion arithmetic")
        if type(self.window_bits) is not int or self.window_bits < 1:
            raise ValueError("window bits must be a positive integer")
        if self.operand_strategy not in ("paired", "fused"):
            raise ValueError("operand strategy must be paired or fused")
        if self.adder_chunk is not None and (type(self.adder_chunk) is not int or self.adder_chunk < 1):
            raise ValueError("adder chunk must be positive or None")
        if type(self.fused_block) is not int or self.fused_block < 1:
            raise ValueError("fused block must be positive")
        if (type(self.unlookup_low_bits) is not int or
                not 0 <= self.unlookup_low_bits <= self.window_bits):
            raise ValueError("unlookup low bits must be within the window")

    @classmethod
    def tuned(cls, n: int = U, p: int = PRIME, bits: int = 4, *,
              operand_strategy: str = "paired", adder_chunk: int | str | None = "half",
              fused_block: int | str = "half", unlookup_low_bits: int | None = None,
              inversion: InversionConfig | None = None):
        inversion = InversionConfig.tuned(n, p) if inversion is None else inversion
        half = (n + 2) // 2
        return cls(n, p, bits, operand_strategy,
                   half if adder_chunk == "half" else adder_chunk,
                   half if fused_block == "half" else fused_block,
                   bits // 2 if unlookup_low_bits is None else unlookup_low_bits, inversion)

    def resolved(self, n, p, bits):
        if (self.n, self.p, self.window_bits) != (n, p, bits):
            raise ValueError("window configuration does not match operand field/width/address")
        return self

    def metadata(self):
        result = asdict(self)
        result.pop("inversion")
        return result


def _snapshot_bindings(circ):
    return {name: Register(list(reg.ids), reg.name) for name, reg in circ.inputs.items()}


def _rebind_coordinates(circ, bindings, old_x, old_y, x, y):
    # GCD clears a consumed operand's named view and recycles its ids. Restore
    # pre-call bindings before measured unlookup checks scratch ownership.
    mapping = dict(zip(old_x.ids + old_y.ids, x.ids + y.ids))
    for name, reg in bindings.items():
        if any(q in mapping for q in reg):
            circ.rebind(name, Register([mapping.get(q, q) for q in reg], reg.name))


def _packed_sub(circ, address, accumulators, columns, p, const_kw, low_bits, regions):
    n = len(accumulators[0])
    words = tuple(sum(column[row] << (i*n) for i, column in enumerate(columns))
                  for row in range(1 << len(address)))
    with _region(circ, regions, "lookup"):
        operand = circ.alloc(n * len(columns))
        lookup_xor_unary(circ, address, operand, words)
    with _region(circ, regions, "arithmetic"):
        for i, acc in enumerate(accumulators):
            modsub_direct(circ, operand[i*n:(i+1)*n], acc, p, **const_kw)
    with _region(circ, regions, "unlookup"):
        unlookup_measure(circ, address, operand, words, low_bits=low_bits)


def add_window_affine_optimized_inplace(
    circ: Circuit, x: Register, y: Register, address: Register, table: WindowTable, *,
    config: OptimizedWindowConfig | None = None,
    const_kw: dict | None = None, sqr_kw: dict | None = None,
    regions: list | None = None,
) -> tuple[Register, Register]:
    """Add one selected point with measured lookup cleanup; return coordinate handles.

    Matching named coordinate views are rebound after each GCD region. A paired
    window has three lookup/unlookup stages; streamed operands need more calls.
    """
    if not isinstance(table, WindowTable):
        raise ValueError("table must be a validated WindowTable")
    if not all(isinstance(r, Register) for r in (x, y, address)):
        raise TypeError("window operands must be Registers")
    n, p, bits = len(x), table.p, table.bits
    if len(y) != n or len(address) != bits or p >= 1 << n:
        raise ValueError("window operand widths must match")
    _validate_registers(circ, address, Register(x.ids + y.ids))
    if circ._cond_stack:
        raise ValueError("measured windows cannot be wrapped in a classical condition")
    config = (OptimizedWindowConfig.tuned(n, p, bits) if config is None else config).resolved(n, p, bits)
    kw = tuned_point_kwargs(n, p, config.inversion)
    const = kw["const_kw"] if const_kw is None else dict(const_kw)
    square = kw["sqr_kw"] if sqr_kw is None else dict(sqr_kw)

    if bits == 1 and config.operand_strategy == "paired":
        bindings = _snapshot_bindings(circ)
        old_x, old_y = x, y
        x, y = add_window_affine_inplace(
            circ, x, y, address, table, config=config.inversion,
            const_kw=const, sqr_kw=square, regions=regions)
        _rebind_coordinates(circ, bindings, old_x, old_y, x, y)
        return x, y

    const["add_fn"] = (gidney_add if config.adder_chunk is None else
                       partial(gidney_add_chunked, chunk=config.adder_chunk, cmp_bits=None))
    points = (table.points[1],) + table.points[1:]
    xs, ys = tuple(pt[0] for pt in points), tuple(pt[1] for pt in points)
    f = (1 << n) - p
    ks = (0,) + tuple((f - 1 + 3*xp) % p for xp in xs[1:])

    def subtract(accumulators, columns):
        if config.operand_strategy == "paired":
            _packed_sub(circ, address, accumulators, columns, p, const,
                        config.unlookup_low_bits, regions)
        else:
            from .fused import modsub_lookup_inplace
            for acc, values in zip(accumulators, columns):
                modsub_lookup_inplace(
                    circ, address, acc, values, p, block=config.fused_block,
                    carry_window=const.get("carry_window"), cmp_bits=const.get("cmp_bits"),
                    low_bits=config.unlookup_low_bits, regions=regions)

    subtract((x, y), (xs, ys))
    with _region(circ, regions, "arithmetic"):
        bindings = _snapshot_bindings(circ)
        old_x, old_y = x, y
        x, y = moddiv_inplace(circ, x, y, p, config=config.inversion)
        _rebind_coordinates(circ, bindings, old_x, old_y, x, y)
    with _enabled(circ, address, regions) as flag:
        with _region(circ, regions, "arithmetic"):
            bindings = _snapshot_bindings(circ)
            old_x, old_y = x, y
            x = modsqrsub_inplace(circ, y, x, p, ctrl=flag, **square)
            _rebind_coordinates(circ, bindings, old_x, old_y, x, y)
            for q in x:
                circ.cx(flag, q)
    subtract((x,), (ks,))
    with _region(circ, regions, "arithmetic"):
        bindings = _snapshot_bindings(circ)
        old_x, old_y = x, y
        x, y = modmul_inplace(circ, x, y, p, config=config.inversion)
        _rebind_coordinates(circ, bindings, old_x, old_y, x, y)
    subtract((x, y), (tuple((-v) % p for v in xs), tuple((-v) % p for v in ys)))
    with _enabled(circ, address, regions) as flag:
        with _region(circ, regions, "arithmetic"):
            for q in y:
                circ.cx(flag, q)
            sub_const_mod(circ, y, (f-1) % p, p, ctrl=flag, **const)
    return x, y


def build_optimized_window_case(bits: int = 4, *, table: WindowTable | None = None,
                                config: OptimizedWindowConfig | None = None,
                                legacy_square_ripple: bool = False) -> Case:
    table = make_window_table(bits) if table is None else table
    if (table.bits != bits or
            (table.p, table.a, table.b) != (PRIME, curve.CURVE.A, curve.CURVE.B)):
        raise ValueError("benchmark table must match the active field/curve and window width")
    config = (OptimizedWindowConfig.tuned(U, PRIME, bits) if config is None else config).resolved(U, PRIME, bits)
    circ = Circuit()
    j, x, y = circ.input("j", bits), circ.input("x", U), circ.input("y", U)
    regions = []
    square = tuned_point_kwargs(U, PRIME, config.inversion,
                                legacy_square_ripple=legacy_square_ripple)["sqr_kw"]
    x, y = add_window_affine_optimized_inplace(circ, x, y, j, table, config=config,
                                               sqr_kw=square, regions=regions)
    circ.rebind("x", x)
    circ.rebind("y", y)
    case = _window_case(circ, table, config.inversion, regions, square["ripple"])
    specialized = bits == 1 and config.operand_strategy == "paired"
    case.meta["lookup_backend"] = "constant-control.v1" if specialized else "unary-and.v1"
    case.meta["tuning"]["window"] = config.metadata()
    case.meta["circuit_config"] = {
        "unlookup_backend": "constant-control.v1" if specialized else "measured-onehot.v1",
        "operand_strategy": config.operand_strategy,
        "adder_chunk": config.adder_chunk,
        "fused_block": config.fused_block,
        "unlookup_low_bits": config.unlookup_low_bits,
        "one_bit_specialization": specialized,
    }
    return case


@circuit_version("scalarmul.add_window_affine_w4_inplace.v2")
def _build_optimized() -> Case:
    # Registered before the F-independent square ripple; pinned for reproducibility.
    return build_optimized_window_case(4, legacy_square_ripple=True)
