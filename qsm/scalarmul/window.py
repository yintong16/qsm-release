"""Quantum-indexed unsigned windows on the experimental tuned affine circuit.

Entry zero is logically the identity, but its arithmetic uses a finite dummy
addend and still has the tuned divisor/numerator preconditions and error modes.
Lookup is exact; the full window is not. No selected operand spans a GCD region.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.modarith import modsub_direct
from ..inversion.config import InversionConfig
from ..inversion.inplace import moddiv_inplace, modmul_inplace
from ..mul.accumulate import modsqrsub_inplace
from ..params import PRIME, U
from ..point import active as curve
from ..point.affine import add_const_affine_inplace, sub_const_mod, tuned_point_kwargs
from ..point.group import Point, scalar_mul
from .lookup import _validate_registers, lookup_xor, xor_nonzero


@dataclass(frozen=True)
class WindowTable:
    p: int
    a: int
    b: int
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if type(self.p) is not int or self.p <= 3 or not self.p & 1:
            raise ValueError("window table requires an odd prime greater than three")
        if type(self.a) is not int or type(self.b) is not int:
            raise ValueError("curve coefficients must be integers")
        object.__setattr__(self, "a", self.a % self.p)
        object.__setattr__(self, "b", self.b % self.p)
        if (4 * self.a**3 + 27 * self.b**2) % self.p == 0:
            raise ValueError("window table requires a nonsingular curve")
        points = tuple(None if pt is None else tuple(pt) for pt in self.points)
        object.__setattr__(self, "points", points)
        size = len(points)
        if size < 2 or size & (size - 1) or points[0] is not None:
            raise ValueError("table length must be 2**w with w>=1 and entry zero the identity")
        for pt in points[1:]:
            if pt is None or len(pt) != 2:
                raise ValueError("nonzero window entries must be finite points")
            x, y = pt
            if any(type(v) is not int or not 0 <= v < self.p for v in pt):
                raise ValueError("table coordinates must be canonical integers")
            if (y*y - x*x*x - self.a*x - self.b) % self.p:
                raise ValueError("window entry is not on the declared curve")

    @property
    def bits(self) -> int:
        return (len(self.points) - 1).bit_length()

    @property
    def fingerprint(self) -> str:
        data = json.dumps((self.p, self.a, self.b, self.points), separators=(",", ":"))
        return hashlib.sha256(data.encode()).hexdigest()

    def operand(self, digit: int) -> tuple[int, int]:
        if type(digit) is not int or not 0 <= digit < len(self.points):
            raise ValueError("window digit is out of range")
        return self.points[digit or 1]


def make_window_table(bits: int, base: tuple[int, int] | None = None, *,
                      p: int = PRIME, a: int | None = None,
                      b: int | None = None) -> WindowTable:
    """Build [j]B classically; no scalar reduction or infinity encoding is assumed."""
    if type(bits) is not int or bits < 1:
        raise ValueError("window width must be a positive integer")
    if (a is None or b is None or base is None) and p != PRIME:
        raise ValueError("a non-active field requires explicit curve coefficients and base")
    a = curve.CURVE.A if a is None else a
    b = curve.CURVE.B if b is None else b
    base = (curve.GX, curve.GY) if base is None else base
    WindowTable(p, a, b, (None, base))
    points = (None,) + tuple(scalar_mul(j, *base, p=p, a=a) for j in range(1, 1 << bits))
    return WindowTable(p, a, b, points)


@contextmanager
def _region(circ, regions, label):
    start = len(circ.ops)
    yield
    if regions is not None:
        regions.append((label, start, len(circ.ops)))


def _table_sub(circ, address, acc, values, p, kw, regions):
    with _region(circ, regions, "lookup"):
        operand = circ.alloc(len(acc))
        lookup_xor(circ, address, operand, values)
    with _region(circ, regions, "arithmetic"):
        modsub_direct(circ, operand, acc, p, **kw)
    with _region(circ, regions, "unlookup"):
        lookup_xor(circ, address, operand, values)
        circ.free(operand)


@contextmanager
def _enabled(circ, address, regions):
    with _region(circ, regions, "selection"):
        flag = circ.alloc(1)
        xor_nonzero(circ, address, flag[0])
    yield flag[0]
    with _region(circ, regions, "selection"):
        xor_nonzero(circ, address, flag[0])
        circ.free(flag)


def add_window_affine_inplace(
    circ: Circuit, x: Register, y: Register, address: Register, table: WindowTable,
    *, config: InversionConfig | None = None, const_kw: dict | None = None,
    sqr_kw: dict | None = None, regions: list | None = None,
) -> tuple[Register, Register]:
    """Add the selected classical point; preserve address and return coordinate handles.

    Registers are disjoint. Both coordinate differences must be nonzero, including
    in the dummy zero-digit branch; enabled output x must differ from the addend x.
    These algebraic conditions do not certify the tuned arithmetic distribution.
    """
    if not isinstance(table, WindowTable):
        raise ValueError("table must be a validated WindowTable")
    if not all(isinstance(r, Register) for r in (x, y, address)):
        raise TypeError("window operands must be Registers")
    n, p = len(x), table.p
    if len(y) != n or len(address) != table.bits or p >= 1 << n:
        raise ValueError("window operands require disjoint registers with matching widths")
    _validate_registers(circ, address, Register(list(x.ids) + list(y.ids)))
    kw = tuned_point_kwargs(n, p, config)
    const_kw = kw["const_kw"] if const_kw is None else dict(const_kw)
    sqr_kw = kw["sqr_kw"] if sqr_kw is None else dict(sqr_kw)
    inv_kw = kw["inv_kw"]

    if table.bits == 1:
        with _region(circ, regions, "arithmetic"):
            return add_const_affine_inplace(
                circ, x, y, *table.points[1], ctrl=address[0], p=p,
                const_kw=const_kw, sqr_kw=sqr_kw, inv_kw=inv_kw)

    points = (table.points[1],) + table.points[1:]
    xs, ys = tuple(pt[0] for pt in points), tuple(pt[1] for pt in points)
    f = (1 << n) - p
    ks = (0,) + tuple((f - 1 + 3 * xp) % p for xp in xs[1:])
    _table_sub(circ, address, x, xs, p, const_kw, regions)
    _table_sub(circ, address, y, ys, p, const_kw, regions)
    with _region(circ, regions, "arithmetic"):
        x, y = moddiv_inplace(circ, x, y, p, **inv_kw)
    with _enabled(circ, address, regions) as flag:
        with _region(circ, regions, "arithmetic"):
            x = modsqrsub_inplace(circ, y, x, p, ctrl=flag, **sqr_kw)
            for q in x:
                circ.cx(flag, q)
    _table_sub(circ, address, x, ks, p, const_kw, regions)
    with _region(circ, regions, "arithmetic"):
        x, y = modmul_inplace(circ, x, y, p, **inv_kw)
    _table_sub(circ, address, x, tuple((-v) % p for v in xs), p, const_kw, regions)
    _table_sub(circ, address, y, tuple((-v) % p for v in ys), p, const_kw, regions)
    with _enabled(circ, address, regions) as flag:
        with _region(circ, regions, "arithmetic"):
            for q in y:
                circ.cx(flag, q)
            sub_const_mod(circ, y, (f - 1) % p, p, ctrl=flag, **const_kw)
    return x, y


def add_window_affine_inplace_ref(x: int, y: int, digit: int,
                                 table: WindowTable) -> tuple[int, int]:
    """Ideal group-law residues, with the tuned arithmetic's algebraic exclusions."""
    p = table.p
    xp, yp = table.operand(digit)
    u, v = (x - xp) % p, (y - yp) % p
    assert u, "exceptional: input x equals selected (or dummy) x"
    assert v, "exceptional: zero numerator"
    if digit == 0:
        return x % p, y % p
    slope = v * pow(u, -1, p) % p
    xr = (slope*slope - x - xp) % p
    assert xr != xp, "exceptional: output x equals selected x"
    return xr, (slope*(xp - xr) - yp) % p


def build_window_case(bits: int = 4, *, table: WindowTable | None = None,
                      config: InversionConfig | None = None,
                      legacy_square_ripple: bool = False) -> Case:
    table = make_window_table(bits) if table is None else table
    if (table.bits != bits or
            (table.p, table.a, table.b) != (PRIME, curve.CURVE.A, curve.CURVE.B)):
        raise ValueError("benchmark table must match the active field/curve and window width")
    config = (InversionConfig.tuned(U, PRIME) if config is None else config).resolved(U, PRIME)
    circ = Circuit()
    digit = circ.input("j", bits)
    x, y = circ.input("x", U), circ.input("y", U)
    regions = []
    square = tuned_point_kwargs(U, PRIME, config, legacy_square_ripple=legacy_square_ripple)["sqr_kw"]
    x, y = add_window_affine_inplace(circ, x, y, digit, table, config=config,
                                     sqr_kw=square, regions=regions)
    circ.rebind("x", x)
    circ.rebind("y", y)
    return _window_case(circ, table, config, regions, square["ripple"])


def _window_case(circ: Circuit, table: WindowTable, config: InversionConfig,
                 regions: list, square_ripple: int) -> Case:
    bits = table.bits

    def ref(v):
        xr, yr = add_window_affine_inplace_ref(v["x"], v["y"], v["j"], table)
        return {"j": v["j"], "x": xr, "y": yr}

    def gen(rng: random.Random):
        while True:
            point = curve.mul(rng.randrange(1, curve.ORDER), curve.GX, curve.GY)
            if point is None:
                continue
            v = {"j": rng.randrange(1 << bits), "x": point[0], "y": point[1]}
            try:
                ref(v)
            except AssertionError:
                continue
            return v

    return Case(circ, ref, gen, meta={
        "empirical_only": True,
        "fail_analytic": None,
        "input_distribution": "independent prime-subgroup points and uniform window digits, conditioned only on algebraic preconditions",
        "arithmetic_contract": "tuned lazy incomplete affine; finite dummy T[1] for digit zero; no transferable analytic window-error model",
        "output_moduli": {"x": PRIME, "y": PRIME},
        "window_bits": bits,
        "lookup_backend": "unitary-prefix.v1",
        "table_fingerprint": table.fingerprint,
        "dummy_policy": "T[1]",
        "tuning": {"inversion": asdict(config), "point": {
            "carry_window": tuned_point_kwargs(U, PRIME, config)["const_kw"]["carry_window"],
            "comparator_bits": tuned_point_kwargs(U, PRIME, config)["const_kw"]["cmp_bits"],
            "square": "flattened-triangular", "square_ripple": square_ripple}},
        "resource_regions": regions,
    }, analysis_context={"window_table": table})


@circuit_version("scalarmul.add_window_affine_w4_inplace.v1")
def _build_window() -> Case:
    # Registered before the F-independent square ripple; pinned for reproducibility.
    return build_window_case(4, legacy_square_ripple=True)
