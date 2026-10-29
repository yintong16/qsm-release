"""Isolated carry-free recoding and finite shifted first-point preparation.

Only public curve points and a known odd subgroup order enter table generation.
There is no carry tape, scalar copy, discrete-log input, or registry mutation.
The first point is retained, NOT measured away. Its public shift must be included
in the final oracle identity; no correction point is silently applied here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from ..bench.registry import Case
from ..core.circuit import Circuit, Register
from ..params import PRIME
from ..point import active as curve
from ..point.group import scalar_mul
from .lookup import lookup_xor_unary
from .signed_recoding import twos_complement_to_sign_address_inplace
from .signed_window import _group_add, conditional_negate_nonzero_inplace
from .window import _region


TABLE_VERSION = "local-half-shift.v1"
LOOKUP_BACKEND = "unary-and.v1"


def _width(word_bits: int) -> None:
    if type(word_bits) is not int or word_bits < 1:
        raise ValueError("word_bits must be a positive integer")


def _word(circ: Circuit, word: Register, inverse: bool = False) -> None:
    if not isinstance(word, Register):
        raise TypeError("word must be a Register")
    _width(len(word))
    if type(inverse) is not bool:
        raise TypeError("inverse must be a bool")
    if any(type(q) is not int or q not in circ._live_qids for q in word):
        raise ValueError("word wires must be live integer ids")
    if len(set(word)) != len(word):
        raise ValueError("word wires must be distinct")
    if circ._cond_stack:
        raise ValueError("local recoding cannot run inside a classical condition")


def bias_recode_inplace(circ: Circuit, word: Register, inverse: bool = False) -> None:
    """u -> endpoint-aware sign/address for d=u-2**(w-1); inverse restores u.

    Low bits hold abs(d) modulo B, sign is the MSB. Thus (sign,address)=(1,0)
    means -B, not negative zero. All transient CDKM padding/carry is cleaned;
    there is no persistent scratch. For w=1 the sign alone encodes -1 or 0.
    """
    _word(circ, word, inverse)
    if not inverse:
        circ.x(word[-1])
    if len(word) > 1:
        twos_complement_to_sign_address_inplace(circ, word)
    if inverse:
        circ.x(word[-1])


def half_shift_recode_inplace(circ: Circuit, word: Register,
                               inverse: bool = False) -> None:
    """Encode u-B+1/2 as (-1)**sign*(address+1/2), with no scratch.

    sign=1-original_highbit; address=low if highbit else complement(low).
    This is NOT the integer bias recoder: its negative branch has no increment.
    """
    _word(circ, word, inverse)
    if not inverse:
        circ.x(word[-1])
    for bit in word[:-1]:
        circ.cx(word[-1], bit)
    if inverse:
        circ.x(word[-1])


@dataclass(frozen=True)
class FirstLookupTable:
    """Validated odd-half multiples of a public base in its odd-order subgroup.

    Primality of p and exactness of the supplied known order are caller facts.
    We additionally check [order]base=O and EVERY physical table point, so even
    an annihilating non-exact order cannot silently introduce an infinity row.
    """
    word_bits: int
    signed: bool
    p: int
    a: int
    b: int
    base: tuple[int, int]
    order: int
    points: tuple[tuple[int, int], ...]

    def __post_init__(self):
        _width(self.word_bits)
        if type(self.signed) is not bool:
            raise TypeError("signed must be a bool")
        if type(self.p) is not int or self.p <= 3 or not self.p & 1:
            raise ValueError("p must be an odd prime greater than three")
        if type(self.a) is not int or type(self.b) is not int:
            raise ValueError("curve coefficients must be integers")
        object.__setattr__(self, "a", self.a % self.p)
        object.__setattr__(self, "b", self.b % self.p)
        if (4*self.a**3 + 27*self.b**2) % self.p == 0:
            raise ValueError("curve must be nonsingular")
        if type(self.order) is not int or self.order < 3 or not self.order & 1:
            raise ValueError("known subgroup order must be odd and greater than one")
        if 2*self.entries-1 >= self.order:
            raise ValueError("largest odd table multiplier must be less than order")
        base = self._point(self.base)
        object.__setattr__(self, "base", base)
        if scalar_mul(self.order, *base, p=self.p, a=self.a) is not None:
            raise ValueError("declared order does not annihilate base")
        points = tuple(self._point(pt) for pt in self.points)
        if len(points) != self.entries:
            raise ValueError("table length does not match word_bits and signed mode")
        object.__setattr__(self, "points", points)
        expected = self.half_base
        for point in points:
            if point != expected:
                raise ValueError("table must contain (2j+1)*(base/2)")
            expected = _group_add(expected, base, self.p, self.a)

    def _point(self, point):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("base and every table entry must be finite points")
        if any(type(v) is not int or not 0 <= v < self.p for v in point):
            raise ValueError("point coordinates must be canonical integers")
        x, y = point
        if (y*y-x**3-self.a*x-self.b) % self.p:
            raise ValueError("point is not on the declared curve")
        if y == 0:
            raise ValueError("odd-subgroup table points must have nonzero y")
        return x, y

    @property
    def address_bits(self) -> int:
        return self.word_bits-int(self.signed)

    @property
    def entries(self) -> int:
        return 1 << self.address_bits

    @property
    def bias(self) -> int:
        return (1 << (self.word_bits-1)) if self.signed else 0

    @property
    def half_base(self) -> tuple[int, int]:
        return scalar_mul((self.order+1)//2, *self.base, p=self.p, a=self.a)

    @property
    def shift_numerator(self) -> int:
        return 1-2*self.bias

    @property
    def shift_denominator(self) -> int:
        return 2

    @property
    def shift_scalar(self) -> int:
        return (self.shift_numerator * ((self.order+1)//2)) % self.order

    @property
    def shift_point(self) -> tuple[int, int]:
        return scalar_mul(self.shift_scalar, *self.base, p=self.p, a=self.a)

    @property
    def fingerprint(self) -> str:
        payload = (TABLE_VERSION, self.word_bits, self.signed, self.p, self.a,
                   self.b, self.base, self.order, self.points)
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def point_for_word(self, word: int) -> tuple[int, int]:
        if type(word) is not int or not 0 <= word < 1 << self.word_bits:
            raise ValueError("word does not fit word_bits")
        # Independent complete classical multiplication, not an address decoder.
        point = scalar_mul((word+self.shift_scalar) % self.order, *self.base,
                           p=self.p, a=self.a)
        if point is None:
            raise ValueError("shifted word unexpectedly maps to infinity")
        return point

    def metadata(self) -> dict:
        return {
            "table_version": TABLE_VERSION, "table_fingerprint": self.fingerprint,
            "word_bits": self.word_bits, "address_bits": self.address_bits,
            "table_entries": self.entries, "signed": self.signed, "bias": self.bias,
            "curve": {"p": self.p, "a": self.a, "b": self.b,
                      "base": list(self.base), "order": self.order},
            "half_base": list(self.half_base),
            "shift_numerator": self.shift_numerator, "shift_denominator": 2,
            "shift_scalar": self.shift_scalar, "shift_point": list(self.shift_point),
            "oracle_identity": "output = [original_word]base + shift_point",
            "correction_applied": False,
        }


def make_first_lookup_table(word_bits: int, base=None, *, order: int | None = None,
                            p: int = PRIME, a: int | None = None,
                            b: int | None = None, signed: bool = True) -> FirstLookupTable:
    """Signed B-row table for (u-B+1/2)P, or unsigned 2**w-row (u+1/2)P.

    Supply the actual (possibly short top) word width and already-weighted public
    base for a scalar schedule. No knowledge of a secret scalar is required.
    """
    _width(word_bits)
    if type(signed) is not bool:
        raise TypeError("signed must be a bool")
    if p != PRIME and any(v is None for v in (base, order, a, b)):
        raise ValueError("non-active fields require explicit base, order, a and b")
    base = (curve.GX, curve.GY) if base is None else base
    order = curve.ORDER if order is None else order
    a = curve.CURVE.A if a is None else a
    b = curve.CURVE.B if b is None else b
    # Validate public constants before doing group arithmetic or allocating a table.
    if (type(p) is not int or p <= 3 or not p & 1 or type(a) is not int or
            type(b) is not int or type(order) is not int or order < 3 or not order & 1):
        raise ValueError("invalid field, coefficients or odd subgroup order")
    if (not isinstance(base, (tuple, list)) or len(base) != 2 or
            any(type(v) is not int or not 0 <= v < p for v in base)):
        raise ValueError("base must be a finite canonical point")
    if (base[1]**2-base[0]**3-a*base[0]-b) % p or (4*a**3+27*b**2) % p == 0:
        raise ValueError("base must be on the declared nonsingular curve")
    entries = 1 << (word_bits-int(signed))
    if 2*entries-1 >= order:
        raise ValueError("largest odd table multiplier must be less than order")
    if scalar_mul(order, *base, p=p, a=a) is not None:
        raise ValueError("declared order does not annihilate base")
    half = scalar_mul((order+1)//2, *base, p=p, a=a)
    points, point = [], half
    for _ in range(entries):
        if point is None:
            raise ValueError("every first-lookup table entry must remain finite")
        points.append(point)
        point = _group_add(point, tuple(base), p, a)
    return FirstLookupTable(word_bits, signed, p, a, b, tuple(base), order, tuple(points))


def _prepare(circ, word, x, y, table, regions):
    with _region(circ, regions, "recode"):
        if table.signed:
            half_shift_recode_inplace(circ, word)
    with _region(circ, regions, "lookup"):
        target = Register(x.ids+y.ids)
        values = tuple(xx | (yy << len(x)) for xx, yy in table.points)
        if table.address_bits:
            lookup_xor_unary(circ, word[:table.address_bits], target, values)
        else:
            circ.xor_word(target, values[0])
    with _region(circ, regions, "sign"):
        if table.signed:
            conditional_negate_nonzero_inplace(circ, y, word[-1], table.p)
    with _region(circ, regions, "restore_word"):
        if table.signed:
            half_shift_recode_inplace(circ, word, inverse=True)
    return x, y


def _validate_initialization(circ, word, table):
    _word(circ, word)
    if not isinstance(table, FirstLookupTable) or len(word) != table.word_bits:
        raise ValueError("first-lookup table must match word width")


def initialize_first_lookup(circ: Circuit, word: Register, table: FirstLookupTable,
                            *, regions: list | None = None) -> tuple[Register, Register]:
    """Allocate zero x/y, prepare the shifted point, and restore the original word.

    Returned canonical coordinates remain live, with no output measurement or
    measured unlookup. Only decoder/negation scratch is consumed and phase-clean.
    """
    _validate_initialization(circ, word, table)
    n = table.p.bit_length()
    x, y = circ.alloc(n), circ.alloc(n)
    return _prepare(circ, word, x, y, table, regions)


def build_first_lookup_case(word_bits: int = 5, *, table: FirstLookupTable | None = None,
                            signed: bool = True) -> Case:
    """Emitted first lookup with named x=y=0 inputs, retained as point outputs.

    Named zero destinations accommodate Case's input/output naming convention;
    integration should use initialize_first_lookup, which allocates zeros itself.
    """
    _width(word_bits)
    if type(signed) is not bool:
        raise TypeError("signed must be a bool")
    table = make_first_lookup_table(word_bits, signed=signed) if table is None else table
    if (not isinstance(table, FirstLookupTable) or table.word_bits != word_bits or
            table.signed != signed):
        raise ValueError("first-lookup table must match requested width and signed mode")
    circ = Circuit()
    word = circ.input("word", word_bits)
    x, y = circ.input("x", table.p.bit_length()), circ.input("y", table.p.bit_length())
    regions = []
    _prepare(circ, word, x, y, table, regions)

    def reference(values):
        if values["x"] != 0 or values["y"] != 0:
            raise ValueError("first-lookup destinations must initially be zero")
        xx, yy = table.point_for_word(values["word"])
        return {"word": values["word"], "x": xx, "y": yy}

    return Case(circ, reference,
                lambda rng: {"word": rng.randrange(1 << word_bits), "x": 0, "y": 0},
                meta={**table.metadata(), "lookup_backend": LOOKUP_BACKEND,
                      "fail_analytic": 0.0, "input_domain": "unsigned word; x=y=0",
                      "input_distribution": "uniform original unsigned word",
                      "output_representation": "canonical finite affine point",
                      "phase_contract": "exact for every measurement outcome",
                      "point_output_retained": True, "measured_output_unlookup": False,
                      "resource_regions": regions},
                analysis_context={"first_lookup_table": table})


def build_bias_recode_case(word_bits: int = 5, *, roundtrip: bool = False) -> Case:
    """Emitted in-place integer recoder, optionally followed by its true inverse."""
    _width(word_bits)
    if type(roundtrip) is not bool:
        raise TypeError("roundtrip must be a bool")
    circ = Circuit()
    word = circ.input("word", word_bits)
    bias_recode_inplace(circ, word)
    if roundtrip:
        bias_recode_inplace(circ, word, inverse=True)
    bound = 1 << (word_bits-1)

    def reference(values):
        u = values["word"]
        if type(u) is not int or not 0 <= u < 1 << word_bits:
            raise ValueError("word does not fit word_bits")
        digit = u-bound
        packed = (int(digit < 0) << (word_bits-1)) | (abs(digit) % bound)
        return {"word": u if roundtrip else packed}

    return Case(circ, reference, lambda rng: {"word": rng.randrange(1 << word_bits)},
                meta={"word_bits": word_bits, "bias": bound, "roundtrip": roundtrip,
                      "digit_interval": [-bound, bound-1], "fail_analytic": 0.0,
                      "digit_encoding": "sign MSB; abs(d) mod B low bits; (1,0)=-B",
                      "persistent_scratch": 0, "input_distribution": "uniform unsigned word"})
