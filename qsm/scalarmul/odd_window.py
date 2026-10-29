"""Unregistered ONE-addition odd-only fixed-radix prototype, primarily Wei25519.

For the ORIGINAL unsigned w-bit word u, D=2u-(2**w-1) is always odd and
nonzero.  H=[2**-1 mod order]base is a known CLASSICAL point, and the physical
positive-magnitude table holds (2j+1)H.  Odd refers to the multiplier, not to
coordinate parity.  This is the same 2**(w-1) row count as the existing signed
construction at radix w/address w-1, NOT another table-halving optimization.

No zero enable, dummy row, endpoint correction, quantum halving, initialization,
or scalar schedule is implemented here.  The ideal group action is Q+[D]H;
it differs from Q+[u]base by the fixed public shift [-(2**w-1)]H.
The reused tuned affine arithmetic remains lazy/incomplete and can carry value,
dirty-ancilla and measurement-phase errors even at full width.  Conjugation is
exact p-y mod 2**n, not canonical negation of a lazy y>p.  Neither the reference
nor the sampler screens those circuit hazards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import random

from ..bench.registry import Case
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add
from ..field.add_chunked import gidney_add_chunked
from ..inversion.inplace import moddiv_inplace, modmul_inplace
from ..mul.accumulate import modsqrsub_inplace
from ..params import PRIME, U
from ..point import active as curve
from ..point.affine import sub_const_mod, tuned_point_kwargs
from ..point.group import scalar_mul
from .local_signed import FirstLookupTable, half_shift_recode_inplace, make_first_lookup_table
from .lookup import _validate_registers
from .signed_window import _group_add, conditional_negate_nonzero_inplace
from .window import _region
from .window_opt import (
    OptimizedWindowConfig, _packed_sub, _rebind_coordinates, _snapshot_bindings,
)


def _width(word_bits):
    if type(word_bits) is not int or word_bits < 2:
        raise ValueError("odd window word_bits must be an integer >= 2")


@dataclass(frozen=True)
class OddWindowTable:
    """Validated finite odd multiples of the known classical half-base.

    Primality of p and EXACT subgroup order are caller facts.  Validation checks
    nonsingularity, canonical on-curve points, odd order > largest odd digit,
    [order]base=O, and every row against (2j+1)H.  Thus an annihilating but
    non-exact order cannot silently admit an infinity row.  No primality/order
    proof or claim about odd coordinate values is made.
    """
    word_bits: int
    p: int
    a: int
    b: int
    base: tuple[int, int]
    order: int
    points: tuple[tuple[int, int], ...]

    def __post_init__(self):
        _width(self.word_bits)
        # Reuse the independently validated local odd-half-multiple contract,
        # without exposing its unsigned/initialization variants through this API.
        checked = FirstLookupTable(self.word_bits, True, self.p, self.a, self.b,
                                   self.base, self.order, self.points)
        for name in ("a", "b", "base", "points"):
            object.__setattr__(self, name, getattr(checked, name))

    @property
    def address_bits(self):
        return self.word_bits-1

    @property
    def bits(self):
        """Address width (not the original raw word width)."""
        return self.address_bits

    @property
    def entries(self):
        return 1 << self.address_bits

    @property
    def half_base(self):
        """H=[inverse2 mod order]base, already checked against physical row zero."""
        return self.points[0]

    @property
    def fingerprint(self):
        payload = ("odd-fixed-radix.v1", self.word_bits, self.p, self.a, self.b,
                   self.base, self.order, self.points)
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def digit(self, word: int) -> int:
        if type(word) is not int or not 0 <= word < 1 << self.word_bits:
            raise ValueError("original word must fit word_bits")
        return 2*word-((1 << self.word_bits)-1)

    def point_for_word(self, word: int) -> tuple[int, int]:
        """Independent classical group oracle, not a lookup/recoding replay."""
        k = self.digit(word)*pow(2, -1, self.order) % self.order
        point = scalar_mul(k, *self.base, p=self.p, a=self.a)
        if point is None:  # Also excluded by table validation and signed symmetry.
            raise ValueError("odd word unexpectedly maps to infinity")
        return point


def make_odd_window_table(word_bits: int = 16, base=None, *, order: int | None = None,
                          p: int = PRIME, a: int | None = None,
                          b: int | None = None) -> OddWindowTable:
    """Construct positive odd magnitudes; non-active curves require all constants."""
    _width(word_bits)
    table = make_first_lookup_table(word_bits, base, order=order, p=p, a=a, b=b,
                                    signed=True)
    return OddWindowTable(word_bits, table.p, table.a, table.b, table.base,
                          table.order, table.points)


def add_odd_window_inplace(
    circ: Circuit, x: Register, y: Register, word: Register, table: OddWindowTable, *,
    config: OptimizedWindowConfig | None = None,
    const_kw: dict | None = None, sqr_kw: dict | None = None,
    regions: list | None = None,
) -> tuple[Register, Register]:
    """Add [2u-(2**w-1)]H, restoring ORIGINAL word; return rebound coordinates.

    config.window_bits is w-1, the physical address width.  Exactly three positive
    paired lookup/unlookup stages operate on (X,Y), K, and (-X,-Y) field words;
    K=(2**n-p-1+3X) mod p at EVERY row, including row zero.  Sign acts only by
    accumulator conjugation.  Square and final negation are unconditional.

    Regions append disjoint (label,start_op,stop_op) spans, covering all emitted
    operations, with labels recoding/sign/lookup/arithmetic/unlookup.  No nested
    spans or setup inputs are counted.  Both forward and inverse recoding use
    label recoding; both conjugations use sign.  Named coordinate views are
    rebound after consuming arithmetic; callers must use the returned handles.
    """
    if not isinstance(table, OddWindowTable):
        raise ValueError("table must be a validated OddWindowTable")
    if not all(isinstance(r, Register) for r in (x, y, word)):
        raise TypeError("coordinate and original word operands must be Registers")
    n, p, bits = len(x), table.p, table.address_bits
    if len(y) != n or len(word) != table.word_bits or p >= 1 << n:
        raise ValueError("odd window operand widths must match")
    _validate_registers(circ, word, Register(x.ids+y.ids))
    if circ._cond_stack:
        raise ValueError("measured odd windows cannot run inside a classical condition")
    config = (OptimizedWindowConfig.tuned(n, p, bits) if config is None else
              config).resolved(n, p, bits)
    if config.operand_strategy != "paired":
        raise ValueError("odd prototype supports only the matched paired lookup strategy")
    kwargs = tuned_point_kwargs(n, p, config.inversion)
    const = dict(kwargs["const_kw"] if const_kw is None else const_kw)
    square = dict(kwargs["sqr_kw"] if sqr_kw is None else sqr_kw)
    const["add_fn"] = (gidney_add if config.adder_chunk is None else
                       partial(gidney_add_chunked, chunk=config.adder_chunk, cmp_bits=None))
    address, sign = word[:-1], word[-1]
    xs = tuple(pt[0] for pt in table.points)
    ys = tuple(pt[1] for pt in table.points)
    f = (1 << n)-p
    ks = tuple((f-1+3*xp) % p for xp in xs)

    def subtract(accumulators, columns):
        _packed_sub(circ, address, accumulators, columns, p, const,
                    config.unlookup_low_bits, regions)

    def rebind_call(fn):
        nonlocal x, y
        bindings = _snapshot_bindings(circ)
        old_x, old_y = x, y
        x, y = fn(circ, x, y, p, config=config.inversion)
        _rebind_coordinates(circ, bindings, old_x, old_y, x, y)

    with _region(circ, regions, "recoding"):
        half_shift_recode_inplace(circ, word)
    with _region(circ, regions, "sign"):
        conditional_negate_nonzero_inplace(circ, y, sign, p)
    subtract((x, y), (xs, ys))
    with _region(circ, regions, "arithmetic"):
        rebind_call(moddiv_inplace)
        bindings = _snapshot_bindings(circ)
        old_x, old_y = x, y
        x = modsqrsub_inplace(circ, y, x, p, ctrl=None, **square)
        _rebind_coordinates(circ, bindings, old_x, old_y, x, y)
        for q in x:
            circ.x(q)
    subtract((x,), (ks,))
    with _region(circ, regions, "arithmetic"):
        rebind_call(modmul_inplace)
    subtract((x, y), (tuple((-v) % p for v in xs), tuple((-v) % p for v in ys)))
    with _region(circ, regions, "arithmetic"):
        for q in y:
            circ.x(q)
        sub_const_mod(circ, y, (f-1) % p, p, ctrl=None, **const)
    with _region(circ, regions, "sign"):
        conditional_negate_nonzero_inplace(circ, y, sign, p)
    with _region(circ, regions, "recoding"):
        half_shift_recode_inplace(circ, word, inverse=True)
    return x, y


def odd_window_reference(x: int, y: int, word: int,
                         table: OddWindowTable) -> tuple[int, int]:
    """Ideal complete group result, with ONLY algebraic interface exclusions.

    No lazy trajectory, phase, or canonical-boundary checks are used to admit
    an input.  Consequently accepted inputs can fail the actual tuned circuit.
    """
    p = table.p
    if any(type(v) is not int or not 0 <= v < p for v in (x, y)):
        raise ValueError("reference input coordinates must be canonical")
    if (y*y-x*x*x-table.a*x-table.b) % p:
        raise ValueError("reference input must be on the declared curve")
    xp, yp = table.point_for_word(word)
    assert x != xp, "exceptional: input x equals selected x"
    assert y != yp, "exceptional: zero numerator"
    result = _group_add((x, y), (xp, yp), p, table.a)
    assert result is not None, "exceptional: point at infinity"
    assert result[0] != xp, "exceptional: output x equals selected x"
    return result


def build_odd_window_case(
    word_bits: int = 16, *, base=None, order: int | None = None,
    table: OddWindowTable | None = None, config: OptimizedWindowConfig | None = None,
    const_kw: dict | None = None, sqr_kw: dict | None = None,
) -> Case:
    """Unregistered one-addition case, inputs/outputs named word,x,y.

    Explicit toy tables and full-width tuned config, const_kw={}, sqr_kw={} are
    useful for correctness diagnostics, NOT a promise of coherent exactness.
    With an explicit table, base/order (if supplied) must agree with it.
    """
    _width(word_bits)
    table = (make_odd_window_table(word_bits, base, order=order)
             if table is None else table)
    if not isinstance(table, OddWindowTable) or table.word_bits != word_bits:
        raise ValueError("odd table must match the requested original word width")
    if base is not None and tuple(base) != table.base:
        raise ValueError("base disagrees with the supplied odd table")
    if order is not None and (type(order) is not int or order != table.order):
        raise ValueError("order disagrees with the supplied odd table")
    n, bits = (U if table.p == PRIME else table.p.bit_length()), table.address_bits
    config = (OptimizedWindowConfig.tuned(n, table.p, bits) if config is None else
              config).resolved(n, table.p, bits)
    circ = Circuit()
    word, x, y = circ.input("word", word_bits), circ.input("x", n), circ.input("y", n)
    regions = []
    x, y = add_odd_window_inplace(circ, x, y, word, table, config=config,
                                 const_kw=const_kw, sqr_kw=sqr_kw, regions=regions)
    circ.rebind("x", x)
    circ.rebind("y", y)

    def reference(values):
        xr, yr = odd_window_reference(values["x"], values["y"], values["word"], table)
        return {"word": values["word"], "x": xr, "y": yr}

    active = (table.p, table.a, table.b) == (PRIME, curve.CURVE.A, curve.CURVE.B)

    def gen_input(rng: random.Random):
        for _ in range(10000 if active else 10000*table.p):
            if active:
                point = scalar_mul(rng.randrange(1, curve.ORDER), curve.GX, curve.GY,
                                   p=table.p, a=table.a)
                if point is None:
                    continue
                xx, yy = point
            else:
                xx, yy = rng.randrange(table.p), rng.randrange(table.p)
                if (yy*yy-xx**3-table.a*xx-table.b) % table.p:
                    continue
            values = {"word": rng.randrange(1 << word_bits), "x": xx, "y": yy}
            try:
                reference(values)
            except AssertionError:
                continue
            return values
        raise ValueError("could not draw an algebraically admissible odd-window input")

    kw = tuned_point_kwargs(n, table.p, config.inversion)
    const = kw["const_kw"] if const_kw is None else dict(const_kw)
    square = kw["sqr_kw"] if sqr_kw is None else dict(sqr_kw)
    bound = (1 << word_bits)-1
    return Case(circ, reference, gen_input, meta={
        "empirical_only": True, "fail_analytic": None,
        "input_distribution": ("independent uniform active prime-subgroup points" if active else
                               "uniform finite on-curve toy points") +
                              " and uniform original words, conditioned only on algebraic preconditions",
        "input_domain": "canonical finite on-curve coordinates; nonzero selected x difference and numerator; output x differs from operand x",
        "arithmetic_contract": "tuned lazy incomplete affine; no transferable analytic odd-window error model",
        "sign_domain": "canonical nonzero accumulator y at both negation boundaries; 0 and p exchange; noncanonical y>p is an unresolved interface hazard",
        "output_representation": "lazy residues; not guaranteed canonical or phase-clean",
        "output_moduli": {"x": table.p, "y": table.p},
        "word_bits": word_bits, "radix_bits": word_bits, "address_bits": bits,
        "window_bits": bits, "table_entries": table.entries,
        "digit_interval": [-bound, bound], "digit_stride": 2,
        "digit_encoding": "D=2*original_word-(2**word_bits-1); sign=1-highbit; j=low if highbit else complement(low)",
        "oracle_identity": "output = input_point + [D]H = input_point + [original_word]base - [2**word_bits-1]H",
        "half_base": list(table.half_base), "quantum_halving": False,
        "oddness_contract": "odd nonzero scalar multipliers, NOT odd coordinates",
        "original_word_restored": True, "enable_policy": "unconditional; every addend nonzero",
        "dummy_policy": "none", "endpoint_correction": "none; folded K valid at every row",
        "comparison_baseline": "current signed construction at same radix_bits/address_bits/table_entries; no further table halving",
        "table_fingerprint": table.fingerprint,
        "curve": {"p": table.p, "a": table.a, "b": table.b,
                  "base": list(table.base), "order": table.order},
        "lookup_backend": "unary-and.v1", "sign_strategy": "conjugate",
        "circuit_config": {"unlookup_backend": "measured-onehot.v1", "operand_strategy": "paired",
                           "unlookup_low_bits": config.unlookup_low_bits, "adder_chunk": config.adder_chunk,
                           "lookup_stages": 3, "one_bit_specialization": False,
                           "square_control": None, "zero_enable": False, "endpoint_xor": False,
                           "sign_strategy": "conjugate", "sign_truncation": None},
        "tuning": {"inversion": asdict(config.inversion), "window": config.metadata(),
                   "point": {"carry_window": const.get("carry_window"), "comparator_bits": const.get("cmp_bits"),
                             "square": "flattened-triangular" if sqr_kw is None else "explicit-override",
                             "square_window": square.get("window"), "square_ripple": square.get("ripple"),
                             "square_comparator_bits": square.get("cmp_bits")}},
        "resource_regions": regions,
    }, analysis_context={"window_table": table, "odd_window_table": table})
