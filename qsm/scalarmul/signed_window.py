"""Unregistered signed-magnitude, three-lookup affine-window experiment.

Physical T[0]=[2**a]P and T[j]=[j]P. Logical (s,j)=(0,0) is zero,
(1,0) is the negative endpoint, and other digits are (-1)**s*j.

Lookup, sign and selection operations are exact on their stated domains. The
point arithmetic remains tuned, lazy and incomplete: full-width tuning does NOT
repair its inherited measurement-phase errors. In particular conjugation uses
p-y mod 2**n, a bitspace involution agreeing with canonical nonzero field
negation. A noncanonical intermediate y>p can therefore cause a NEW value error;
this is an explicit interface hazard, not something the sampler filters away.
No whole-window analytic failure model or uniform-success claim is supplied.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import random

from ..bench.registry import Case
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add
from ..field.add_chunked import gidney_add_chunked
from ..field.canonical import neg_inplace as canonical_neg_inplace
from ..field.constadd import sub_const_inplace
from ..field.modarith import modsub_direct
from ..inversion.inplace import moddiv_inplace, modmul_inplace
from ..inversion.zero_safe import moddiv_zero_safe_inplace, modmul_zero_safe_inplace
from ..mul.accumulate import modsqrsub_inplace
from ..params import PRIME, U
from ..point import active as curve
from ..point.affine import sub_const_mod, tuned_point_kwargs
from ..point.group import scalar_mul
from .lookup import _validate_registers, lookup_xor_unary, unlookup_measure, xor_nonzero
from .window import _region
from .window_opt import OptimizedWindowConfig, _rebind_coordinates, _snapshot_bindings


def _group_add(pt, qt, p, a):
    """Complete classical group law, independent of the lazy circuit schedule."""
    if pt is None:
        return qt
    if qt is None:
        return pt
    x, y = pt
    xx, yy = qt
    if x == xx:
        if (y + yy) % p == 0:
            return None
        slope = (3*x*x + a) * pow(2*y, -1, p) % p
    else:
        slope = (yy-y) * pow(xx-x, -1, p) % p
    xr = (slope*slope - x - xx) % p
    return xr, (slope*(x-xr) - y) % p


@dataclass(frozen=True)
class SignedWindowTable:
    """Validated finite magnitude multiples; the endpoint is NOT a logical zero.

    As with the other curve-table APIs, primality of p is a caller assumption.
    Every entry, including the endpoint, is checked on-curve and against [j]P.
    """
    p: int
    a: int
    b: int
    points: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if type(self.p) is not int or self.p <= 3 or not self.p & 1:
            raise ValueError("signed table requires an odd prime greater than three")
        if type(self.a) is not int or type(self.b) is not int:
            raise ValueError("curve coefficients must be integers")
        object.__setattr__(self, "a", self.a % self.p)
        object.__setattr__(self, "b", self.b % self.p)
        if (4*self.a**3 + 27*self.b**2) % self.p == 0:
            raise ValueError("signed table requires a nonsingular curve")
        points = tuple(None if pt is None else tuple(pt) for pt in self.points)
        object.__setattr__(self, "points", points)
        if len(points) < 2 or len(points) & (len(points)-1):
            raise ValueError("signed table length must be 2**a with a>=1")
        for pt in points:
            if pt is None or len(pt) != 2:
                raise ValueError("every signed table entry must be finite")
            if any(type(v) is not int or not 0 <= v < self.p for v in pt):
                raise ValueError("signed table coordinates must be canonical integers")
            x, y = pt
            if (y*y - x*x*x - self.a*x - self.b) % self.p:
                raise ValueError("signed table entry is not on the declared curve")
        point = None
        for j in range(1, len(points)+1):
            point = _group_add(point, points[1], self.p, self.a)
            if point != points[j % len(points)]:
                raise ValueError("signed table must store T[0]=[B]P, T[j]=[j]P")

    @property
    def bits(self):
        return (len(self.points)-1).bit_length()

    @property
    def fingerprint(self):
        encoded = json.dumps(("signed-endpoint.v1", self.p, self.a, self.b, self.points),
                             separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def digit(self, address: int, sign: int) -> int:
        if type(address) is not int or not 0 <= address < len(self.points):
            raise ValueError("signed address is out of range")
        if type(sign) is not int or sign not in (0, 1):
            raise ValueError("sign must be a bit")
        return (-len(self.points)*sign if address == 0 else
                (-address if sign else address))

    def operand(self, address: int, sign: int = 0) -> tuple[int, int]:
        """Finite arithmetic operand, including the dummy on logical zero."""
        self.digit(address, sign)
        x, y = self.points[address]
        return x, (-y) % self.p if sign else y


def make_signed_window_table(address_bits: int, base=None, *, p: int = PRIME,
                             a: int | None = None, b: int | None = None):
    """Build sequential magnitude multiples, without assuming a subgroup order."""
    if type(address_bits) is not int or address_bits < 1:
        raise ValueError("address_bits must be a positive integer")
    if (base is None or a is None or b is None) and p != PRIME:
        raise ValueError("non-active fields require explicit base and curve coefficients")
    base = (curve.GX, curve.GY) if base is None else tuple(base)
    a = curve.CURVE.A if a is None else a
    b = curve.CURVE.B if b is None else b
    if (type(p) is not int or p <= 3 or not p & 1 or len(base) != 2 or
            any(type(v) is not int or not 0 <= v < p for v in base) or
            type(a) is not int or type(b) is not int):
        raise ValueError("invalid signed base/field")
    if (base[1]**2-base[0]**3-a*base[0]-b) % p:
        raise ValueError("base is not on the declared curve")
    multiples, point = [], None
    for _ in range(1 << address_bits):
        point = _group_add(point, base, p, a)
        if point is None:
            raise ValueError("signed magnitude multiples must remain finite")
        multiples.append(point)
    return SignedWindowTable(p, a, b, (multiples[-1], *multiples[:-1]))


def conditional_negate_nonzero_inplace(circ: Circuit, value: Register,
                                       sign: int, p: int) -> Register:
    """Exact controlled p-y mod 2**n; field negation on 1<=y<p.

    This is an involution on ALL n-bit words, with exact measurement cleanup.
    Zero and p exchange (both represent residue zero). Words above p generally
    do not map to their field negatives. No truncated carry/comparison is used.
    """
    _validate_registers(circ, Register([sign]), value)
    if type(p) is not int or not 1 < p < 1 << len(value) or not p & 1:
        raise ValueError("negation requires an odd modulus fitting the register")
    if circ._cond_stack:
        raise ValueError("sign primitive cannot run inside a classical condition")
    for q in value:
        circ.cx(sign, q)
    sub_const_inplace(circ, value, (1 << len(value))-p-1, ctrl=sign, window=None)
    return value


@contextmanager
def _signed_enabled(circ, address, sign, regions=None):
    with _region(circ, regions, "selection"):
        flag = circ.alloc(1)
        combined = Register(list(address.ids) + [sign])
        xor_nonzero(circ, combined, flag[0])
    yield flag[0]
    with _region(circ, regions, "selection"):
        xor_nonzero(circ, combined, flag[0])
        circ.free(flag)


def _endpoint_xor(circ, address, sign, target, value):
    """XOR a constant iff sign AND address==0; erase all flags unitarily."""
    zero, endpoint = circ.alloc(1), circ.alloc(1)
    circ.x(zero[0])
    xor_nonzero(circ, address, zero[0])
    circ.ccx(sign, zero[0], endpoint[0])
    circ.xor_word(tuple(target), value, ctrl=endpoint[0])
    circ.ccx(sign, zero[0], endpoint[0])
    xor_nonzero(circ, address, zero[0])
    circ.x(zero[0])
    circ.free(endpoint)
    circ.free(zero)


def _signed_packed_sub(circ, address, sign, accumulators, columns, p, const,
                       low_bits, regions, *, sign_y=False, endpoint=None):
    """Restore selected Y / endpoint XOR before measured base-table unlookup."""
    n = len(accumulators[0])
    words = tuple(sum(column[j] << (i*n) for i, column in enumerate(columns))
                  for j in range(1 << len(address)))
    with _region(circ, regions, "lookup"):
        operand = circ.alloc(n*len(columns))
        lookup_xor_unary(circ, address, operand, words)

    def adjust():
        if sign_y:
            with _region(circ, regions, "sign"):
                yy = operand[n:2*n]
                # Loaded words are canonical by construction. Zero table words
                # need the true canonical 0->0 map rather than 0->p.
                if 0 in columns[1]:
                    canonical_neg_inplace(circ, yy, p, ctrl=sign)
                else:
                    conditional_negate_nonzero_inplace(circ, yy, sign, p)
        if endpoint is not None:
            with _region(circ, regions, "selection"):
                _endpoint_xor(circ, address, sign, operand, endpoint)

    adjust()
    with _region(circ, regions, "arithmetic"):
        for i, acc in enumerate(accumulators):
            modsub_direct(circ, operand[i*n:(i+1)*n], acc, p, **const)
    adjust()  # Involutions; unlookup must see the UNSIGNED BASE word.
    with _region(circ, regions, "unlookup"):
        unlookup_measure(circ, address, operand, words, low_bits=low_bits)


def add_signed_window_inplace(
    circ: Circuit, x: Register, y: Register, address: Register,
    sign: Register | int, table: SignedWindowTable, *,
    sign_strategy: str = "conjugate", config: OptimizedWindowConfig | None = None,
    const_kw: dict | None = None, sqr_kw: dict | None = None,
    regions: list | None = None, zero_safe: bool = False,
) -> tuple[Register, Register]:
    """Add the signed selected point; preserve address/sign and rebind coordinates.

    Three paired QROM stages, also for one address bit (no unsigned shortcut).
    All algebraic incomplete-addition exclusions still apply to the signed
    operand, including the finite dummy when disabled. Conjugation additionally
    requires canonical nonzero y at both sign boundaries for canonical field
    negation. Lazy tuned trajectories are NOT checked or repaired at runtime.
    Caller must use the returned handles when constructing subsequent windows.

    zero_safe=True permits literal-zero numerators with the existing replay tags
    and canonical positive restoration. It supports conjugate signs only, requires
    canonical restoration payloads, and retains upstream approximation hazards.
    """
    if type(zero_safe) is not bool:
        raise ValueError("zero_safe must be a bool")
    if zero_safe and sign_strategy != "conjugate":
        raise ValueError("zero-safe signed windows require conjugate signs")
    if not isinstance(table, SignedWindowTable):
        raise ValueError("table must be a validated SignedWindowTable")
    if not all(isinstance(r, Register) for r in (x, y, address)):
        raise TypeError("coordinate and address operands must be Registers")
    if isinstance(sign, Register):
        if len(sign) != 1:
            raise ValueError("sign register must contain one bit")
        sign = sign[0]
    n, p, bits = len(x), table.p, table.bits
    if len(y) != n or len(address) != bits or p >= 1 << n:
        raise ValueError("signed window operand widths must match")
    _validate_registers(circ, Register(address.ids + [sign]), Register(x.ids + y.ids))
    if circ._cond_stack:
        raise ValueError("measured signed windows cannot run inside a classical condition")
    if sign_strategy not in ("conjugate", "selected"):
        raise ValueError("sign_strategy must be conjugate or selected")
    config = (OptimizedWindowConfig.tuned(n, p, bits) if config is None else
              config).resolved(n, p, bits)
    if config.operand_strategy != "paired":
        raise ValueError("signed prototype supports only the matched paired lookup strategy")
    kwargs = tuned_point_kwargs(n, p, config.inversion)
    const = dict(kwargs["const_kw"] if const_kw is None else const_kw)
    square = dict(kwargs["sqr_kw"] if sqr_kw is None else sqr_kw)
    const["add_fn"] = (gidney_add if config.adder_chunk is None else
                       partial(gidney_add_chunked, chunk=config.adder_chunk, cmp_bits=None))
    xs, ys = tuple(pt[0] for pt in table.points), tuple(pt[1] for pt in table.points)
    f = (1 << n)-p
    ks = (0,) + tuple((f-1+3*v) % p for v in xs[1:])
    endpoint = (f-1+3*xs[0]) % p

    def sign_accumulator():
        if sign_strategy == "conjugate":
            with _region(circ, regions, "sign"):
                conditional_negate_nonzero_inplace(circ, y, sign, p)

    def subtract(accumulators, columns, **kw):
        _signed_packed_sub(circ, address, sign, accumulators, columns, p, const,
                           config.unlookup_low_bits, regions, **kw)

    def rebind_call(fn, **kw):
        nonlocal x, y
        bindings = _snapshot_bindings(circ)
        old_x, old_y = x, y
        x, y = fn(circ, x, y, p, **kw)
        _rebind_coordinates(circ, bindings, old_x, old_y, x, y)

    sign_accumulator()
    subtract((x, y), (xs, ys), sign_y=sign_strategy == "selected")
    with _region(circ, regions, "arithmetic"):
        rebind_call(moddiv_zero_safe_inplace if zero_safe else moddiv_inplace,
                    config=config.inversion)
    with _signed_enabled(circ, address, sign, regions) as enabled:
        with _region(circ, regions, "arithmetic"):
            bindings = _snapshot_bindings(circ)
            old_x, old_y = x, y
            x = modsqrsub_inplace(circ, y, x, p, ctrl=enabled, **square)
            _rebind_coordinates(circ, bindings, old_x, old_y, x, y)
            for q in x:
                circ.cx(enabled, q)
    subtract((x,), (ks,), endpoint=endpoint)
    with _region(circ, regions, "arithmetic"):
        rebind_call(modmul_zero_safe_inplace if zero_safe else modmul_inplace,
                    config=config.inversion)
    if zero_safe:
        from .zero_point import _restore_positive
        _restore_positive(circ, address, x, y, xs, ys, p, const["add_fn"],
                          config.unlookup_low_bits, regions, unlookup="paired")
    else:
        subtract((x, y), (tuple((-v) % p for v in xs), tuple((-v) % p for v in ys)),
                 sign_y=sign_strategy == "selected")
    with _signed_enabled(circ, address, sign, regions) as enabled:
        with _region(circ, regions, "arithmetic"):
            for q in y:
                circ.cx(enabled, q)
            sub_const_mod(circ, y, (f-1) % p, p, ctrl=enabled, **const)
    sign_accumulator()
    return x, y


def signed_window_reference(x: int, y: int, address: int, sign: int,
                            table: SignedWindowTable, *, zero_safe: bool = False) -> tuple[int, int]:
    """Ideal complete group result, rejecting ONLY algebraic interface exceptions.

    Deliberately does not imitate or screen lazy arithmetic/sign trajectories.
    Consequently an accepted reference input can still fail the actual circuit.
    """
    p = table.p
    if any(type(v) is not int or not 0 <= v < p for v in (x, y)):
        raise ValueError("reference input coordinates must be canonical")
    if (y*y-x*x*x-table.a*x-table.b) % p:
        raise ValueError("reference input must be on the declared curve")
    digit = table.digit(address, sign)
    xp, yp = table.operand(address, sign)
    assert x != xp, "exceptional: input x equals selected (or dummy) x"
    if not zero_safe:
        assert y != yp, "exceptional: zero numerator"
    if digit == 0:
        return x, y
    # Use the complete group operation, not the complement/folded-K schedule.
    result = _group_add((x, y), (xp, yp), p, table.a)
    assert result is not None, "exceptional: point at infinity"
    assert result[0] != xp, "exceptional: output x equals selected x"
    return result


def build_signed_window_case(
    address_bits: int = 4, *, sign_strategy: str = "conjugate",
    table: SignedWindowTable | None = None, config: OptimizedWindowConfig | None = None,
    const_kw: dict | None = None, sqr_kw: dict | None = None,
    zero_safe: bool = False,
) -> Case:
    """Unregistered complete circuit; default active field, explicit toy tables allowed.

    For small tests pass an explicit field-resolved full-width tuned inversion
    configuration and const_kw={}, sqr_kw={}. This removes truncations but NOT
    inherited lazy-arithmetic measurement phases; it is not a conservative mode.
    """
    if type(zero_safe) is not bool:
        raise ValueError("zero_safe must be a bool")
    if zero_safe and sign_strategy != "conjugate":
        raise ValueError("zero-safe signed windows require conjugate signs")
    table = make_signed_window_table(address_bits) if table is None else table
    if not isinstance(table, SignedWindowTable) or table.bits != address_bits:
        raise ValueError("signed table must match the requested address width")
    n = U if table.p == PRIME else table.p.bit_length()
    config = (OptimizedWindowConfig.tuned(n, table.p, address_bits) if config is None else
              config).resolved(n, table.p, address_bits)
    circ = Circuit()
    j, s = circ.input("j", address_bits), circ.input("s", 1)
    x, y = circ.input("x", n), circ.input("y", n)
    regions = []
    x, y = add_signed_window_inplace(circ, x, y, j, s, table,
                                    sign_strategy=sign_strategy, config=config,
                                    const_kw=const_kw, sqr_kw=sqr_kw, regions=regions,
                                    zero_safe=zero_safe)
    circ.rebind("x", x)
    circ.rebind("y", y)

    def reference(values):
        xr, yr = signed_window_reference(values["x"], values["y"], values["j"],
                                         values["s"], table, zero_safe=zero_safe)
        return {"j": values["j"], "s": values["s"], "x": xr, "y": yr}

    active = (table.p, table.a, table.b) == (PRIME, curve.CURVE.A, curve.CURVE.B)

    def gen_input(rng: random.Random):
        # Coordinate-pair rejection accepts only about one proposal in p.
        attempts = 10000 if active else 10000 * table.p
        for _ in range(attempts):
            if active:
                point = scalar_mul(rng.randrange(1, curve.ORDER), curve.GX, curve.GY,
                                   p=table.p, a=table.a)
            else:
                # Toy-only sampling law: uniform on-curve finite points by
                # rejection from coordinate pairs, no subgroup/order assumption.
                xx, yy = rng.randrange(table.p), rng.randrange(table.p)
                if (yy*yy-xx**3-table.a*xx-table.b) % table.p:
                    continue
                point = xx, yy
            if point is None:
                continue
            values = {"j": rng.randrange(1 << address_bits), "s": rng.randrange(2),
                      "x": point[0], "y": point[1]}
            try:
                reference(values)
            except AssertionError:
                continue
            return values
        raise ValueError("could not draw an algebraically admissible signed-window input")

    kw = tuned_point_kwargs(n, table.p, config.inversion)
    const = kw["const_kw"] if const_kw is None else dict(const_kw)
    square = kw["sqr_kw"] if sqr_kw is None else dict(sqr_kw)
    metadata = {
        "empirical_only": True, "fail_analytic": None,
        "input_distribution": ("independent uniform prime-subgroup points" if active else
                               "uniform finite on-curve toy points") +
                              " and uniform sign/address states, conditioned only on algebraic preconditions",
        "arithmetic_contract": "tuned lazy incomplete affine; no transferable analytic signed-window error model",
        "input_domain": "canonical finite on-curve coordinates; nonzero signed/dummy x difference and numerator; enabled output x differs from operand x",
        "sign_domain": ("canonical nonzero accumulator y at both negation boundaries; 0 and p exchange; noncanonical y>p is an unresolved interface hazard"
                        if sign_strategy == "conjugate" else
                        "canonical selected Y and -Y words, restored exactly before measured unlookup; zero words use exact canonical negation"),
        "output_representation": "lazy residues; not guaranteed canonical or phase-clean",
        "output_moduli": {"x": table.p, "y": table.p},
        "window_bits": address_bits, "address_bits": address_bits,
        "radix_bits": address_bits+1, "table_entries": len(table.points),
        "digit_interval": [-len(table.points), len(table.points)-1],
        "digit_encoding": "j!=0: (-1)**s*j; j==0: -s*2**address_bits",
        "enable_policy": "s OR (j!=0)",
        "dummy_policy": "T[0]=[2**address_bits]P for (s,j)=(0,0); real negative endpoint for (1,0)",
        "endpoint_correction": "base K[0]=0; XOR endpoint K iff s AND j==0; restored before measured unlookup",
        "table_fingerprint": table.fingerprint,
        "curve": {"p": table.p, "a": table.a, "b": table.b, "base": list(table.points[1])},
        "lookup_backend": "unary-and.v1", "sign_strategy": sign_strategy,
        "circuit_config": {"unlookup_backend": "measured-onehot.v1", "operand_strategy": "paired",
                           "unlookup_low_bits": config.unlookup_low_bits, "adder_chunk": config.adder_chunk,
                           "sign_strategy": sign_strategy, "sign_truncation": None,
                           "lookup_stages": 3, "one_bit_specialization": False},
        "tuning": {"inversion": asdict(config.inversion), "window": config.metadata(),
                   "point": {"carry_window": const.get("carry_window"), "comparator_bits": const.get("cmp_bits"),
                             "square": "flattened-triangular" if sqr_kw is None else "explicit-override",
                             "square_window": square.get("window"), "square_ripple": square.get("ripple"),
                             "square_comparator_bits": square.get("cmp_bits")}},
        "resource_regions": regions,
    }
    if zero_safe:
        metadata.update({
            "zero_safe": True,
            "input_domain": "canonical finite on-curve coordinates; nonzero signed/dummy x difference; enabled output x differs from operand x; zero numerator allowed",
            "zero_contract": "literal 0 only; literal p zero representation NOT supported",
            "arithmetic_contract": "tuned incomplete affine with canonical-zero replay tags and positive restoration; inherited hazards retained",
            "coordinate_restoration": "add",
            "restoration_adder": "modadd.borrowed_pseudo_mersenne",
            "restoration_unlookup": "paired",
            "restoration_exact_reduction": True,
            "restoration_carry_window": None,
            "restoration_comparator_bits": None,
            "restoration_contract": "canonical operands required; exact local addition; upstream lazy/phase failures remain",
            "phase_correction": False,
            "output_representation": "canonical restoration on canonical payloads and nonzero-y sign domain; upstream approximate hazards retained",
        })
    return Case(circ, reference, gen_input, meta=metadata,
                analysis_context={"window_table": table, "signed_window_table": table})
