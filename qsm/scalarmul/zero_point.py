"""Odd point additions with zero-safe replay and explicit restoration policies.

The historical default subtracts negative coordinates and repairs its final
borrow phase. Positive restoration instead adds the original coordinates with
an exact canonical adder and needs no subtraction-specific equality patch.
Both retain the tuned GCD/square, their representation and phase hazards, and
the nonzero canonical-y promise of sign conjugation. Canonical restoration is
not a clean canonicalizer of arbitrary lazy inputs: its accumulator must be
canonical. References and samplers do not filter circuit failures.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import partial
import random

from ..bench.registry import Case
from ..core.circuit import Circuit, Register
from ..field.add import _and, _unand, gidney_add
from ..field.add_chunked import gidney_add_chunked
from ..field.compare import cmp_lt_phase
from ..field.constadd import sub_const_inplace
from ..field.modarith import modadd, modsub_direct
from ..field.sub import sub
from ..inversion.zero_safe import moddiv_zero_safe_inplace, modmul_zero_safe_inplace
from ..mul.accumulate import modsqrsub_inplace
from ..params import PRIME, U
from ..point import active as curve
from ..point.affine import sub_const_mod, tuned_point_kwargs
from ..point.group import scalar_mul
from .local_signed import half_shift_recode_inplace
from .lookup import _validate_registers, lookup_xor_unary, unlookup_measure
from .odd_window import OddWindowTable, _width, make_odd_window_table
from .signed_window import _group_add, conditional_negate_nonzero_inplace
from .window import _region
from .window_opt import (
    OptimizedWindowConfig, _packed_sub, _rebind_coordinates, _snapshot_bindings,
)


def _modsub_direct_borrow(circ, a, b, p, add_fn=gidney_add,
                          carry_window=None, cmp_bits=None):
    """Exact modsub_direct decomposition, returning ITS borrow cbit.

    Kept deliberately gate-for-gate identical to the original primitive, with
    regression tests to detect drift. In particular this is not a replacement
    comparator or a guess based on a measurement index/last allocated cbit.
    """
    n = len(a)
    assert n == len(b) and 0 < p < 1 << n
    borrow = circ.alloc(1)
    extended = Register(b.ids + [borrow[0]])
    sub(circ, a, extended, add_fn)
    sub_const_inplace(circ, b, (1 << n)-p, ctrl=borrow[0], window=carry_window)
    m = circ.measure(borrow[0])
    for q in a:
        circ.x(q)
    cmp_lt_phase(circ, b, a, m, k=cmp_bits)
    for q in a:
        circ.x(q)
    return m


def _equality_phase(circ: Circuit, output: Register, operand: Register, m: int):
    """Apply (-1)**(m*[output==operand]), restoring both words exactly.

    n-1 CCX and n-1 measured AND erasures, each with its own CZ correction.
    XOR into the live operand instead of keeping a zero flag across arithmetic.
    The ladder is unconditional; ONLY its phase is conditioned on the borrow.
    """
    n = len(output)
    if n < 1 or len(operand) != n:
        raise ValueError("equality needs nonempty equal-width words")
    _validate_registers(circ, output, operand)
    if circ._cond_stack or not 0 <= m < circ._next_cbit:
        raise ValueError("equality needs an existing cbit and no outer condition")
    for a, b in zip(output, operand):
        circ.cx(a, b)
        circ.x(b)
    ladder = circ.alloc(n-1) if n > 1 else Register([])
    prev = operand[0]
    triples = []
    for j in range(1, n):
        target = ladder[j-1]
        _and(circ, prev, operand[j], target)
        triples.append((prev, operand[j], target))
        prev = target
    with circ.condition(m):
        circ.z(prev)
    for a, b, target in reversed(triples):
        _unand(circ, a, b, target)
    for a, b in zip(output, operand):
        circ.x(b)
        circ.cx(a, b)


def _restore_positive(circ, address, x, y, xs, ys, p, add_fn, low_bits,
                      regions, *, unlookup):
    """Restore from positive coordinates; incoming payloads must be canonical.

    Borrow only the significant bits of F for the exact +/-p operations, so
    the configured chunked adder also bounds their workspace. This does not
    change the first two subtractions or widen any inversion/square tuning.
    """
    n = len(x)
    packed = tuple(a | (b << n) for a, b in zip(xs, ys))
    with _region(circ, regions, "lookup"):
        operand = circ.alloc(2*n)
        lookup_xor_unary(circ, address, operand, packed)
    for index, accumulator in enumerate((x, y)):
        view = operand[index*n:(index+1)*n]
        with _region(circ, regions, "arithmetic"):
            modadd(circ, view, accumulator, p, add_fn=add_fn,
                   inplace_const=False, pseudo_mersenne=True)
        if unlookup == "split":
            with _region(circ, regions, "unlookup"):
                unlookup_measure(circ, address, view, (xs, ys)[index], low_bits=low_bits)
    if unlookup == "paired":
        with _region(circ, regions, "unlookup"):
            unlookup_measure(circ, address, operand, packed, low_bits=low_bits)


def add_zero_point_inplace(
    circ: Circuit, x: Register, y: Register, word: Register, table: OddWindowTable, *,
    config: OptimizedWindowConfig | None = None,
    const_kw: dict | None = None, sqr_kw: dict | None = None,
    phase_correction: bool = True, restoration: str = "subtract",
    addition_unlookup: str = "paired", regions: list | None = None,
) -> tuple[Register, Register]:
    """Same odd table, recoder, conjugation and three forward lookups.

    With phase_correction=False this is the original point schedule with only
    the two zero-safe field tags changed. With correction enabled, split the
    final measured unlookup: release X before borrowing the Y word for equality.
    restoration="add" requires phase_correction=False and uses canonical
    positive-coordinate addition; ordinary arithmetic/lookup phase corrections
    remain intact. addition_unlookup selects paired or split cleanup there.
    """
    if not isinstance(table, OddWindowTable):
        raise ValueError("table must be a validated OddWindowTable")
    if type(phase_correction) is not bool:
        raise ValueError("phase_correction must be a bool")
    if restoration not in ("subtract", "add"):
        raise ValueError("restoration must be subtract or add")
    if addition_unlookup not in ("paired", "split"):
        raise ValueError("addition_unlookup must be paired or split")
    if restoration == "add" and phase_correction:
        raise ValueError("positive restoration requires phase_correction=False; no subtraction equality patch")
    if restoration == "subtract" and addition_unlookup != "paired":
        raise ValueError("addition_unlookup only applies to positive restoration")
    if not all(isinstance(r, Register) for r in (x, y, word)):
        raise TypeError("coordinate and original word operands must be Registers")
    n, p, bits = len(x), table.p, table.address_bits
    if len(y) != n or len(word) != table.word_bits or p >= 1 << n:
        raise ValueError("odd window operand widths must match")
    _validate_registers(circ, word, Register(x.ids+y.ids))
    if circ._cond_stack:
        raise ValueError("measured odd windows cannot run inside a classical condition")
    if any(yp == 0 for _, yp in table.points):
        raise ValueError("zero-point correction requires nonzero table Y")
    config = (OptimizedWindowConfig.tuned(n, p, bits) if config is None else
              config).resolved(n, p, bits)
    if config.operand_strategy != "paired":
        raise ValueError("zero-point prototype supports only paired lookup")
    kw = tuned_point_kwargs(n, p, config.inversion)
    const = dict(kw["const_kw"] if const_kw is None else const_kw)
    square = dict(kw["sqr_kw"] if sqr_kw is None else sqr_kw)
    const["add_fn"] = (gidney_add if config.adder_chunk is None else
                       partial(gidney_add_chunked, chunk=config.adder_chunk, cmp_bits=None))
    address, sign = word[:-1], word[-1]
    xs, ys = tuple(pt[0] for pt in table.points), tuple(pt[1] for pt in table.points)
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

    def final_negate():
        for q in y:
            circ.x(q)
        sub_const_mod(circ, y, (f-1) % p, p, ctrl=None, **const)

    with _region(circ, regions, "recoding"):
        half_shift_recode_inplace(circ, word)
    with _region(circ, regions, "sign"):
        conditional_negate_nonzero_inplace(circ, y, sign, p)
    subtract((x, y), (xs, ys))
    with _region(circ, regions, "arithmetic"):
        rebind_call(moddiv_zero_safe_inplace)
        bindings = _snapshot_bindings(circ)
        old_x, old_y = x, y
        x = modsqrsub_inplace(circ, y, x, p, ctrl=None, **square)
        _rebind_coordinates(circ, bindings, old_x, old_y, x, y)
        for q in x:
            circ.x(q)
    subtract((x,), (ks,))
    with _region(circ, regions, "arithmetic"):
        rebind_call(modmul_zero_safe_inplace)
    if restoration == "add":
        _restore_positive(circ, address, x, y, xs, ys, p, const["add_fn"],
                          config.unlookup_low_bits, regions, unlookup=addition_unlookup)
        with _region(circ, regions, "arithmetic"):
            final_negate()
    else:
        xneg, yneg = tuple(-v % p for v in xs), tuple(-v % p for v in ys)
        if not phase_correction:
            subtract((x, y), (xneg, yneg))
            with _region(circ, regions, "arithmetic"):
                final_negate()
        else:
            packed = tuple(a | (b << n) for a, b in zip(xneg, yneg))
            with _region(circ, regions, "lookup"):
                operand = circ.alloc(2*n)
                lookup_xor_unary(circ, address, operand, packed)
            xo, yo = operand[:n], operand[n:]
            with _region(circ, regions, "arithmetic"):
                modsub_direct(circ, xo, x, p, **const)
            with _region(circ, regions, "unlookup"):
                unlookup_measure(circ, address, xo, xneg, low_bits=config.unlookup_low_bits)
            with _region(circ, regions, "arithmetic"):
                m = _modsub_direct_borrow(circ, yo, y, p, **const)
                final_negate()
                _equality_phase(circ, y, yo, m)
            with _region(circ, regions, "unlookup"):
                unlookup_measure(circ, address, yo, yneg, low_bits=config.unlookup_low_bits)
    with _region(circ, regions, "sign"):
        conditional_negate_nonzero_inplace(circ, y, sign, p)
    with _region(circ, regions, "recoding"):
        half_shift_recode_inplace(circ, word, inverse=True)
    return x, y


def zero_point_reference(x: int, y: int, word: int,
                         table: OddWindowTable) -> tuple[int, int]:
    """Original odd reference with ONLY its zero-numerator exclusion removed.

    This is an ideal-domain reference, not a circuit-success or phase filter.
    The documented sign-domain and lazy-trajectory caveats remain unchanged.
    """
    p = table.p
    if any(type(v) is not int or not 0 <= v < p for v in (x, y)):
        raise ValueError("reference input coordinates must be canonical")
    if (y*y-x*x*x-table.a*x-table.b) % p:
        raise ValueError("reference input must be on the declared curve")
    xp, yp = table.point_for_word(word)
    assert x != xp, "exceptional: input x equals selected x"
    result = _group_add((x, y), (xp, yp), p, table.a)
    assert result is not None, "exceptional: point at infinity"
    assert result[0] != xp, "exceptional: output x equals selected x"
    return result


def build_zero_point_case(
    word_bits: int = 16, *, table: OddWindowTable | None = None,
    config: OptimizedWindowConfig | None = None, const_kw: dict | None = None,
    sqr_kw: dict | None = None, phase_correction: bool = True,
    restoration: str = "subtract", addition_unlookup: str = "paired",
) -> Case:
    """Unregistered odd-only point variant; same word,x,y interface as baseline."""
    _width(word_bits)
    table = make_odd_window_table(word_bits) if table is None else table
    if not isinstance(table, OddWindowTable) or table.word_bits != word_bits:
        raise ValueError("odd table must match the requested original word width")
    n, bits = (U if table.p == PRIME else table.p.bit_length()), table.address_bits
    config = (OptimizedWindowConfig.tuned(n, table.p, bits) if config is None else
              config).resolved(n, table.p, bits)
    circ = Circuit()
    word, x, y = circ.input("word", word_bits), circ.input("x", n), circ.input("y", n)
    regions = []
    x, y = add_zero_point_inplace(circ, x, y, word, table, config=config,
                                 const_kw=const_kw, sqr_kw=sqr_kw,
                                 phase_correction=phase_correction, restoration=restoration,
                                 addition_unlookup=addition_unlookup, regions=regions)
    circ.rebind("x", x)
    circ.rebind("y", y)

    def reference(values):
        xr, yr = zero_point_reference(values["x"], values["y"], values["word"], table)
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
        raise ValueError("could not draw an algebraically admissible zero-point input")

    kw = tuned_point_kwargs(n, table.p, config.inversion)
    const = kw["const_kw"] if const_kw is None else dict(const_kw)
    square = kw["sqr_kw"] if sqr_kw is None else dict(sqr_kw)
    bound = (1 << word_bits)-1
    return Case(circ, reference, gen_input, meta={
        "empirical_only": True, "fail_analytic": None,
        "input_distribution": ("independent uniform active prime-subgroup points" if active else
                               "uniform finite on-curve toy points") +
                              " and uniform original words, conditioned only on algebraic preconditions",
        "input_domain": "canonical finite on-curve coordinates; input/output x differs from selected x; zero numerator allowed",
        "arithmetic_contract": "tuned lazy incomplete affine with canonical-zero replay tags; inherited hazards retained",
        "zero_contract": "literal 0 only; literal p zero representation NOT supported",
        "phase_correction": phase_correction,
        "coordinate_restoration": restoration,
        "restoration_adder": "modadd.borrowed_pseudo_mersenne" if restoration == "add" else "modsub_direct",
        "restoration_unlookup": addition_unlookup if restoration == "add" else ("split" if phase_correction else "paired"),
        "restoration_contract": ("canonical operands required; exact local addition; upstream lazy/phase failures remain"
                                 if restoration == "add" else "historical negative-coordinate subtraction"),
        "restoration_exact_reduction": restoration == "add",
        "restoration_carry_window": None if restoration == "add" else const.get("carry_window"),
        "restoration_comparator_bits": None if restoration == "add" else const.get("cmp_bits"),
        "phase_correction_contract": ("borrow_m * [positive-addition output_y == -Y_T], before outer sign; correct canonical trajectory and nonzero x factors required"
                                      if restoration == "subtract" else None),
        "sign_domain": "canonical nonzero accumulator y at both negation boundaries; 0 and p exchange; noncanonical y>p is an unresolved interface hazard",
        "output_representation": ("canonical restoration on canonical payloads and nonzero-y sign domain; upstream approximate hazards retained"
                                  if restoration == "add" else "lazy residues; not guaranteed canonical or phase-clean"),
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
        "table_fingerprint": table.fingerprint,
        "curve": {"p": table.p, "a": table.a, "b": table.b,
                  "base": list(table.base), "order": table.order},
        "lookup_backend": "unary-and.v1", "sign_strategy": "conjugate",
        "circuit_config": {"unlookup_backend": "measured-onehot.v1", "operand_strategy": "paired",
                           "unlookup_low_bits": config.unlookup_low_bits, "adder_chunk": config.adder_chunk,
                           "lookup_stages": 3,
                           "unlookup_stages": (4 if addition_unlookup == "split" else 3) if restoration == "add" else (4 if phase_correction else 3),
                           "one_bit_specialization": False, "square_control": None,
                           "zero_enable": False, "endpoint_xor": False,
                           "sign_strategy": "conjugate", "sign_truncation": None},
        "tuning": {"inversion": asdict(config.inversion), "window": config.metadata(),
                   "point": {"carry_window": const.get("carry_window"), "comparator_bits": const.get("cmp_bits"),
                             "square": "flattened-triangular" if sqr_kw is None else "explicit-override",
                             "square_window": square.get("window"), "square_ripple": square.get("ripple"),
                             "square_comparator_bits": square.get("cmp_bits")}},
        "resource_regions": regions,
    }, analysis_context={"window_table": table, "odd_window_table": table})
