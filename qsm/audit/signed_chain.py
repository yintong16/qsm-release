"""Complete SMALL signed scalar chains, not full-size scalar resource estimates.

All radix additions, carry addition, scalar-preserving recoding and inverse
cleanup are physical gates in one circuit. All scalar/digit/carry wires remain
live until their actual cleanup. No streaming or semiclassical space saving is
assumed. The inherited tuned affine primitives remain incomplete and can have
measurement-phase errors even with truncation disabled.

Bias has an explicit *prepared-input* contract: the supplied accumulator R is
already P + [C]G. The circuit returns R + [k-C]G = P + [k]G. Preparation of an
arbitrary P + [C]G is NOT emitted or priced here. No first-addition saving is
claimed. Both variants execute every radix addition, including dummy zero slots.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.circuit import Circuit
from ..inversion.config import InversionConfig
from ..scalarmul.signed_recoding import recode_scalar, signed_recode_inplace
from ..scalarmul.signed_window import add_signed_window_inplace, make_signed_window_table
from ..scalarmul.window import WindowTable
from ..scalarmul.window_opt import (
    OptimizedWindowConfig, add_window_affine_optimized_inplace,
)


Point = tuple[int, int] | None


def group_add(left: Point, right: Point, p: int, a: int) -> Point:
    """Independent complete classical group law, including doubling and infinity."""
    if left is None:
        return right
    if right is None:
        return left
    x, y = left
    xx, yy = right
    if x == xx and (y + yy) % p == 0:
        return None
    slope = (((3*x*x + a) * pow(2*y, -1, p)) if left == right else
             ((yy-y) * pow(xx-x, -1, p))) % p
    out_x = (slope*slope-x-xx) % p
    return out_x, (slope*(x-out_x)-y) % p


def group_mul(scalar: int, point: Point, p: int, a: int) -> Point:
    """Double-and-add oracle; does not use a circuit or signed-recoding reference."""
    if scalar < 0:
        return group_mul(-scalar, None if point is None else
                         (point[0], -point[1] % p), p, a)
    result = None
    while scalar:
        if scalar & 1:
            result = group_add(result, point, p, a)
        point = group_add(point, point, p, a)
        scalar >>= 1
    return result


def _exact_config(p: int, address_bits: int) -> OptimizedWindowConfig:
    # Full deterministic GCD length, no width/comparison/carry/replay truncation.
    # "Exact parameters" does NOT repair the inherited lazy vent predicates.
    n = p.bit_length()
    iters = 2*n + 1
    inversion = InversionConfig.tuned(
        n, p, iters=iters, widths=(n,)*iters, gcd_cmp_bits=None,
        replay_cmp_bits=None, replay_chunk=None, replay_chunk_cmp_bits=None,
        carry_window=None, double_window=None)
    return OptimizedWindowConfig.tuned(
        n, p, address_bits, inversion=inversion, adder_chunk=None)


@dataclass
class SmallSignedChain:
    circuit: Circuit
    scalar_bits: int
    address_bits: int
    variant: str
    p: int
    a: int
    b: int
    base: tuple[int, int]
    tables: tuple
    terminal_table: WindowTable | None
    bias: int
    metadata: dict

    def reference(self, inputs: dict[str, int]) -> dict[str, int]:
        """Independent oracle: (input_point - [bias]G) + [scalar]G.

        This validates mathematical inputs, not the incomplete arithmetic domain.
        An infinity output has no representation in the finite-coordinate circuit.
        """
        scalar = inputs["scalar"]
        if type(scalar) is not int or not 0 <= scalar < 1 << self.scalar_bits:
            raise ValueError("scalar outside declared small domain")
        point = (inputs["x"], inputs["y"])
        if any(type(v) is not int or not 0 <= v < self.p for v in point):
            raise ValueError("input coordinates must be canonical")
        if (point[1]**2-point[0]**3-self.a*point[0]-self.b) % self.p:
            raise ValueError("input point is not on the declared curve")
        logical = group_add(point, group_mul(-self.bias, self.base, self.p, self.a),
                            self.p, self.a)
        expected = group_add(logical, group_mul(scalar, self.base, self.p, self.a),
                             self.p, self.a)
        if expected is None:
            raise ValueError("infinity output is outside finite-coordinate domain")
        return {"scalar": scalar, "x": expected[0], "y": expected[1]}

    def domain_exclusion(self, inputs: dict[str, int]) -> str | None:
        """Check ideal intermediate points BEFORE simulation, never observed phases.

        The declared domain excludes zero coordinates at slot boundaries, input
        x matching the real/dummy operand, zero slope numerator, and output x
        matching an enabled operand. Nonzero coordinates avoid the lazy p-as-zero
        representative at sign-negation boundaries. These conditions are NOT a
        certification of the inherited arithmetic's phase predicates.
        """
        try:
            self.reference(inputs)
        except ValueError as error:
            return str(error)
        point = (inputs["x"], inputs["y"])
        recoding = recode_scalar(inputs["scalar"], self.scalar_bits,
                                 self.address_bits, self.variant)
        slots = []
        for table, (sign, address) in zip(self.tables, recoding.encoded):
            operand = table.points[address]
            signed = (operand[0], (-operand[1] if sign else operand[1]) % self.p)
            slots.append((signed, bool(sign or address)))
        if self.terminal_table is not None:
            slots.append((self.terminal_table.points[1], bool(recoding.terminal_carry)))
        for index, (operand, enabled) in enumerate(slots):
            if point is None or not all(point):
                return f"slot {index}: infinity or zero boundary coordinate"
            if point[0] == operand[0]:
                return f"slot {index}: input x equals selected/dummy x"
            if point[1] == operand[1]:
                return f"slot {index}: zero slope numerator"
            output = group_add(point, operand, self.p, self.a) if enabled else point
            if output is None or not all(output):
                return f"slot {index}: infinity or zero output coordinate"
            if enabled and output[0] == operand[0]:
                return f"slot {index}: output x equals selected x"
            point = output
        expected = self.reference(inputs)
        assert point == (expected["x"], expected["y"]), "recoding/group identity disagrees"
        return None


def build_small_signed_chain(
    scalar_bits: int = 4, address_bits: int = 1, *, variant: str = "carry",
    p: int = 127, a: int = 3, b: int = 4, base: tuple[int, int] = (0, 2),
    sign_strategy: str = "conjugate",
) -> SmallSignedChain:
    """Emit at least two radix slots, and a real constant carry slot for S-carry.

    Hard toy-size limits prevent interpreting this audit helper as a full-size
    builder. The carry slot is emitted even when a partial top chunk makes its
    control identically zero; metadata states this deliberate overhead. There is
    no table lookup initialization and no bias-preparation circuit.
    """
    if type(p) is not int or p < 5 or p.bit_length() > 10:
        raise ValueError("SMALL chains require a prime of at most ten bits")
    if any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
        raise ValueError("SMALL chain field modulus must be prime")
    if p != (1 << p.bit_length()) - 1:
        raise ValueError("SMALL chain audit supports only F=1 Mersenne fields")
    if type(scalar_bits) is not int or not 1 <= scalar_bits <= 12:
        raise ValueError("SMALL scalar_bits must be between one and twelve")
    if type(address_bits) is not int or not 1 <= address_bits <= 3:
        raise ValueError("SMALL magnitude addresses must have one to three bits")
    # Validate mode and compute schedule size/bias without constructing any gates.
    first = recode_scalar(0, scalar_bits, address_bits, variant)
    width, slots = address_bits + 1, len(first.digits)
    if slots < 2:
        raise ValueError("a complete SMALL chain must contain at least two radix slots")
    config = _exact_config(p, address_bits)
    tables = []
    for index in range(slots):
        shifted = group_mul(1 << (width*index), base, p, a)
        if shifted is None:
            raise ValueError("a radix base is infinity in this toy group")
        tables.append(make_signed_window_table(address_bits, shifted, p=p, a=a, b=b))
    terminal_table = None
    if variant == "carry":
        endpoint = group_mul(1 << (width*slots), base, p, a)
        if endpoint is None:
            raise ValueError("terminal carry point is infinity in this toy group")
        terminal_table = WindowTable(p, a, b, (None, endpoint))

    circ = Circuit()
    scalar = circ.input("scalar", scalar_bits)
    x, y = circ.input("x", p.bit_length()), circ.input("y", p.bit_length())
    digits = circ.alloc(slots*width)
    terminal = circ.alloc(1) if variant == "carry" else None
    terminal_id = None if terminal is None else terminal[0]
    recoder_wires = tuple(digits) + (() if terminal is None else tuple(terminal))
    recode_start = len(circ.ops)
    signed_recode_inplace(circ, scalar, digits, address_bits,
                         variant=variant, terminal=terminal_id)
    recode_stop = len(circ.ops)
    addition_ranges = []
    for index, table in enumerate(tables):
        word = digits[index*width:(index+1)*width]
        start = len(circ.ops)
        x, y = add_signed_window_inplace(
            circ, x, y, word[:address_bits], word[address_bits], table,
            sign_strategy=sign_strategy, config=config, const_kw={}, sqr_kw={})
        if circ.inputs["x"].ids != x.ids or circ.inputs["y"].ids != y.ids:
            raise RuntimeError("signed window did not preserve named coordinate bindings")
        addition_ranges.append((start, len(circ.ops)))
    terminal_range = None
    if terminal_table is not None:
        start = len(circ.ops)
        # One-bit paired specialization really emits a controlled constant-point
        # addition, not a sign-extended magnitude lookup or a Python branch.
        x, y = add_window_affine_optimized_inplace(
            circ, x, y, terminal, terminal_table, config=_exact_config(p, 1),
            const_kw={}, sqr_kw={})
        if circ.inputs["x"].ids != x.ids or circ.inputs["y"].ids != y.ids:
            raise RuntimeError("terminal addition did not preserve coordinate bindings")
        terminal_range = (start, len(circ.ops))
    cleanup_start = len(circ.ops)
    signed_recode_inplace(circ, scalar, digits, address_bits, variant=variant,
                         terminal=terminal_id, inverse=True)
    circ.free(digits)
    if terminal is not None:
        circ.free(terminal)

    metadata = {
        "scope": "complete SMALL gate-composed chain only; no full-size counts",
        "scalar_domain": f"0 <= scalar < 2**{scalar_bits}",
        "radix_slots": slots, "radix_bits": width,
        "terminal_constant_slots": int(terminal is not None),
        "terminal_policy": "always emitted for carry, even if top carry is impossible",
        "bias": first.bias,
        "bias_policy": "prepared input R=P+[bias]G; output P+[scalar]G",
        "bias_preparation_emitted": False,
        "bias_preparation_cost_included": False,
        "initialization_lookup_slots": 0,
        "input_preparation_cost_included": False,
        "first_addition_omitted": False,
        "scalar_and_recoder_wires_included": True,
        "recoder_outputs_freed": recoder_wires,
        "recode_range": (recode_start, recode_stop),
        "cleanup_range": (cleanup_start, len(circ.ops)),
        "addition_ranges": tuple(addition_ranges),
        "terminal_range": terminal_range,
        "inversion": asdict(config.inversion),
        "const_kw": {}, "sqr_kw": {}, "sign_strategy": sign_strategy,
        "phase_contract": "inherited lazy vent errors possible; no all-input phase-clean claim",
        "domain_policy": "ideal intermediate algebraic exclusions, never simulation-selected",
    }
    return SmallSignedChain(circ, scalar_bits, address_bits, variant, p, a, b,
                            base, tuple(tables), terminal_table, first.bias, metadata)
