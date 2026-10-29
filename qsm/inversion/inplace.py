"""Tuned experimental and canonical conservative field composites, each ONE circuit.

The historical discussion below describes ONLY the experimental lazy functions
`modmul_inplace` and `moddiv_inplace`. The separately registered conservative
functions use full-width GCD and canonical arithmetic, accept zero/structured
second operands, and do not inherit these representation/distribution limitations.
Simulation diagnostics are not implementable quantum branches: a dirty release
reveals a violated construction contract, it does not halt an executing device.

Phase 1 writes a tape, Phase 2 spends it, Phase 1 backwards eats it. Those three regions
have been measured separately for a while and their Toffoli counts do add up -- but their
PEAKS do not, and doc 10 §2 is the record of getting that wrong. A region's measured peak
is the composite's peak only if nothing else is live while it runs, and in this
construction something always is: the forward tape-builder runs with `y` beside it and the
reverse one runs with the product beside it. Reading three separate peaks and taking the
max quietly assumed an empty machine three times over.

So nothing here is assembled from numbers. Each operation is built as a single `Circuit`,
handed to the same `count` as everything else, and its peak is whatever the harness says it
is with every wire that exists at that instant on the books.

    modmul_inplace(x, y)   |x, y> -> |x, x*y mod p>        Phase 2 forwards
    moddiv_inplace(x, z)   |x, z> -> |x, z/x mod p>        Phase 2 backwards

There is deliberately no `modinv_inplace`. `moddiv_inplace` with a CONSTANT numerator does
not work at this tuning, the reason is not a bug, and doc 12 is the record of it; the short
version is in `moddiv_inplace`'s own docstring under "the numerator must be generic".

The `_inplace` names are honest about the second register and only the second register: `x`
survives all three, because the tape has to be un-built to give it back and un-building it
is what the third region is. There is no cheaper arrangement that keeps `x` -- the tape IS
`x`, in a different alphabet, and it is consumed by the time Phase 2 starts.

**Both operands must be non-zero, and the reason is the same one in both directions.**
Phase 2's emptied register comes back holding `p` rather than `|0>` -- the loop never
canonicalises -- and that is exactly what makes the composite possible, because `p` is a
CONSTANT and a constant clears with X gates and frees for nothing. But which constant it
lands on depends on whether the result is zero:

    modmul   (y, 0) -> (p, x*y)        emptied register is `p`   ... unless y = 0
    moddiv   (p, z) -> (z/x, 0)        emptied register is `0`   ... unless z = 0

Those are opposite constants, and they are opposite for a reason rather than by accident:
the multiply's zero case maps `(0, 0) -> (0, 0)`, so `(p, 0)` is simply not in its image
and the division has nothing to undo there. Each operation therefore clears the constant
its non-zero branch produces, and a zero operand leaves a live qubit that `circ.free`
refuses -- `DirtyAncillaError`, at the end, loudly.

That is the right failure and not a limitation to work around. Verified exhaustively at
p = 31 and p = 127: every one of the 900 and 15,876 non-zero pairs is exact in both
directions, and the raises are exactly the p-1 zero cases and nothing else. At 255 bits the
`(p, x*y)` landing is confirmed on 200 random pairs. The narrowing window is not involved
-- the same counts appear with it switched off.

The point layer must discharge the experimental operation's domain before using it.
A simulator can report a dirty release at zero; a quantum implementation has no
corresponding exception branch. The conservative operation removes this second-
operand restriction but still requires a nonzero divisor.

Algorithm from [GRLGS23] PRL 131(4) 040602 and [KSGZ+25] arXiv:2510.10967; written from
the papers, not from either reference implementation (both are AGPL-3.0).
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add_ctrl_threaded
from ..field.add_chunked import gidney_add_ctrl_chunked_threaded
from ..params import (
    CARRY_WINDOW,
    CMP_BITS,
    PRIME,
    U,
    gcd_width_clip_fail_prob,
    replay_j2_fail_prob,
    un_replay_j2_fail_prob,
)
from . import pack, schedule
from .config import InversionConfig
from .gcd import PackedTapeJ2, to_dialog_j2
from .reference import to_dialog_jump as to_dialog_jump_ref
from .replay import apply_dialog_j2
from .undo import PackedUnTapeJ2, un_to_dialog_j2
from .unreplay import un_apply_dialog_j2

__all__ = [
    "modmul_inplace",
    "moddiv_inplace",
    "modmul_conservative_inplace",
    "moddiv_conservative_inplace",
    "InversionConfig",
]


def _clear_constant(circ: Circuit, r: Register, c: int) -> None:
    """Free a register known to hold the constant `c`. No Toffoli, no measurement.

    The one place the composite touches Phase 2's leftover, and it is deliberately the
    cheapest thing that can be written: `c` is classical, so undoing it is an X layer on
    its set bits. If the value is anything else the X layer leaves a 1 somewhere and
    `circ.free` raises rather than dropping a live qubit -- which is the entire safety
    argument for the precondition in the module docstring.
    """
    for j in range(len(r)):
        if (c >> j) & 1:
            circ.x(r[j])
    circ.free(r)


def _phase1(circ: Circuit, x: Register, p: int, iters: int, widths, **kw) -> Register:
    return to_dialog_j2(circ, x, PackedTapeJ2(circ), p,
                        iters=iters, widths=widths, **kw)


def _un_phase1(circ: Circuit, d: Register, n: int, p: int, iters: int, widths,
               **kw) -> Register:
    return un_to_dialog_j2(circ, PackedUnTapeJ2(circ, d, iters), n, p,
                           iters=iters, widths=widths, **kw)


def modmul_inplace(
    circ: Circuit,
    x: Register,
    y: Register,
    p: int = PRIME,
    iters: int | None = None,
    widths: list[int] | None = None,
    gcd_kw: dict | None = None,
    replay_kw: dict | None = None,
    *,
    config: InversionConfig | None = None,
) -> tuple[Register, Register]:
    """|x, y> -> |x, x*y mod p>, one circuit, one measured peak. `x` must be non-zero.

    The product lands in a register allocated here rather than in `y`'s own wires, and `y`
    is consumed on the way: the replay reads `(a, b) = (y, 0)` and leaves `(p, x*y)`. So
    "in place" means the operand count does not grow, not that the wires are literally
    reused -- `y`'s wires come back to the pool at the moment the replay empties them, and
    the allocator hands most of them straight to the product.

    `gcd_kw` tunes both Phase 1 halves and `replay_kw` tunes Phase 2; they are kept apart
    because the two phases do not take the same knobs (Phase 1 has no modulus and so no
    `double_window`) and because mixing them is how a forward and its reverse end up built
    with different windows, which fails loudly but confusingly.
    """
    n = len(x)
    if config is not None:
        config.resolved(n, p)
        if config.arithmetic_family != "tuned":
            raise ValueError("use the separately named conservative composite for canonical arithmetic")
        if any(value is not None for value in (iters, widths, gcd_kw, replay_kw)):
            raise ValueError("config cannot be mixed with legacy inversion tuning kwargs")
        iters, widths = config.iters, config.widths
        gcd_kw, replay_kw = config.gcd_kwargs(), config.replay_kwargs()
    if iters is None:
        iters = schedule.iterations_j2(n)
    if widths is None:
        widths = schedule.widths_j2(n, iters)
    gcd_kw, replay_kw = dict(gcd_kw or {}), dict(replay_kw or {})

    d = _phase1(circ, x, p, iters, widths, **gcd_kw)
    a, b = apply_dialog_j2(circ, d, y, circ.alloc(n), iters, p, **replay_kw)
    _clear_constant(circ, a, p)                  # the lazy zero, which is always p
    return _un_phase1(circ, d, n, p, iters, widths, **gcd_kw), b


def moddiv_inplace(
    circ: Circuit,
    x: Register,
    z: Register,
    p: int = PRIME,
    iters: int | None = None,
    widths: list[int] | None = None,
    gcd_kw: dict | None = None,
    replay_kw: dict | None = None,
    *,
    config: InversionConfig | None = None,
) -> tuple[Register, Register]:
    """|x, z> -> |x, z/x mod p>. The same three regions, middle one reversed.

    The asymmetry with `modmul_inplace` is in which register carries the constant and when.
    The multiply ALLOCATES a `|0>` and gets `p` handed back; the division has to SUPPLY the
    `p` up front and gets an honest `|0>` back, so the X layer moves from after Phase 2 to
    before it. Both cost nothing, and between them they are the only difference in the
    composite -- everything else is `un_apply_dialog_j2` where `apply_dialog_j2` stood.

    That the emptied register really is exactly `|0>` here rather than lazy is not an
    assumption: it is what `test_the_whole_tape_reversed_is_the_divide` measures, and it is
    why the free below needs no X layer of its own.

    **The numerator must be GENERIC, and that is a stronger requirement than non-zero.**
    Every `-F` in the reverse replay -- the gated one inside the halving and step R2 of
    `modsub_lazy_ctrl` -- is truncated to the low `carry_window` bits, so it is wrong
    whenever the operand's low `carry_window` bits fall below `F`. For a uniform operand
    that is `19/2^45` per firing and the whole replay carries `2^-31.6` of it, which is the
    term `un_replay_j2_fail_prob` already prices. For a STRUCTURED numerator it is not a
    rate at all: the operand sequence is produced by halving, halving pulls the top bits
    down, and a numerator with nothing up there stays in the danger set for iteration after
    iteration. Measured over six tapes at n = 255 (`reference.un_apply_dialog_jump_trunc_
    faults`): a uniform `z` takes 0 faults, `z = p + 1` takes 15, `z = 2^128` takes 125,
    `z = 19` takes 315.

    So there is no constant numerator this circuit can be handed, and in particular no
    `modinv_inplace`. Doc 12 has the full account including the exact circuit's own
    narrower hole (`b < a` with `b < F`, which widening the window does not close) and the
    blinding construction that would fix it and what it would cost.

    Quantum input is NOT uniform by construction. Arbitrary amplitudes, structured
    numerators, and correlated arithmetic intermediates need not obey this tuning's
    generic/uniform marginal assumptions. The analytic estimate is distribution-
    conditioned, not a worst-case quantum error bound. Use the separately named
    canonical conservative composite when these assumptions cannot be justified.
    """
    n = len(x)
    if config is not None:
        config.resolved(n, p)
        if config.arithmetic_family != "tuned":
            raise ValueError("use the separately named conservative composite for canonical arithmetic")
        if any(value is not None for value in (iters, widths, gcd_kw, replay_kw)):
            raise ValueError("config cannot be mixed with legacy inversion tuning kwargs")
        iters, widths = config.iters, config.widths
        gcd_kw, replay_kw = config.gcd_kwargs(), config.replay_kwargs()
    if iters is None:
        iters = schedule.iterations_j2(n)
    if widths is None:
        widths = schedule.widths_j2(n, iters)
    gcd_kw, replay_kw = dict(gcd_kw or {}), dict(replay_kw or {})

    d = _phase1(circ, x, p, iters, widths, **gcd_kw)
    lazy = circ.alloc(n)
    for j in range(n):
        if (p >> j) & 1:
            circ.x(lazy[j])                      # the lazy zero the reverse replay wants
    q, empty = un_apply_dialog_j2(circ, d, lazy, z, iters, p, **replay_kw)
    circ.free(empty)                             # exactly |0>, not lazy -- see docstring
    return _un_phase1(circ, d, n, p, iters, widths, **gcd_kw), q


def _conservative_config(x: Register, y: Register, p: int,
                         config: InversionConfig | None) -> InversionConfig:
    if len(x) != len(y) or set(x.ids) & set(y.ids):
        raise ValueError("inversion requires disjoint equal-width operands")
    cfg = (config if config is not None else
           InversionConfig.conservative(len(x), p)).resolved(len(x), p)
    if cfg.arithmetic_family != "canonical":
        raise ValueError("conservative composite requires a canonical configuration")
    return cfg


def modmul_conservative_inplace(
    circ: Circuit, x: Register, y: Register, p: int = PRIME,
    config: InversionConfig | None = None,
) -> tuple[Register, Register]:
    """|x,y> -> |x,x*y mod p> for 1<=x<p and 0<=y<p, p odd prime.

    Full-width terminating GCD, canonical replay, packed tape undo. Both
    returned handles are authoritative; x's tape and y's relabeling consume
    their original wire layouts. No distribution-conditioned truncations.
    """
    from .replay import apply_dialog_canonical_inplace

    cfg = _conservative_config(x, y, p, config)
    d = _phase1(circ, x, p, cfg.iters, cfg.widths, **cfg.gcd_kwargs())
    empty, product = apply_dialog_canonical_inplace(
        circ, d, y, circ.alloc(cfg.n), cfg.iters, p)
    circ.free(empty)  # canonical zero, for y=0 as well as every nonzero y
    recovered = _un_phase1(circ, d, cfg.n, p, cfg.iters, cfg.widths,
                          **cfg.gcd_kwargs())
    return recovered, product


def moddiv_conservative_inplace(
    circ: Circuit, x: Register, z: Register, p: int = PRIME,
    config: InversionConfig | None = None,
) -> tuple[Register, Register]:
    """|x,z> -> |x,z/x mod p>, canonical nonzero x and any canonical z.

    In particular zero, constant one, and small structured numerators are
    supported. This supplies an actual |0>, not the experimental lazy p.
    """
    from .unreplay import un_apply_dialog_canonical_inplace

    cfg = _conservative_config(x, z, p, config)
    d = _phase1(circ, x, p, cfg.iters, cfg.widths, **cfg.gcd_kwargs())
    quotient, empty = un_apply_dialog_canonical_inplace(
        circ, d, circ.alloc(cfg.n), z, cfg.iters, p)
    circ.free(empty)
    recovered = _un_phase1(circ, d, cfg.n, p, cfg.iters, cfg.widths,
                          **cfg.gcd_kwargs())
    return recovered, quotient


# ---------------------------------------------------------------------------
# Registered versions (real field width).
#
# The settled tuning throughout: jump 2, packed tape, threaded controls in both Phase 1
# halves, and the chunk-43 / cmp-32 threaded replay doc 10 §6 landed on. Each version is
# one circuit with one peak; there is no arithmetic in this section and that is the point.
# ---------------------------------------------------------------------------
_N = U
_CONFIG = InversionConfig.tuned(_N, PRIME)
_ITERS_J2 = _CONFIG.iters
_WIDTHS_J2 = list(_CONFIG.widths)

#: Phase 1, both directions. `compare_bits` and not CMP_BITS -- inside the narrowing window
#: the top bits are padding, and comparing padding against padding is a coin flip.
_GCD_KW = _CONFIG.gcd_kwargs()

#: Compatibility views of the single resolved configuration, not separate tunings.
_REPLAY_KW = _CONFIG.replay_kwargs()
_REPLAY_KW["add_ctrl_fn"].__name__ = "gidney_add_ctrl_chunked_threaded43"


def _phase1_fail() -> float:
    """Both Phase 1 halves, which carry identical terms and are both present in all three.

    Iteration budget, window clip and comparator, twice -- the forward's and the reverse's
    are the same three numbers, because the reverse runs the same loop over the same
    schedule with the same truncation. Doubling rather than re-deriving is deliberate: if
    the two ever stop matching, that is a bug in the reverse and not a refinement here.
    """
    return _CONFIG.phase1_fail_prob()


def _mul_case() -> Case:
    circ = Circuit()
    x = circ.input("x", _N)
    y = circ.input("y", _N)
    x_out, prod = modmul_inplace(circ, x, y, PRIME, config=_CONFIG)
    circ.rebind("x", x_out)
    circ.rebind("y", prod)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"x": v["x"], "y": (v["x"] * v["y"]) % PRIME}

    def gen(rng: random.Random) -> dict[str, int]:
        # BOTH non-zero, and the module docstring says why: a zero operand leaves the
        # emptied register on the other constant and the free raises.
        return {"x": rng.randrange(1, PRIME), "y": rng.randrange(1, PRIME)}

    fail = _CONFIG.fail_prob()
    return Case(circ, ref, gen, meta={"fail_analytic": fail, "tolerates_phase": True,
                                     "arithmetic_contract": _CONFIG.assumptions,
                                     "input_distribution": "independent uniform nonzero field operands"})


def _div_case() -> Case:
    circ = Circuit()
    x = circ.input("x", _N)
    z = circ.input("z", _N)
    x_out, quot = moddiv_inplace(circ, x, z, PRIME, config=_CONFIG)
    circ.rebind("x", x_out)
    circ.rebind("z", quot)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"x": v["x"], "z": (v["z"] * pow(v["x"], -1, PRIME)) % PRIME}

    def gen(rng: random.Random) -> dict[str, int]:
        # Both uniform on [1, p), and for `z` that is a PRECONDITION and not a convenience:
        # a structured numerator sits in the truncated `-F`'s danger set for many
        # iterations running. See `moddiv_inplace` and doc 12.
        return {"x": rng.randrange(1, PRIME), "z": rng.randrange(1, PRIME)}

    # The reverse replay's total is the forward's, exactly; only the split between phase
    # and value moves. See `params.un_replay_j2_fail_prob`.
    fail = _CONFIG.fail_prob(reverse=True)
    # `tolerates_phase` is still set, but it is Phase 1's tolerance and not Phase 2's:
    # the reverse replay never measures, so its disagreements arrive as stranded flags
    # rather than as phase, while the two tape-builders on either side of it do vent. The
    # composite therefore leaks strictly less phase than the multiply does and the flag
    # cannot say so -- it is a boolean. `qsm.audit rates` carries the split.
    return Case(circ, ref, gen, meta={"fail_analytic": fail, "tolerates_phase": True,
                                     "arithmetic_contract": _CONFIG.assumptions,
                                     "input_distribution": "independent uniform nonzero field operands"})


@circuit_version("inv.modmul_inplace.v1")
def _build_modmul_inplace_v1() -> Case:
    """|x, y> -> |x, x*y mod p>, measured end to end at last.

    The number doc 10 predicted by adding three regions was 754,544 @ 1,173. This is the
    same construction built as one circuit, so the gate count had better agree exactly --
    gates do add, and a difference there would mean a region had been mis-attributed. The
    peak is the open question, because that is the quantity the region arithmetic had no
    right to.
    """
    return _mul_case()


@circuit_version("inv.moddiv_inplace.v1")
def _build_moddiv_inplace_v1() -> Case:
    """|x, z> -> |x, z/x mod p>. The multiply plus 522 Toffoli, and nothing else.

    522 is 0.07% of the composite, which is the quantitative form of the claim both
    baselines make qualitatively when they implement division by flipping a direction flag.

    `z` is drawn uniform on [1, p) and that is the circuit's domain, not the harness being
    lazy -- see `moddiv_inplace`. There is no companion `inv.modinv_inplace` version for
    the same reason: 1 is not in the domain and neither is any other constant.
    """
    return _div_case()


def _conservative_case(reverse: bool) -> Case:
    config = InversionConfig.conservative(_N, PRIME)
    circ = Circuit()
    x, y = circ.input("x", _N), circ.input("y", _N)
    fn = moddiv_conservative_inplace if reverse else modmul_conservative_inplace
    x_out, y_out = fn(circ, x, y, PRIME, config=config)
    circ.rebind("x", x_out)
    circ.rebind("y", y_out)

    def ref(values: dict[str, int]) -> dict[str, int]:
        factor = pow(values["x"], -1, PRIME) if reverse else values["x"]
        return {"x": values["x"], "y": values["y"] * factor % PRIME}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"x": rng.randrange(1, PRIME), "y": rng.randrange(PRIME)}

    return Case(circ, ref, gen, meta={
        "fail_analytic": config.fail_prob(reverse=reverse),
        "arithmetic_contract": config.assumptions,
        "input_distribution": "any supported canonical inputs; benchmark samples uniformly",
        "automatic_resource_winner": False,
    })


@circuit_version("inv.modmul_conservative_inplace.v1")
def _build_modmul_conservative_inplace_v1() -> Case:
    """Exact canonical full-width composite; structured and zero second operands."""
    return _conservative_case(False)


@circuit_version("inv.moddiv_conservative_inplace.v1")
def _build_moddiv_conservative_inplace_v1() -> Case:
    """Exact canonical division, including zero and constant-one numerators."""
    return _conservative_case(True)
