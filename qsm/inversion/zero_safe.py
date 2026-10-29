"""Isolated conditional-tag prototypes for the existing *tuned* replay.

The intended input domain is canonical ``1 <= x < p, 0 <= y,z < p`` for
odd prime p. Zero support refers only to the second operand, NOT to x=0.
In particular the lazy encoding p is not accepted as another input zero:
merging 0 and p would not be reversible on arbitrary input superpositions.

Only the replay boundary changes: tau(v) is p for nonzero v and 0 for zero.
The GCD, replay, their truncations and measurement phases are unchanged.
This is NOT a correctness repair of tuned arithmetic: inherited value,
representation, and phase failures (including structured inputs) remain.
Both returned register handles are authoritative, as in inplace.py. Omitted
config resolves to InversionConfig.tuned; baseline comparisons must pass that
same config explicitly (inplace.py also retains a different legacy-kwargs path).
"""
from __future__ import annotations

from ..core.circuit import Circuit, Register
from ..params import PRIME
from ..scalarmul.lookup import xor_nonzero
from .config import InversionConfig
from .inplace import _phase1, _un_phase1
from .replay import apply_dialog_j2
from .unreplay import un_apply_dialog_j2

__all__ = ["modmul_zero_safe_inplace", "moddiv_zero_safe_inplace"]


def _config(circ: Circuit, x: Register, y: Register, p: int,
            config: InversionConfig | None) -> InversionConfig:
    if circ._cond_stack:
        raise ValueError("zero-safe inversion cannot run inside a classical condition")
    wires = x.ids + y.ids
    if len(x) != len(y) or len(set(wires)) != len(wires):
        raise ValueError("inversion requires disjoint equal-width operands")
    if any(q not in circ._live_qids for q in wires):
        raise ValueError("inversion requires live operand wires")
    cfg = (config if config is not None else InversionConfig.tuned(len(x), p))
    cfg.resolved(len(x), p)
    if cfg.arithmetic_family != "tuned":
        raise ValueError("zero-safe prototypes require tuned arithmetic, not canonical fallback")
    return cfg


def _prepare_tag(circ: Circuit, value: Register, p: int) -> Register:
    """Allocate tau(value), using the p low bit as its only temporary flag.

    Compute nonzero before allocating the remaining n-1 tag wires, so its
    n-2 scratch wires fit inside the eventual tag footprint. Since p is odd,
    the flag is already tag[0]; there is no extra flag across reverse replay.
    """
    flag = circ.alloc(1)
    xor_nonzero(circ, value, flag[0])
    rest = circ.alloc(len(value) - 1)
    tag = Register([flag[0], *rest.ids])
    for j in range(1, len(tag)):
        if (p >> j) & 1:
            circ.cx(tag[0], tag[j])
    return tag


def _erase_tag(circ: Circuit, tag: Register, value: Register, p: int) -> None:
    """Erase a promised tau(value), without measuring a zero/nonzero flag.

    Compress 0/p to its low bit; free the other n-1 known-zero wires *before*
    exact nonzero needs n-2 scratch. Dirty frees expose a violated tag promise
    rather than silently discarding a representation or arithmetic error.
    """
    for j in range(1, len(tag)):
        if (p >> j) & 1:
            circ.cx(tag[0], tag[j])
    circ.free(tag[1:])
    xor_nonzero(circ, value, tag[0])
    circ.free(tag[:1])


def modmul_zero_safe_inplace(
    circ: Circuit, x: Register, y: Register, p: int = PRIME, *,
    config: InversionConfig | None = None,
) -> tuple[Register, Register]:
    """Experimental |x,y> -> |x,x*y mod p>, allowing canonical y=0.

    Assumes the unchanged tuned replay yields (tau(product), product).
    No persistent zero flag crosses replay, and no tuning is silently widened.
    Canonical x must be nonzero; arbitrary lazy p-as-zero is out of contract.
    Inherited tuned value/phase failures are not corrected by this boundary.
    """
    cfg = _config(circ, x, y, p, config)
    gcd_kw, replay_kw = cfg.gcd_kwargs(), cfg.replay_kwargs()
    d = _phase1(circ, x, p, cfg.iters, cfg.widths, **gcd_kw)
    tag, product = apply_dialog_j2(
        circ, d, y, circ.alloc(cfg.n), cfg.iters, p, **replay_kw)
    _erase_tag(circ, tag, product, p)
    recovered = _un_phase1(circ, d, cfg.n, p, cfg.iters, cfg.widths, **gcd_kw)
    return recovered, product


def moddiv_zero_safe_inplace(
    circ: Circuit, x: Register, z: Register, p: int = PRIME, *,
    config: InversionConfig | None = None,
) -> tuple[Register, Register]:
    """Experimental |x,z> -> |x,z/x mod p>, allowing canonical z=0.

    Supply (tau(z), z), not an unconditional p, to unchanged reverse replay.
    The exact predicate's flag becomes a tag wire before replay begins.
    Canonical x must be nonzero. This neither supports arbitrary lazy p-as-zero
    nor repairs the tuned reverse's structured-numerator and phase failures.
    """
    cfg = _config(circ, x, z, p, config)
    gcd_kw, replay_kw = cfg.gcd_kwargs(), cfg.replay_kwargs()
    d = _phase1(circ, x, p, cfg.iters, cfg.widths, **gcd_kw)
    tag = _prepare_tag(circ, z, p)
    quotient, empty = un_apply_dialog_j2(
        circ, d, tag, z, cfg.iters, p, **replay_kw)
    circ.free(empty)
    recovered = _un_phase1(circ, d, cfg.n, p, cfg.iters, cfg.widths, **gcd_kw)
    return recovered, quotient
