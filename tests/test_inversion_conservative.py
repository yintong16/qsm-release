"""Deterministic GCD and canonical multiplication/division, not lazy round trips."""
import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import gidney_add_ctrl_threaded
from qsm.inversion import pack, schedule
from qsm.inversion.config import InversionConfig
from qsm.inversion.gcd import PackedTapeJ2, gcd_loop_j2, to_dialog_j2
from qsm.inversion.inplace import moddiv_conservative_inplace, modmul_conservative_inplace
from qsm.inversion.replay import apply_dialog_canonical_inplace
from qsm.inversion.unreplay import un_apply_dialog_canonical_inplace
from qsm.inversion.undo import PackedUnTapeJ2, un_to_dialog_j2


def build(p, reverse=False, config=None):
    c = Circuit()
    x, y = c.input("x", p.bit_length()), c.input("y", p.bit_length())
    fn = moddiv_conservative_inplace if reverse else modmul_conservative_inplace
    xo, yo = fn(c, x, y, p, config=config)
    c.rebind("x", xo)
    c.rebind("y", yo)
    return c


@pytest.mark.parametrize("p", [13, 29, 31])
@pytest.mark.parametrize("reverse", [False, True])
def test_all_canonical_pairs_include_zero_second_operand(p, reverse):
    c = build(p, reverse)
    for x in range(1, p):
        factor = pow(x, -1, p) if reverse else x
        for y in range(p):
            for seed in (0, 1):
                result = simulate(c, {"x": x, "y": y}, rng=random.Random(seed))
                assert result.outputs == {"x": x, "y": y * factor % p}
                assert result.phase == 0


@pytest.mark.parametrize("p", [13, 29, 31])
@pytest.mark.parametrize("extra", [0, 1, 2])
def test_fullwidth_gcd_boundary_terminal_padding_and_undo(p, extra):
    from dataclasses import replace
    n = p.bit_length()
    cfg = InversionConfig.conservative(n, p)
    cfg = replace(cfg, iters=cfg.iters + extra, widths=(n,) * (cfg.iters + extra))
    # Packed tapes with every possible final group size must be accepted.
    c = Circuit()
    x = c.input("x", n)
    d = to_dialog_j2(c, x, PackedTapeJ2(c), p, iters=cfg.iters,
                     widths=cfg.widths, **cfg.gcd_kwargs())
    saved = c.alloc(len(d))
    for a, b in zip(d, saved):
        c.cx(a, b)
    # Standalone replay must preserve the tape bit for bit.
    y = c.input("y", n)
    a, b = apply_dialog_canonical_inplace(c, d, y, c.alloc(n), cfg.iters, p)
    c.free(a)
    # Uncompute snapshot against the preserved tape (not against decoded residues).
    for a, z in zip(d, saved):
        c.cx(a, z)
    c.free(saved)
    xo = un_to_dialog_j2(c, PackedUnTapeJ2(c, d, cfg.iters), n, p,
                         iters=cfg.iters, widths=cfg.widths, **cfg.gcd_kwargs())
    c.rebind("x", xo)
    c.rebind("y", b)
    for xv in range(1, p):
        for yv in (0, 1, p - 1):
            result = simulate(c, {"x": xv, "y": yv}, rng=random.Random(xv + yv))
            assert result.outputs == {"x": xv, "y": xv * yv % p}
            assert result.phase == 0


@pytest.mark.parametrize("p", [13, 29, 31])
def test_reverse_replay_preserves_packed_tape_independently(p):
    from qsm.inversion.reference import to_dialog_jump

    n = p.bit_length()
    iters = schedule.conservative_iterations_j2(n)
    c = Circuit()
    d = c.input("d", pack.packed_bits_j2(iters))
    z = c.input("z", n)
    quotient, empty = un_apply_dialog_canonical_inplace(
        c, d, c.alloc(n), z, iters, p)
    c.free(empty)
    c.rebind("z", quotient)
    for x in range(1, p):
        tape, u, v = to_dialog_jump(p, x, iters)
        assert (u, v) == (1, 0)
        encoded = pack.pack_dialog_j2(tape, boundary=True)
        for z in range(p):
            result = simulate(c, {"d": encoded, "z": z}, rng=random.Random(x + z))
            assert result.outputs == {"d": encoded, "z": z * pow(x, -1, p) % p}
            assert result.phase == 0


@pytest.mark.parametrize("p", [13, 29, 31, 61, 127])
def test_gcd_bound_invariants_at_every_iteration(p):
    n = p.bit_length()
    bound = schedule.conservative_iterations_j2(n)
    for x in range(1, p):
        u, v = p, x
        # Model the implementation boundary explicitly, including x odd and /4.
        for _ in range(2):
            if not v & 1:
                v //= 2
        if v & 1:
            u, v = v, u - v
        assert u > 0 and u & 1 and not v & 1 and 0 <= v < 1 << n
        for _ in range(1, bound + 3):
            old = u * v
            v //= 2
            if not v & 1:
                v //= 2
            if v & 1:
                if v < u:
                    u, v = v, u
                v -= u
            assert u > 0 and u & 1 and not v & 1 and v >= 0
            assert u * v <= old // 2
        assert (u, v) == (1, 0)
    # Compare actual circuit boundary + fixed terminal with the integer invariants.
    for iters in (1, bound, bound + 1):
        c = Circuit()
        u, v = c.input("u", n), c.input("v", n)
        tape = PackedTapeJ2(c)
        uo, vo = gcd_loop_j2(c, u, v, tape, iters=iters, widths=[n] * iters,
                             add_ctrl=gidney_add_ctrl_threaded)
        c.rebind("u", uo)
        c.rebind("v", vo)
        for x in range(1, p):
            result = simulate(c, {"u": p, "v": x}, rng=random.Random(x))
            assert result.outputs["u"] > 0 and result.outputs["u"] & 1
            assert not result.outputs["v"] & 1
            if iters >= bound:
                assert result.outputs == {"u": 1, "v": 0}
            assert result.phase == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_zero_divisor_is_out_of_domain(reverse):
    c = build(13, reverse)
    with pytest.raises(DirtyAncillaError):
        simulate(c, {"x": 0, "y": 1})


@pytest.mark.slow
@pytest.mark.parametrize("p", [(1 << 255) - 19, (1 << 256) - (1 << 32) - 977])
@pytest.mark.parametrize("reverse", [False, True])
def test_fullwidth_structured_operands(p, reverse):
    n, f = p.bit_length(), (1 << p.bit_length()) - p
    c = build(p, reverse)
    resources = count(c)
    print(f"canonical n={n} reverse={reverse}: {resources}")
    values = sorted({0, 1, f - 1, f, f + 1, 1 << (n // 2), p - 1})
    pairs = {(x, y) for x in values if x for y in values}
    # Exact-replay deficit band from the original audit, with nontrivial divisors.
    pairs.update((x, y) for x in (2, 3, p - 2) for y in (1, 3, 17, 19, 20, 44))
    for seed, (x, y) in enumerate(sorted(pairs)):
        result = simulate(c, {"x": x, "y": y}, rng=random.Random(seed))
        factor = pow(x, -1, p) if reverse else x
        assert result.outputs == {"x": x, "y": y * factor % p}
        assert result.phase == 0
