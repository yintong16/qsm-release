"""One immutable configuration drives construction and its declared model."""
from dataclasses import FrozenInstanceError, replace

import pytest

from qsm import params
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.inversion import schedule
from qsm.inversion.config import InversionConfig
from qsm.inversion.inplace import modmul_inplace, modmul_conservative_inplace


def test_immutable_and_field_resolved():
    cfg = InversionConfig.conservative(5, 29)
    assert cfg.resolved(5, 29) is cfg
    assert cfg.iters == 11
    assert cfg.widths == (5,) * 11
    assert cfg.fail_prob() == cfg.fail_prob(reverse=True) == 0
    with pytest.raises(FrozenInstanceError):
        cfg.iters = 12
    with pytest.raises(ValueError, match="width/modulus"):
        cfg.resolved(5, 31)
    mutable = list(cfg.widths)
    copied = replace(cfg, widths=mutable)
    mutable[0] = 1
    assert copied.widths == cfg.widths


@pytest.mark.parametrize("change", [
    {"iters": 10, "widths": (5,) * 10},
    {"gcd_cmp_bits": 3}, {"double_window": 3},
    {"widths": (5,) * 10 + (4,)}, {"arithmetic_family": "unknown"},
    {"p": 32}, {"p": 30}, {"iters": 0},
])
def test_conservative_rejects_inexact_or_invalid_configuration(change):
    with pytest.raises(ValueError):
        replace(InversionConfig.conservative(5, 29), **change)


def test_defaults_match_existing_component_model():
    cfg = InversionConfig.tuned()
    assert cfg.iters == schedule.iterations_j2(params.U)
    assert cfg.widths == tuple(schedule.widths_j2(params.U, cfg.iters))
    expected = 2 * (schedule.iteration_fail_prob_j2(params.U, cfg.iters)
                    + params.gcd_width_clip_fail_prob(n=params.U, jump=2)
                    + schedule.compare_fail_prob(schedule.compare_bits(params.U)
                                                 - schedule.width_pad(params.U)))
    assert cfg.phase1_fail_prob() == expected
    assert cfg.fail_prob() == expected + params.replay_j2_fail_prob(
        cfg.iters, cmp_bits=params.CMP_BITS, chunk=43, chunk_cmp_bits=32)
    assert "distribution-conditioned" in cfg.assumptions


def test_retuning_changes_constructor_and_model_together():
    # At tiny widths full-window arithmetic allows a cheap construction test.
    base = InversionConfig.tuned(5, 31, iters=8)
    tuned = InversionConfig.tuned(5, 31, iters=9, replay_chunk=4,
                                  replay_chunk_cmp_bits=3, carry_window=4)
    def build(cfg):
        c = Circuit()
        x, y = c.input("x", 5), c.input("y", 5)
        xo, yo = modmul_inplace(c, x, y, 31, config=cfg)
        c.rebind("x", xo)
        c.rebind("y", yo)
        return count(c)
    assert build(base).toffoli != build(tuned).toffoli
    assert base.phase1_fail_prob() != tuned.phase1_fail_prob()
    assert base.replay_fail_prob() != tuned.replay_fail_prob()
    assert tuned.replay_kwargs()["carry_window"] == tuned.carry_window
    assert tuned.gcd_kwargs()["cmp_bits"] == tuned.gcd_cmp_bits
    assert tuned.replay_fail_prob() == params.replay_j2_fail_prob(
        tuned.iters, double_window=tuned.double_window, carry_window=tuned.carry_window,
        cmp_bits=tuned.replay_cmp_bits, u=5, f=1, chunk=4, chunk_cmp_bits=3)


def test_custom_narrowing_has_no_silent_default_analytic_model():
    cfg = InversionConfig.tuned()
    widths = list(cfg.widths)
    widths[-1] -= 1
    custom = replace(cfg, widths=widths)
    for model in (custom.phase1_fail_prob, custom.replay_fail_prob, custom.fail_prob):
        with pytest.raises(ValueError, match="custom inversion width schedule"):
            model()


def test_two_fields_have_their_own_replay_error():
    a = InversionConfig.tuned(255, (1 << 255) - 19)
    b = InversionConfig.tuned(256, (1 << 256) - (1 << 32) - 977)
    assert b.replay_fail_prob() == params.replay_j2_fail_prob(
        b.iters, double_window=b.double_window, carry_window=b.carry_window,
        cmp_bits=b.replay_cmp_bits, u=b.n, f=(1 << b.n) - b.p,
        chunk=b.replay_chunk, chunk_cmp_bits=b.replay_chunk_cmp_bits)
    assert a.replay_fail_prob() != b.replay_fail_prob()


def test_default_windows_resolve_from_supplied_field_not_process_profile():
    small = InversionConfig.tuned(255, (1 << 255) - 19)
    large = InversionConfig.tuned(256, (1 << 256) - (1 << 32) - 977)
    assert small.carry_window == small.double_window == 45
    assert large.carry_window == large.double_window == 73
    assert large.replay_kwargs()["carry_window"] == 73
    assert large.replay_kwargs()["double_window"] == 73
    exact = InversionConfig.tuned(256, large.p, carry_window=None, double_window=None)
    assert exact.carry_window is exact.double_window is None
    explicit = InversionConfig.tuned(256, large.p, carry_window=81, double_window=82)
    assert (explicit.carry_window, explicit.double_window) == (81, 82)
    # Exact integer logarithm at deficit one and just below a power of two.
    assert InversionConfig.tuned(5, 31).carry_window == params.ERROR_BUDGET
    assert InversionConfig.tuned(5, 17).carry_window == params.ERROR_BUDGET + 4
    assert large.replay_fail_prob() == params.replay_j2_fail_prob(
        large.iters, double_window=73, carry_window=73,
        cmp_bits=large.replay_cmp_bits, u=large.n, f=(1 << 32) + 977,
        chunk=large.replay_chunk, chunk_cmp_bits=large.replay_chunk_cmp_bits)


def test_config_cannot_conflict_with_legacy_kwargs_or_family():
    c = Circuit()
    x, y = c.input("x", 5), c.input("y", 5)
    cfg = InversionConfig.tuned(5, 31)
    with pytest.raises(ValueError, match="legacy"):
        modmul_inplace(c, x, y, 31, iters=cfg.iters, config=cfg)
    with pytest.raises(ValueError, match="canonical configuration"):
        modmul_conservative_inplace(c, x, y, 31, config=cfg)
    with pytest.raises(ValueError, match="separately named"):
        modmul_inplace(c, x, y, 31, config=InversionConfig.conservative(5, 31))


@pytest.mark.slow
def test_tuned_registered_costs_are_unchanged():
    from qsm.bench.registry import get_builder
    expected = {"inv.modmul_inplace.v1": 754544, "inv.moddiv_inplace.v1": 755066}
    if params.U != 255 or params.F != 19:
        pytest.skip("published baseline belongs to the 25519 field")
    for key, toffoli in expected.items():
        resources = count(get_builder(key)().circuit)
        assert resources.toffoli == toffoli
        assert resources.peak_qubits == 1173
