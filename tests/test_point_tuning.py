"""The point builder and its analytic declaration use the same resolved configuration."""

import pytest

from qsm.inversion.config import InversionConfig
from qsm.params import PRIME, U
from qsm.point import affine


def test_point_construction_and_accounting_share_configuration(monkeypatch):
    used = []

    def point(circ, x, y, *args, inv_kw, **kwargs):
        used.append(inv_kw["config"])
        return x, y

    monkeypatch.setattr(affine, "add_const_affine_inplace", point)
    default = InversionConfig.tuned(U, PRIME)
    changed = InversionConfig.tuned(U, PRIME, iters=default.iters + 3)
    first = affine._affine_case(False, default)
    second = affine._affine_case(False, changed)
    assert used == [default, changed]
    assert first.meta["fail_analytic"] == affine._fail(default)
    assert second.meta["fail_analytic"] == affine._fail(changed)
    assert first.meta["fail_analytic"] != second.meta["fail_analytic"]
    remainder = affine.TRISUB_TRUNC_R + affine._CONST_SUBS * affine._const_sub_fail()
    for config, case in ((default, first), (changed, second)):
        assert case.meta["fail_analytic"] == pytest.approx(
            config.fail_prob() + config.fail_prob(reverse=True) + remainder)
    assert "distribution-conditioned" in first.meta["arithmetic_contract"]


def test_point_rejects_configuration_from_another_field():
    wrong = InversionConfig.tuned(5, 31)
    with pytest.raises((ValueError, AssertionError)):
        affine._fail(wrong)


def test_square_ripple_is_field_independent_and_v1_stays_pinned(monkeypatch):
    from qsm.params import CARRY_WINDOW, SQUARE_RIPPLE
    seen = []

    def point(circ, x, y, *args, sqr_kw, **kwargs):
        seen.append(sqr_kw["ripple"])
        return x, y

    monkeypatch.setattr(affine, "add_const_affine_inplace", point)
    # Mocked emitters must not populate conftest's persistent registry resource cache.
    for build, ripple in [(affine._build_add_const_affine_inplace_v1, CARRY_WINDOW),
                          (affine._build_add_const_affine_inplace_ctrl_v1, CARRY_WINDOW),
                          (affine._build_add_const_affine_inplace_v2, SQUARE_RIPPLE),
                          (affine._build_add_const_affine_inplace_ctrl_v2, SQUARE_RIPPLE)]:
        case = build()
        assert seen[-1] == ripple == case.meta["square_ripple"]
    legacy = affine._fail(legacy_square_ripple=True)
    current = affine._fail()
    assert legacy - current == pytest.approx(affine.TRISUB_TRUNC - affine.TRISUB_TRUNC_R)
