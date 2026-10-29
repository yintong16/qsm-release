"""Preserve the experimental counterexamples separately from the canonical regressions."""

import random

import pytest

from qsm.bench.registry import get_builder
from qsm.core.circuit import DirtyAncillaError
from qsm.core.simulator import simulate
from qsm.params import FIELD
from qsm.point import curve
from qsm.point.affine import add_const_affine_inplace_ref


@pytest.mark.slow
@pytest.mark.skipif(FIELD != "25519", reason="fixed Wei25519 subgroup counterexamples")
def test_tuned_point_error_model_is_not_a_worst_case_guarantee():
    points = [
        (54612104369102466295186403110110132856877110769130571327805934252953404584066,
         curve.GY + 20),
        (14442679257967034402166127258426498254760573270987925111286464008990023459040,
         curve.GY + 44),
    ]
    case = get_builder("point.add_const_affine_inplace.v1")()
    assert "distribution-conditioned" in case.meta["arithmetic_contract"]
    for x, y in points:
        assert curve.is_on_curve(x, y)
        assert curve.mul(curve.ORDER, x, y) is None
        assert add_const_affine_inplace_ref(x, y, curve.GX, curve.GY) == curve.add(
            x, y, curve.GX, curve.GY)
        with pytest.raises(DirtyAncillaError):
            simulate(case.circuit, {"x": x, "y": y}, rng=random.Random(0))
