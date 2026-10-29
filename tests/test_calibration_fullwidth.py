"""Slow emitted-gate cross-checks of full-width all-outcome analysis."""
import random

import pytest

from qsm.core.circuit import DirtyAncillaError
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.params import FIELD, PRIME, U
from qsm.point import active as curve
from qsm.point.sampling import SubgroupSampler
from qsm.scalarmul.window_opt import OptimizedWindowConfig, build_optimized_window_case


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["paired", "fused"])
def test_fullwidth_batch_vs_emitted_measurement_shots(strategy):
    config = OptimizedWindowConfig.tuned(U, PRIME, 4, operand_strategy=strategy)
    case = build_optimized_window_case(4, config=config)
    plan = compile_symbolic(case.circuit)
    sampler = SubgroupSampler(PRIME, curve.CURVE.A, curve.CURVE.B,
                              curve.ORDER, curve.CURVE.COFACTOR)
    rng = random.Random(7310)
    inputs, expected = [], []
    while len(inputs) < 8:
        x, y = sampler.sample(rng)
        inp = {"j": rng.randrange(16), "x": x, "y": y}
        try:
            out = case.reference(inp)
        except AssertionError:
            continue
        inputs.append(inp)
        expected.append(out)
    result = simulate_batch(plan, inputs, expected, output_moduli={"x": PRIME, "y": PRIME})
    for lane, (inp, out) in enumerate(zip(inputs, expected)):
        for seed in (17, 29):
            try:
                actual = simulate(case.circuit, inp, rng=random.Random(seed))
            except DirtyAncillaError:
                assert result.bad_mask >> lane & 1
                continue
            if not (result.bad_mask >> lane & 1):
                assert actual.outputs == out
                assert actual.phase == 0
            assert all(actual.outputs[key] == result.outputs[lane][key] for key in out)

    if FIELD == "25519":
        points = [
            (54612104369102466295186403110110132856877110769130571327805934252953404584066,
             curve.GY + 20),
            (14442679257967034402166127258426498254760573270987925111286464008990023459040,
             curve.GY + 44),
        ]
        inputs = [{"j": 1, "x": x, "y": y} for x, y in points]
        expected = [case.reference(inp) for inp in inputs]
        result = simulate_batch(plan, inputs, expected, output_moduli={"x": PRIME, "y": PRIME})
        assert result.bad_count == 2
        assert result.dirty_count == 2
        zeros = [0] * sum(op.kind == "MEASURE" for op in case.circuit.ops)
        for inp in inputs:
            with pytest.raises(DirtyAncillaError):
                simulate(case.circuit, inp, measurements=zeros)
