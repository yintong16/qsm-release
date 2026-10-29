"""Wide measured words exercise parity bits beyond machine-word boundaries."""
import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.simulator import simulate
from qsm.scalarmul.lookup import lookup_xor_unary, unlookup_measure


@pytest.mark.parametrize("width", [255, 256, 510, 512])
@pytest.mark.parametrize("split", [0, 1, 2])
def test_wide_word_parity_and_repeated_shots(width, split):
    rng = random.Random(381 + width)
    rows = (0, 1 << (width-1), (1 << width)-1, rng.getrandbits(width))
    c = Circuit()
    address = c.input("j", 2)
    target = c.alloc(width)
    lookup_xor_unary(c, address, target, rows)
    unlookup_measure(c, address, target, rows, low_bits=split)
    assert c._live == 2
    for digit in range(4):
        for seed in range(4):
            result = simulate(c, {"j": digit}, rng=random.Random(41*digit + seed))
            assert result.outputs == {"j": digit}
            assert result.phase == 0
