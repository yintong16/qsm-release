"""Canonical means exact values, clean releases AND phase, including zero."""
import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.simulator import simulate
from qsm.field import canonical


@pytest.mark.parametrize("p", [13, 29, 31])
@pytest.mark.parametrize("name,sign", [("add_inplace", 1), ("sub_inplace", -1)])
@pytest.mark.parametrize("controlled", [False, True])
def test_binary_exhaustive(p, name, sign, controlled):
    c = Circuit()
    n = p.bit_length()
    a, b = c.input("a", n), c.input("b", n)
    ctrl = c.input("ctrl", 1)[0] if controlled else None
    c.rebind("b", getattr(canonical, name)(c, a, b, p, ctrl=ctrl))
    for av in range(p):
        for bv in range(p):
            for cv in ([0, 1] if controlled else [1]):
                values = {"a": av, "b": bv}
                if controlled:
                    values["ctrl"] = cv
                for seed in (0, 1):
                    result = simulate(c, values, rng=random.Random(seed))
                    assert result.outputs == {**values, "b": (bv + cv * sign * av) % p}
                    assert result.phase == 0


@pytest.mark.parametrize("p", [13, 29, 31])
@pytest.mark.parametrize("name", ["double_inplace", "halve_inplace", "neg_inplace"])
@pytest.mark.parametrize("controlled", [False, True])
def test_unary_exhaustive(p, name, controlled):
    c = Circuit()
    b = c.input("b", p.bit_length())
    ctrl = c.input("ctrl", 1)[0] if controlled else None
    c.rebind("b", getattr(canonical, name)(c, b, p, ctrl=ctrl))
    factor = {"double_inplace": 2, "halve_inplace": pow(2, -1, p), "neg_inplace": -1}[name]
    for bv in range(p):
        for cv in ([0, 1] if controlled else [1]):
            values = {"b": bv}
            if controlled:
                values["ctrl"] = cv
            for seed in range(4):
                result = simulate(c, values, rng=random.Random(seed))
                assert result.outputs == {**values, "b": (bv * factor) % p if cv else bv}
                assert result.phase == 0


def test_relabelled_chain_and_alias_guard():
    c = Circuit()
    b = c.input("b", 5)
    for _ in range(12):
        b = canonical.double_inplace(c, b, 29)
    for _ in range(12):
        b = canonical.halve_inplace(c, b, 29)
    c.rebind("b", b)
    for v in range(29):
        result = simulate(c, {"b": v}, rng=random.Random(v))
        assert result.outputs == {"b": v}
        assert result.phase == 0
    with pytest.raises(ValueError, match="disjoint"):
        canonical.add_inplace(c, b, b, 29)
