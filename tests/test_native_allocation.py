"""Native subgroup-sized controls leave the arithmetic windows unchanged."""
import random

import pytest

from qsm.audit.full_scalar import active_group, _initial_points, shift_point, windows
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.params import FIELD
from qsm.point.group import scalar_mul
from qsm.scalarmul.lookup import lookup_xor_unary


@pytest.mark.parametrize("u_bits,top_bits", [(253, 13), (255, 15), (256, 16)])
def test_native_and_reference_schedule_partitions(u_bits, top_bits):
    schedule = windows(u_bits, 208, 16)
    assert len(schedule) == 29
    assert schedule[0] == {"axis": "u", "offset": 240, "bits": top_bits}
    assert schedule[16] == {"axis": "v", "offset": 192, "bits": 16}
    assert all(slot["bits"] == 16 for slot in schedule[1:])
    for axis, width in (("u", u_bits), ("v", 208)):
        slots = [s for s in schedule if s["axis"] == axis]
        assert sum(s["bits"] for s in slots) == width
        covered = [bit for s in slots for bit in range(s["offset"], s["offset"] + s["bits"])]
        assert sorted(covered) == list(range(width))


def test_removing_first_register_padding_preserves_every_arithmetic_slot():
    native = windows(253, 208, 16)
    old = windows(255, 208, 16)
    assert native[1:] == old[1:]
    assert (1 << old[0]["bits"]) - (1 << native[0]["bits"]) == 24_576


@pytest.mark.parametrize("width", [253, 208])
def test_odd_digit_bias_identity_at_native_register_widths(width):
    rng = random.Random(20261014)
    values = [0, 1, 1 << (width-1), (1 << width)-1,
              *(rng.getrandbits(width) for _ in range(4))]
    slots = [s for s in windows(width, 208, 16) if s["axis"] == "u"]
    for value in values:
        total = 0
        for slot in slots:
            raw = (value >> slot["offset"]) & ((1 << slot["bits"])-1)
            digit = 2*raw - ((1 << slot["bits"])-1)
            total += digit << slot["offset"]
        assert total == 2*value - ((1 << width)-1)


@pytest.mark.parametrize("variant", ["signed", "odd_torsion"])
def test_native_fixed_translation_matches_closed_group_formula(variant):
    if variant == "odd_torsion" and FIELD != "25519":
        pytest.skip("the native torsion construction requires Wei25519's cofactor")
    group = active_group(torsion=variant == "odd_torsion")
    a = group.order.bit_length()
    schedule = windows(a, 208, 16)
    if variant == "odd_torsion":
        half = pow(2, -1, group.order)
        expected = group.add(group.torsion, group.add(
            group.mul(-((1 << a)-1)*half, group.P),
            group.mul(-((1 << 208)-1)*half, group.Q)))
    else:
        p_bias = sum(1 << (s["offset"]+s["bits"]-1)
                     for s in schedule if s["axis"] == "u")
        q_bias = sum(1 << (s["offset"]+s["bits"]-1)
                     for s in schedule if s["axis"] == "v")
        expected = group.add(group.mul(pow(2, -1, group.order)-p_bias, group.P),
                             group.mul(-q_bias, group.Q))
    assert shift_point(group, schedule, variant) == expected


@pytest.mark.parametrize("variant", ["signed", "odd_torsion"])
@pytest.mark.parametrize("u_bits,top_bits", [(253, 13), (255, 15), (256, 16)])
def test_actual_initializer_cost_and_sampled_lookup_semantics(variant, u_bits, top_bits):
    if variant == "odd_torsion" and FIELD != "25519":
        pytest.skip("the native torsion construction requires Wei25519's cofactor")
    group = active_group(torsion=variant == "odd_torsion")
    slot = windows(u_bits, 208, 16)[0]
    points = _initial_points(group, slot, variant)
    assert len(points) == 1 << top_bits
    assert all(point is not None for point in points)
    c = Circuit()
    word = c.input("word", top_bits)
    packed = c.input("point", 2*group.n)
    lookup_xor_unary(c, word, packed, tuple(x | (y << group.n) for x, y in points))
    resources = count(c)
    assert resources.toffoli == resources.measurements == (1 << top_bits)-2
    assert resources.peak_qubits == 2*group.n + 2*top_bits-1
    weighted = group.mul(1 << slot["offset"], group.P)
    half = pow(2, -1, group.order)
    for raw in (0, 1, 1 << (top_bits-1), (1 << top_bits)-1):
        if variant == "odd_torsion":
            expected = group.add(group.torsion,
                group.mul((2*raw-((1 << top_bits)-1))*half, weighted))
            projection = scalar_mul(group.order, *expected, p=group.p, a=group.a)
            assert projection == scalar_mul(group.order, *group.torsion, p=group.p, a=group.a)
        else:
            expected = group.add(group.mul(half, group.P),
                                 group.mul(raw-(1 << (top_bits-1)), weighted))
        assert points[raw] == expected
        result = simulate(c, {"word": raw, "point": 0}, rng=random.Random(20261014))
        assert result.outputs == {"word": raw, "point": expected[0] | (expected[1] << group.n)}
        assert result.phase == 0
