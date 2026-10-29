"""Wider fixed-torsion initialization without assuming phase-exact arithmetic."""
from itertools import product

import coset_fixtures as fixture
from qsm.audit.full_scalar import Group, _initial_points, run, windows
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.point.group import scalar_mul
from qsm.scalarmul.lookup import lookup_xor_unary


def test_two_bit_initializer_all_words_and_measurement_outcomes():
    group = Group(**fixture.group_kwargs(), torsion=fixture.T)
    slot = windows(4,4,2)[0]
    points = _initial_points(group,slot,"odd_torsion")
    c = Circuit()
    word,x,y = c.input("word",2),c.input("x",5),c.input("y",5)
    lookup_xor_unary(c,word,Register(x.ids+y.ids),tuple(xx | yy << 5 for xx,yy in points))
    resources = count(c)
    assert resources.toffoli == resources.measurements == 2
    projection = scalar_mul(group.order,*group.torsion,p=group.p,a=group.a)
    for raw in range(4):
        expected = group.add(group.torsion,group.mul((2*raw-3)*4*pow(2,-1,group.order),group.P))
        assert points[raw] == expected
        assert scalar_mul(group.order,*expected,p=group.p,a=group.a) == projection
        for outcomes in product((0,1), repeat=resources.measurements):
            result = simulate(c,{"word":raw,"x":0,"y":0},measurements=list(outcomes))
            assert result.outputs == {"word":raw,"x":expected[0],"y":expected[1]}
            assert result.phase == 0


def test_four_by_four_chain_retains_all_inputs_and_correct_values():
    group = Group(**fixture.group_kwargs(),torsion=fixture.T)
    pairs = list(product(range(16),repeat=2))
    result = run(group=group,u_bits=4,v_bits=4,window_bits=2,variant="odd_torsion",
                 input_pairs=pairs,full_width=True,recycled=True)
    assert result["stages"][0]["bits"] == 2
    assert result["point_additions"] == 3
    assert result["resources"]["toffoli"] == 15278
    assert len(result["diagnostics"]) == 256
    for row in result["diagnostics"]:
        expected = group.add(group.torsion,group.add(
            group.mul((2*row["u"]-15)*pow(2,-1,group.order),group.P),
            group.mul((2*row["v"]-15)*pow(2,-1,group.order),group.Q)))
        assert row["expected"] == row["output"] == expected
        assert not any(row[key] for key in ("domain_failure","dirty_failure","value_failure","control_failure","noncanonical_stages"))
    assert any(row["horizontal_stages"] for row in result["diagnostics"])
    assert any(not row["observed_pass"] for row in result["diagnostics"])
