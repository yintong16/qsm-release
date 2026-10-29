"""Classical domain witnesses, not quantum-circuit completeness tests."""
import pytest

from qsm.point.group import scalar_mul
from qsm.scalarmul.signed_window import _group_add


P = (1 << 255)-19
R = (1 << 252)+27742317777372353535851937790883648493
A = (3-486662**2)*pow(3, -1, P) % P
B = (2*486662**3-9*486662)*pow(27, -1, P) % P
G = ((9+486662*pow(3, -1, P)) % P,
     43114425171068552920764898935933967039370386198203806730763910166200978582548)
T0 = (19624287790469256669057814461137428606839005560001276145469620721820115669041,
      25869741026945134960544184956460972567356779614910045322022475500191642319642)


def mul(k, point):
    return None if point is None else scalar_mul(k, *point, p=P, a=A)


def add(left, right):
    return _group_add(left, right, P, A)


def test_order_eight_offset_and_prime_subgroup_are_distinct():
    assert (T0[1]**2-T0[0]**3-A*T0[0]-B) % P == 0
    assert (G[1]**2-G[0]**3-A*G[0]-B) % P == 0
    assert mul(R, G) is None
    assert mul(8, T0) is None
    assert mul(4, T0) is not None
    assert mul(R, T0) is not None


@pytest.mark.parametrize("scalar", [0, 1, 2, R-1])
@pytest.mark.parametrize("digit", [-9, -1, 1, 9])
def test_coset_additions_and_sign_conjugation_preserve_required_domain(scalar, digit):
    acc = add(T0, mul(scalar, G))
    operand = mul(digit, G)
    output = add(acc, operand)
    assert acc is not None and operand is not None and output is not None
    assert acc[0] != operand[0] and output[0] != operand[0]
    assert acc[1] != 0 and output[1] != 0
    assert mul(R, acc) == mul(R, output) == mul(R, T0)
    if digit < 0:
        transient = add(mul(-1, acc), mul(-1, operand))
        assert mul(R, transient) == mul(-R, T0)
        assert mul(-1, transient) == output


@pytest.mark.parametrize("kind,u,last_word,last_digit", [
    ("cancel", ((1 << 255)-1+R)//2, 59894, 54253),
    ("double", ((1 << 255)-1+R)//2-54253, 5641, -54253),
    ("output_x", ((1 << 255)-1+2*R-21485)//2, 43510, 21485),
])
def test_natural_all_odd_final_window_witnesses(kind, u, last_word, last_digit):
    assert 0 <= u < 1 << 255
    half = mul(pow(2, -1, R), G)
    acc = None
    coefficient = 0
    for offset in range(240, -1, -16):
        width = 15 if offset == 240 else 16
        word = (u >> offset) & ((1 << width)-1)
        digit = 2*word-((1 << width)-1)
        operand = mul(digit*(1 << offset), half)
        output = add(acc, operand)
        coefficient += digit*(1 << offset)
        if offset == 240:
            assert output is not None
        elif offset:
            assert acc is not None and output is not None
            assert acc[0] != operand[0] and output[0] != operand[0]
            assert acc[1] != operand[1]
        else:
            assert word == last_word and digit == last_digit
            if kind == "cancel":
                assert acc == mul(-1, operand) and output is None
            elif kind == "double":
                assert acc == operand
            else:
                assert acc == mul(-2, operand)
                assert output == mul(-1, operand)
                assert acc[0] != operand[0] and acc[1] != operand[1]
                assert output[0] == operand[0]
        acc = output
    assert coefficient == 2*u-((1 << 255)-1)
    assert acc == mul(coefficient, half)
