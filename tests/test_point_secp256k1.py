"""secp256k1's constants, and that its group law is the same function `curve`'s is.

Wei25519's constants are DERIVED from the Montgomery form, so `test_point_affine.py` can
check them against a published document. secp256k1's are transcribed, so the check has to
be internal: pin them against the curve equation and the group order, which between them
leave no freedom. A typo in any digit of p, b, Gx, Gy or n fails at least one of these.
"""

from __future__ import annotations

import random

import pytest

from qsm.point import secp256k1 as s


def test_p_is_the_pseudo_mersenne_form_the_circuits_assume():
    assert s.P == (1 << 256) - (1 << 32) - 977
    f = (1 << 256) - s.P
    assert f == (1 << 32) + 977
    assert f & 1, "odd F is assumed throughout the field layer"
    assert pow(2, s.P - 1, s.P) == 1, "p is not prime"
    assert s.P.bit_length() == 256


def test_the_base_point_is_on_the_curve():
    assert s.is_on_curve(s.GX, s.GY)
    assert (s.A, s.B) == (0, 7)


def test_the_order_is_prime_and_kills_the_base_point():
    assert pow(2, s.ORDER - 1, s.ORDER) == 1
    assert s.mul(s.ORDER, s.GX, s.GY) is None
    assert s.mul(s.ORDER + 1, s.GX, s.GY) == (s.GX, s.GY)


def test_an_incorrect_claimed_order_does_not_kill_the_base_point(monkeypatch):
    monkeypatch.setattr(s, "ORDER", s.ORDER + 1)
    assert s.mul(s.ORDER, s.GX, s.GY) == (s.GX, s.GY)


def test_cofactor_one_means_every_point_is_in_the_prime_subgroup():
    """Unlike 25519's h = 8. Nothing in the circuit depends on it; recorded because the
    absence of small-subgroup structure is one more thing the two curves do NOT share."""
    assert s.COFACTOR == 1


@pytest.mark.parametrize("k", [1, 2, 3, 7, 12345, 2**200 + 1])
def test_scalar_multiples_stay_on_the_curve(k: int):
    pt = s.mul(k, s.GX, s.GY)
    assert pt is not None
    assert s.is_on_curve(*pt)


def test_incomplete_addition_agrees_with_double_and_add():
    """`add` is the function the circuit computes; `mul` is the group-law oracle."""
    rng = random.Random(2026)
    for _ in range(20):
        i, j = rng.randrange(2, 10**6), rng.randrange(2, 10**6)
        if i == j:
            continue
        pi, pj = s.mul(i, s.GX, s.GY), s.mul(j, s.GX, s.GY)
        assert s.add(*pi, *pj) == s.mul(i + j, s.GX, s.GY)


def test_add_refuses_the_exceptional_inputs_exactly_as_wei25519_does():
    """The precondition is the same one, which is what makes the port a fair experiment."""
    pt = s.mul(5, s.GX, s.GY)
    with pytest.raises(AssertionError):
        s.add(*pt, *pt)                       # doubling
    with pytest.raises(AssertionError):
        s.add(*pt, *s.neg(*pt))               # the identity


def test_negation_is_an_involution_and_lands_on_the_curve():
    pt = s.mul(9, s.GX, s.GY)
    assert s.neg(*s.neg(*pt)) == pt
    assert s.is_on_curve(*s.neg(*pt))
