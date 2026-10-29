"""Classical scalar oracles checked without assuming the claimed group order."""

import pytest

from qsm.point import curve, secp256k1


def _add_repeated(pt, qt, p, a):
    """Independent affine addition for the linear-time, small-field reference.

    No scalar multiplication, order reduction, or production group helper is used.
    Fermat inversion keeps this reference separate from the oracle's inverse call.
    """
    if pt is None:
        return qt
    if qt is None:
        return pt
    x, y = pt
    u, v = qt
    if x == u and (y + v) % p == 0:
        return None
    numerator, denominator = ((3 * x * x + a, 2 * y) if pt == qt
                              else (v - y, u - x))
    slope = numerator * pow(denominator % p, p - 2, p) % p
    rx = (slope * slope - x - u) % p
    return rx, (slope * (x - rx) - y) % p


@pytest.mark.parametrize("p,a,b", [(5, 0, 1), (7, 2, 3), (13, 1, 0), (17, 2, 2)])
def test_small_curve_scalar_multiplication_matches_repeated_additions(p, a, b):
    from qsm.point.group import scalar_mul

    assert (4 * a**3 + 27 * b**2) % p != 0
    points = [(x, y) for x in range(p) for y in range(p)
              if (y * y - x**3 - a * x - b) % p == 0]
    # Traverse more than two full group orders, visiting identity intermediates
    # as well as every point. The helper gets p and a, never a claimed order.
    for point in points:
        positive = negative = None
        inverse = (point[0], -point[1] % p)
        for k in range(2 * (len(points) + 1) + 2):
            assert scalar_mul(k, *point, p=p, a=a) == positive, (p, point, k)
            assert scalar_mul(-k, *point, p=p, a=a) == negative, (p, point, -k)
            positive = _add_repeated(positive, point, p, a)
            negative = _add_repeated(negative, inverse, p, a)


@pytest.mark.parametrize("model", [curve, secp256k1], ids=["wei25519", "secp256k1"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 7, 12345])
def test_public_scalar_oracles_handle_signed_scalars(model, k):
    positive = model.mul(k, model.GX, model.GY)
    expected = None if positive is None else model.neg(*positive)
    assert model.mul(-k, model.GX, model.GY) == expected


@pytest.mark.parametrize("model", [curve, secp256k1], ids=["wei25519", "secp256k1"])
def test_public_scalar_oracles_preserve_explicit_modulus(model):
    p = 13
    a = model.A % p
    # b is irrelevant to addition; choose a nonsingular curve containing the point.
    x, y = next((x, y) for x in range(p) for y in range(1, p)
                if (4 * a**3 + 27 * ((y * y - x**3 - a * x) % p)**2) % p)
    expected = None
    for k in range(32):
        assert model.mul(k, x + p, y + p, p) == expected
        expected = _add_repeated(expected, (x, y), p, a)
