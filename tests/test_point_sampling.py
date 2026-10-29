"""Exact proposal counting, with an independent small-curve group oracle."""
from collections import Counter
import json
import random

import pytest

from qsm.point.group import scalar_mul
from qsm.point.sampling import SubgroupSampler, sample_subgroup_point, sqrt_mod


# Deliberately no production group helper in the exhaustive small-curve oracle.
def add_points(left, right, p, a):
    if left is None:
        return right
    if right is None:
        return left
    x, y = left
    u, v = right
    if x == u and (y + v) % p == 0:
        return None
    if left == right:
        numerator, denominator = 3 * x * x + a, 2 * y
    else:
        numerator, denominator = v - y, u - x
    # Independent inverse by exhaustive search, not the production pow(-1) path.
    inverse = next(i for i in range(1, p) if denominator * i % p == 1)
    slope = numerator * inverse % p
    z = (slope * slope - x - u) % p
    return z, (slope * (x - z) - y) % p


def repeated_add(k, point, p, a):
    answer = None
    for _ in range(k):
        answer = add_points(answer, point, p, a)
    return answer


def affine_points(p, a, b):
    return [(x, y) for x in range(p) for y in range(p)
            if (y * y - x**3 - a * x - b) % p == 0]


class RejectedProposal(Exception):
    pass


class OneProposal:
    """Stop at the second x request so a rejection loop cannot mask proposal mass."""
    def __init__(self, p, x, sign):
        self.p, self.x, self.sign = p, x, sign
        self.used_x = self.used_sign = False

    def randrange(self, stop):
        if stop == self.p:
            if self.used_x:
                raise RejectedProposal
            self.used_x = True
            return self.x
        assert stop == 2 and self.used_x and not self.used_sign
        self.used_sign = True
        return self.sign


@pytest.mark.parametrize("p", [7, 11, 13, 19, 29, 43])
def test_sqrt_all_residues_and_nonresidues(p):
    squares = {y * y % p for y in range(p)}
    for value in range(-p, 2 * p):
        root = sqrt_mod(value, p)
        if value % p not in squares:
            assert root is None
        else:
            assert type(root) is int and 0 <= root < p
            assert root * root % p == value % p
    assert sqrt_mod(0, p) == 0


@pytest.mark.parametrize("p", [2**255 - 19, 2**256 - 2**32 - 977])
def test_full_width_sqrt_both_field_branches(p):
    rng = random.Random(123)
    for _ in range(20):
        value = rng.randrange(p)
        root = sqrt_mod(value * value, p)
        assert root is not None and 0 <= root < p
        assert root * root % p == value * value % p
    nonsquare = next(x for x in range(2, 100) if pow(x, (p - 1) // 2, p) == p - 1)
    assert sqrt_mod(nonsquare, p) is None


@pytest.mark.parametrize("p,a,b,r,h", [
    (7, 0, 1, 3, 4), (11, 1, 5, 11, 1), (13, 0, 2, 19, 1),
    (29, 0, 1, 3, 10), (43, 0, 2, 13, 4),
])
def test_exact_proposal_distribution_and_cofactor_fibres(p, a, b, r, h):
    points = affine_points(p, a, b)
    assert len(points) + 1 == h * r
    subgroup = {None} | {point for point in points if repeated_add(r, point, p, a) is None}
    assert len(subgroup) == r
    # Each nonidentity has h finite preimages; infinity has h-1 finite preimages.
    fibres = Counter(repeated_add(h, point, p, a) for point in points)
    assert set(fibres) <= subgroup
    assert all(fibres[point] == h for point in subgroup if point is not None)
    assert fibres[None] == h - 1

    outputs = Counter()
    affine_proposals = Counter()
    for x in range(p):
        root = sqrt_mod((x**3 + a * x + b) % p, p)
        for sign in (0, 1):
            if root is not None and not (root == 0 and sign):
                point = (x, (-root if sign else root) % p)
                affine_proposals[point] += 1
                expected = repeated_add(h, point, p, a)
            else:
                expected = None
            sampler = SubgroupSampler(p, a, b, r, h)
            try:
                actual = sampler.sample(OneProposal(p, x, sign))
            except RejectedProposal:
                assert expected is None
            else:
                assert actual == expected
                outputs[actual] += 1
    assert affine_proposals == Counter({point: 1 for point in points})
    assert outputs == Counter({point: h for point in subgroup if point is not None})


def test_zero_root_half_rejection_and_identity_rejection_counters():
    # y^2=x^3+1 over F_7: (3,0) is 2-torsion and h=4 kills it.
    for sign in (0, 1):
        sampler = SubgroupSampler(7, 0, 1, 3, 4)
        with pytest.raises(RejectedProposal):
            sampler.sample(OneProposal(7, 3, sign))
        assert sampler.zero_root_reject == sign
        assert sampler.identity_reject == 1 - sign
        assert sampler.nonsquare_reject == 0
    sampler = SubgroupSampler(43, 0, 2, 13, 4)
    nonsquare_x = next(x for x in range(43) if sqrt_mod(x**3 + 2, 43) is None)
    with pytest.raises(RejectedProposal):
        sampler.sample(OneProposal(43, nonsquare_x, 0))
    assert sampler.nonsquare_reject == sampler.proposal_reject == 1


def test_cofactor_one_skips_scalar_multiplication(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("cofactor one must bypass scalar_mul")
    monkeypatch.setattr("qsm.point.sampling.scalar_mul", unexpected)
    sampler = SubgroupSampler(13, 0, 2, 19, 1)
    assert sampler.sample(random.Random(15)) in affine_points(13, 0, 2)


def test_deterministic_wrapper_cache_and_json_safe_stats():
    kwargs = dict(p=43, a=0, b=2, order=13, cofactor=4)
    sampler = SubgroupSampler(**kwargs)
    first, second = random.Random(25), random.Random(25)
    for _ in range(40):
        assert sampler.sample(first) == sample_subgroup_point(second, **kwargs)
    stats = sampler.stats
    assert json.loads(json.dumps(stats)) == stats
    assert stats["accepted"] == 40
    assert stats["attempted_x"] == 40 + stats["proposal_reject"] + stats["identity_reject"]
    stats["accepted"] = -1
    assert sampler.stats["accepted"] == 40
    # Canonicalizing coefficients does not change the output law or RNG use.
    assert SubgroupSampler(43, 43, -41, 13, 4).sample(random.Random(2)) == \
        sampler.sample(random.Random(2))


@pytest.mark.parametrize("field", ["25519", "secp256k1"])
def test_full_width_classical_membership_only(field):
    if field == "25519":
        p = 2**255 - 19
        m = 486662
        a = (3 - m * m) * pow(3, -1, p) % p
        b = (2 * m**3 - 9 * m) * pow(27, -1, p) % p
        r, h = 2**252 + 27742317777372353535851937790883648493, 8
    else:
        from qsm.point import secp256k1 as curve
        p, a, b, r, h = curve.P, curve.A, curve.B, curve.ORDER, curve.COFACTOR
    sampler = SubgroupSampler(p, a, b, r, h)
    rng = random.Random(12345)
    points = [sampler.sample(rng) for _ in range(12)]
    assert len(set(points)) == len(points)
    for x, y in points:
        assert 0 <= x < p and 0 <= y < p
        assert (y * y - x**3 - a * x - b) % p == 0
        assert scalar_mul(r, x, y, p=p, a=a) is None


@pytest.mark.parametrize("overrides", [
    {"p": True}, {"a": False}, {"b": 1.5}, {"order": True}, {"cofactor": False},
    {"p": 2}, {"p": 17}, {"p": 18}, {"order": 2}, {"order": 4}, {"order": 0},
    {"cofactor": 0}, {"cofactor": -1}, {"cofactor": 13}, {"order": 101},
    {"cofactor": 1}, {"a": 0, "b": 0},
])
def test_invalid_parameters_and_obvious_order_cofactor_mismatches(overrides):
    kwargs = dict(p=43, a=0, b=2, order=13, cofactor=4)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        SubgroupSampler(**kwargs)


@pytest.mark.parametrize("value,p", [(True, 7), (1.0, 7), (1, True), (1, 17), (1, 2)])
def test_sqrt_rejects_bad_types_and_unsupported_branches(value, p):
    with pytest.raises(ValueError):
        sqrt_mod(value, p)


def test_basic_validation_is_explicitly_not_point_count_certification():
    # 3*3=9 is in Hasse's interval for p=7 but is not this curve's order 12;
    # gcd catches this example. A coprime wrong order within the interval does
    # pass the necessary checks, and must not be presented as a certification.
    with pytest.raises(ValueError, match="coprime"):
        SubgroupSampler(7, 0, 1, 3, 3)
    unproven = SubgroupSampler(7, 0, 1, 5, 2)
    assert unproven.order * unproven.cofactor == 10
    assert len(affine_points(7, 0, 1)) + 1 == 12
