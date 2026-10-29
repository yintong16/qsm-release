"""Algebra-only order-eight coset fixture for the 3/4/2 odd positional chain.

Bounded search used p in (31, 127), lexicographic (a, b), the first nonzero
order-r point, then lexicographic order-eight T and Q_LOG in range(2, r).
The first eligible curve and seventh (T, Q_LOG) choice gave horizontal chords
on the initial classical 3/3/2 census (26 events on 19/64 pairs). The corrected
3/4/2 schedule below has no one-bit arithmetic slot and needs no reselection.
Selection used only complete classical arithmetic, never circuit success.
No quantum circuit is built here; no full_scalar.Group dependency is needed.
"""

from collections import Counter

from qsm.point.group import scalar_mul
from qsm.scalarmul.signed_window import _group_add


MODULUS = 31
A = 1
B = 29
ORDER = 5
COFACTOR = 8
P = (5, 2)
T = (18, 12)
Q_LOG = 2
Q = (10, 27)

U_BITS, V_BITS, WINDOW_BITS = 3, 4, 2
# (axis, positional offset, actual word width), high words first on each axis.
# Only the first initializer is one bit; every arithmetic slot is two bits.
SCHEDULE = (("u", 2, 1), ("u", 0, 2), ("v", 2, 2), ("v", 0, 2))
INPUT_PAIRS = tuple((u, v) for u in range(1 << U_BITS) for v in range(1 << V_BITS))

# All horizontal events, not a sample: (u, v, zero-based addition stage).
# Stage zero initializes by lookup, so it is not an arithmetic addition event.
HORIZONTAL_LOCATIONS = (
    (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1),
    (0, 4, 1), (0, 5, 1), (0, 6, 1), (0, 7, 1),
    (0, 8, 1), (0, 8, 2), (0, 9, 1), (0, 9, 2),
    (0, 10, 1), (0, 10, 2), (0, 10, 3), (0, 11, 1), (0, 11, 2),
    (0, 12, 1), (0, 13, 1), (0, 14, 1), (0, 15, 1),
    (1, 2, 3), (1, 9, 3), (2, 1, 3), (2, 14, 3), (3, 6, 3), (3, 13, 3),
    (4, 4, 2), (4, 5, 2), (4, 5, 3), (4, 6, 2), (4, 7, 2),
    (5, 8, 2), (5, 9, 2), (5, 10, 2), (5, 10, 3), (5, 11, 2),
    (6, 2, 3), (6, 9, 3), (7, 1, 3), (7, 14, 3),
)
# Both signs occur. Each tuple is (accumulator, operand, sum).
HORIZONTAL_CHORDS = (
    ((4, 2), (5, 2), (22, 29)),
    ((22, 29), (5, 29), (4, 2)),
)


def group_kwargs():
    """Fresh constructor data; T is separate from the prime-order group."""
    return {"p": MODULUS, "a": A, "b": B, "order": ORDER,
            "P": P, "Q": Q, "name": "order5-coset8-test"}


def _mul(k, point):
    # Do not reduce modulo ORDER: doing so would invalidate torsion checks.
    return scalar_mul(k, *point, p=MODULUS, a=A)


def _add(left, right):
    return _group_add(left, right, MODULUS, A)


def odd_trace(u, v):
    """Complete ideal trace, initialized T + D_0 * weighted_P / 2.

    D = 2*raw - (2**bits-1); division by two is solely in the odd subgroup.
    In particular T is never halved or reduced modulo the subgroup order.
    Each returned row contains the input, actual signed operand and output.
    """
    if any(type(k) is not int or not 0 <= k < 1 << bits
           for k, bits in ((u, U_BITS), (v, V_BITS))):
        raise ValueError("fixture scalars require unsigned three-bit u and four-bit v")
    point = T
    rows = []
    for stage, (axis, offset, bits) in enumerate(SCHEDULE):
        raw = ((u if axis == "u" else v) >> offset) & ((1 << bits)-1)
        digit = 2*raw - ((1 << bits)-1)
        operand = _mul(digit*(1 << offset)*pow(2, -1, ORDER),
                       P if axis == "u" else Q)
        result = _add(point, operand)
        rows.append({"stage": stage, "axis": axis, "offset": offset,
                     "bits": bits, "raw": raw, "digit": digit,
                     "before": point, "operand": operand, "after": result})
        point = result
    return tuple(rows)


def horizontal_witnesses():
    """Every reachable nondegenerate horizontal addition, with coordinates."""
    return tuple({"u": u, "v": v, **row}
                 for u, v in INPUT_PAIRS for row in odd_trace(u, v)[1:]
                 if row["before"][0] != row["operand"][0]
                 and row["before"][1] == row["operand"][1])


def validate_fixture():
    """Independently census the curve and check all 128 complete ideal chains.

    Enumeration uses the curve equation, not an assumed order. Repeated
    _group_add cross-checks scalar_mul for every subgroup/torsion multiple.
    The Cartesian torsion/subgroup sum must equal the entire enumerated curve.
    """
    for prime in (MODULUS, ORDER):
        assert prime > 3
        assert all(prime % d for d in range(2, int(prime**0.5)+1))
    assert (4*A**3 + 27*B**2) % MODULUS != 0
    affine = {(x, y) for x in range(MODULUS) for y in range(MODULUS)
              if (y*y-x*x*x-A*x-B) % MODULUS == 0}
    curve = affine | {None}
    assert len(curve) == COFACTOR*ORDER == 40
    assert all(point in affine for point in (P, Q, T))
    assert _mul(ORDER, P) is None
    assert all(_mul(k, P) is not None for k in range(1, ORDER))
    assert _mul(8, T) is None
    assert all(_mul(k, T) is not None for k in range(1, 8))
    assert Q == _mul(Q_LOG, P)
    assert _mul(ORDER, Q) is None

    for base, period in ((P, ORDER), (T, 8)):
        point = None
        for k in range(period+1):
            assert point == _mul(k, base)
            point = _add(point, base)
    subgroup = {_mul(k, P) for k in range(ORDER)}
    torsion = {_mul(k, T) for k in range(8)}
    assert len(subgroup) == ORDER and len(torsion) == 8
    assert subgroup & torsion == {None}
    assert {_add(left, right) for left in subgroup for right in torsion} == curve
    coset = {_add(T, point) for point in subgroup}
    assert coset.isdisjoint(subgroup)
    assert all(point is not None and point[1] != 0 for point in coset)

    half = pow(2, -1, ORDER)
    for u, v in INPUT_PAIRS:
        trace = odd_trace(u, v)
        for row in trace:
            before, operand, after = row["before"], row["operand"], row["after"]
            assert before in coset and after in coset
            assert operand in subgroup and operand is not None and operand[1] != 0
            # Coset separation excludes both forward and reverse denominators.
            assert before[0] != operand[0] and after[0] != operand[0]
            assert _add(after, (operand[0], -operand[1] % MODULUS)) == before
        expected = _add(T, _add(_mul((2*u-7)*half, P), _mul((2*v-15)*half, Q)))
        assert trace[-1]["after"] == expected

    witnesses = horizontal_witnesses()
    assert tuple((row["u"], row["v"], row["stage"]) for row in witnesses) == HORIZONTAL_LOCATIONS
    assert {(row["before"], row["operand"], row["after"]) for row in witnesses} == set(HORIZONTAL_CHORDS)
    counts = dict(Counter(row["stage"] for row in witnesses))
    pairs = len({(row["u"], row["v"]) for row in witnesses})
    assert len(witnesses) == 41 and pairs == 34 and counts == {1: 16, 2: 12, 3: 13}
    return {"curve_order": len(curve), "subgroup_order": len(subgroup),
            "torsion_order": len(torsion), "input_pairs": len(INPUT_PAIRS),
            "horizontal_events": len(witnesses), "horizontal_pairs": pairs,
            "horizontal_by_stage": counts, "exceptional_events": 0}


def test_coset_fixture_group_and_schedule():
    """Runnable directly by pytest, or imported by integration tests."""
    validate_fixture()
