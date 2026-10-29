"""Which inputs break a truncation — the failure predicates, and the shape they imply.

The rates in qsm.params assume UNIFORM operands. These predicates exist so a caller can
replay its ACTUAL value sequence and measure instead, which is what turns a guessed
hybrid cut-off (Schrottenloher's ITER_CAN_BE_Q = 50) into a counted one.

The load-bearing test here is `test_predicate_agrees_with_the_real_circuit`: a predicate
that does not match the circuit it describes is worse than none.
"""

import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.simulator import simulate
from qsm.field.constadd import add_const_inplace
from qsm.field.compare import cmp_lt
from qsm.params import (
    CARRY_WINDOW,
    CMP_BITS,
    F,
    PRIME,
    U,
    audit_carry_truncation,
    carry_truncation_fails,
    cmp_truncation_blind,
)


# ---------------------------------------------------------------------------
# the predicate must describe the actual circuit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,L,c", [(8, 4, 3), (8, 5, 7), (10, 4, 5), (10, 6, 3)])
def test_predicate_agrees_with_the_real_circuit(n: int, L: int, c: int):
    """Exhaustive: carry_truncation_fails predicts exactly which inputs the truncated
    constant adder gets wrong. No false positives, no false negatives."""
    circ = Circuit()
    b = circ.input("b", n)
    add_const_inplace(circ, b, c, window=L)
    mask = (1 << n) - 1
    for bv in range(1 << n):
        actually_wrong = simulate(circ, {"b": bv}).outputs["b"] != ((bv + c) & mask)
        predicted = carry_truncation_fails(bv, L, c)
        assert actually_wrong == predicted, \
            f"n={n} L={L} c={c} b={bv}: circuit wrong={actually_wrong}, predicted={predicted}"


@pytest.mark.parametrize("n,k", [(8, 3), (8, 4), (10, 4)])
def test_blindness_predicate_brackets_the_real_comparator(n: int, k: int):
    """`blind` marks EXPOSURE, not error: every wrong answer must be blind, but not
    every blind case is wrong (the discarded bits decide it ~50/50)."""
    circ = Circuit()
    a = circ.input("a", n)
    b = circ.input("b", n)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=k)
    blind = wrong = 0
    for av in range(1 << n):
        for bv in range(1 << n):
            got = simulate(circ, {"a": av, "b": bv, "flag": 0}).outputs["flag"]
            is_blind = cmp_truncation_blind(av, bv, n, k)
            is_wrong = got != int(bv < av)
            assert not (is_wrong and not is_blind), \
                f"wrong but not blind at ({av},{bv}) -- predicate misses a failure"
            blind += is_blind
            wrong += is_wrong
    assert wrong > 0 and blind > wrong, "blind should strictly over-approximate wrong"
    assert abs(wrong / blind - 0.5) < 0.15, "about half of blind cases should be wrong"


# ---------------------------------------------------------------------------
# the SHAPE of the danger set (this is the part a formula does not convey)
# ---------------------------------------------------------------------------
def test_danger_is_below_a_power_of_two_not_small_values():
    """The counter-intuitive part: small structured constants are SAFE; values just
    below a power of two are not. Base point x = 9 is fine, p//2 is not."""
    safe = {"Montgomery base x=9": 9, "a24=121665": 121665, "0": 0, "1": 1,
            "2^254": 1 << 254, "p-1": PRIME - 1, "p": PRIME}
    dangerous = {"p//2 = 2^254-10": PRIME // 2, "2^254-1": (1 << 254) - 1,
                 "2^128-1": (1 << 128) - 1, "2^45-1": (1 << CARRY_WINDOW) - 1}
    for name, v in safe.items():
        assert not carry_truncation_fails(v, CARRY_WINDOW, F - 1), f"{name} should be safe"
    for name, v in dangerous.items():
        assert carry_truncation_fails(v, CARRY_WINDOW, F - 1), f"{name} should be dangerous"


def test_p_sits_exactly_on_the_boundary_between_modneg_and_modadd():
    """p mod 2^45 == ceiling - 19, so p is safe for modneg (+F-1) by ONE and fails
    modadd (+F). Two primitives one bit apart, with the modulus falling in the gap."""
    assert PRIME % (1 << CARRY_WINDOW) == (1 << CARRY_WINDOW) - F
    assert not carry_truncation_fails(PRIME, CARRY_WINDOW, F - 1), "safe for modneg"
    assert carry_truncation_fails(PRIME, CARRY_WINDOW, F), "but FAILS modadd"


# ---------------------------------------------------------------------------
# auditing a real sequence, which is the point of having the predicate
# ---------------------------------------------------------------------------
def test_audit_recovers_the_uniform_rate_on_uniform_inputs():
    rng = random.Random(0)
    L = 12                                    # small enough to be measurable
    vals = [rng.randrange(PRIME) for _ in range(200_000)]
    a = audit_carry_truncation(vals, L, F)
    assert abs(a["measured_rate"] - a["uniform_rate"]) < 0.25 * a["uniform_rate"]


def test_audit_flags_a_structured_sequence_as_far_worse_than_uniform():
    """The whole reason the predicate exists: a structured sequence can blow the budget
    by orders of magnitude, and only replaying it reveals that."""
    structured = [(1 << k) - 1 for k in range(CARRY_WINDOW, 200)]
    a = audit_carry_truncation(structured, CARRY_WINDOW, F)
    assert a["measured_rate"] == 1.0, "every 2^k-1 lands in the bad set"
    assert a["measured_rate"] > 1e9 * a["uniform_rate"]
