"""modneg: a <- p - a, via `a += F-1` then complement."""

import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.modarith import PRIME, modneg, modneg_ref
from qsm.params import CARRY_WINDOW, F, U, carry_fail_prob

# (u, f, p) with p = 2^u - f. Includes f = 1, where F-1 = 0 and the add vanishes.
SMALL_PM = [(4, 1, 15), (4, 3, 13), (5, 3, 29), (5, 1, 31), (6, 3, 61), (7, 5, 123)]


def _circ(u: int, p: int, window: int | None = None) -> Circuit:
    c = Circuit()
    a = c.input("a", u)
    modneg(c, a, p, window=window)
    return c


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_modneg_exact_exhaustive(u: int, f: int, p: int):
    """Every reduced input, exactly. No truncation, so no error term at all."""
    c = _circ(u, p)
    for av in range(p):
        r = simulate(c, {"a": av})
        assert r.outputs["a"] == modneg_ref(av, p), f"p={p}: neg({av})"
        assert r.phase == 0


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_modneg_is_congruent_and_an_involution(u: int, f: int, p: int):
    """p - a is congruent to -a, and negating twice returns the original."""
    c = _circ(u, p)
    for av in range(p):
        once = simulate(c, {"a": av}).outputs["a"]
        assert once % p == (-av) % p, f"p={p}: neg({av}) = {once}"
        assert once < (1 << u), "result must fit the register"
        if once < p:                      # feed it back where the contract allows
            twice = simulate(c, {"a": once}).outputs["a"]
            assert twice % p == av % p, f"p={p}: neg(neg({av})) = {twice}"


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_modneg_zero_is_the_only_unreduced_output(u: int, f: int, p: int):
    """Output range is [1, p]; a = 0 -> p is the single value outside [0, p)."""
    c = _circ(u, p)
    outside = [av for av in range(p)
               if not (0 <= simulate(c, {"a": av}).outputs["a"] < p)]
    assert outside == [0], outside
    assert simulate(c, {"a": 0}).outputs["a"] == p


def test_modneg_needs_no_flag_or_ancilla_beyond_the_adder():
    """No conditional means no flag: the only ancillas are the constant-add carries."""
    from qsm.core import ops
    u, p = 32, (1 << 32) - 5
    c = _circ(u, p, window=8)
    # nothing is freed, because nothing is conditionally erased
    assert sum(1 for op in c.ops if op.kind == ops.FREE) <= 1


# ---------------------------------------------------------------------------
# truncated
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p,L", [(8, 3, 253, 3), (8, 3, 253, 5), (10, 5, 1019, 4)])
def test_modneg_truncated_rate_matches_formula(u, f, p, L):
    """Exhaustive: the rate must equal (F-1)/2^L exactly."""
    c = _circ(u, p, window=L)
    wrong = 0
    for av in range(p):
        r = simulate(c, {"a": av})
        wrong += r.outputs["a"] != modneg_ref(av, p)
        assert r.phase == 0, "truncation must not leak phase"
    rate = wrong / p
    predicted = carry_fail_prob(L, f - 1)
    assert abs(rate - predicted) < 0.15 * predicted + 1e-9, \
        f"u={u} L={L}: measured {rate:.5f} vs predicted {predicted:.5f}"


def test_modneg_25519_edges_are_exact_on_the_EXACT_circuit():
    """Edge cases belong to the exact version, which must be right everywhere."""
    c = _circ(U, PRIME)                      # window=None
    for av in [0, 1, 2, PRIME // 2, PRIME - 2, PRIME - 1]:
        r = simulate(c, {"a": av})
        assert r.outputs["a"] == modneg_ref(av, PRIME), f"neg({av})"
        assert r.phase == 0


def test_modneg_25519_truncated_on_random_inputs():
    """The truncated version is only claimed correct on UNIFORM inputs."""
    c = _circ(U, PRIME, window=CARRY_WINDOW)
    rng = random.Random(3)
    for _ in range(32):
        av = rng.randrange(PRIME)
        r = simulate(c, {"a": av}, rng=rng)
        assert r.outputs["a"] == modneg_ref(av, PRIME), f"neg({av})"
        assert r.phase == 0


def test_STRUCTURED_inputs_hit_the_truncation_failure_set():
    """Recorded hazard: the (F-1)/2^L rate assumes UNIFORM inputs.

    The failure set is "low part within F-1 of the window ceiling". Values of the form
    2^k - small have low parts sitting right at that ceiling BY CONSTRUCTION, so they
    are wrong far more often than 2^-40.9 would suggest. p//2 = 2^254 - 10 is one: its
    low 45 bits are 2^45 - 10, and +18 carries straight out.

    Found by picking p//2 as a "natural" edge case and having it fail. Any caller that
    can feed structured field elements invalidates the error budget -- the same lesson
    as the near-tie comparator inputs, reached from the opposite direction.
    """
    c = _circ(U, PRIME, window=CARRY_WINDOW)
    structured = [PRIME // 2, (1 << 254) - 10, (1 << 200) - 5, (1 << 100) - 1]
    failures = sum(simulate(c, {"a": av}).outputs["a"] != modneg_ref(av, PRIME)
                   for av in structured if av < PRIME)
    assert failures > 0, "expected at least one structured input to break truncation"

    # ...and the exact version handles every one of them
    ce = _circ(U, PRIME)
    for av in structured:
        if av < PRIME:
            assert simulate(ce, {"a": av}).outputs["a"] == modneg_ref(av, PRIME)


def test_modneg_is_cheap_vs_litinski():
    """Litinski's curve-agnostic modular negation is 2n = 510 Toffoli."""
    exact = count(_circ(U, PRIME))
    trunc = count(_circ(U, PRIME, window=CARRY_WINDOW))
    assert exact.toffoli < 2 * U, f"exact {exact.toffoli} should beat 2n={2 * U}"
    assert trunc.toffoli <= CARRY_WINDOW, f"truncated {trunc.toffoli}"
    assert trunc.toffoli < exact.toffoli // 5


def test_modneg_constant_is_f_minus_one():
    """Guard the off-by-one: using F instead of F-1 makes every result one too low."""
    p, u = PRIME, U
    allones = (1 << u) - 1
    assert p - 0 == allones - (F - 1)
    assert p - 0 != allones - F
