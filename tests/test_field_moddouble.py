"""moddouble: a <- 2a mod p. The cheapest circuit in Layer 0, and structurally so.

Doubling hands over two things that addition has to pay for:

    the shift        FREE      -- relabelling wires is not an operation
    the erasure      ONE CX    -- the flag is the PARITY of the result

The second is the interesting one. 2a is even and p is odd, so subtracting p flips the
parity: bit 0 of the result IS the reduction flag, exactly, at every width. modadd needs a
full-width `result < a` for the same job; modsub_direct needs `result > ~a` and settles
for an approximate predicate. Here the predicate is exact and free.

Net: 45 Toffoli truncated against modadd's 339, at peak 300 against 766.
"""

import random
from math import log2

import pytest

from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import gidney_add
from qsm.field.modarith import (
    PRIME,
    modadd_lazy,
    double_wide,
    double_wide_ref,
    moddouble,
    moddouble_inplace,
    moddouble_lazy_wide,
    moddouble_lazy_wide_ref,
    moddouble_ref,
)
from qsm.params import (
    CARRY_WINDOW,
    CMP_BITS,
    F,
    U,
    carry_fail_prob,
    lazy_unreduced_prob,
    moddouble_carry_fail_prob,
    moddouble_lazy_input_carry_fail_prob,
    moddouble_lazy_overflow_prob,
    moddouble_overflows,
)

SMALL_PM = [(6, 3, 61), (7, 5, 123), (8, 3, 253), (8, 5, 251), (10, 5, 1019)]


def _circ(fn, u, p, window=None):
    c = Circuit()
    a = c.input("a", u)
    res = fn(c, a, p, window)
    if res is not None:                 # `moddouble` relabels; tell the simulator
        c.rebind("a", res)
    return c


def _lazy_wide_circ(u, p):
    c = Circuit()
    a = c.input("a", u)
    c.rebind("a", moddouble_lazy_wide(c, a, p))
    return c


# ---------------------------------------------------------------------------
# correctness, exhaustive, both variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [moddouble, moddouble_inplace],
                         ids=["relabel", "inplace"])
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_exact_exhaustive(fn, u, f, p):
    c = _circ(fn, u, p)
    for av in range(p):
        r = simulate(c, {"a": av})
        assert r.outputs["a"] == moddouble_ref(av, p, u), f"p={p}: 2*{av}"
        assert r.phase == 0
        assert r.outputs["a"] % p == (2 * av) % p


@pytest.mark.parametrize("fn", [moddouble, moddouble_inplace],
                         ids=["relabel", "inplace"])
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_output_is_lazy_but_always_in_range(fn, u, f, p):
    """Same lazy contract as modadd, and for the same reason: the free bit answers
    `2a >= 2^u`, not `2a >= p`."""
    c = _circ(fn, u, p)
    outs = [simulate(c, {"a": av}).outputs["a"] for av in range(p)]
    assert all(o < (1 << u) for o in outs), "must stay inside the register"
    assert any(o >= p for o in outs), "and should sometimes be unreduced (that is lazy)"


def test_both_variants_agree():
    u, p = 8, 253
    c1, c2 = _circ(moddouble, u, p), _circ(moddouble_inplace, u, p)
    for av in range(p):
        assert (simulate(c1, {"a": av}).outputs["a"]
                == simulate(c2, {"a": av}).outputs["a"])


# ---------------------------------------------------------------------------
# the two structural wins
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_the_flag_is_the_PARITY_of_the_result(u, f, p):
    """Pure arithmetic, and the reason the erasure is one CX.

    2a is even; p is odd; so subtracting p flips the parity. The result's low bit
    therefore equals the reduction flag EXACTLY -- no window, no approximation, at every
    width. Compare modadd (`result < a`, a full comparator) and modsub_direct
    (`result > ~a`, approximate).
    """
    for a in range(p):
        flag = a >> (u - 1)                      # does 2a reach 2^u?
        r = moddouble_ref(a, p, u)
        assert (r & 1) == flag, f"p={p}, a={a}: parity {r & 1} != flag {flag}"


def test_the_shift_itself_costs_nothing():
    """relabel and inplace do identical arithmetic; the only difference is whether the
    value is slid back onto the original wires. So they must have the SAME Toffoli
    count, and differ by exactly 2n CNOTs."""
    for window in (None, CARRY_WINDOW):
        rel = count(_circ(moddouble, U, PRIME, window))
        inp = count(_circ(moddouble_inplace, U, PRIME, window))
        assert rel.toffoli == inp.toffoli, "the shift must cost zero Toffoli"
        moves = inp.clifford - rel.clifford
        assert moves == 2 * U, f"expected 2n={2 * U} CNOTs of moves, got {moves}"


def test_erasure_is_a_single_cx_not_a_comparator():
    """Structural: the whole Toffoli cost must be the constant adder. Compare against a
    bare controlled `+F` -- moddouble should not exceed it."""
    from qsm.field.constadd import add_const_inplace

    for window in (None, CARRY_WINDOW):
        c = Circuit()
        a = c.input("a", U)
        ctrl = c.input("c", 1)
        add_const_inplace(c, a, F, ctrl=ctrl[0], window=window)
        bare = count(c).toffoli
        got = count(_circ(moddouble, U, PRIME, window)).toffoli
        assert got <= bare + 1, f"moddouble {got} vs a bare controlled +F {bare}"


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p,L", [(8, 3, 253, 3), (8, 5, 251, 4),
                                     (10, 5, 1019, 4), (10, 3, 1021, 3),
                                     (10, 5, 1019, 5)])
def test_truncation_here_is_SAFER_than_the_generic_rate(u, f, p, L):
    """Doubling gets ~2x truncation safety for free, and the generic F/2^L over-states
    the risk by ~3x at these widths.

      (a) the `+F` only fires when the flag is set -- about half of [0, p);
      (b) the shifted value has bit 0 = 0, so its low L bits are always EVEN, and only
          floor(F/2) of the F dangerous residues just below 2^L are even.

    Another instance of the running theme: a truncation's rate is a property of
    (circuit, input distribution), not of the window alone.
    """
    c = _circ(moddouble, u, p, L)
    wrong = sum(simulate(c, {"a": av}).outputs["a"] != moddouble_ref(av, p, u)
                for av in range(p))
    rate = wrong / p
    predicted = moddouble_carry_fail_prob(L, f, u, p)
    assert abs(rate - predicted) < 0.1 * predicted + 1e-9, \
        f"u={u} f={f} L={L}: measured {rate:.5f} vs predicted {predicted:.5f}"
    assert rate < 0.6 * carry_fail_prob(L, f), \
        f"should be well below the generic {carry_fail_prob(L, f):.5f}, got {rate:.5f}"


@pytest.mark.parametrize("fn", [moddouble, moddouble_inplace],
                         ids=["relabel", "inplace"])
def test_truncation_never_dirties_or_leaks_phase(fn):
    u, p = 10, 1019
    for L in (3, 4, 6):
        c = _circ(fn, u, p, L)
        for av in range(0, p, 3):
            assert simulate(c, {"a": av}).phase == 0     # DirtyAncillaError would raise


# ---------------------------------------------------------------------------
# arbitrary lazy input: retain the second carry instead of wrapping it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_lazy_wide_exact_is_injective_over_the_FULL_lazy_range(u, f, p):
    """The extra output bit is information, not workspace: keeping it turns the
    full-lazy map back into an injection."""
    c = _lazy_wide_circ(u, p)
    outputs = []
    for av in range(1 << u):
        r = simulate(c, {"a": av})
        expected = moddouble_lazy_wide_ref(av, p, u)
        assert r.outputs["a"] == expected, f"p={p}: 2*{av} once-folded"
        assert r.phase == 0
        assert r.outputs["a"] % p == (2 * av) % p
        outputs.append(r.outputs["a"])

    assert len(set(outputs)) == 1 << u, "retaining the carry must preserve injectivity"
    assert sum(out >> u for out in outputs) == f // 2, \
        "exactly the top floor(F/2) lazy inputs must set the retained carry"


@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_lazy_wide_agrees_with_cheap_moddouble_on_CANONICAL_inputs(u, f, p):
    narrow = _circ(moddouble, u, p)
    wide = _lazy_wide_circ(u, p)
    for av in range(p):
        narrow_out = simulate(narrow, {"a": av}).outputs["a"]
        wide_out = simulate(wide, {"a": av}).outputs["a"]
        assert wide_out == narrow_out
        assert wide_out >> u == 0, "a canonical input can never need the second carry"


@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (8, 5, 251), (10, 5, 1019)])
def test_lazy_wide_retains_exactly_the_nonclosure_overflow(u, f, p):
    c = _lazy_wide_circ(u, p)
    B = 1 << u
    first = B - f // 2
    for av in range(first - 2, B):
        out = simulate(c, {"a": av}).outputs["a"]
        assert (out >> u) == (av >= first), f"a={av}, out={out}"
        assert out == moddouble_lazy_wide_ref(av, p, u)


@pytest.mark.parametrize("u,f,p,L", [(8, 3, 253, 3), (8, 5, 251, 4),
                                     (10, 5, 1019, 4), (10, 3, 1021, 3),
                                     (11, 19, 2029, 6), (11, 19, 2029, 7)])
def test_a_windowed_lazy_wide_CANNOT_retain_the_carry(u, f, p, L):
    """The dead end, kept measured. `field.moddouble_lazy_wide_trunc.v1` existed briefly.

    The `+F` enters at bit 0, so a carry reaches bit U only by crossing every bit in
    between; a window of L bits is exactly the refusal to let it cross bit L-1. So a
    windowed build does not *risk* losing the retained carry -- bit U is untouched by
    construction and can never be set, at ANY window up to and including U. What is left
    is `moddouble` with an extra always-|0> qubit.

    Hence `moddouble_lazy_wide` takes no `window` argument. This rebuilds the deleted
    circuit inline so the claim stays a measurement rather than a memory.
    """
    from qsm.field.constadd import add_const_inplace

    c = Circuit()
    a = c.input("a", u)
    fresh = c.alloc(1)
    wide = Register([fresh[0]] + list(a.ids), a.name)
    saved = c.alloc(1)
    c.cx(a[u - 1], saved[0])
    c.cx(saved[0], a[u - 1])
    add_const_inplace(c, wide, f, ctrl=saved[0], window=L)   # the truncation
    c.cx(wide[0], saved[0])
    c.free(saved)
    c.rebind("a", wide)

    narrow = _circ(moddouble, u, p, L)
    hi_set = 0
    for av in range(1 << u):
        out = simulate(c, {"a": av}).outputs["a"]
        hi_set += out >> u
        assert out == simulate(narrow, {"a": av}).outputs["a"], \
            f"a={av}: a windowed lazy-wide is just moddouble_trunc"
    assert hi_set == 0, "a truncated build must never set the retained carry bit"

    # ...whereas the exact build sets it on exactly the non-closure inputs.
    assert sum(simulate(_lazy_wide_circ(u, p), {"a": av}).outputs["a"] >> u
               for av in range(1 << u)) == f // 2


@pytest.mark.parametrize("u,f,p,L", [(8, 3, 253, 3), (8, 5, 251, 4),
                                     (10, 5, 1019, 4), (10, 5, 1019, 5),
                                     (11, 19, 2029, 6), (12, 19, 4077, 7)])
def test_lazy_inputs_shift_the_truncation_rate(u, f, p, L):
    """Same circuit, different input distribution, different rate -- exactly.

    `moddouble_carry_fail_prob` assumes a canonical input, where the flag fires with
    probability `(p - 2^(u-1))/p`. Over the full lazy range it fires with probability
    exactly 1/2, so the rate rises to `floor(f/2)/2^L`. Small, but the two formulas are
    not interchangeable, and "lazy throughout" is the regime that picks the second.
    """
    trunc, exact = _circ(moddouble, u, p, L), _circ(moddouble, u, p, None)
    wrong = sum(simulate(trunc, {"a": av}).outputs["a"]
                != simulate(exact, {"a": av}).outputs["a"] for av in range(1 << u))

    assert wrong / (1 << u) == moddouble_lazy_input_carry_fail_prob(L, f)
    assert wrong == (f // 2) * (1 << (u - L))
    assert moddouble_lazy_input_carry_fail_prob(L, f) > moddouble_carry_fail_prob(L, f, u, p)


def test_lazy_wide_resource_delta_is_only_the_retained_carry_stage():
    canonical_exact = count(_circ(moddouble, U, PRIME, None))
    wide_exact = count(_lazy_wide_circ(U, PRIME))

    assert wide_exact.toffoli == canonical_exact.toffoli + 1
    assert wide_exact.peak_qubits == canonical_exact.peak_qubits + 2


def test_only_the_circuits_are_registered_not_the_relabelling_or_the_dead_end():
    """Three contracts, two of which are circuits, one of which is exact-only.

    `field.double_wide.v1` was registered and should not have been: it emits no gates, so
    its row was zeros except a `peak_qubits` that echoed the input width back as if
    measured. `field.moddouble_lazy_wide_trunc.v1` was registered and should not have
    been either: it computed the same function as `moddouble_trunc` for one extra qubit,
    under a name promising it handled lazy inputs.
    """
    from qsm.bench.registry import all_versions, get_builder

    versions = set(all_versions())
    assert "field.moddouble_lazy_wide.v1" in versions
    assert "field.double_wide.v1" not in versions, \
        "a relabelling that emits no gates must not be a registered circuit version"
    assert "field.moddouble_lazy_wide_trunc.v1" not in versions, \
        "truncation cannot retain the carry; that version was moddouble_trunc in disguise"

    exact = get_builder("field.moddouble_lazy_wide.v1")()
    assert len(exact.circuit.inputs["a"]) == U + 1
    assert exact.meta.get("fail_analytic", 0.0) == 0.0

    class UpperEndpoint:
        def randrange(self, stop):
            assert stop == 1 << U, "lazy-wide generators must cover the full U-bit range"
            return stop - 1

    top = {"a": (1 << U) - 1}
    assert exact.gen_input(UpperEndpoint()) == top
    assert exact.reference(top)["a"] == moddouble_lazy_wide_ref(top["a"], PRIME, U)


# ---------------------------------------------------------------------------
# composition: a doubling chain is what a ladder actually runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (8, 5, 251), (10, 5, 1019)])
def test_moddouble_is_NOT_closed_under_the_lazy_contract(u, f, p):
    """The one Layer-0 circuit whose output contract does not survive being fed back in.

    Given a lazy input a in [p, 2^u), the reduced result 2a - p can exceed 2^u - 1 and
    wrap. Exactly when 2a - p >= 2^u, i.e. a >= 2^u - f/2 -- the top floor(f/2) values.

    modadd escapes this because it has TWO operands: canonical+lazy is closed, and only
    lazy+lazy has a hole (at 2^-502.6). Doubling has one operand and therefore no
    canonical partner to lean on, so a doubling CHAIN feeds lazy into lazy every step and
    pays this every step.

    9/2^255 = 2^-251.8 at real width -- a ladder of ~2^8 doublings totals ~2^-243.8, so
    it does not move the budget. But it is a genuine term, and it is the first place the
    lazy contract fails to compose.
    """
    overflowing = [a for a in range(1 << u)
                   if not 0 <= (2 * a - p if a >> (u - 1) else 2 * a) < (1 << u)]
    assert len(overflowing) == f // 2, f"expected floor(f/2)={f // 2}, got {len(overflowing)}"
    assert all(a >= p for a in overflowing), "only LAZY inputs can overflow"
    assert all(2 * a - p >= (1 << u) for a in overflowing)
    assert moddouble_lazy_overflow_prob(u, f) == (f // 2) / (1 << u)


@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (8, 5, 251), (10, 5, 1019),
                                   (11, 19, 2029), (12, 19, 4077)])
def test_the_overflow_count_is_floor_F_over_2_because_the_RESULT_IS_ODD(u, f, p):
    """Why the rate is 9/2^255 = 2^-251.8 and not 19/2^255 = 2^-250.8.

    The band above 2^u that a once-folded result can reach is f wide, which invites f.
    But 2a is even and p is odd, so `2a - p` is ALWAYS odd, and only the odd half of that
    band is reachable -- floor(f/2) values. Same parity fact as
    `moddouble_carry_fail_prob`, from the other side: there the addend is even, here the
    result is odd.
    """
    B = 1 << u
    overflow = [a for a in range(B) if moddouble_lazy_wide_ref(a, p, u) >= B]
    outs = [moddouble_lazy_wide_ref(a, p, u) for a in overflow]

    assert len(overflow) == f // 2
    assert all(o & 1 for o in outs), "a once-folded result is always odd"
    assert sorted(o - B for o in outs) == list(range(1, f - 1, 2))
    # the even offsets in the same band are unreachable, which is the whole point
    assert not (set(range(2, f - 1, 2)) & {o - B for o in outs})

    # and the exponent that falls out of it
    assert moddouble_lazy_overflow_prob(u, f) == (f // 2) / B
    if (u, f) == (11, 19):
        assert round(log2(moddouble_lazy_overflow_prob(255, 19)), 2) == -251.83
        assert round(log2(lazy_unreduced_prob(255, 19)), 2) == -250.75


@pytest.mark.parametrize("u,f,p", [(6, 3, 61), (8, 3, 253), (8, 5, 251),
                                   (10, 5, 1019), (12, 5, 4091)])
def test_a_CANONICAL_input_can_never_overflow(u, f, p):
    """The mitigation, and it is a scheduling constraint rather than a circuit change.

    Overflow needs x >= 2^u - f/2; canonical means x < p = 2^u - f; and f > f/2, so
    x < 2^u - f < 2^u - f/2. Never. The whole failure set lives in the lazy band
    [p, 2^u), and within that only its top half.

    So anything that guarantees a canonical input makes moddouble exact. A
    multiplication's reduction produces canonical output, so a double immediately after a
    multiply is always safe; only a double fed by another lazy operation is exposed.
    """
    assert not any(moddouble_overflows(x, u, f) for x in range(p)), \
        "no canonical input may overflow"
    assert all(moddouble_overflows(x, u, f) for x in range((1 << u) - f // 2, 1 << u)), \
        "the top floor(f/2) lazy values must all overflow"
    # and the predicate agrees with the arithmetic, exhaustively
    for x in range(1 << u):
        r = 2 * x - p if x >> (u - 1) else 2 * x
        assert moddouble_overflows(x, u, f) == (r >= (1 << u)), f"x={x}"


@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (10, 5, 1019)])
def test_a_chain_of_doublings_is_congruent_except_on_that_overflow_set(u, f, p):
    """k doublings give 2^k * a mod p, and every deviation is traceable to the overflow
    above -- nothing else goes wrong along the way."""
    k = 12
    c = Circuit()
    a = c.input("a", u)
    reg = a
    for _ in range(k):
        reg = moddouble(c, reg, p, None)
    c.rebind("a", reg)

    def hits_overflow(av):
        """Replay classically: did any step receive an input that overflows?"""
        x = av
        for _ in range(k):
            if x >= (1 << u) - f / 2:
                return True
            x = 2 * x - p if x >> (u - 1) else 2 * x
        return False

    rng = random.Random(0)
    unexplained = 0
    for _ in range(300):
        av = rng.randrange(p)
        out = simulate(c, {"a": av}).outputs["a"]
        assert out < (1 << u), "must stay inside the register"
        if out % p != (av << k) % p and not hits_overflow(av):
            unexplained += 1
    assert unexplained == 0, f"{unexplained} deviations not explained by the overflow set"


def test_a_chain_is_exact_when_every_step_stays_canonical(u=10, f=5, p=1019):
    """The positive statement: where the overflow set is avoided, chaining is exact."""
    k = 12
    c = Circuit()
    a = c.input("a", u)
    reg = a
    for _ in range(k):
        reg = moddouble(c, reg, p, None)
    c.rebind("a", reg)

    def safe(av):
        x = av
        for _ in range(k):
            if x >= (1 << u) - f / 2:
                return False
            x = 2 * x - p if x >> (u - 1) else 2 * x
        return True

    checked = 0
    for av in range(p):
        if not safe(av):
            continue
        checked += 1
        assert simulate(c, {"a": av}).outputs["a"] % p == (av << k) % p
    assert checked > 0.9 * p, f"only {checked}/{p} inputs were safe -- expected almost all"


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def test_moddouble_is_far_cheaper_than_doubling_via_modadd():
    """The in-house baseline: a curve-agnostic design without a dedicated doubler would
    reach for modular addition. (No verified Litinski figure for doubling is recorded
    here, so this deliberately compares against our own modadd rather than a number we
    have not checked.)"""
    dbl = count(_circ(moddouble, U, PRIME, CARRY_WINDOW))

    c = Circuit()
    x = c.input("x", U)
    y = c.input("y", U)
    modadd_lazy(c, x, y, PRIME, gidney_add, CARRY_WINDOW, CMP_BITS)
    add = count(c)

    assert dbl.toffoli * 5 < add.toffoli, f"{dbl.toffoli} vs {add.toffoli}"
    assert dbl.peak_qubits * 2 < add.peak_qubits, \
        f"peak {dbl.peak_qubits} vs {add.peak_qubits}"


# ---------------------------------------------------------------------------
# not reducing at all: free, and exactly how far it pays
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3, 5])
def test_double_wide_is_literally_free(k):
    """Not "cheap" -- zero gates of any kind. Declining to reduce removes the flag, the
    constant add and the erasure all at once; the register just grows a bit."""
    u, p = 8, 253
    c = Circuit()
    a = c.input("a", u)
    reg = a
    for _ in range(k):
        reg = double_wide(c, reg)
    c.rebind("a", reg)

    res = count(c)
    assert res.toffoli == 0, "a shift is a relabelling, not an operation"
    assert res.clifford == 0
    assert len(reg) == u + k, "the register grows one bit per doubling"
    for av in range(p):
        assert simulate(c, {"a": av}).outputs["a"] == double_wide_ref(av, k)


def test_deferring_saves_45_per_doubling_but_costs_a_qubit_each():
    k = 8
    c = Circuit()
    a = c.input("a", U)
    reg = a
    for _ in range(k):
        reg = moddouble(c, reg, PRIME, CARRY_WINDOW)
    reduced = count(c).toffoli

    c2 = Circuit()
    a2 = c2.input("a", U)
    reg2 = a2
    for _ in range(k):
        reg2 = double_wide(c2, reg2)

    assert count(c2).toffoli == 0
    assert reduced == 45 * k, f"expected 45 per doubling, got {reduced}"
    assert len(reg2) - U == k, "one qubit per deferred doubling"


def test_deferral_stops_paying_at_the_next_MULTIPLICATION():
    """Where the idea breaks, quantitatively.

    Extra width is LINEAR in an addition (+k Toffoli) but QUADRATIC in a multiplication:
    a schoolbook U x U costs ~U^2 partial products, so U+k costs ~(U+k)^2 and the penalty
    is ~2Uk. At U = 255 that is ~510 per deferred bit, against the 45 it saved -- an
    order of magnitude the wrong way, from a SINGLE deferred bit.

    So: defer freely across doublings and additions, never carry a deferred bit into a
    multiply.

    The 2Uk figure is an ESTIMATE from the schoolbook shape, not a measurement -- the
    multiplication layer does not exist yet. The conclusion is robust to the constant
    (it would take a ~11x better multiplier to change the sign), but the number should be
    replaced with a measured one once there is a multiplier to measure.
    """
    saved_per_bit = 45
    add_penalty_per_bit = 1              # linear
    mul_penalty_per_bit = 2 * U          # quadratic: (U+k)^2 - U^2 ~ 2Uk

    assert add_penalty_per_bit < saved_per_bit, "deferring across additions pays"
    assert mul_penalty_per_bit > 10 * saved_per_bit, \
        "deferring into a multiplication should lose by an order of magnitude"


def test_truncation_buys_the_same_factor_here_as_elsewhere():
    exact = count(_circ(moddouble, U, PRIME, None)).toffoli
    trunc = count(_circ(moddouble, U, PRIME, CARRY_WINDOW)).toffoli
    assert trunc < exact // 5, f"{trunc} vs {exact}"
