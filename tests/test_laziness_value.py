"""What does *laziness* buy, separately from what *truncation* buys?

`modadd_lazy` changes two things at once against the canonical `modadd_cmp`, and the
headline "1022 -> 339 Toffoli" credits both to laziness. This file separates them by
building the missing control: a CANONICAL modadd that is vented and truncated exactly
like the lazy one, so the only remaining difference is where the reduction flag comes
from.

The finding is that laziness is worth much less than the headline in Toffoli terms
(40 of the 683), and that its real value is structural: it moves the approximation out
of the CONTROL path and into the ERASURE path, which is what makes the carry-window
truncation safe at all.

    canonical: flag = (a+b >= p)      computed by a comparator, so it can be WRONG;
                                      it then CONTROLS the subtraction, so a wrong flag
                                      corrupts the value AND leaves ovf dirty.
    lazy:      flag = (a+b >= 2^U)    IS the adder's carry-out. Free, and never wrong.
                                      Truncation only ever touches the +F carry (wrong
                                      value) and the erasure predicate (vented -> phase).
"""

import random

import pytest

from qsm.core.circuit import Circuit, DirtyAncillaError, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import gidney_add
from qsm.field.compare import cmp_ge_const, cmp_lt_phase
from qsm.field.constadd import addsub_p
from qsm.field.modarith import PRIME, modadd_cmp, modadd_lazy, modadd_lazy_ref
from qsm.params import CARRY_WINDOW, CMP_BITS, F, U, lazy_unreduced_prob

SMALL_PM = [(6, 3, 61), (7, 5, 123), (8, 3, 253)]


def canonical_vented(circ, a, b, p, carry_window=None, cmp_bits=None):
    """The control: `modadd_cmp` with the same venting and truncation knobs as
    `modadd_lazy`, so the two differ ONLY in where the reduction flag comes from.

    Not registered as a circuit version -- it exists to be measured against, and one of
    its configurations is unsafe (see the dirty-ancilla tests below).
    """
    n = len(a)
    f = (1 << n) - p
    ovf = circ.alloc(1)
    flag = circ.alloc(1)
    B = Register(list(b.ids) + [ovf[0]])

    gidney_add(circ, a, B)                                              # 1
    cmp_ge_const(circ, B, p, flag[0], k=cmp_bits)                       # 2  <-- the flag
    addsub_p(circ, B, n, f, adding=False, ctrl=flag[0], inplace=True,
             add_fn=gidney_add, window=carry_window)                    # 3
    m = circ.measure(flag[0])                                           # 4
    cmp_lt_phase(circ, a, B, m, k=cmp_bits)
    circ.free(ovf)


def _canon(u, p, carry_window=None, cmp_bits=None):
    c = Circuit()
    a = c.input("a", u)
    b = c.input("b", u)
    canonical_vented(c, a, b, p, carry_window, cmp_bits)
    return c


def _lazy(u, p, carry_window=None, cmp_bits=None):
    c = Circuit()
    a = c.input("a", u)
    b = c.input("b", u)
    modadd_lazy(c, a, b, p, gidney_add, carry_window=carry_window,
                cmp_bits=cmp_bits, vent=True)
    return c


# ---------------------------------------------------------------------------
# the control has to be a fair one: exact, canonically reduced, clean
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", SMALL_PM)
def test_control_is_exact_and_canonically_reduced(u, f, p):
    """Unlike the lazy circuit, the control's output is in [0, p) -- a STRICTLY better
    contract. That is what makes it the honest thing to price laziness against."""
    c = _canon(u, p)
    for av in range(p):
        for bv in range(p):
            r = simulate(c, {"a": av, "b": bv})
            assert r.outputs["b"] == (av + bv) % p, f"p={p}: {av}+{bv}"
            assert r.outputs["b"] < p, "control must be canonically reduced"
            assert r.phase == 0


# ---------------------------------------------------------------------------
# the actual accounting
# ---------------------------------------------------------------------------
def test_laziness_buys_exactly_one_comparator_when_exact():
    """Exact regime: the saving is one full-width comparator, ~n Toffoli.

    Canonical spends n Toffoli computing `a+b >= p`. Lazy reads the adder's own carry
    out, which costs nothing.
    """
    canon = count(_canon(U, PRIME)).toffoli
    lazy = count(_lazy(U, PRIME)).toffoli
    saved = canon - lazy
    assert 0.9 * U <= saved <= 1.1 * U, \
        f"expected ~{U} saved (one comparator), got {saved} ({canon} -> {lazy})"


def test_laziness_buys_far_less_once_BOTH_are_truncated():
    """The headline number is misleading. At a MATCHED approximation level -- only the
    erasure truncated, the one knob both designs can turn safely -- laziness is worth
    ~k Toffoli, not ~n, because truncation already made that comparator cheap.

    Laziness and truncation are SUBSTITUTES for the flag-computation cost, not
    complements. This is the number the paper should quote for laziness alone.
    """
    canon = count(_canon(U, PRIME, cmp_bits=CMP_BITS)).toffoli
    lazy = count(_lazy(U, PRIME, cmp_bits=CMP_BITS)).toffoli
    saved = canon - lazy
    assert saved < 0.25 * U, f"expected a small saving, got {saved} ({canon} -> {lazy})"
    assert saved > 0, f"laziness should still be a win, got {saved}"


def test_the_big_win_is_truncation_which_only_laziness_makes_safe():
    """Most of 1022 -> 339 is the carry window, not laziness. But the carry window is
    only *available* to the lazy design (see the dirty-ancilla tests), so the two are
    not separable in the final circuit -- only in the accounting.
    """
    canon = count(_canon(U, PRIME, cmp_bits=CMP_BITS)).toffoli
    lazy_exact = count(_lazy(U, PRIME, cmp_bits=CMP_BITS)).toffoli
    lazy_trunc = count(_lazy(U, PRIME, CARRY_WINDOW, CMP_BITS)).toffoli

    from_laziness = canon - lazy_exact          # the free flag
    from_carry_window = lazy_exact - lazy_trunc  # what laziness UNLOCKS
    assert from_carry_window > 4 * from_laziness, \
        f"carry window ({from_carry_window}) should dominate laziness ({from_laziness})"


# ---------------------------------------------------------------------------
# WHY the canonical design cannot use the carry window: the flag is in the control path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", [(12, 5, 4091)])
def test_truncating_the_CANONICAL_flag_leaves_dirty_ancillas(u, f, p):
    """A truncated `a+b >= p` is not merely sometimes-wrong: because the flag CONTROLS
    the subtraction, a wrong flag subtracts p when it should not (or vice versa), and
    the result no longer clears the overflow bit. `ovf` is freed at |1>.

    That is the failure mode no output check can see, at rate ~2^-k. It is why the flag
    must be exact -- and being exact-for-free is precisely what laziness provides.
    """
    k = 4
    c = _canon(u, p, carry_window=None, cmp_bits=k)
    rng = random.Random(11)
    dirty = 0
    for _ in range(4000):
        av, bv = rng.randrange(p), rng.randrange(p)
        try:
            simulate(c, {"a": av, "b": bv}, rng=rng)
        except DirtyAncillaError:
            dirty += 1
    rate = dirty / 4000
    assert rate > 0.5 * 2 ** -k, f"expected ~2^-{k} dirty, saw {rate}"


@pytest.mark.parametrize("u,f,p", [(12, 5, 4091)])
def test_truncating_the_CANONICAL_carry_also_dirties_but_only_on_the_narrow_set(u, f, p):
    """Even with an exact flag, the canonical carry window is unsafe -- just far more
    rarely. The bad set is a+b in [p, 2^u): there the `-p` needs its carry to propagate
    all the way through bit u to clear it, and the window cuts that chain.

    Rate ~f/2^u = 2^-250.8 at real width, so this alone would not sink the design; the
    flag truncation above is what does. Recorded because the two have the same symptom
    and very different magnitudes.
    """
    c = _canon(u, p, carry_window=6, cmp_bits=None)

    def dirties(av, bv):
        try:
            simulate(c, {"a": av, "b": bv})
            return False
        except DirtyAncillaError:
            return True

    # every sum in the band [p, 2^u) dirties, whatever split of it we pick
    for s in range(p, 1 << u):
        av = s // 2
        assert dirties(av, s - av), f"a+b={s} in [p,2^u) should dirty ovf"

    # and sums just outside the band, on either side, do not
    rng = random.Random(13)
    for s in list(range(p - 40, p)) + list(range(1 << u, (1 << u) + 40)):
        av = rng.randrange(max(0, s - p + 1), min(p, s + 1))
        assert not dirties(av, s - av), f"a+b={s} is outside the band, should be clean"


def _one_line_apart(circ, a, b, p, carry_window, lazy_flag):
    """The two designs written as ONE circuit with a single switched line, so nothing
    else can be blamed for the difference."""
    n = len(a)
    f = (1 << n) - p
    ovf = circ.alloc(1)
    flag = circ.alloc(1)
    B = Register(list(b.ids) + [ovf[0]])

    gidney_add(circ, a, B)
    if lazy_flag:
        circ.cx(B[n], flag[0])                        # (a+b >= 2^u): 0 Toffoli, exact
    else:
        cmp_ge_const(circ, B, p, flag[0], k=None)     # (a+b >= p):   n Toffoli, exact
    addsub_p(circ, B, n, f, adding=False, ctrl=flag[0], inplace=True,
             add_fn=gidney_add, window=carry_window)
    m = circ.measure(flag[0])
    cmp_lt_phase(circ, a, B, m, k=None)
    circ.free(ovf)


def test_ONLY_the_flag_source_decides_whether_the_carry_window_is_safe():
    """THE load-bearing experiment for the laziness claim.

    Both flags here are EXACT -- no comparator truncation anywhere. The only difference
    is which question the flag answers, and that alone decides whether the carry window
    is usable:

        flag = (a+b >= p)     -> truncating the carry DIRTIES ovf
        flag = (a+b >= 2^u)   -> truncating the carry is clean

    Why: `B -= p` is "flip bit u, then += f". Split on where a+b lands.
      [0, p)      neither reduces. Nothing happens.
      [2^u, 2p)   both reduce. Bit u is 1, flipping CLEARS it, and the += f stays inside
                  the low bits. No long carry. Truncatable.
      [p, 2^u)    THE BAND WHERE THE FLAGS DISAGREE. Bit u is 0, so canonical's flip SETS
                  it, and only a carry running the full width u can clear it again --
                  a+b is within f of 2^u, so the low bits are nearly all ones and the
                  chain is maximal. The window cuts it, bit u survives, ovf is dirty.
                  Lazy simply declines to reduce here, which is the whole lazy contract.

    So the carry window is not a second, independent trick. The only branch that needs a
    long carry is exactly the branch laziness deletes. "Canonical with a truncated carry"
    is not a design that exists: make the carry local and you have stopped reducing on
    [p, 2^u), and that IS lazy.
    """
    u, f, p = 12, 5, 4091
    window = 6

    def run(lazy_flag, carry_window):
        c = Circuit()
        a = c.input("a", u)
        b = c.input("b", u)
        _one_line_apart(c, a, b, p, carry_window, lazy_flag)
        dirty = wrong = total = 0
        for av in range(0, p, 7):
            for bv in range(0, p, 13):
                total += 1
                try:
                    r = simulate(c, {"a": av, "b": bv})
                    wrong += r.outputs["b"] % p != (av + bv) % p
                except DirtyAncillaError:
                    dirty += 1
        return dirty / total, wrong / total

    # exact carry: both designs are clean and correct
    for lazy_flag in (False, True):
        d, w = run(lazy_flag, None)
        assert d == 0 and w == 0, f"lazy_flag={lazy_flag} should be exact"

    can_dirty, can_wrong = run(False, window)
    lazy_dirty, lazy_wrong = run(True, window)

    assert can_dirty > 0, "canonical flag + carry window must dirty ovf"
    assert lazy_dirty == 0, "lazy flag + carry window must stay clean"
    # and it is NOT that laziness hides the arithmetic error -- both pay the same one
    assert abs(can_wrong - lazy_wrong) < 1e-9, \
        f"same carry-truncation error expected, got {can_wrong} vs {lazy_wrong}"


def test_lazy_stays_clean_under_both_truncations():
    """The same two knobs, on the lazy design, never dirty anything: the flag is the
    true carry, and it is vented by measurement rather than freed."""
    u, p = 12, 4091
    c = _lazy(u, p, carry_window=6, cmp_bits=4)
    rng = random.Random(12)
    for _ in range(4000):
        av, bv = rng.randrange(p), rng.randrange(p)
        simulate(c, {"a": av, "b": bv}, rng=rng)      # DirtyAncillaError would raise


# ---------------------------------------------------------------------------
# what laziness COSTS: the output contract
# ---------------------------------------------------------------------------
def _unreduced_pairs(u: int, p: int) -> int:
    """Exact count of (a, b) in [0,p)^2 whose lazy sum is unreduced, in closed form.

    #{(a,b) : a+b = s} = p - |s - (p-1)|, and the output is unreduced exactly when
    s lands in [p, 2^u).
    """
    return sum(p - abs(s - (p - 1)) for s in range(p, 1 << u))


@pytest.mark.parametrize("u,f,p", [(6, 3, 61), (7, 5, 123)])
def test_closed_form_pair_count_matches_brute_force(u, f, p):
    brute = sum(1 for av in range(p) for bv in range(p)
                if modadd_lazy_ref(av, bv, p, u) >= p)
    assert brute == _unreduced_pairs(u, p)


@pytest.mark.parametrize("u,f,p", [(6, 3, 61), (8, 3, 253), (8, 5, 251),
                                   (10, 5, 1019), (12, 5, 4091), (20, 5, 1048571)])
def test_lazy_unreduced_rate_is_f_over_2_to_the_u(u, f, p):
    """The price of laziness is that outputs live in [0, 2^u), not [0, p).

    The rate is f/2^u, NOT f/2^(u+1): the sum a+b is spread over [0, 2p), not over the
    full (u+1)-bit range, so the unreduced window [p, 2^u) is a fraction of 2^u. That
    factor of 2 was wrong in the docstrings until this test was written.
    """
    measured = _unreduced_pairs(u, p) / p ** 2
    predicted = lazy_unreduced_prob(u, f)
    assert abs(measured - predicted) < 0.05 * predicted, \
        f"p=2^{u}-{f}: measured {measured:.3e} vs predicted {predicted:.3e}"
    # and it is NOT the half-as-large figure msb_reduce_fail_prob would give
    from qsm.params import msb_reduce_fail_prob
    assert abs(measured - msb_reduce_fail_prob(u, f)) > 0.4 * predicted


# ---------------------------------------------------------------------------
# does laziness COMPOSE? (the "stay lazy throughout, canonicalise once at the end" plan)
# ---------------------------------------------------------------------------
def _chain(u, p, k, carry_window=None, cmp_bits=None):
    c = Circuit()
    b = c.input("b", u)
    addends = [c.input(f"a{i}", u) for i in range(k)]
    for reg in addends:
        modadd_lazy(c, reg, b, p, gidney_add, carry_window=carry_window,
                    cmp_bits=cmp_bits, vent=True)
    return c


@pytest.mark.parametrize("u,f,p", [(9, 3, 509), (10, 5, 1019)])
def test_laziness_composes_values_stay_congruent_and_range_does_not_creep(u, f, p):
    """The load-bearing property for "stay lazy throughout":

      * values remain EXACTLY congruent mod p over a long chain (no drift), and
      * the unreduced rate does NOT accumulate -- it stays at f/2^u no matter how many
        lazy additions have been composed, because each step re-derives the range from
        its own overflow bit rather than inheriting the previous one's slack.

    The second point is the non-obvious one: a redundant representation that *widened*
    with every operation would be unusable.
    """
    c = _chain(u, p, 16)
    rng = random.Random(4)
    trials = 3000
    unreduced = 0
    for _ in range(trials):
        vals = {f"a{i}": rng.randrange(p) for i in range(16)}
        b0 = rng.randrange(p)
        vals["b"] = b0
        r = simulate(c, dict(vals), rng=rng)          # DirtyAncillaError would raise
        got = r.outputs["b"]
        want = (b0 + sum(vals[f"a{i}"] for i in range(16))) % p
        assert got % p == want, "lazy chain must stay exactly congruent"
        assert got < (1 << u), "must stay inside the register"
        unreduced += got >= p
    rate = unreduced / trials
    assert rate < 3 * lazy_unreduced_prob(u, f), \
        f"unreduced rate {rate:.4f} should not accumulate past ~{f}/2^{u}"


@pytest.mark.parametrize("u,f,p", [(9, 3, 509), (10, 5, 1019)])
def test_composing_laziness_costs_phase_at_half_f_over_2u_per_LATER_call(u, f, p):
    """What composition DOES cost, and it is not free.

    A lazily-reduced *input* (in [p, 2^u)) breaks step 3's erasure predicate `b < a`, so
    the vented comparator kicks back a phase. Measured law, with EVERY truncation knob
    off:

        P(phase) ~ (K - 1)/2 * f/2^u

    Note the shape: K-1, not K -- the FIRST call gets canonical inputs and is exactly
    clean (measured 0). And /2, because an unreduced input flips the predicate only about
    half the time.

    Values stay perfectly correct throughout; this is a phase-only cost, which is exactly
    what venting was for. At 25519 with ~2^20 modadds it is 2^20 * 2^-251.8 = 2^-231.8.
    """
    rng_seed = 5
    per_call = lazy_unreduced_prob(u, f)

    # K = 1: canonical inputs, so exactly zero exposure
    c1 = _chain(u, p, 1)
    rng = random.Random(rng_seed)
    for _ in range(2000):
        av, bv = rng.randrange(p), rng.randrange(p)
        assert simulate(c1, {"a0": av, "b": bv}, rng=rng).phase == 0, \
            "the first call has canonical inputs and must be phase-clean"

    # K > 1: linear in the number of calls that receive a lazy input
    for k in (8, 16):
        c = _chain(u, p, k)
        rng = random.Random(rng_seed)
        trials = 4000
        phase = 0
        for _ in range(trials):
            vals = {f"a{i}": rng.randrange(p) for i in range(k)}
            vals["b"] = rng.randrange(p)
            phase += simulate(c, dict(vals), rng=rng).phase != 0
        measured = phase / trials
        predicted = (k - 1) / 2 * per_call
        assert 0.5 * predicted < measured < 2.0 * predicted, \
            f"k={k}: phase {measured:.5f} vs predicted {predicted:.5f}"


@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (9, 3, 509), (8, 5, 251)])
def test_the_lazy_range_does_not_creep_even_on_adversarial_inputs(u, f, p):
    """The property that makes a redundant representation usable at all.

    Feed BOTH operands from the top of the lazy range -- the worst case -- and the
    output is still < 2^u. The range is re-established every step: `b` is a u-bit
    register and step 2 writes the reduced value back into it, so the width is enforced
    by the hardware, not by an invariant we have to prove inductively.

    So "lazy adds another f of slack each time" is NOT what happens. The slack is a
    fixed f, not a growing one.
    """
    c = _chain(u, p, 1)
    top = max(simulate(c, {"a0": av, "b": bv}).outputs["b"]
              for av in range(1 << u) for bv in range(1 << u))
    assert top < (1 << u), f"lazy output {top} escaped the register"


@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (9, 3, 509), (8, 5, 251)])
def test_lazy_PLUS_lazy_has_a_hole_at_the_top_but_lazy_PLUS_canonical_does_not(u, f, p):
    """A genuine gap in the lazy contract, found by adversarial input rather than by
    random sampling (the chain test missed it -- its addends were canonical).

    `modadd_lazy` writes its result into a u-bit register, so the step-2 `+f` can
    overflow if the reduced value is already within f of 2^u. That needs

        a + b >= 2^(u+1) - f

    which requires BOTH operands to be unreduced. Exactly f(f-1)/2 pairs, i.e.
    f(f-1)/2 / 2^(2u) = 2^-502.6 at 25519 -- twice as deep as the lazy contract itself.

    The important half is the second assertion: **canonical (+) lazy is closed**. With
    a < p and b < 2^u the worst case is (p-1) + (2^u - 1) = 2^(u+1) - f - 2, which
    clears the threshold by 2. So an accumulator pattern -- lazy accumulator, canonical
    addends -- never touches this hole at all, and that is the pattern a scalar
    multiplication actually uses.
    """
    c = _chain(u, p, 1)

    def wrong(av, bv):
        return simulate(c, {"a0": av, "b": bv}).outputs["b"] % p != (av + bv) % p

    bad = [(av, bv) for av in range(1 << u) for bv in range(1 << u) if wrong(av, bv)]
    assert len(bad) == f * (f - 1) // 2, f"expected f(f-1)/2 bad pairs, got {len(bad)}"
    assert all(av + bv >= (1 << (u + 1)) - f for av, bv in bad)
    assert all(av >= p and bv >= p for av, bv in bad), "both operands must be unreduced"

    # canonical x lazy is closed -- no hole at all
    assert not any(wrong(av, bv) for av in range(p) for bv in range(1 << u))


@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (9, 5, 507), (10, 3, 1021)])
def test_REJECTED_free_comparator_still_cannot_be_truncated(u, f, p):
    """Rejected alternative: merge intervals 2 and 3 cheaply via `s >= p <=> s+f >= 2^u`.

    The flag `s >= p` normally costs a full-width comparator. But adding f first makes
    bit u of the result equal that flag exactly, so the comparator looks free -- this is
    the classic pseudo-Mersenne conditional-subtract, and it IS exactly correct.

    It does not survive truncation, because THE FLAG IS THE CARRY OUT OF THE LOW u BITS.
    Truncating the carry and destroying the flag are the same operation, not two
    problems. The failures land on exactly interval 2, [p, 2^u).

    The general statement, which is why no implementation can rescue the 2+3 merge:
    truncation to a window L means "do not propagate information further than L bits",
    so it is compatible only with predicates of bounded bit-depth.

        s >= 2^u   (lazy)       depends on 1 bit
        s >= 2^u-f (canonical)  needs bits u-1..log2(f) ALL ONES -> ~u-5 bits

    Laziness does not compute the hard predicate more cleverly; it moves the boundary to
    a power of two so the predicate becomes one bit deep.

    This is a pure-arithmetic test: it rejects a whole class of circuits, so building one
    to measure would be wasted work.
    """
    full = 1 << (u + 1)

    def addf_then_bit(s, window):
        if window is None:
            t = (s + f) % full
        else:
            w = (1 << window) - 1
            lo, hi = s & ((1 << u) - 1), s >> u
            lo = (lo & ~w) | (((lo & w) + f) & w)     # carry out of the window discarded
            t = (hi << u) | lo
        flag = (t >> u) & 1
        return (t - (1 << u) if flag else t - f), flag

    # exact: the free comparator is genuinely correct
    assert all(addf_then_bit(s, None)[1] == (s >= p) for s in range(p))
    assert all(addf_then_bit(s, None)[0] % p == s % p for s in range(2 * p))

    # truncated: the flag itself breaks, on exactly interval 2
    window = max(2, (f - 1).bit_length() + 1)
    bad = [s for s in range(2 * p) if addf_then_bit(s, window)[1] != (s >= p)]
    assert bad, "truncating the +f must break the flag"
    assert all(p <= s < (1 << u) for s in bad), \
        f"failures should be exactly interval 2 [{p},{1 << u}), got {bad[:8]}"


def test_TRUNCATION_error_unlike_laziness_cannot_be_fixed_at_the_end():
    """The asymmetry that makes "lazy throughout, exact at the end" only half a plan.

    Laziness is a REPRESENTATION: every lazy output is exactly congruent to the true
    value, so one canonical reduction at the end recovers the right answer. Truncation is
    an ARITHMETIC ERROR: the value is not congruent to anything useful, and no later step
    can repair it. Its budget is per-call, N * 2^-40, and that is the knob that actually
    constrains the design.
    """
    u, f, p = 10, 5, 1019
    rng = random.Random(6)

    # lazy-but-exact: every output is congruent, so a final `% p` is a complete fix
    c_exact = _chain(u, p, 8)
    # lazy + truncated: outputs are simply wrong, and `% p` cannot help
    c_trunc = _chain(u, p, 8, carry_window=5, cmp_bits=None)

    exact_bad = trunc_bad = 0
    for _ in range(2000):
        vals = {f"a{i}": rng.randrange(p) for i in range(8)}
        vals["b"] = rng.randrange(p)
        want = sum(vals.values()) % p
        exact_bad += simulate(c_exact, dict(vals), rng=rng).outputs["b"] % p != want
        trunc_bad += simulate(c_trunc, dict(vals), rng=rng).outputs["b"] % p != want

    assert exact_bad == 0, "laziness alone is always repairable by a final reduction"
    assert trunc_bad > 0, "truncation error survives any amount of final canonicalising"


# ---------------------------------------------------------------------------
# why there is no canonicalize circuit (docs/decisions/04-*.md)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u,f,p", [(8, 3, 253), (8, 5, 251), (9, 3, 509)])
def test_canonicalization_is_not_a_unitary_so_no_circuit_can_do_it_cleanly(u, f, p):
    """The argument behind `04-no-in-circuit-canonicalization.md`.

    Every flag we uncompute elsewhere is REDUNDANT -- recoverable from the result, so
    recomputing and XOR-ing it away is exact and free. Canonicalize's flag is not: it is
    the bit that distinguishes `r` from `r + p`, i.e. exactly what the operation destroys.

    Three checks, in the order the argument needs them:
      1. `x -> x mod p` is 2-to-1 on exactly f residues, so it is not a permutation.
      2. No function of the result recovers the flag -- the ambiguous residues carry BOTH
         flag values. This is what kills the "just use a full-width comparator" repair:
         after canonicalization r < p by construction, so any comparison against any
         constant answers identically for both preimages. Precision is not the issue.
      3. (result, flag) <-> x IS a bijection, which is the positive statement of the same
         fact: the flag is not garbage, it is the missing information.

    Contrast `modadd`, checked here too, where the flag IS a function of the result.
    """
    # 1. two-to-one on exactly f residues
    preimages = {}
    for x in range(1 << u):
        preimages.setdefault(x % p, []).append(x)
    doubled = [r for r, xs in preimages.items() if len(xs) > 1]
    assert len(doubled) == f, f"expected {f} doubled residues, got {len(doubled)}"
    assert all(len(preimages[r]) == 2 for r in doubled)
    assert doubled == list(range(f)), "the doubled residues are exactly [0, f)"

    # 2. no function of the result recovers the flag
    flags_by_result = {}
    for x in range(1 << u):
        flags_by_result.setdefault(x % p, set()).add(x >= p)
    ambiguous = [r for r, s in flags_by_result.items() if len(s) > 1]
    assert ambiguous == doubled, "ambiguity is exactly the doubled residues"
    # and specifically: comparing the result against p (the proposed repair) is constant
    assert all(x % p < p for x in range(1 << u)), "r < p by construction, so the compare is vacuous"

    # 3. (result, flag) <-> x is a bijection
    pairs = {(x % p, x >= p) for x in range(1 << u)}
    assert len(pairs) == (1 << u), "flag carries exactly the missing information"

    # contrast: modadd's flag IS a function of its result, which is why IT uncomputes
    for a in (0, 1, p // 3, p - 1):
        assert all(((a + b) % p < a) == (a + b >= p) for b in range(p)), \
            "modadd's flag must be recoverable from the result"


def test_lazy_contract_cost_is_negligible_at_25519():
    """2^-250.8 -- 210 orders of magnitude below the truncation error it enables."""
    assert lazy_unreduced_prob(U, F) < 1e-75
    from qsm.params import carry_fail_prob
    assert lazy_unreduced_prob(U, F) < 1e-60 * carry_fail_prob(CARRY_WINDOW, F)
