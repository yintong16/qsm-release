"""Comparators: write a predicate into a flag qubit without disturbing the operands.

A comparison is an addition that throws the sum away and keeps only the carry-out. So
these run the same Gidney carry chain as the adders, copy the top carry into `flag`, then
run the chain backwards -- and the backward pass erases every carry by measurement, at
**zero Toffoli**. A comparison therefore costs about one adder pass (~n Toffoli) and
leaves its operands exactly as it found them.

Identities used:

    carry-out of (B + (2^n - c))  ==  (B >= c)        unsigned, n-bit B
    carry-out of (~B + a)         ==  (B <  a)        borrow-out of B - a

The second holds because the borrow recurrence of B - a is the carry recurrence of
~B + a: borrow_{i+1} = MAJ(~B_i, a_i, borrow_i), with borrow_0 = carry_0 = 0.

These exist so modular reduction can *test* whether it needs to subtract p, rather than
subtracting speculatively and adding back -- which is what costs `modadd` v1/v2 two extra
adder passes.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import CMP_BITS, PRIME, U, cmp_fail_prob, cmp_ge_const_fail_prob
from .add import _and, _unand


def carry_out_const(circ: Circuit, B: Register, c: int, flag: int) -> None:
    """flag ^= carry-out of (B + c) for classical c. B is restored exactly.

    ~n-1-k Toffoli and n-1-k ancillas, where k is c's lowest set bit (carries below it
    are known |0>, so those positions are skipped entirely). The top carry lands straight
    in `flag` rather than in an ancilla of its own, which is where the -1 comes from.
    """
    n = len(B)
    if n == 0:
        return
    c &= (1 << n) - 1
    if c == 0:
        return                                  # B + 0 never carries out

    bits = [(c >> i) & 1 for i in range(n)]
    k = next(i for i in range(n) if bits[i])

    if k == n - 1:                              # carry_n = MAJ(B_{n-1}, 1, 0) = B_{n-1}
        circ.cx(B[n - 1], flag)
        return

    m = n - 1 - k                               # carries carry_{k+1} .. carry_{n-1}
    g = circ.alloc(m)                           # g[j] == carry_{k+1+j}

    # carry_{k+1} = MAJ(B_k, 1, 0) = B_k -- a copy, no Toffoli.
    circ.cx(B[k], g[0])

    for i in range(k + 1, n - 1):               # forward carry chain
        j = i - k
        circ.cx(g[j - 1], B[i])                 # B_i ^= carry_i
        if bits[i]:
            circ.x(g[j - 1])                    # read c_i ^ carry_i
        _and(circ, g[j - 1], B[i], g[j])
        if bits[i]:
            circ.x(g[j - 1])
        circ.cx(g[j - 1], g[j])                 # ^= carry_i -> carry_{i+1}

    # Top carry lands straight in `flag`. Unlike every other carry it is never
    # uncomputed -- it *is* the answer -- so it needs no erasable ancilla of its own.
    # Same one Toffoli, one fewer qubit, one fewer measurement.
    i = n - 1
    j = i - k
    circ.cx(g[j - 1], B[i])
    if bits[i]:
        circ.x(g[j - 1])
    _and(circ, g[j - 1], B[i], flag)            # flag ^= (c_i ^ carry_i)(B_i ^ carry_i)
    if bits[i]:
        circ.x(g[j - 1])
    circ.cx(g[j - 1], flag)                     # flag ^= carry_i  -> flag ^= carry_n
    circ.cx(g[j - 1], B[i])                     # restore B_{n-1}

    for i in range(n - 2, k, -1):               # unwind: 0 Toffoli, restores B
        j = i - k
        circ.cx(g[j - 1], g[j])
        if bits[i]:
            circ.x(g[j - 1])
        _unand(circ, g[j - 1], B[i], g[j])
        if bits[i]:
            circ.x(g[j - 1])
        circ.cx(g[j - 1], B[i])                 # restore B_i

    circ.cx(B[k], g[0])
    circ.free(g[0:1])                           # the rest were measured away


def _flag_and(
    circ: Circuit, ctrl: int | None, x: int, y: int, flag: int
) -> None:
    """flag ^= x & y, or flag ^= ctrl & x & y. 1 Toffoli, or 2 and a scratch qubit."""
    if ctrl is None:
        _and(circ, x, y, flag)
        return
    t = circ.alloc(1)
    _and(circ, x, y, t[0])
    circ.ccx(ctrl, t[0], flag)
    _unand(circ, x, y, t[0])                    # free; consumes t


def _flag_cx(circ: Circuit, ctrl: int | None, src: int, flag: int) -> None:
    """flag ^= src, or flag ^= ctrl & src."""
    if ctrl is None:
        circ.cx(src, flag)
    else:
        circ.ccx(ctrl, src, flag)


def carry_out(
    circ: Circuit, a: Register, B: Register, flag: int, ctrl: int | None = None
) -> None:
    """flag ^= carry-out of (B + a), both operands restored. len(a) may be < len(B).

    n Toffoli, n-1 ancillas (`alloc(n-1)`: the top carry is written to `flag`, not to an
    ancilla).

    With `ctrl` given this becomes `flag ^= ctrl & carry-out`, for **two extra Toffoli
    total** -- not two per bit. The carry ladder is what costs n Toffoli and it does not
    need controlling at all: it is computed and then erased whatever `ctrl` is, and the
    only place the answer escapes is the single write into `flag`. Gating that one write
    turns the top block's `flag ^= c_{n-1} ^ (a^c)(B^c)` into a 3-input AND plus one
    Toffoli, i.e. 3 Toffoli where the uncontrolled version spends 1.

    This is why the dialog loop's `b0 & (u > v)` does not cost a second comparator pass:
    computing the predicate into an ancilla and ANDing afterwards would pay for the
    ladder twice.
    """
    n = len(B)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(B)"
    if n == 0 or na == 0:
        return
    if n == 1:
        _flag_and(circ, ctrl, a[0], B[0], flag)  # carry_1 = a_0 & B_0
        return

    g = circ.alloc(n - 1)                       # g[i] == carry_{i+1}, i = 0..n-2
    _and(circ, a[0], B[0], g[0])                # carry_1 = a_0 & B_0

    for i in range(1, n - 1):
        if i < na:
            circ.cx(g[i - 1], a[i])
            circ.cx(g[i - 1], B[i])
            _and(circ, a[i], B[i], g[i])
            circ.cx(g[i - 1], g[i])
        else:                                   # a_i == 0
            circ.cx(g[i - 1], B[i])
            _and(circ, g[i - 1], B[i], g[i])
            circ.cx(g[i - 1], g[i])

    # Top carry straight into `flag` -- it is the answer, so it is never uncomputed and
    # needs no ancilla. Same one Toffoli, one fewer qubit, one fewer measurement.
    i = n - 1
    if i < na:
        circ.cx(g[i - 1], a[i])
        circ.cx(g[i - 1], B[i])
        _flag_and(circ, ctrl, a[i], B[i], flag)
        _flag_cx(circ, ctrl, g[i - 1], flag)    # ^= carry_{n-1}  -> flag ^= carry_n
        circ.cx(g[i - 1], a[i])
        circ.cx(g[i - 1], B[i])
    else:                                       # a_{n-1} == 0
        circ.cx(g[i - 1], B[i])
        _flag_and(circ, ctrl, g[i - 1], B[i], flag)
        _flag_cx(circ, ctrl, g[i - 1], flag)
        circ.cx(g[i - 1], B[i])

    for i in range(n - 2, 0, -1):               # unwind, restoring a and B
        circ.cx(g[i - 1], g[i])
        if i < na:
            _unand(circ, a[i], B[i], g[i])
            circ.cx(g[i - 1], a[i])
            circ.cx(g[i - 1], B[i])
        else:
            _unand(circ, g[i - 1], B[i], g[i])
            circ.cx(g[i - 1], B[i])
    _unand(circ, a[0], B[0], g[0])
    # every g qubit was consumed by a measurement


def cmp_ge_const(
    circ: Circuit, B: Register, c: int, flag: int, k: int | None = None
) -> None:
    """flag ^= (B >= c) for classical c. B restored.

    `k` truncates the comparison to the top k bits of B (and the correspondingly shifted
    top bits of c), costing ~k Toffoli instead of ~n. Exact when k is None or k >= n.

    Accuracy depends on c's LOW bits, not just on k. On a tie in the top k bits this
    answers ">=", so it is wrong exactly when the discarded low parts disagree:

        P = c_low / 2^n,    c_low = c mod 2^(n-k)

    For c = p = 2^U - F the low part is nearly maximal, so a tie almost always means we
    answered wrongly -- P collapses to about 2^-k with none of the usual factor-2 relief.
    If what you actually want is "did this overflow past 2^U", read bit U instead: free,
    and wrong only on the F values in [p, 2^U). See qsm.params.msb_reduce_fail_prob.
    """
    n = len(B)
    if n == 0:
        return
    if c <= 0:
        circ.x(flag)                            # unsigned B >= 0 always
        return
    if c >= (1 << n):
        return                                  # unreachable; flag unchanged

    if k is None or k >= n:
        carry_out_const(circ, B, (1 << n) - c, flag)
        return

    lo = n - k                                  # keep bits [lo, n)
    c_top = c >> lo
    if c_top == 0:
        circ.x(flag)                            # B_top >= 0 always
        return
    carry_out_const(circ, B[lo:n], (1 << k) - c_top, flag)


def cmp_lt(
    circ: Circuit, a: Register, B: Register, flag: int, k: int | None = None,
    ctrl: int | None = None,
) -> None:
    """flag ^= (B < a), i.e. the borrow-out of B - a. Both operands restored.

    `ctrl` gives `flag ^= ctrl & (B < a)` for two extra Toffoli; see `carry_out`. The
    NOTs on B are unconditional either way -- they are self-inverse and cancel, so only
    the flag write has to be gated.

    `k` truncates to the top k bits of both operands, costing ~k Toffoli instead of ~n.
    When the top k bits differ the answer is *exactly* right -- the highest differing bit
    settles the ordering. The only exposure is a tie, and then the discarded low bits
    decide it 50/50, so for uniform independent operands

        P = 2^-k * 1/2 = 2^-(k+1)

    (One bit better than the 2^-k usually quoted.) A shorter `a` is zero-extended, so the
    slices need not be the same width.
    """
    n = len(B)
    if n == 0 or len(a) == 0:
        return

    if k is None or k >= n:
        Bs, A = B, a
    else:
        lo = n - k
        Bs = B[lo:n]
        A = a[lo:len(a)] if len(a) > lo else Register([])
        if len(A) == 0:
            return                              # a's top k bits are all zero: B >= a
    for q in Bs:
        circ.x(q)
    carry_out(circ, A, Bs, flag, ctrl=ctrl)
    for q in Bs:
        circ.x(q)


def cmp_lt_phase(
    circ: Circuit,
    a: Register,
    B: Register,
    m: int,
    k: int | None = None,
    ctrl: int | None = None,
) -> None:
    """Phase-only form of `flag ^= (B < a)`, for measurement-based flag erasure.

    Applies phase (-1)^(m * ctrl * [B < a]) for a *classical* bit `m`, touching no
    qubit's computational-basis value. Pair it with a preceding `m = circ.measure(flag)`:

        m = circ.measure(flag)        # erases flag UNCONDITIONALLY, 0 Toffoli
        cmp_lt_phase(circ, a, B, m, k=k)

    The measurement leaves phase (-1)^(flag * m); this contributes (-1)^(m * [B<a]);
    they cancel exactly when flag == [B<a]. So a wrong comparison costs a PHASE error
    rather than leaving `flag` stuck at |1>. That is the difference between
    ecdsa.fail's construction (`hmr` then `cz_if_bit`) and writing the comparison into
    the flag with a CX, which fails open and frees a dirty ancilla.

    `ctrl` -- a qubit -- is for venting a flag that was itself computed under a control,
    as in `modadd_lazy_ctrl`, where `flag = ctrl & (a+B >= 2^n)`. **It is not optional
    there.** With ctrl = 0 the accumulator is untouched, so `flag` is 0 while `[B < a]`
    is whatever the data says -- a coin flip -- and the two stop cancelling on half of
    all *correct* runs. The resulting phase is invisible to every value comparison, which
    is exactly how it survived the replay's first small-prime run. It costs one Toffoli:
    the chain below never materialises the top carry, so the only place `ctrl` has to
    reach is the pair of top-of-chain phase gates.

    Never materialises the top carry. Using carry_n = c_{n-1} ^ (a'&B'):

        phase ^= m*c_{n-1}       via a conditioned Z
        phase ^= m*(a' & B')     via a conditioned CZ

    so the top step costs ZERO Toffoli (the value form spends one on `_and`). The rest
    of the chain is computed and then exactly gate-reversed, which returns its scratch
    to |0> for every input regardless of truncation -- truncation changes *which* value
    the carry holds, not *whether* the reverse cleans it.
    """
    n = len(B)
    na = len(a)
    if n == 0 or na == 0:
        return

    if k is None or k >= n:
        Bs, A = B, a
    else:
        lo = n - k
        Bs = B[lo:n]
        A = a[lo:len(a)] if len(a) > lo else Register([])
        if len(A) == 0:
            return                                  # a's top k bits are zero: B >= a
    for q in Bs:
        circ.x(q)                                   # borrow-out of B-a == carry of ~B+a
    _carry_out_phase(circ, A, Bs, m, ctrl=ctrl)
    for q in Bs:
        circ.x(q)


def _cz_ctrl(circ: Circuit, m: int, x: int, y: int, ctrl: int | None) -> None:
    """phase ^= m * ctrl * x * y. One Toffoli when `ctrl` is a qubit, zero when it is not."""
    if ctrl is None:
        with circ.condition(m):
            circ.cz(x, y)
        return
    t = circ.alloc(1)[0]
    _and(circ, ctrl, x, t)
    with circ.condition(m):
        circ.cz(t, y)
    _unand(circ, ctrl, x, t)                        # phase-clean, 0 Toffoli


def _carry_out_phase(
    circ: Circuit, a: Register, B: Register, m: int, ctrl: int | None = None
) -> None:
    """phase ^= m * ctrl * carry-out of (B + a). Both operands restored, no flag written."""
    n = len(B)
    na = len(a)
    if n == 0 or na == 0:
        return
    if n == 1:
        _cz_ctrl(circ, m, a[0], B[0], ctrl)         # carry_1 = a_0 & B_0
        return

    g = circ.alloc(n - 1)
    _and(circ, a[0], B[0], g[0])
    for i in range(1, n - 1):
        if i < na:
            circ.cx(g[i - 1], a[i])
            circ.cx(g[i - 1], B[i])
            _and(circ, a[i], B[i], g[i])
            circ.cx(g[i - 1], g[i])
        else:
            circ.cx(g[i - 1], B[i])
            _and(circ, g[i - 1], B[i], g[i])
            circ.cx(g[i - 1], g[i])

    # Top carry as PHASE, never as a value: carry_n = c_{n-1} ^ (a' & B').
    i = n - 1
    if i < na:
        circ.cx(g[i - 1], a[i])
        circ.cx(g[i - 1], B[i])
        _cz_ctrl(circ, m, a[i], B[i], ctrl)
        with circ.condition(m):
            circ.z(g[i - 1]) if ctrl is None else circ.cz(ctrl, g[i - 1])
        circ.cx(g[i - 1], a[i])
        circ.cx(g[i - 1], B[i])
    else:
        circ.cx(g[i - 1], B[i])
        _cz_ctrl(circ, m, g[i - 1], B[i], ctrl)
        with circ.condition(m):
            circ.z(g[i - 1]) if ctrl is None else circ.cz(ctrl, g[i - 1])
        circ.cx(g[i - 1], B[i])

    for i in range(n - 2, 0, -1):                   # exact unwind
        circ.cx(g[i - 1], g[i])
        if i < na:
            _unand(circ, a[i], B[i], g[i])
            circ.cx(g[i - 1], a[i])
            circ.cx(g[i - 1], B[i])
        else:
            _unand(circ, g[i - 1], B[i], g[i])
            circ.cx(g[i - 1], B[i])
    _unand(circ, a[0], B[0], g[0])


# ---------------------------------------------------------------------------
# Registered benchmark versions.
# ---------------------------------------------------------------------------
_N = U


@circuit_version("field.cmp_ge_p.v1")
def _build_cmp_ge_p_v1() -> Case:
    """flag ^= (B >= p) on a 256-bit register -- modadd v3's step 2."""
    circ = Circuit()
    b = circ.input("b", _N + 1)
    flag = circ.input("flag", 1)
    cmp_ge_const(circ, b, PRIME, flag[0])

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"b": v["b"], "flag": v["flag"] ^ int(v["b"] >= PRIME)}

    def gen(rng: random.Random) -> dict[str, int]:
        # straddle the boundary so both branches are exercised
        b = rng.choice([rng.randrange(PRIME), PRIME + rng.randrange(1 << 40),
                        rng.getrandbits(_N + 1)])
        return {"b": b % (1 << (_N + 1)), "flag": rng.getrandbits(1)}

    return Case(circ, ref, gen)


@circuit_version("field.cmp_lt_trunc.v1")
def _build_cmp_lt_trunc_v1() -> Case:
    """
    APPROXIMATE. flag ^= (B < a) on the top CMP_BITS bits only.

    Wrong only when those bits TIE (prob 2^-k); given a tie the discarded low bits decide
    it 50/50, so the generic rate is P = 2^-(k+1). When the top bits differ the answer is
    exactly right -- there is no error term at all in that case.

    Cost: one Toffoli per compared bit, so CMP_BITS instead of ~256.

    For THIS benchmark's operand shape the rate is 2^-k = 2^-40, not the generic
    2^-(k+1) = 2^-41, and the reason is the operand range
    rather than anything about ties: both operands are < 2^U inside a (U+1)-bit register,
    so the top window bit is 0 for both and carries no information. Only k-1 of the k
    compared bits inform the result, hence `informative_bits=CMP_BITS - 1`.
    (Correlation between the operands -- modadd compares (a+b) mod p against a -- was
    measured and does NOT matter; the range does.)

    Worst case is far worse: feed it near-ties (b = a, a+-1) and it is wrong ~34% of the
    time -- and that average hides the real shape, since b = a and b = a+1 are answered
    correctly every time while b = a-1 is wrong every time. See tests/test_field_compare.py::
    test_truncated_cmp_is_catastrophic_on_near_ties and
    ::test_operand_range_costs_one_window_bit.
    """
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N + 1)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=CMP_BITS)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": v["b"], "flag": v["flag"] ^ int(v["b"] < v["a"])}

    def gen(rng: random.Random) -> dict[str, int]:
        # exactly what modadd step 4 sees, not an idealised independent pair
        a = rng.randrange(PRIME)
        return {"a": a, "b": (a + rng.randrange(PRIME)) % PRIME,
                "flag": rng.getrandbits(1)}

    return Case(circ, ref, gen, meta={
        "fail_analytic": cmp_fail_prob(CMP_BITS, informative_bits=CMP_BITS - 1)})


@circuit_version("field.cmp_ge_p_trunc.v1")
def _build_cmp_ge_p_trunc_v1() -> Case:
    """
    APPROXIMATE, and deliberately registered to show a TRAP.

    Truncating `B >= p` to the top CMP_BITS bits is a bad deal: accuracy depends on the
    constant's LOW bits, not on k. Here P = p_low / 2^n with p_low = p mod 2^(n-k), and
    p = 2^255-19 has a nearly maximal low part -- so a tie almost always means a wrong
    answer, and none of the usual factor-2 relief applies.

    The right construction for this particular test is to read bit U instead (free, and
    wrong only on the 19 values in [p, 2^255) -> ~2^-252). That is what modadd.v4 will
    do; this version exists so the table shows why.
    """
    circ = Circuit()
    b = circ.input("b", _N + 1)
    flag = circ.input("flag", 1)
    cmp_ge_const(circ, b, PRIME, flag[0], k=CMP_BITS)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"b": v["b"], "flag": v["flag"] ^ int(v["b"] >= PRIME)}

    def gen(rng: random.Random) -> dict[str, int]:
        # uniform, matching the assumption behind the analytic rate. (The boundary
        # probes used by the exact version sit inside this circuit's blind spot.)
        return {"b": rng.getrandbits(_N + 1), "flag": rng.getrandbits(1)}

    return Case(circ, ref, gen,
                meta={"fail_analytic": cmp_ge_const_fail_prob(PRIME, _N + 1, CMP_BITS)})


@circuit_version("field.cmp_lt.v1")
def _build_cmp_lt_v1() -> Case:
    """flag ^= (B < a), 255-bit a against a 256-bit B -- modadd v3's step 4."""
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N + 1)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0])

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": v["b"], "flag": v["flag"] ^ int(v["b"] < v["a"])}

    def gen(rng: random.Random) -> dict[str, int]:
        a = rng.randrange(PRIME)
        b = rng.choice([a, a - 1 if a else 0, a + 1, rng.randrange(PRIME)])
        return {"a": a, "b": b % (1 << (_N + 1)), "flag": rng.getrandbits(1)}

    return Case(circ, ref, gen)


def _cmp_lt_ctrl_case(k: int | None) -> Case:
    circ = Circuit()
    ctrl = circ.input("ctrl", 1)
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    flag = circ.input("flag", 1)
    cmp_lt(circ, a, b, flag[0], k=k, ctrl=ctrl[0])

    def ref(v: dict[str, int]) -> dict[str, int]:
        bit = v["ctrl"] & int(v["b"] < v["a"])
        return {"ctrl": v["ctrl"], "a": v["a"], "b": v["b"], "flag": v["flag"] ^ bit}

    def gen_near_tie(rng: random.Random) -> dict[str, int]:
        a = rng.randrange(PRIME)
        b = rng.choice([a, a - 1 if a else 0, a + 1, rng.randrange(PRIME)])
        return {"ctrl": rng.getrandbits(1), "a": a, "b": b % (1 << _N),
                "flag": rng.getrandbits(1)}

    def gen_uniform(rng: random.Random) -> dict[str, int]:
        # The truncated version must be scored against the operand shape its analytic
        # rate assumes. Near-ties are its documented catastrophic case, not its average
        # -- see `field.cmp_lt_trunc.v1` and test_truncated_cmp_is_catastrophic_on_near_ties.
        return {"ctrl": rng.getrandbits(1), "a": rng.randrange(PRIME),
                "b": rng.randrange(PRIME), "flag": rng.getrandbits(1)}

    if k is None:
        return Case(circ, ref, gen_near_tie)
    return Case(circ, ref, gen_uniform,
                meta={"fail_analytic": cmp_fail_prob(k, _N, informative_bits=k)})


@circuit_version("field.cmp_lt_ctrl.v1")
def _build_cmp_lt_ctrl_v1() -> Case:
    """flag ^= ctrl & (B < a), exact. The dialog loop's `b0 & (u > v)`.

    Two Toffoli more than `field.cmp_lt.v1`, not two per bit: only the write into
    `flag` is gated, because the carry ladder is erased regardless of `ctrl`.
    """
    return _cmp_lt_ctrl_case(None)


@circuit_version("field.cmp_lt_ctrl_trunc.v1")
def _build_cmp_lt_ctrl_trunc_v1() -> Case:
    """The same, truncated to the top CMP_BITS -- what Phase 1 actually runs."""
    return _cmp_lt_ctrl_case(CMP_BITS)

    return Case(circ, ref, gen)
