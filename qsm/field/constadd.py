"""Addition of a *classical* constant to a quantum register.

This is the primitive the 25519 modular layer leans on, because p = 2^255 - 19 is a
pseudo-Mersenne prime: on an (n+1)-bit register,

    b - p  ==  b - 2^255 + 19  ==  (b XOR 2^255) + 19   (mod 2^256)

i.e. subtracting the 255-bit prime is *one X gate on the top bit plus a 5-bit constant
add*. A generic prime would need a full-width constant subtraction. That asymmetry is
exactly the structural advantage Wei25519/Ed25519 has over a random-prime curve, so this
module is where a big part of the paper's claim gets cashed out.

v1 here is the obvious-and-obviously-correct baseline: load the constant into a borrowed
register and reuse the register-register adder. It costs `c.bit_length()` extra ancillas
-- 5 for c = 19, only the full n for a dense constant like p itself -- which is
wasteful -- an in-place constant adder needs none, since the constant's bits are known at
compile time and collapse the carry logic (c_i = 0 -> carry & b_i; c_i = 1 -> carry | b_i).
That optimisation is deliberately left for a v2 so the modular layer has a *verified*
floor to be measured against first.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import CARRY_WINDOW, F, PRIME, U, carry_fail_prob
from .add import _unand, cdkm_add, gidney_add

AddFn = Callable[[Circuit, Register, Register], None]


def add_const(circ: Circuit, b: Register, c: int, add_fn: AddFn = gidney_add) -> None:
    """In place: b += c (mod 2^n) for classical c. Baseline: c.bit_length() ancillas.

    Only the *significant* bits of c get qubits -- a k-bit constant needs a k-qubit
    register, not an n-qubit one, because the adders zero-extend a short first operand.
    For c = 19 into a 255-bit register that is 5 qubits instead of 255. The carry chain
    still runs the full width of b (adding 1 to 0b11..1 ripples to the top), so this is
    a pure space saving, not a Toffoli one.
    """
    n = len(b)
    if n == 0:
        return
    c &= (1 << n) - 1
    if c == 0:
        return

    w = c.bit_length()            # high bits of c are zero -> no qubits needed for them
    tmp = circ.alloc(w)
    set_bits = [i for i in range(w) if (c >> i) & 1]
    for i in set_bits:
        circ.x(tmp[i])
    add_fn(circ, tmp, b)          # adders preserve their first operand...
    for i in set_bits:
        circ.x(tmp[i])            # ...so undoing the loads returns tmp to |0>
    circ.free(tmp)


def sub_const(circ: Circuit, b: Register, c: int, add_fn: AddFn = gidney_add) -> None:
    """In place: b -= c (mod 2^n). Two's complement of the constant; same cost."""
    n = len(b)
    if n == 0:
        return
    add_const(circ, b, (-c) % (1 << n), add_fn)


def add_const_controlled(
    circ: Circuit, ctrl: int, b: Register, c: int, add_fn: AddFn = gidney_add
) -> None:
    """In place: b += c iff qubit `ctrl` is |1>. ctrl is preserved.

    The control is applied to the *constant load* rather than to the adder itself, so an
    uncontrolled adder suffices: tmp holds (ctrl ? c : 0), and adding 0 is the identity.
    This keeps us from needing a controlled-adder construction at all.

    As in `add_const`, only c.bit_length() qubits are materialised. Note this saves
    nothing for modular reduction, whose constants (+-p mod 2^(n+1)) have their top bit
    set and so are full-width however few bits they have set.
    """
    n = len(b)
    if n == 0:
        return
    c &= (1 << n) - 1
    if c == 0:
        return

    w = c.bit_length()
    tmp = circ.alloc(w)
    set_bits = [i for i in range(w) if (c >> i) & 1]
    for i in set_bits:
        circ.cx(ctrl, tmp[i])
    add_fn(circ, tmp, b)
    for i in set_bits:
        circ.cx(ctrl, tmp[i])
    circ.free(tmp)


def sub_const_controlled(
    circ: Circuit, ctrl: int, b: Register, c: int, add_fn: AddFn = gidney_add
) -> None:
    """In place: b -= c iff qubit `ctrl` is |1>. ctrl is preserved.

    Currently unreachable from any registered version: its only caller is `modadd`'s
    `sub_p`, which is invoked solely with `ctrl=None`, and then only on the
    `pseudo_mersenne=False` path that nothing selects. Kept because it completes the v1
    constant-adder family that `add_const_controlled` belongs to, and because the
    naive-form comparison in `modadd` is the baseline the pseudo-Mersenne decomposition is
    measured against. Verified correct by `qsm.audit.refs`, which exercises it directly.
    """
    n = len(b)
    if n == 0:
        return
    add_const_controlled(circ, ctrl, b, (-c) % (1 << n), add_fn)


# ---------------------------------------------------------------------------
# v2: in-place constant addition -- no constant register at all.
# ---------------------------------------------------------------------------
def add_const_inplace(
    circ: Circuit,
    b: Register,
    c: int,
    ctrl: int | None = None,
    window: int | None = None,
) -> None:
    """In place: b += c (mod 2^n) with NO constant register. ~n-1 Toffoli.

    `window` truncates the addition to the low `window` bits, dropping the carry out of
    bit window-1. Cost falls from ~n to ~window Toffoli. This is wrong exactly when the
    addition would have carried out of the window:

        (b mod 2^window) + c >= 2^window   <=>   b mod 2^window >= 2^window - c

    which is exactly `c` of the 2^window possible low parts (the top c values), so for
    uniform b:

        P(wrong) = c / 2^window            (exact, not asymptotic)

    For c = F = 19 and window = CARRY_WINDOW = 45 that is 19/2^45 ~ 2^-40.8. Truncation
    is implemented simply by operating on the register slice -- the circuit stays an
    exact unitary on those qubits, so ancillas and phase remain clean; only the function
    being computed moves.


    If `ctrl` is a qubit id, the whole addition is conditioned on it (ctrl preserved).
    Controlling costs only +2 Toffoli: every gate that injects a *classical* constant
    bit becomes a CX from ctrl, and the two carry-seed CX gates become CCX. When ctrl
    is |0> the seed stays |0>, so the entire carry chain collapses to |0> and every
    conditioned flip is skipped -- the identity, as required.


    Same Gidney carry recurrence as `gidney_add`, but the second operand's bits are
    known at compile time, so instead of a quantum register we fold each classical bit
    into the carry with an X:

        carry_{i+1} = carry_i ^ (c_i ^ carry_i)(b_i ^ carry_i)
                    = b_i AND carry_i      when c_i = 0
                    = b_i OR  carry_i      when c_i = 1

    Two structural savings over the load-a-register baseline: the n-qubit constant
    register disappears entirely, and every bit below c's lowest set bit is a no-op
    (the carry is still |0> there, so those bits pass through untouched).

    Toffoli count is *not* improved -- a carry chain still has to run to the top of the
    register, since adding 1 to 0b11..1 propagates all the way. The win is space.
    """
    n = len(b)
    if n == 0:
        return
    if window is not None and window < n:
        assert c < (1 << window), (
            f"constant {c} does not fit in a {window}-bit window; raise the window"
        )
        b = b[0:window]                        # truncation is just a register slice
        n = window
    c &= (1 << n) - 1
    if c == 0:
        return

    bits = [(c >> i) & 1 for i in range(n)]
    k = next(i for i in range(n) if bits[i])   # lowest set bit; carry_i = 0 for i <= k

    def flip(t: int) -> None:
        """Inject a classical 1 into qubit t -- gated on ctrl when controlled."""
        if ctrl is None:
            circ.x(t)
        else:
            circ.cx(ctrl, t)

    def seed(src: int, tgt: int) -> None:
        """tgt ^= src (gated on ctrl): seeds and later clears the carry chain."""
        if ctrl is None:
            circ.cx(src, tgt)
        else:
            circ.ccx(ctrl, src, tgt)

    if k == n - 1:                             # only the top bit survives: pure XOR
        flip(b[n - 1])
        return

    m = n - 1 - k                              # carries carry_{k+1} .. carry_{n-1}
    g = circ.alloc(m)                          # g[j] holds carry_{k+1+j}

    # carry_{k+1} = MAJ(b_k, 1, 0) = b_k -- a copy, not an AND.
    seed(b[k], g[0])

    # --- forward: carry_{i+1} into g[i-k] ---
    for i in range(k + 1, n - 1):
        j = i - k
        circ.cx(g[j - 1], b[i])                # b_i ^= carry_i
        if bits[i]:
            flip(g[j - 1])                     # g[j-1] now reads c_i ^ carry_i
        circ.ccx(g[j - 1], b[i], g[j])
        if bits[i]:
            flip(g[j - 1])                     # restore carry_i
        circ.cx(g[j - 1], g[j])                # ^= carry_i  ->  carry_{i+1}

    # --- top sum bit ---
    circ.cx(g[m - 1], b[n - 1])                # b ^= carry_{n-1}
    if bits[n - 1]:
        flip(b[n - 1])

    # --- backward: erase carries (0 Toffoli), write sums ---
    for i in range(n - 2, k, -1):
        j = i - k
        circ.cx(g[j - 1], g[j])                # back to the pure AND
        if bits[i]:
            flip(g[j - 1])
        _unand(circ, g[j - 1], b[i], g[j])     # measurement-based erasure
        if bits[i]:
            flip(g[j - 1])
        if bits[i]:
            flip(b[i])                         # b_i ^ carry_i ^ c_i = s_i

    seed(b[k], g[0])                           # g[0] was a copy of b_k -> back to |0>
    circ.free(g[0:1])                          # the rest were consumed by measurement
    flip(b[k])                                 # s_k = NOT b_k


def sub_const_inplace(
    circ: Circuit,
    b: Register,
    c: int,
    ctrl: int | None = None,
    window: int | None = None,
) -> None:
    """In place: b -= c (mod 2^n), no constant register. Optionally controlled/truncated.

    Uses ~(~b + c) = b - c rather than adding the two's complement, so the constant that
    actually reaches the carry chain stays *sparse*: for c = 19 on 256 bits the
    complement (-19) mod 2^256 has 254 set bits, while 19 has 3. The X layers cancel when
    ctrl is |0>, so they need no control of their own.
    """
    n = len(b)
    if n == 0:
        return
    if window is not None and window < n:
        b = b[0:window]
    for q in b:
        circ.x(q)
    add_const_inplace(circ, b, c, ctrl)
    for q in b:
        circ.x(q)


# ---------------------------------------------------------------------------
# +/- p for a pseudo-Mersenne prime, via the small constant f.
# ---------------------------------------------------------------------------
def addsub_p(
    circ: Circuit,
    B: Register,
    u: int,
    f: int,
    adding: bool = False,
    ctrl: int | None = None,
    inplace: bool = True,
    add_fn: AddFn = gidney_add,
    window: int | None = None,
) -> None:
    """B +/-= p on a register of at least u+1 bits, for pseudo-Mersenne p = 2^u - f.

        B - p = (B ^ 2^u) + f            B + p = (B ^ 2^u) - f

    Flipping bit u realises both +2^u and -2^u (they are congruent mod 2^(u+1)), so the
    only arithmetic left is the *small* constant f. This is the reason `p` itself never
    has to be materialised: p has 253 set bits and its two's complement 2^u + f has
    bit_length u+1, so neither is sparse-and-narrow -- but f = 19 is both.

    Crucially it is also what makes truncation possible at all. Adding p into a
    45-bit window is meaningless (p does not fit); adding f into one is fine. With
    `window = L`:

        P(wrong) = f / 2^L        (both signs: `-p` overflows the window when the low
                                   part is within f of the ceiling, `+p` underflows it
                                   when the low part is below f -- f cases either way)

    `window` never interacts with the bit-u flip, which stays exact: L << u.
    """
    if ctrl is None:
        circ.x(B[u])
    else:
        circ.cx(ctrl, B[u])

    if not adding:
        _const_add(circ, B, f, ctrl, inplace, add_fn, window)      # -p: just += f
        return

    # +p: -= f, via ~(~B + f) = B - f. The X layers cancel when ctrl = |0>, so they
    # need no control of their own. Under a window they cover only the windowed slice.
    S = B[0:window] if (window is not None and window < len(B)) else B
    for q in S:
        circ.x(q)
    _const_add(circ, B, f, ctrl, inplace, add_fn, window)
    for q in S:
        circ.x(q)


def _const_add(
    circ: Circuit,
    B: Register,
    c: int,
    ctrl: int | None,
    inplace: bool,
    add_fn: AddFn,
    window: int | None = None,
) -> None:
    """B += c, dispatched to the chosen constant-adder strategy."""
    if inplace:
        add_const_inplace(circ, B, c, ctrl=ctrl, window=window)
        return
    S = B[0:window] if (window is not None and window < len(B)) else B
    if ctrl is None:
        add_const(circ, S, c, add_fn)
    else:
        add_const_controlled(circ, ctrl, S, c, add_fn)


# ---------------------------------------------------------------------------
# Registered benchmark versions.
# ---------------------------------------------------------------------------
_N = U


def _const_case(c: int, add_fn: AddFn, width: int = _N) -> Case:
    circ = Circuit()
    b = circ.input("b", width)
    add_const(circ, b, c, add_fn)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"b": (v["b"] + c) % (1 << width)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"b": rng.getrandbits(width)}

    return Case(circ, ref, gen)


@circuit_version("field.addconst_p.v1")
def _build_addconst_p_v1() -> Case:
    """
    Worst case for the baseline: p has 253 set bits, so 253 X loads.
    Load p into a register before add

    """
    return _const_case(PRIME, gidney_add)


@circuit_version("field.addconst_19.v1")
def _build_addconst_19_v1() -> Case:
    """
    The constant the pseudo-Mersenne trick actually needs (5 bits, 3 set).
    Load 19 into a register before add
    b = 255 bits, carries = 254 ancillas -> peak 509.
    loading 19 into a register costs 5 ancillas -> total peak 514.
    """
    return _const_case(19, gidney_add)


def _const_case_inplace(c: int, width: int = _N) -> Case:
    circ = Circuit()
    b = circ.input("b", width)
    add_const_inplace(circ, b, c)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"b": (v["b"] + c) % (1 << width)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"b": rng.getrandbits(width)}

    return Case(circ, ref, gen)


@circuit_version("field.addconst_p.v2")
def _build_addconst_p_v2() -> Case:
    """
    In place: b += p (mod 2^255) with no constant register. ~253 Toffoli.
    b = 255 bits, carries = 254 ancillas -> peak 509.
    """
    return _const_case_inplace(PRIME)


def _p_case(adding: bool, window: int | None) -> Case:
    """B -/+= p on a 256-bit register, via the (B ^ 2^U) -+ F decomposition."""
    circ = Circuit()
    b = circ.input("b", _N + 1)
    addsub_p(circ, b, _N, F, adding=adding, window=window)
    mod = 1 << (_N + 1)
    delta = PRIME if adding else -PRIME

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"b": (v["b"] + delta) % mod}          # the EXACT answer, deliberately

    def gen(rng: random.Random) -> dict[str, int]:
        return {"b": rng.getrandbits(_N + 1)}

    meta = {"fail_analytic": carry_fail_prob(window, F)} if window else {}
    return Case(circ, ref, gen, meta=meta)


@circuit_version("field.subconst_p_decomp.v1")
def _build_subconst_p_decomp_v1() -> Case:
    """
    EXACT. b -= p as (b ^ 2^255) + 19 -- the decomposition, full-width carry chain.
    Registered so the truncated version below has an exact baseline to be measured
    against, and to show what the decomposition alone buys over `addconst_p.v2`.
    """
    return _p_case(adding=False, window=None)


@circuit_version("field.subconst_p_trunc.v1")
def _build_subconst_p_trunc_v1() -> Case:
    """
    APPROXIMATE. b -= p as (b ^ 2^255) + 19, with the +19 confined to the low
    CARRY_WINDOW bits.

    This is the version that could not exist without the decomposition: truncating
    `b -= p` directly is meaningless because p does not fit in any small window, but
    once p is rewritten as a bit-flip plus 19, the 19 does.

        P(wrong) = F / 2^CARRY_WINDOW = 19 / 2^45 ~ 2^-40.8
    """
    return _p_case(adding=False, window=CARRY_WINDOW)


@circuit_version("field.addconst_p_trunc.v1")
def _build_addconst_p_trunc_v1() -> Case:
    """
    APPROXIMATE. b += p as (b ^ 2^255) - 19, with the -19 confined to the low
    CARRY_WINDOW bits (via the complement trick, so the constant stays sparse).

        P(wrong) = F / 2^CARRY_WINDOW ~ 2^-40.8
    """
    return _p_case(adding=True, window=CARRY_WINDOW)


@circuit_version("field.addconst_19_trunc.v1")
def _build_addconst_19_trunc_v1() -> Case:
    """
    APPROXIMATE. b += 19 into the low CARRY_WINDOW bits only; carries past bit
    CARRY_WINDOW-1 are dropped.

    P(wrong) = 19 / 2^CARRY_WINDOW for uniform b (~2^-40.8 at the 25519 profile).

    The literal 19 stays fixed under field selection; only the register width and
    carry window follow the profile. This version is not addition of the selected F.
    """
    circ = Circuit()
    b = circ.input("b", _N)
    add_const_inplace(circ, b, 19, window=CARRY_WINDOW)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"b": (v["b"] + 19) % (1 << _N)}         # the EXACT answer, deliberately

    def gen(rng: random.Random) -> dict[str, int]:
        return {"b": rng.getrandbits(_N)}

    return Case(circ, ref, gen, meta={"fail_analytic": carry_fail_prob(CARRY_WINDOW, 19)})


@circuit_version("field.addconst_19.v2")
def _build_addconst_19_v2() -> Case:
    """
    In place: b += 19 (mod 2^255) with no constant register. ~253 Toffoli.
    b = 255 bits, carries = 254 ancillas -> peak 509 (same as p: the carry chain
    must still reach the top, so a small constant saves Cliffords/depth, not space).
    """
    return _const_case_inplace(19)
