"""Blocked Gidney addition: trade Toffoli for peak qubits at a chosen chunk width.

`gidney_add` holds all n-1 carries at once, so its peak is ~3n. Splitting the register
into blocks of `chunk` bits makes only ONE block's carries live at a time -- peak drops
to ~2n + chunk -- at the cost of one boundary carry per block that has to be created and
then destroyed again.

Destroying it is the whole problem, and the solution is the reason this file exists.
After block j has run, its operand slice A_j survives but B_j is gone: b holds the sum
S_j. The carry out is nonetheless *recoverable* from what is left, because

    A_j + B_j + c_in = c_out * 2^w + S_j,     0 <= B_j <= 2^w - 1

forces c_out = 1 exactly when S_j < A_j + c_in. Note the carry IN on the right. Without
it the predicate would be ambiguous on S_j == A_j (B_j = 0 with no carry out, or
B_j = 2^w - 1 with one); with it the two cases separate cleanly and the identity is
EXACT. That is what lets the exact variant stay exact -- it is not a comparison that has
been truncated to something affordable, it is an equality.

Two variants, and the difference is only in how the boundary carry dies:

  exact   `_carry_out_block` XORs the predicate straight back in, returning the qubit to
          |0>. ~w Toffoli per boundary; no measurement, no phase exposure.

  vented  measure the carry, then apply the predicate as a PHASE on the top `cmp_bits`
          bits of the block (`cmp_lt_phase`). ~k Toffoli per boundary, and a wrong
          predicate costs a phase rather than a dirty ancilla. The vent drops the carry-in
          term, so it is wrong on a tie even at k = w -- see
          `params.chunked_add_boundary_fail_prob`.

WHEN THIS PAYS. Volume is peak x Toffoli, so chunking wins only when the adder's carries
actually dominate the peak and the extra boundary Toffoli are small next to the circuit's
total. That is the profile of a circuit doing a FEW wide adds at a high peak. Our
accumulate schedule is the opposite -- 256 adds, each already cheap -- so the prediction
is that it loses here and would win inside something like ecdsa.fail's squaring. The
prediction is tested, not assumed; see docs/decisions/08-chunked-addition.md.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..params import CMP_BITS, U, chunked_add_boundary_fail_prob
from .add import (
    _and,
    _carry_out_block,
    _gidney_add_block,
    _gidney_add_block_ctrl_threaded,
    _unand,
    gidney_add_ctrl,
    gidney_add_ctrl_threaded,
)
from .compare import cmp_lt_phase


def gidney_add_chunked(
    circ: Circuit,
    a: Register,
    b: Register,
    chunk: int = 64,
    cmp_bits: int | None = None,
) -> None:
    """In place: b += a (mod 2^n), a preserved -- in blocks of `chunk` bits.

    Same contract as `gidney_add` (including a short `a`, zero-extended for free), but
    peak ~2n + chunk instead of ~3n. `cmp_bits=None` selects the exact boundary
    uncompute; an int selects the measurement vent truncated to that many bits, which is
    cheaper and approximate.

    Cost, for m = ceil(n/chunk) blocks:

        Toffoli  ~ n + (m-1) * (1 + boundary),   boundary = chunk (exact) or cmp_bits
        qubits   ~ 2n + chunk + m

    The `m` is the boundary carries themselves: c_in of block j is needed to kill c_out
    of block j, so they are all held to the end and then unwound top-down. m is 4 at
    n=255, chunk=64, which is why holding them is not worth being clever about.
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    assert chunk >= 1, "chunk width must be positive"
    if n == 0 or na == 0:
        return
    if chunk >= n:
        _gidney_add_block(circ, a, b)            # one block == plain gidney_add
        return

    bounds = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    live: list[tuple[int, int, int | None, int]] = []   # (lo, hi, c_in, c_out)
    cin: int | None = None

    for lo, hi in bounds:
        last = hi == n
        cout = None if last else circ.alloc(1)[0]
        _gidney_add_block(circ, a[lo:hi], b[lo:hi], cin=cin, cout=cout)
        if cout is not None:
            live.append((lo, hi, cin, cout))
            cin = cout

    # Unwind top-down: killing c_out of block j reads c_in of block j, which is c_out of
    # block j-1 and so must still be alive. Reversed order is the only order that works.
    for lo, hi, c_in, c_out in reversed(live):
        _kill_boundary_carry(circ, a[lo:hi], b[lo:hi], c_in, c_out, cmp_bits)


def _kill_boundary_carry(
    circ: Circuit,
    A: Register,
    S: Register,
    cin: int | None,
    cout: int,
    cmp_bits: int | None,
) -> None:
    """Return `cout` to |0>, given that it holds [S < A + cin]. See the module docstring.

    S is this block's slice of `b`, which now holds the SUM; A is its slice of `a`, still
    the original operand. Both are restored.
    """
    if cmp_bits is None:
        # Exact: borrow-out of S - A - cin is the carry-out of ~S + A + cin.
        for q in S:
            circ.x(q)
        _carry_out_block(circ, A, S, cin, cout)  # XORs the same bit back in -> |0>
        for q in S:
            circ.x(q)
        circ.free(Register([cout]))
        return
    # Vented: the measurement erases `cout` with certainty and the predicate only drives
    # a phase, so a wrong answer is a phase error rather than a dirty ancilla. The
    # carry-in is dropped here -- it can only matter on an exact tie.
    m = circ.measure(cout)
    cmp_lt_phase(circ, A, S, m, k=cmp_bits)


def gidney_add_ctrl_chunked(
    circ: Circuit,
    ctrl: int,
    a: Register,
    b: Register,
    chunk: int = 64,
    cmp_bits: int | None = None,
) -> None:
    """In place: b += a (mod 2^n) if ctrl -- in blocks of `chunk` bits.

    The middle of the controlled-adder Pareto, and the dialog replay is why it exists.
    `gidney_add_ctrl` masks the whole operand and then holds every carry: 509 ancillas at
    255 bits, on a circuit whose tape and operands already fill `PEAK_BUDGET` exactly.
    `cdkm_add_ctrl` holds one ancilla and pays 4n Toffoli. This holds one block's worth of
    each -- `2*chunk + blocks` -- for `n + n + boundaries` Toffoli.

    The mask is per block, which is what makes that work: `t_i = ctrl & a_i` for this
    block only, added by `_gidney_add_block`, then unmasked by measurement at zero
    Toffoli. Total masking Toffoli is `na` however the register is cut, so chunking costs
    nothing on the mask and everything on the boundaries.

    **The boundary predicate needs the control, and it can read the UNMASKED operand.**
    `c_out = [S_j < mask_j + c_in]`; with ctrl = 1 the mask is `a`, and with ctrl = 0 both
    the sum and every carry are untouched, so gating the predicate on `ctrl` is exactly
    right and lets the mask die before the boundary is killed. Ungated it would be a coin
    flip on every ctrl = 0 run -- a dirty ancilla in the exact variant, a phase in the
    vented one. See `compare.cmp_lt_phase`.

    `cmp_bits=None` re-masks the block and kills the carry exactly (2*chunk Toffoli per
    boundary, no new failure term); an int vents it as a phase against the unmasked
    operand (`min(cmp_bits, chunk)` + 1 Toffoli, and
    `params.chunked_add_boundary_fail_prob` per addition). Note that the vent is never
    exact however wide `cmp_bits` is -- it drops the `+ c_in` -- so in a loop of 399
    additions `cmp_bits` has to be read as `chunk`, not as `CMP_BITS`.
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    assert chunk >= 1, "chunk width must be positive"
    if n == 0 or na == 0:
        return
    if chunk >= n:
        gidney_add_ctrl(circ, ctrl, a, b)
        return

    bounds = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    live: list[tuple[int, int, int | None, int]] = []
    cin: int | None = None

    for lo, hi in bounds:
        cout = None if hi == n else circ.alloc(1)[0]
        A = a[lo:min(hi, na)] if lo < na else Register([])
        if len(A):
            t = circ.alloc(len(A))
            for i in range(len(A)):
                _and(circ, ctrl, A[i], t[i])         # this block's mask only
            _gidney_add_block(circ, t, b[lo:hi], cin=cin, cout=cout)
            for i in range(len(A)):
                _unand(circ, ctrl, A[i], t[i])       # free; consumes t
        else:
            # `a` is spent: the block still has to propagate the carry in.
            _gidney_add_block(circ, A, b[lo:hi], cin=cin, cout=cout)
        if cout is not None:
            live.append((lo, hi, cin, cout))
            cin = cout

    for lo, hi, c_in, c_out in reversed(live):
        A = a[lo:min(hi, na)] if lo < na else Register([])
        _kill_boundary_carry_ctrl(circ, ctrl, A, b[lo:hi], c_in, c_out, cmp_bits)


def _kill_boundary_carry_ctrl(
    circ: Circuit,
    ctrl: int,
    A: Register,
    S: Register,
    cin: int | None,
    cout: int,
    cmp_bits: int | None,
) -> None:
    """`_kill_boundary_carry` for a masked block. `A` is the ORIGINAL operand slice."""
    if cmp_bits is None:
        # Exact, so the predicate has to see what was actually added: re-mask.
        if len(A):
            t = circ.alloc(len(A))
            for i in range(len(A)):
                _and(circ, ctrl, A[i], t[i])
            for q in S:
                circ.x(q)
            _carry_out_block(circ, t, S, cin, cout)
            for q in S:
                circ.x(q)
            for i in range(len(A)):
                _unand(circ, ctrl, A[i], t[i])
        else:
            for q in S:
                circ.x(q)
            _carry_out_block(circ, A, S, cin, cout)
            for q in S:
                circ.x(q)
        circ.free(Register([cout]))
        return
    # Vented: the mask is already gone, so gate the phase on ctrl instead of rebuilding it.
    m = circ.measure(cout)
    cmp_lt_phase(circ, A, S, m, k=cmp_bits, ctrl=ctrl)


def gidney_add_ctrl_chunked_threaded(
    circ: Circuit,
    ctrl: int,
    a: Register,
    b: Register,
    chunk: int = 64,
    cmp_bits: int | None = None,
) -> None:
    """In place: b += a (mod 2^n) if ctrl -- threaded control, in blocks of `chunk`.

    `gidney_add_ctrl_chunked` with the per-block mask removed, which is the whole point:
    that one holds `2*chunk + blocks` ancillas, this holds **`chunk + blocks + 1`** -- a
    saving of `chunk - 1`, the one qubit back being the scratch the gated carry-out needs
    (see `_gidney_add_block_ctrl_threaded`). Everything else -- the block cut, the boundary
    carries, the vent -- is identical, and so is the failure model
    (`params.chunked_add_boundary_fail_prob`).

    A short `a` is zero-extended but costs a Toffoli per bit above `na`, exactly as in
    `gidney_add_ctrl_threaded`; the masked version is still cheaper for a genuinely narrow
    operand.

    **This one is a trade, not a dominance, and that is the difference from the unchunked
    pair.** `gidney_add_ctrl_threaded` is strictly better than `gidney_add_ctrl` -- same
    `2n - 1` Toffoli, `n` fewer qubits. Chunking hands part of that back at the boundaries:
    the carry out has to be gated inside the block (2 Toffoli, see
    `_gidney_add_block_ctrl_threaded`) where the mask got it for free from a zeroed chain,
    and only one of those two is offset by the masking AND that is no longer paid. Measured
    at n=255 (chunk 38 and 32), 128/32, 64/16 and 12/6, the gap is exactly

        toffoli(threaded) - toffoli(masked) = blocks - 1 + (n - na)

    -- 6 extra Toffoli for 37 qubits at chunk 38, which is why the inversion takes it. The
    `n - na` term is the zero-extension above a short `a`, which the mask does not pay at
    all; for a genuinely narrow operand the masked version is the cheaper circuit on both
    counts, and this one is worth it only from about `na = n - 1` up.
    """
    n = len(b)
    na = len(a)
    assert 0 <= na <= n, "len(a) must not exceed len(b)"
    assert chunk >= 1, "chunk width must be positive"
    if n == 0:
        return
    if chunk >= n:
        gidney_add_ctrl_threaded(circ, ctrl, a, b)
        return

    bounds = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    live: list[tuple[int, int, int | None, int]] = []
    cin: int | None = None

    for lo, hi in bounds:
        cout = None if hi == n else circ.alloc(1)[0]
        A = a[lo:min(hi, na)] if lo < na else Register([])
        _gidney_add_block_ctrl_threaded(circ, ctrl, A, b[lo:hi], cin=cin, cout=cout)
        if cout is not None:
            live.append((lo, hi, cin, cout))
            cin = cout

    for lo, hi, c_in, c_out in reversed(live):
        A = a[lo:min(hi, na)] if lo < na else Register([])
        _kill_boundary_carry_threaded(circ, ctrl, A, b[lo:hi], c_in, c_out, cmp_bits)


def _kill_boundary_carry_threaded(
    circ: Circuit,
    ctrl: int,
    A: Register,
    S: Register,
    cin: int | None,
    cout: int,
    cmp_bits: int | None,
) -> None:
    """`_kill_boundary_carry_ctrl` with no mask to rebuild.

    `cout` holds `ctrl & [A + S_orig + cin overflowed]`, and `S` now holds the *gated* sum
    -- so on a ctrl = 0 run `cout` is already |0> and `S` is untouched. Both branches below
    are therefore gated on `ctrl` rather than on a reconstructed mask.
    """
    if cmp_bits is None:
        # Exact: the same borrow-out identity as `_kill_boundary_carry`, but the bit the
        # block wrote was `ctrl & c`, so the erase has to be gated too. `_carry_out_block`
        # has no control, so land it on a scratch, gate that into `cout`, then run it
        # again to clear the scratch -- 2 carry blocks and 1 Toffoli per boundary. This is
        # the price of not having a zeroable mask, and it is why the vented branch is the
        # one the inversion actually uses.
        s = circ.alloc(1)[0]
        for q in S:
            circ.x(q)
        _carry_out_block(circ, A, S, cin, s)
        circ.ccx(ctrl, s, cout)                  # cout ^= ctrl & c  ->  |0>
        _carry_out_block(circ, A, S, cin, s)     # s  ->  |0>
        for q in S:
            circ.x(q)
        circ.free(Register([s, cout]))
        return
    # Vented: same phase trick as the masked version, against the same unmasked operand.
    m = circ.measure(cout)
    cmp_lt_phase(circ, A, S, m, k=cmp_bits, ctrl=ctrl)


def gidney_add_chunked_ref(a: int, b: int, n: int) -> int:
    """Exact semantics: the same mod-2^n sum any adder computes."""
    return (a + b) % (1 << n)


def boundary_carries(a: int, b: int, n: int, chunk: int) -> list[int]:
    """The carry crossing each block boundary -- what the circuit's `live` qubits hold.

    The carry into bit `hi` is decided entirely by the operands BELOW it:
    (a_low + b_low) >> hi. Exposed so tests can pin the intermediate state rather than
    only the final sum, which is what makes a boundary bug distinguishable from a
    ladder bug.
    """
    out = []
    for hi in range(chunk, n, chunk):
        mask = (1 << hi) - 1
        out.append(((a & mask) + (b & mask)) >> hi)
    return out


# ---------------------------------------------------------------------------
# Registered benchmark versions (real field width).
# ---------------------------------------------------------------------------
_N = U


def _chunked_case(chunk: int, cmp_bits: int | None) -> Case:
    circ = Circuit()
    a = circ.input("a", _N)
    b = circ.input("b", _N)
    gidney_add_chunked(circ, a, b, chunk=chunk, cmp_bits=cmp_bits)

    def ref(v: dict[str, int]) -> dict[str, int]:
        return {"a": v["a"], "b": gidney_add_chunked_ref(v["a"], v["b"], _N)}

    def gen(rng: random.Random) -> dict[str, int]:
        return {"a": rng.getrandbits(_N), "b": rng.getrandbits(_N)}

    # `{}`, not None: the harness reads `meta.get(...)` unconditionally, so an exact
    # circuit declares an empty dict rather than no dict.
    meta: dict = {}
    if cmp_bits is not None:
        meta = {"fail_analytic": chunked_add_boundary_fail_prob(_N, chunk, cmp_bits)}
    return Case(circ, ref, gen, meta=meta)


@circuit_version("field.add_gidney_chunked_32.v1")
def _build_add_gidney_chunked_32_v1() -> Case:
    """EXACT. 8 blocks of 32: the aggressive end of the trade -- lowest peak, most
    boundaries, so the most extra Toffoli."""
    return _chunked_case(32, None)


@circuit_version("field.add_gidney_chunked_64.v1")
def _build_add_gidney_chunked_64_v1() -> Case:
    """EXACT. 4 blocks of 64."""
    return _chunked_case(64, None)


@circuit_version("field.add_gidney_chunked_128.v1")
def _build_add_gidney_chunked_128_v1() -> Case:
    """EXACT. 2 blocks of 128: one boundary, the cheapest chunking that chunks at all."""
    return _chunked_case(128, None)


@circuit_version("field.add_gidney_chunked_64_trunc.v1")
def _build_add_gidney_chunked_64_trunc_v1() -> Case:
    """
    APPROXIMATE. 4 blocks of 64, boundary carries vented on a top-CMP_BITS compare.

    Cheaper than the exact boundary by (chunk - cmp_bits) Toffoli each, and wrong with
    probability ~3 * 2^-(CMP_BITS+1) per addition -- which is the SAME order as the
    truncations already inside `modadd_lazy`, so it costs nothing in an approximate
    schedule and is unusable in an exact one.
    """
    return _chunked_case(64, CMP_BITS)
