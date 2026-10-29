"""`|x, y> -> |x, y> + P` for a CLASSICAL P, in place, optionally controlled. Wei25519.

This is Layer 3's experimental baseline: a candidate primitive for a later scalar schedule.
Everything here is one registered circuit measured on the live state,
for the reason doc 10 section 2 gives -- a peak summed over regions is not a peak.

THE SHAPE. Affine addition needs `lambda = (y - y_P)/(x - x_P)`, and `lambda` occupies a
register that has to be emptied again before the step can repeat. Emptying it is not a
bookkeeping detail, it is the second half of the cost: a quantum register cannot be
discarded, and `lambda` is not recoverable from `(x_3, y_3)` for free. So the step pays
TWO expensive in-place field operations, one to create `lambda` and one to destroy it.

What destroys it is the geometry. The line through `Q` and `P` is also the line through
`R = Q + P` and `-P`, so with `u_3 = x_3 - x_P`:

    lambda * u_3  =  -(y_3 + y_P)

The register a division turned into `lambda` is turned back into a y-coordinate by an
in-place multiply. Both are `qsm.inversion.inplace`, the same circuit run in two
directions, and they measure 0.07% apart for this tuning. This prices this construction;
it is not a lower bound on other reversible multiplication or coordinate schedules.

    1. u := x - x_P                        modsub_direct[x_P]
    2. v := y - y_P                        modsub_direct[y_P]
    3. (u, v) -> (u, lambda)               moddiv_inplace            EXPENSIVE
    4. if ctrl: u <- u - lambda^2          modsqrsub_inplace(ctrl)
    5. if ctrl: u <- -u - 3 x_P            CNOT layer + modsub_direct[K1]
    6. (u, lambda) -> (u, lambda u)        modmul_inplace            EXPENSIVE
    7. x := u + x_P                        modsub_direct[q - x_P]
    8. y := lambda u + y_P                 modsub_direct[q - y_P]
    9. if ctrl: y <- -y                    CNOT layer + modsub_direct[F-1]

THE CONTROL IS ALMOST FREE, and that is the point of writing it this way. Steps 3 and 6
are UNCONTROLLED: their ideal arithmetic maps are inverses, so at `ctrl = 0` the ideal
step 6 restores `v`. The tuned implementations still carry both regions' approximation
budgets; independent measurement-vented phases need not cancel. Only steps 4, 5 and 9
carry the control, and only step 4 costs more than a layer of
CNOTs. The control therefore touches ~4% of the circuit rather than all of it.

A CONTROLLED MODULAR CONSTANT SUBTRACT COSTS THE SAME AS AN UNCONTROLLED ONE. There is no
controlled constant adder anywhere below, and none is needed. `modsub_direct(a, b)` takes
its subtrahend in a register; loading a classical `c` there is an X layer, and loading it
with CNOTs from `ctrl` instead gives `c` or `0` -- and subtracting `0` is the exact
identity, flag and phase included (the borrow never fires, so the vented erasure kicks no
phase). The control rides on gates that were free anyway. See `sub_const_mod`.

WHY NOT `sub_const_inplace`. It is `b -= c (mod 2^n)`, and a borrow off the bottom loses
`2^U`, which is `F` mod q -- a NON-congruent result, not a lazy one, at roughly 50% for a
generic constant. The modular version is mandatory here, and `modsub_direct` supplies it
at zero added failure rate: its failure set is exactly `a - b > q`, and with `a` classical
and `a < q` that set is EMPTY.

WHY THE NEGATION FOLDS INTO STEP 5. `modneg` has no controlled form, and adding one would
need a controlled adder. It is not needed: complementing a register is `~w = 2^U - 1 - w`,
which is `F - 1 - w` mod q, so complement-then-subtract-`K` computes `F - 1 - w - K`, and
choosing `K = F - 1 + 3 x_P` makes that `-w - 3 x_P` exactly. Under a control the
complement is a CNOT layer. Step 9 is the same trick with `K = F - 1`.

OUTPUT IS LAZY, congruent to the group law but not canonically reduced. Steps 5, 7, 8 and
9 each end in a `modsub_direct` whose `b` may already be lazy, and `modsub_direct` only
promises canonical OUT for canonical IN. The structural case is `y_3 = 0`, which comes back
as `q`: step 9 complements, and the complement of zero is `2^U - 1`, so subtracting `F - 1`
lands on `q` rather than on `0`. That is exactly the `[1, q]` output range `modneg`
documents, reached for the same reason, and `y_3 = 0` is a point of order two -- it cannot
occur in the prime-order subgroup Shor runs in. Every other route to a lazy output needs
the value to land in the top `F` of the register, `F/2^U = 2^-250.8`.

So the step is closed under its own output: a lazy coordinate fed back into step 1 is fine
(`modsub_direct`'s requirement `a - b <= q` holds for any `b` when `a` is classical and
below `q`), and reaches `moddiv_inplace` lazy only at that same 2^-250.8. The reference
models the laziness exactly; only `curve.add` differs, and only in the representative.

PRECONDITIONS. Three, and only two of them are exceptional cases of the addition law:

  * `x != x_P` -- else `u = 0` and `moddiv_inplace` divides by zero. This is `A = +-P`.
  * `x_3 != x_P` -- else `u_3 = 0` and `modmul_inplace` sees a zero operand. This is
    `A = O` or `A = -2P`.
  * `y != y_P` -- else `lambda = 0`, and BOTH expensive regions refuse a zero operand:
    the division gets a zero numerator and the multiply a zero multiplicand.

The third one is the circuit's own, not the group law's. `A` and `P` sharing a
y-coordinate is a horizontal chord, a perfectly ordinary addition that returns a perfectly
ordinary point -- it is only exceptional HERE, because `lambda = 0` is a value this
register cannot hold. It is also the one that is not a rate: the points sharing `y_P` are
the roots of `x^3 + a x + b = y_P^2`, so there are AT MOST TWO of them besides `P` itself,
named and fixed once `P` is. A uniform accumulator meets any of the three with probability
~2^-254, but a scalar multiplication does not sample uniformly, so what discharges these
is the schedule and not the arithmetic. That is Layer 4's problem and doc 14's.

All three are conditions on the RESIDUE, not on the representative, and `u_3` is where that
bites: it reaches the multiply as a `modsub_direct` of a complement, so it is lazy whenever
the pre-complement value is below `F`, and `u_3 = 0` can present as `q`. At full width the
two conditions are far apart (laziness is 2^-250.8); at the `f = 1` widths the tests sweep
they COINCIDE, which is what exposed it. The reference tests `% p` for that reason.

These are algebraic preconditions, not sufficient guarantees for the tuned arithmetic.
No runtime quantum exceptional-case handling is implemented. `DirtyAncillaError` is a
simulator diagnostic, not a physical branch that can abort the computation safely.

The benchmark below draws independent uniform field coordinates. Its reported error model
is conditional on distributional assumptions about the resulting arithmetic trajectories;
it is not a worst-case guarantee for all on-curve points. In particular, a quantum numerator
need not be uniform. Valid prime-subgroup points with y = GY + 20 and y = GY + 44 expose
structured-numerator faults at full width. A scalar/window schedule must establish its own
input distribution and error analysis before using this experimental variant. The separate
canonical conservative variant avoids these arithmetic approximations on its stated domain.
"""

from __future__ import annotations

import random

from ..bench.registry import Case, circuit_version
from ..core.circuit import Circuit, Register
from ..field.add import gidney_add
from ..field.modarith import AddFn, modsub_direct, modsub_ref
from ..inversion.config import InversionConfig
from ..inversion.inplace import moddiv_inplace, modmul_inplace
from ..mul.accumulate import (
    TRISUB_TRUNC,
    TRISUB_TRUNC_R,
    _flat_diag,
    modsqrsub_inplace,
    modsqrsub_inplace_ref,
)
from ..params import (
    CARRY_WINDOW,
    CMP_BITS,
    ERROR_BUDGET,
    F,
    PRIME,
    SQUARE_RIPPLE,
    U,
    cmp_fail_prob,
    controlled_carry_fail_prob,
)
from . import active as curve

__all__ = [
    "sub_const_mod",
    "add_const_affine_inplace",
    "add_const_affine_inplace_ref",
    "tuned_point_kwargs",
]


def sub_const_mod(
    circ: Circuit,
    b: Register,
    c: int,
    p: int = PRIME,
    ctrl: int | None = None,
    add_fn: AddFn = gidney_add,
    carry_window: int | None = None,
    cmp_bits: int | None = None,
) -> None:
    """b <- (b - c) mod p for a CLASSICAL `c < p`, optionally gated on `ctrl`. Canonical.

    The constant goes into a scratch register and `modsub_direct` does the work. Loading
    and unloading are a layer of X gates on `c`'s set bits -- self-inverse, no Toffoli, no
    measurement -- so the whole cost is one `modsub_direct` and the scratch is transient.

    Under a control the X layer becomes a CNOT layer from `ctrl`, so the scratch holds `c`
    or `0`, and `b - 0` is the EXACT identity: the borrow out of bit U-1 cannot fire, so
    the truncated `-F` is not applied and the vented erasure kicks no phase. A controlled
    modular constant subtract therefore costs exactly what the uncontrolled one costs, and
    Layer 3 never needs a controlled adder.

    ZERO ADDED FAILURE RATE from the precondition side. `modsub_direct`'s failure set is
    exactly `{a - b > p}`, and here `a` is `c` or `0`, both below `p`, so `a - b <= c < p`
    for every `b`. The set is empty. What remains is the version's own truncation and its
    `F/2p` erasure phase, which `_const_sub_fail` prices.

    `circ.free` asserts the scratch is back to |0>, so a load/unload mismatch fails loudly
    rather than dropping a live qubit -- the same safety argument `inplace._clear_constant`
    makes.
    """
    assert 0 <= c < p, "constant must be canonically reduced"
    n = len(b)
    r = circ.alloc(n)
    bits = [j for j in range(n) if (c >> j) & 1]

    def load() -> None:
        for j in bits:
            if ctrl is None:
                circ.x(r[j])
            else:
                circ.cx(ctrl, r[j])

    load()
    modsub_direct(circ, r, b, p, add_fn, carry_window, cmp_bits)
    load()                                   # same layer, self-inverse
    circ.free(r)


def add_const_affine_inplace(
    circ: Circuit,
    x: Register,
    y: Register,
    xp: int,
    yp: int,
    ctrl: int | None = None,
    p: int = PRIME,
    const_kw: dict | None = None,
    sqr_kw: dict | None = None,
    inv_kw: dict | None = None,
) -> tuple[Register, Register]:
    """|x, y> -> |x, y> + [ctrl] (xp, yp) on Wei25519, in place. Returns both, RELABELLED.

    `xp, yp` are classical. `ctrl`, if given, is a single qubit; `None` builds the
    uncontrolled step, which is the baseline the controlled one is priced against.

    The three keyword dicts are kept apart because the three regions do not take the same
    knobs and because mixing them is how a forward and its reverse end up built with
    different windows -- `inplace.modmul_inplace` makes the same split for the same reason.
    `inv_kw` reaches BOTH the division and the multiply. Their arithmetic schedules must
    match; this does not cancel independently sampled approximation or vented-phase errors.
    """
    n = len(x)
    assert len(y) == n, "coordinate widths must match"
    const_kw = dict(const_kw or {})
    sqr_kw = dict(sqr_kw or {})
    inv_kw = dict(inv_kw or {})

    xp %= p
    yp %= p
    #: complement-then-subtract-K computes `F - 1 - w - K`; these make that `-w - 3 x_P`
    #: and `-w` respectively. See the module docstring.
    f = (1 << n) - p
    k_fold = (f - 1 + 3 * xp) % p
    k_neg = (f - 1) % p

    sub_const_mod(circ, x, xp, p, **const_kw)                             # 1. u
    sub_const_mod(circ, y, yp, p, **const_kw)                             # 2. v

    x, y = moddiv_inplace(circ, x, y, p, **inv_kw)                        # 3. lambda

    if ctrl is not None:
        x = modsqrsub_inplace(circ, y, x, p, ctrl=ctrl, **sqr_kw)         # 4. u - lambda^2
        for q in x:
            circ.cx(ctrl, q)                                              # 5a. complement
    else:
        x = modsqrsub_inplace(circ, y, x, p, **sqr_kw)
        for q in x:
            circ.x(q)
    sub_const_mod(circ, x, k_fold, p, ctrl=ctrl, **const_kw)              # 5b. u_3

    x, y = modmul_inplace(circ, x, y, p, **inv_kw)                        # 6. lambda u_3

    sub_const_mod(circ, x, (p - xp) % p, p, **const_kw)                   # 7. x_3
    sub_const_mod(circ, y, (p - yp) % p, p, **const_kw)                   # 8. -y_3

    if ctrl is not None:
        for q in y:
            circ.cx(ctrl, q)                                              # 9a. complement
    else:
        for q in y:
            circ.x(q)
    sub_const_mod(circ, y, k_neg, p, ctrl=ctrl, **const_kw)               # 9b. y_3
    return x, y


# --- exact semantics -------------------------------------------------------
def add_const_affine_inplace_ref(
    x: int,
    y: int,
    xp: int,
    yp: int,
    ctrl: int = 1,
    p: int = PRIME,
    u: int = U,
    ripple: int | None = None,
    trace: list | None = None,
) -> tuple[int, int]:
    """Exact semantics of `add_const_affine_inplace` -- the SCHEDULE, not the group law.

    Step 4 is lazily reduced at every row, so which representative in [0, 2^u) it lands on
    is a fact about the schedule and not about the residue; a reference that returned
    `curve.add(...)` would silently absolve that. `test_point_affine.py` asserts the two
    agree, which is the interesting claim rather than something to assume here.

    `trace` is `modsqrsub_inplace_ref`'s and means the same thing: one entry per
    NON-congruent scaling event. Everything else on this path is exact.
    """
    xp %= p
    yp %= p
    comp = (1 << u) - 1
    f = (1 << u) - p
    k_fold = (f - 1 + 3 * xp) % p

    # The preconditions are on the RESIDUE, not on the representative. `w` below is the
    # only one where that distinction bites: it is a `modsub_ref` of the complement, which
    # is lazy whenever the pre-complement value is under `f`, so `u_3 = 0` can present as
    # `p`. Testing `w != 0` would let those through -- and at f = 1 the two conditions
    # coincide exactly, so the toy widths the tests sweep are precisely where it shows.
    uu = modsub_ref(xp, x, p)                                             # 1
    vv = modsub_ref(yp, y, p)                                             # 2
    assert uu % p != 0, "exceptional: x == x_P"
    assert vv % p != 0, "exceptional: y == y_P (lambda = 0)"
    lam = (vv * pow(uu, -1, p)) % p                                       # 3
    w = modsqrsub_inplace_ref(lam, uu, p, u, ripple, ctrl, trace)         # 4
    if ctrl:
        w = comp - w                                                      # 5a
    w = modsub_ref(k_fold if ctrl else 0, w, p)                           # 5b
    assert w % p != 0, "exceptional: x_3 == x_P"
    prod = (lam * w) % p                                                  # 6
    xo = modsub_ref((p - xp) % p, w, p)                                   # 7
    yo = modsub_ref((p - yp) % p, prod, p)                                # 8
    if ctrl:
        yo = comp - yo                                                    # 9a
    yo = modsub_ref((f - 1) % p if ctrl else 0, yo, p)                    # 9b
    return xo, yo


# ---------------------------------------------------------------------------
# Registered versions (real field width).
#
# The settled tuning throughout, inherited rather than re-chosen: `inplace`'s jump-2
# packed-tape chunk-43/cmp-32 replay for the two expensive regions, the flattened diagonal
# for the square, CARRY_WINDOW/CMP_BITS for every truncated reduction and comparison, and
# the F-independent SQUARE_RIPPLE for the square's narrow integer rows.
# ---------------------------------------------------------------------------
def tuned_point_kwargs(n: int = U, p: int = PRIME,
                       config: InversionConfig | None = None, *,
                       legacy_square_ripple: bool = False) -> dict:
    """Shared fixed-point/window tuning; field windows resolve from the supplied prime.

    `legacy_square_ripple` reproduces versions recorded before 2026-10-11, whose square
    used the reduction window as its narrow-row ripple (45 for 25519, 73 for secp256k1).
    """
    config = (InversionConfig.tuned(n, p) if config is None else config).resolved(n, p)
    if config.arithmetic_family != "tuned":
        raise ValueError("tuned point arithmetic requires a tuned inversion configuration")
    window = ERROR_BUDGET + ((1 << n) - p - 1).bit_length()
    ripple = window if legacy_square_ripple else SQUARE_RIPPLE
    const = dict(add_fn=gidney_add, carry_window=window, cmp_bits=CMP_BITS)
    square = dict(const, window=window, ripple=ripple, diag_add_fn=_flat_diag)
    return dict(const_kw=const, sqr_kw=square, inv_kw=dict(config=config))


_N = U
_INV_CONFIG = InversionConfig.tuned(_N, PRIME)
_DEFAULT_KW = tuned_point_kwargs(_N, PRIME, _INV_CONFIG)
_INV_KW = _DEFAULT_KW["inv_kw"]
_CONST_KW = _DEFAULT_KW["const_kw"]
_SQR_KW = _DEFAULT_KW["sqr_kw"]
_LEGACY_SQR_KW = tuned_point_kwargs(_N, PRIME, _INV_CONFIG, legacy_square_ripple=True)["sqr_kw"]

#: The six `sub_const_mod` calls, whose PRECONDITION term is zero (see `sub_const_mod`);
#: what is left is the version's own truncation and its vented `F/2p` erasure phase.
_CONST_SUBS = 6


def _const_sub_fail() -> float:
    return (controlled_carry_fail_prob(CARRY_WINDOW, F)
            + cmp_fail_prob(CMP_BITS)
            + F / (2.0 * PRIME))


def _fail(config: InversionConfig | None = None, *,
          legacy_square_ripple: bool = False) -> float:
    """Division + multiply + square + six constant subtracts, each at its own declared rate.

    Assembled from the component analytics rather than re-derived, so a retune below moves
    this number with it. The two expensive regions differ only in which replay direction
    they take, and `params` prices those two separately for the reason `un_replay_j2_fail_
    prob` documents -- the totals agree, only the split between phase and value moves.
    """
    config = (_INV_CONFIG if config is None else config).resolved(_N, PRIME)
    div = config.fail_prob(reverse=True)
    mul = config.fail_prob(reverse=False)
    square = TRISUB_TRUNC if legacy_square_ripple else TRISUB_TRUNC_R
    return div + mul + square + _CONST_SUBS * _const_sub_fail()


def _affine_case(ctrl: bool, config: InversionConfig | None = None, *,
                 legacy_square_ripple: bool = False) -> Case:
    config = (_INV_CONFIG if config is None else config).resolved(_N, PRIME)
    sqr_kw = dict(_LEGACY_SQR_KW if legacy_square_ripple else _SQR_KW)
    circ = Circuit()
    c = circ.input("c", 1) if ctrl else None
    x = circ.input("x", _N)
    y = circ.input("y", _N)
    xo, yo = add_const_affine_inplace(
        circ, x, y, curve.GX, curve.GY, None if c is None else c[0], PRIME,
        const_kw=dict(_CONST_KW), sqr_kw=dict(sqr_kw), inv_kw=dict(config=config))
    circ.rebind("x", xo)
    circ.rebind("y", yo)

    def ref(v: dict[str, int]) -> dict[str, int]:
        xr, yr = add_const_affine_inplace_ref(
            v["x"], v["y"], curve.GX, curve.GY, v.get("c", 1), PRIME, _N,
            ripple=sqr_kw["ripple"])
        out = {"x": xr, "y": yr}
        if ctrl:
            out["c"] = v["c"]
        return out

    def gen(rng: random.Random) -> dict[str, int]:
        # This sampling law does not model the correlated operands of a window schedule.
        v = {"x": rng.randrange(1, PRIME), "y": rng.randrange(1, PRIME)}
        if ctrl:
            v["c"] = rng.randrange(2)
        return v

    return Case(circ, ref, gen, meta={
        "fail_analytic": _fail(config, legacy_square_ripple=legacy_square_ripple),
        "tolerates_phase": True, "square_ripple": sqr_kw["ripple"],
        "input_distribution": "independent uniform nonzero field coordinates",
        "arithmetic_contract": "experimental lazy affine; distribution-conditioned error model",
    })


@circuit_version("point.add_const_affine_inplace.v1")
def _build_add_const_affine_inplace_v1() -> Case:
    """(x, y) <- (x, y) + P for classical P, measured end to end. Layer 3's baseline.

    The unit both baselines publish: ecdsa.fail's point addition is 1,299,453 @ 1,151 and
    frenchie's is the `ec-point-addition-master` figure doc 09 records. The construction
    here is two `inv` composites plus one square plus six constant subtracts, so the
    arithmetic is predictable; what is NOT predictable, and is the reason this is one
    registered circuit rather than a sum, is the peak on the live state.

    Pinned to the pre-2026-10-11 square ripple (= the reduction window); see `.v2`.
    """
    return _affine_case(ctrl=False, legacy_square_ripple=True)


@circuit_version("point.add_const_affine_inplace_ctrl.v1")
def _build_add_const_affine_inplace_ctrl_v1() -> Case:
    """The same step gated on one qubit -- the shape a scalar multiplication repeats.

    The claim this version exists to measure: the control is nearly free. Steps 3 and 6
    are uncontrolled because their ideal maps are inverses. At `ctrl = 0` both tuned
    regions' error budgets still apply; the six constant subtracts absorb control into the CNOT
    layer that loads the scratch, at no Toffoli cost at all. Only `modsqrsub_inplace`'s
    `t = ctrl AND x` should show up, U Toffoli and U qubits of peak. Anything more than
    that against the version above is a defect in the schedule, not a price.
    """
    return _affine_case(ctrl=True, legacy_square_ripple=True)


@circuit_version("point.add_const_affine_inplace.v2")
def _build_add_const_affine_inplace_v2() -> Case:
    """`.v1` with the square's narrow-row ripple set to the F-independent SQUARE_RIPPLE."""
    return _affine_case(ctrl=False)


@circuit_version("point.add_const_affine_inplace_ctrl.v2")
def _build_add_const_affine_inplace_ctrl_v2() -> Case:
    """`ctrl.v1` with the square's narrow-row ripple set to SQUARE_RIPPLE."""
    return _affine_case(ctrl=True)
