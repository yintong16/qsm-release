"""The three composites, as circuits rather than as sums of three regions.

Doc 10 §2 is the record of adding three measured peaks together and getting a number the
machine would never have produced. This file exists so that the number quoted for a field
operation is a number the harness measured on one circuit with every live wire on the
books, and so that the region arithmetic is checked against it rather than trusted.

It agrees. `inv.modmul_inplace.v1` is 754,544 Toffoli at peak 1,173, which is the predicted
gate count exactly and the predicted peak exactly. Gates were never in doubt -- gates add.
The peak agreeing is the real result, and it agrees for a reason worth writing down: the
replay is the one region that already holds `d`, `a` and `b` simultaneously, so it was
always going to dominate two tape-builders that hold one operand each, and the threaded
Phase 1 (852 and 853) plus an operand (255) lands at 1,107-1,108, comfortably under it.
Had Phase 1 not been threaded it would not have -- 1,105 + 255 = 1,360 -- which is what
doc 10 caught.

The other result:

    moddiv = modmul + 522        the reverse surcharge, and nothing else

Correctness is checked exhaustively at p = 31 and p = 127, the widths where the lazy domain
closes. Both are exact on every non-zero pair and raise on every zero one, which is the
whole of the zero precondition and is checked as an exact partition rather than as a bound.

The division has a SECOND precondition that the small widths cannot see, because it is a
property of the truncation and the truncation is off at f = 1: its numerator has to be
generic. That one is pinned here at 255 bits against the fault auditor, and doc 12 is the
account. It is why there is no `modinv_inplace` and no `inv.modinv_inplace.v1`.
"""

from __future__ import annotations

import random

import pytest

from qsm import params
from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.inversion import schedule
from qsm.inversion.inplace import moddiv_inplace, modmul_inplace
from qsm.inversion.reference import to_dialog_jump as to_dialog_jump_ref
from qsm.inversion.reference import un_apply_dialog_jump_trunc_faults

#: f = 2^n - p = 1. The lazy domain closes, so the composite is exact on its whole domain
#: and the raise set is exactly the zero cases -- nothing has to be excused. At f > 1 the
#: replay itself is already inexact at these widths (see `test_inversion_unreplay_j2`), so
#: a composite test there would be measuring the small width and not the composition.
CLOSED = [31, 127]

_MUL = "inv.modmul_inplace.v1"
_DIV = "inv.moddiv_inplace.v1"


def _res(name: str):
    from qsm.bench import registry

    return count(registry.get_builder(name)().circuit)


def _mul_circuit(p: int, narrow: bool = True) -> Circuit:
    n = p.bit_length()
    it = schedule.iterations_j2(n)
    w = schedule.widths_j2(n, it) if narrow else [n] * it
    c = Circuit()
    x, y = c.input("x", n), c.input("y", n)
    x_out, prod = modmul_inplace(c, x, y, p, it, w)
    c.rebind("x", x_out)
    c.rebind("y", prod)
    return c


def _div_circuit(p: int) -> Circuit:
    n = p.bit_length()
    it = schedule.iterations_j2(n)
    c = Circuit()
    x, z = c.input("x", n), c.input("z", n)
    x_out, quot = moddiv_inplace(c, x, z, p, it, schedule.widths_j2(n, it))
    c.rebind("x", x_out)
    c.rebind("z", quot)
    return c


# --- values ----------------------------------------------------------------
@pytest.mark.parametrize("p", CLOSED)
def test_the_multiply_is_exact_on_every_nonzero_pair(p: int):
    """|x, y> -> |x, x*y>, all 900 and all 15,876 of them, with `x` given back.

    `x` surviving is not a bonus, it is the contract: the tape IS `x` in another alphabet,
    and the third region exists purely to turn it back. A composite that got the product
    right and lost `x` would have skipped the most expensive third of itself.
    """
    c = _mul_circuit(p)
    for xv in range(1, p):
        for yv in range(1, p):
            r = simulate(c, {"x": xv, "y": yv}, rng=random.Random(xv * 31 + yv))
            assert (r.outputs["x"], r.outputs["y"]) == (xv, (xv * yv) % p), \
                f"p={p} x={xv} y={yv}"


@pytest.mark.parametrize("p", CLOSED)
def test_the_divide_is_exact_on_every_nonzero_pair(p: int):
    """|x, z> -> |x, z/x>. The same circuit with its middle region reversed."""
    c = _div_circuit(p)
    for xv in range(1, p):
        for zv in range(1, p):
            r = simulate(c, {"x": xv, "z": zv}, rng=random.Random(xv * 17 + zv))
            assert (r.outputs["x"], r.outputs["z"]) == (xv, (zv * pow(xv, -1, p)) % p), \
                f"p={p} x={xv} z={zv}"


@pytest.mark.parametrize("p", CLOSED)
def test_the_divide_inverts_when_the_numerator_happens_to_be_one(p: int):
    """`z = 1` is fine HERE, and the fact that it is fine here is the trap.

    At f = 1 the danger band `low bits < F` is `low bits < 1`, i.e. the value is a multiple
    of `2^window`, and the window is off anyway at these widths. So the small primes will
    happily invert and will go on happily inverting no matter how the truncation is tuned
    at 255 bits. Kept as a test precisely so that the 255-bit result below is not read as a
    contradiction of it: the two are measuring different circuits, and only one of them has
    a truncated `-F`.
    """
    c = _div_circuit(p)
    for xv in range(1, p):
        r = simulate(c, {"x": xv, "z": 1}, rng=random.Random(xv))
        assert r.outputs["x"] == xv, f"p={p} x={xv}: the operand did not come back"
        assert (xv * r.outputs["z"]) % p == 1, f"p={p} x={xv}: got {r.outputs['z']}"


@pytest.mark.parametrize("p", CLOSED)
def test_a_zero_operand_raises_and_it_is_exactly_the_zero_operands(p: int):
    """The precondition, as a partition rather than as a bound.

    Asserting "zero operands raise" would be satisfied by a circuit that raised on
    everything, so the count is pinned from both sides: exactly `p - 1` raises for the
    multiply, exactly `p - 1` for the divide, and they are exactly the cases with a zero
    second operand. Nothing else in the domain raises and no zero case survives.

    Why it raises rather than returning garbage is in the module docstring: the multiply's
    emptied register lands on `p` when the product is non-zero and on `0` when it is not,
    and the X layer that clears the first leaves a live qubit on the second. The divide
    lands on the opposite constant for the mirror-image reason. The point layer has to
    handle the exceptional cases of the addition law before the field op, not after.
    """
    for build, key in ((_mul_circuit(p), "y"), (_div_circuit(p), "z")):
        raised_zero = raised_nonzero = 0
        for xv in range(1, p):
            for v in range(p):
                try:
                    simulate(build, {"x": xv, key: v}, rng=random.Random(xv * 7 + v))
                except DirtyAncillaError:
                    if v:
                        raised_nonzero += 1
                    else:
                        raised_zero += 1
        assert raised_zero == p - 1, f"p={p} {key}: {raised_zero} of {p - 1} zero cases"
        assert raised_nonzero == 0, f"p={p} {key}: {raised_nonzero} non-zero cases raised"


@pytest.mark.parametrize("p", CLOSED)
def test_x_equal_zero_raises_too(p: int):
    """The other half of the precondition. 0 has no inverse and the tape for it is junk."""
    c = _mul_circuit(p)
    for yv in (1, 2, p - 1):
        with pytest.raises(DirtyAncillaError):
            simulate(c, {"x": 0, "y": yv}, rng=random.Random(yv))


@pytest.mark.parametrize("p", CLOSED)
def test_the_narrowing_window_is_not_what_makes_the_zero_case_fail(p: int):
    """A control, because "it raises at small width" has an obvious innocent explanation.

    The width schedule's padding is tuned for n = 255 and is brutal at n = 5, so the first
    guess for any small-width raise is that the window clipped. It is not: switching the
    narrowing off entirely leaves the raise count unchanged at both primes. The zero case
    is structural.
    """
    def raises(narrow: bool) -> int:
        c = _mul_circuit(p, narrow=narrow)
        n = 0
        for xv in range(1, p):
            for yv in range(p):
                try:
                    simulate(c, {"x": xv, "y": yv}, rng=random.Random(xv * 3 + yv))
                except DirtyAncillaError:
                    n += 1
        return n

    assert raises(True) == raises(False) == p - 1


@pytest.mark.parametrize("p", CLOSED)
def test_the_multiply_leaks_no_phase_on_its_domain(p: int):
    """Phase 1 vents, so `tolerates_phase` is set -- but at f = 1 there is nothing to vent.

    The flag is a statement about what the circuit MAY do, and this is the measurement of
    what it does where the arithmetic is exact. A leak here would mean a disagreement in a
    region that is provably exact at this width, which would be a bug and not a truncation.
    """
    c = _mul_circuit(p)
    for xv in range(1, p):
        for yv in range(1, p, 3):
            r = simulate(c, {"x": xv, "y": yv}, rng=random.Random(xv * 11 + yv))
            assert r.phase == 0, f"p={p} x={xv} y={yv}"


# --- cost: the claim the file was written for ------------------------------
def test_the_measured_composite_reproduces_the_region_arithmetic():
    """754,544 @ 1,173, built as one circuit. Both halves of the prediction survive.

    The gate count had to agree -- gates add, and a difference would have meant a region
    was mis-attributed rather than that the arithmetic was unsound. The PEAK is the part
    the arithmetic had no right to, and the reason it survives is that the replay already
    holds `d`, `a` and `b` at once while each tape-builder holds only one operand beside
    it: threaded Phase 1 is 852 forward and 853 back, plus 255, which is 1,107-1,108
    against the replay's 1,173. Doc 10's error was reading the UNthreaded Phase 1 (1,105 +
    255 = 1,360) as if it were the composite's.

    So the prediction was right and the method that produced it was still wrong; this test
    is what makes the number a measurement.
    """
    r = _res(_MUL)
    assert r.toffoli == 754544, f"toffoli {r.toffoli}"
    assert r.peak_qubits == 1173, f"peak {r.peak_qubits}"
    assert r.peak_qubits <= params.PEAK_BUDGET


def test_the_division_is_the_multiply_plus_the_reverse_surcharge():
    """+522 and +0 qubits: the reverse replay's price, paid once by the whole composite.

    522 is 2 Toffoli per addition and jump 2 does 261 of them. As a fraction of the
    composite it is 0.07%, which is the quantitative form of what both baselines assert
    qualitatively by implementing division as a direction flag on the same circuit.
    """
    mul, div = _res(_MUL), _res(_DIV)
    assert div.toffoli - mul.toffoli == 2 * schedule.iterations_j2(params.U) == 522
    assert div.peak_qubits == mul.peak_qubits


# --- the domain at 255 bits, where the truncation is on ---------------------
#: Numerators that a caller would reach for if it wanted an inverse or a scaling: the two
#: representatives of 1, the smallest value the exact circuit accepts, and a power of two.
_CONSTANT_NUMERATORS = [1, 19, params.PRIME + 1, 1 << 128]


def test_the_reverse_replay_has_no_constant_numerator():
    """The finding, counted on the real trajectory rather than inferred from the rate.

    `un_replay_j2_fail_prob` prices the truncated `-F` at `19/2^45` a firing, so a whole
    replay carries `2^-31.6` of one. That is a statement about a UNIFORM operand, and the
    operands here are produced rather than drawn: each is the last one halved, halving
    pulls the top bits down, and a numerator with nothing up there to pull stays in the
    danger set. Six tapes, and the difference is four orders of magnitude, not a factor:

        z uniform   0 faults          z = p + 1    15
        z = 1       24                z = 19      315
                                      z = 2^128   125

    The `1` also hits the exact circuit's own hole -- `b < a` with `b < F` -- which is
    checked separately below, because widening the window does not close that one and it
    would be wrong to file the two together.
    """
    it = schedule.iterations_j2(params.U)
    rng = random.Random(11)
    tapes = [to_dialog_jump_ref(params.PRIME, rng.randrange(1, params.PRIME), it)[0]
             for _ in range(6)]

    def trunc_faults(z: int) -> int:
        return sum(sum(un_apply_dialog_jump_trunc_faults(
            t, params.PRIME, z, params.PRIME, params.U, params.CARRY_WINDOW
        )[k] for k in ("halve", "sub")) for t in tapes)

    assert trunc_faults(rng.randrange(1, params.PRIME)) == 0, "a uniform numerator is fine"
    for z in _CONSTANT_NUMERATORS:
        assert trunc_faults(z) >= 10, f"z={z}: only {trunc_faults(z)} faults"


def test_widening_the_window_does_not_rescue_the_numerator_one():
    """`z = 1` is outside the EXACT circuit too, so no amount of window buys it back.

    Two holes, one input. The truncated `-F` is a tuning choice and could in principle be
    tuned away; `b < a` with `b < F` is `un_apply_dialog_jump_circuit`'s documented
    non-totality and is there at every window width, including none. This separates them:
    the `band` column is what survives `window=None`, and for `z = 1` it is not zero.
    """
    it = schedule.iterations_j2(params.U)
    rng = random.Random(11)
    tapes = [to_dialog_jump_ref(params.PRIME, rng.randrange(1, params.PRIME), it)[0]
             for _ in range(6)]

    def band(z: int) -> int:
        return sum(un_apply_dialog_jump_trunc_faults(
            t, params.PRIME, z, params.PRIME, params.U, None
        )["band"] for t in tapes)

    assert band(1) > 0, "z = 1 was expected in the exact circuit's own hole"
    assert band(rng.randrange(1, params.PRIME)) == 0


def test_the_composite_division_refuses_a_constant_numerator():
    """The auditor's counts, confirmed on the circuit: it raises rather than lying.

    Worth having as well as the fault count because a fault is not automatically a raise --
    `un_apply_dialog_j2` alone returns a WRONG value for `z = 19` without raising, and only
    becomes loud once the reverse tape-builder downstream is handed a tape it cannot eat.
    The composite is the thing callers use, and the composite is loud.
    """
    from qsm.bench import registry

    case = registry.get_builder(_DIV)()
    rng = random.Random(3)
    xs = [rng.randrange(1, params.PRIME) for _ in range(2)]
    for z in _CONSTANT_NUMERATORS:
        for xv in xs:
            with pytest.raises(DirtyAncillaError):
                simulate(case.circuit, {"x": xv, "z": z}, rng=random.Random(0))
    for xv in xs:
        v = {"x": xv, "z": rng.randrange(1, params.PRIME)}
        assert simulate(case.circuit, v, rng=random.Random(0)).outputs == case.reference(v)


def test_the_multiplys_second_operand_carries_no_such_restriction():
    """Only the DIVISION's numerator is restricted, and the asymmetry has a cause.

    The forward replay builds its accumulator by DOUBLING, which shifts zeros into the low
    limb, and its truncated constant is a `+F` whose danger set sits just BELOW a power of
    two. Doubling walks away from that set; halving walks into the mirror one. So `y = 4`
    multiplies correctly while `z = 4` would not divide, on the same tape at the same
    tuning, and that is the whole difference between the two directions.
    """
    from qsm.bench import registry

    case = registry.get_builder(_MUL)()
    xv = random.Random(3).randrange(1, params.PRIME)
    for yv in (1, 4, 19, 1 << 128, params.PRIME - 1):
        v = {"x": xv, "y": yv}
        assert simulate(case.circuit, v, rng=random.Random(0)).outputs == case.reference(v), \
            f"y={yv}"


def test_the_first_operand_has_to_be_wide_and_that_is_phase_1_talking():
    """`x` is restricted too, differently, and far more weakly. Not the same finding.

    Phase 1's narrowing schedule allots shrinking widths on the assumption that the pair
    shrinks at the generic rate. A short `x` does not, and the clip shows up as a raise --
    measured, uniform `x` starts raising below about 2^96 and is clean above it. For
    quantum data that is `2^-159` and is already the `gcd_width_clip_fail_prob` term; it is
    recorded here only so that a raise on a hand-picked small `x` is not mistaken for the
    numerator finding above, which is a different mechanism in a different phase.
    """
    from qsm.bench import registry

    case = registry.get_builder(_MUL)()
    rng = random.Random(2)
    yv = rng.randrange(1, params.PRIME)
    for k in (128, 192, 254):
        v = {"x": rng.randrange(1 << (k - 1), 1 << k), "y": yv}
        assert simulate(case.circuit, v, rng=random.Random(0)).outputs == case.reference(v), \
            f"x ~ 2^{k}"


def test_the_composite_beats_both_of_frenchies_operating_points():
    """Where the multiply stands, stated against numbers doc 09 measured and not quoted.

    ecdsa.fail's `tlm_forward_multiply` is 614,570 @ 1,151 and is still ahead: 1.23x on
    gates at n = 256 against our n = 255. This asserts only the frenchie comparison, which
    is the one that is settled; the ecdsa.fail gap is tracked in doc 10 and is not a test,
    because it is a target rather than an invariant.
    """
    r = _res(_MUL)
    assert r.toffoli < 1_000_000
    assert r.peak_qubits < 1_200


# --- the analytic model ----------------------------------------------------
@pytest.mark.parametrize("name", [_MUL, _DIV])
def test_every_composite_carries_a_failure_rate_and_it_is_dominated_by_phase_1(name: str):
    """Both tape-builders, twice each, against one replay -- and Phase 1 is the bigger term.

    Worth pinning because it says where the next bit of accuracy would have to come from.
    The replay's iteration budget was tuned hard and its comparator truncation is the
    visible knob, but the composite's rate is set by the GCD's width-clip term, which is
    paid twice and is not a truncation at all -- it is a wrong answer when the window is too
    narrow to hold the value. See `params.gcd_width_clip_fail_prob`.
    """
    from qsm.bench import registry

    case = registry.get_builder(name)()
    fail = case.meta["fail_analytic"]
    assert 0 < fail < 1e-3, f"{name}: {fail}"

    it = schedule.iterations_j2(params.U)
    replay = params.replay_j2_fail_prob(it, cmp_bits=params.CMP_BITS,
                                        chunk=43, chunk_cmp_bits=32)
    assert fail - replay > replay, \
        f"{name}: Phase 1 contributes {fail - replay}, Phase 2 {replay}"
