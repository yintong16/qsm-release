"""Track which registered circuits exceed the historical point-addition peak.

The conservative canonical baseline is deliberately not constrained by this comparator.
Its wider deterministic schedule is measured explicitly rather than rejected as incorrect.

`PEAK_BUDGET = 1175` is not our number and not a target. It is the peak of Google's
low-qubit Pareto point for the WHOLE point addition (ecdsafail-challenge/README.md), so a
single sub-circuit above it has already lost: the point addition contains several
multiplies and an inversion, and peak is the one resource a scheduler cannot trade away by
running longer. Doc 05 section 1 argues peak is the wrong thing to MINIMISE; this file is
the other half of that argument -- it is the right thing to BOUND.

The test is written as an exact set comparison rather than a per-version assertion, so it
fails in both directions:

  - a NEW version above the budget breaks it, which is the point;
  - a version that DROPS below the budget also breaks it, so the exemption list cannot
    quietly outlive the reason for it.

An entry in OVER_BUDGET is a statement that the version is kept for measurement, not for
scheduling. Every one of them needs a reason on the line.
"""

from __future__ import annotations

import functools
import importlib

import pytest

from qsm.bench.registry import _REGISTRY
from qsm.core.resources import count
from qsm.params import FIELD, PEAK_BUDGET, PEAK_BUDGET_SOURCE, U

for _layer in ("qsm.field", "qsm.mul", "qsm.inversion", "qsm.point", "qsm.scalarmul"):
    try:
        importlib.import_module(_layer)
    except ImportError:
        pass

# Versions above the historical comparator need an explicit accounting reason.
# This includes correctness references and circuits with additional quantum inputs;
# exceeding the comparator does not make a construction invalid.
OVER_BUDGET: dict[str, str] = {
    # The schoolbook multiplication line. All of these materialise a 2U-bit product,
    # which is exactly what puts them ~600 qubits over. Superseded for the general
    # product by mul.modmul_acc_trunc.v1 (peak 1,021); see docs/decisions/07.
    "mul.product_schoolbook.v1": "2U-bit product register",
    "mul.product_square.v1": "2U-bit product register",
    "mul.modmul.v1": "product + fold",
    "mul.modmul_trunc.v1": "product + fold",
    "mul.modmul_canonical.v1": "product + fold + canonicalisation",
    "mul.modmul_karatsuba.v1": "three half-width products live at once; doc 06 section 4",
    # The square is the open case: it is over budget, and it still has the lowest
    # Toffoli count of any square we have. docs/decisions/07 leaves the choice to
    # Layer 2 -- if the point layer picks this one it must schedule around the peak.
    "mul.modsqr.v1": "2U-bit square register; OPEN, see docs/decisions/07",
    # Phase 1 of the dialog-GCD, carrying its 798-qubit tape in a register of its own.
    # These three are measurement instruments: they price the shrinking window and the
    # truncated comparator separately, and they are quoted against frenchie's ToBitVector
    # in docs/decisions/10. inv.to_dialog_pooled.v1 is the same gates with the tape hosted
    # in the slots u and v release, and it is deliberately NOT here -- it fits at 1,129,
    # which is the result those three exist to make legible.
    "inv.to_dialog_wide.v1": "798-qubit tape + full-width window; the exact baseline",
    "inv.to_dialog.v1": "798-qubit tape; prices the shrinking window alone",
    "inv.to_dialog_trunc.v1": "798-qubit tape; prices the truncated comparator on top",
    # The same three instruments at jump 2, over by the same 783-qubit-tape mechanism and
    # not by anything jump 2 does -- their pooled and packed siblings fit at 1,129 and
    # 1,107. Worth keeping as a matched set, because the jump-1/jump-2 gate ratio is only
    # quotable between versions that differ in the jump and nothing else.
    "inv.to_dialog_j2_wide.v1": "783-qubit tape + full-width window; the jump-2 baseline",
    "inv.to_dialog_j2.v1": "783-qubit tape; the shrinking window at jump 2",
    "inv.to_dialog_j2_trunc.v1": "783-qubit tape; truncated comparator at jump 2",
    # Phase 2 at jump 1, and unlike Phase 1 this is not a matter of a badly hosted tape.
    # The replay has to hold d + a + b = 665 + 255 + 255 = 1175 = PEAK_BUDGET
    # simultaneously -- the tape is READ 399 times, so it cannot be staged the way Phase 1
    # writes it -- which leaves zero qubits for the arithmetic. Every jump-1 version is
    # over by construction, and the interesting number is by how much, not whether. See
    # docs/decisions/10; `test_the_shorter_tape_is_what_buys_the_budget` below pins the
    # arithmetic, and pins that jump 2 is the thing that escapes it.
    "inv.apply_dialog.v1": "d+a+b is the whole budget; +509 for Gidney's carry ladder",
    "inv.apply_dialog_trunc.v1": "same, truncated; the ladder is what costs the peak",
    "inv.apply_dialog_cdkm_trunc.v1": "the cheapest at +48, and only 1 of that is the adder",
    "inv.apply_dialog_chunked_trunc.v1": "+134; 64-bit blocks, the volume optimum",
    # Phase 2 at jump 2. d + a + b is 609 + 255 + 255 = 1119, so the arithmetic finally
    # has 56 qubits to work in -- and `inv.apply_dialog_j2_cdkm_trunc.v1` is deliberately
    # NOT here, because it fits at 1,169. It is the first replay in the design that does,
    # and it is the whole reason the jump-2 tape was worth building. The other two want
    # more than 56: Gidney's ladder wants 509 and 64-bit blocks want 130.
    "inv.apply_dialog_j2.v1": "+516 for Gidney's ladder; the exact jump-2 baseline",
    "inv.apply_dialog_j2_trunc.v1": "same ladder, truncated; cheapest on Toffoli anywhere",
    "inv.apply_dialog_j2_chunked_trunc.v1": "64-bit blocks want 130 of the 56 freed",
    # The division. Same peak as the forward it reverses, to the qubit, at every adder --
    # a reversal cannot need a register the forward did not, and the +2 Toffoli per
    # iteration it does cost buys no width. So each of these sits over by exactly the
    # amount its forward does, and the three entries above carry the reason.
    "inv.un_apply_dialog.v1": "the reverse of inv.apply_dialog.v1, at its peak exactly",
    "inv.un_apply_dialog_trunc.v1": "the reverse of inv.apply_dialog_trunc.v1",
    "inv.un_apply_dialog_cdkm_trunc.v1": "the reverse of the cheapest, still +48",
    # The division at jump 2, which is the division the design actually uses. Same
    # statement one jump along: each is its forward's peak to the qubit, so the three
    # jump-2 forwards listed above carry the reasons. The two that matter --
    # `inv.un_apply_dialog_j2_cdkm_trunc.v1` at 1,169 and
    # `inv.un_apply_dialog_j2_threaded_tuned_trunc.v1` at 1,173 -- are deliberately NOT
    # here, because they fit. A division that fits the budget is the point of the exercise.
    "inv.un_apply_dialog_j2.v1": "the reverse of inv.apply_dialog_j2.v1, at its peak",
    "inv.un_apply_dialog_j2_trunc.v1": "the reverse of inv.apply_dialog_j2_trunc.v1",
    "inv.un_apply_dialog_j2_chunked_trunc.v1": "64-bit blocks, backwards; still +80",
    "inv.modmul_conservative_inplace.v1": "full-width worst-case GCD and canonical replay",
    "inv.moddiv_conservative_inplace.v1": "full-width worst-case GCD and canonical reverse replay",
    "point.add_const_affine_conservative_inplace.v1": "canonical point with conservative inversion",
    "point.add_const_affine_conservative_ctrl_inplace.v1": "controlled canonical point baseline",
    "scalarmul.add_window_affine_w4_inplace.v1": "four coherent index qubits remain live through tuned GCD",
    "scalarmul.add_window_affine_w4_inplace.v2": "measured QROM with four index qubits live through tuned GCD",
}


@functools.lru_cache(maxsize=None)
def _peak(name: str) -> int:
    """Count only the requested circuit; focused tests must not build the registry."""
    return count(_REGISTRY[name]().circuit).peak_qubits


@functools.lru_cache(maxsize=1)
def _peaks() -> dict[str, int]:
    # Only the explicit snapshot test walks the whole registry. The dialog-GCD
    # versions are million-op circuits that cost seconds apiece to build.
    peaks = {}
    for name in _REGISTRY:
        if name.startswith("demo."):
            continue
        peaks[name] = _peak(name)
    return peaks


@pytest.mark.slow
@pytest.mark.skipif(FIELD != "25519", reason="OVER_BUDGET is the historical 25519 registry snapshot")
def test_no_unexpected_version_exceeds_the_peak_budget():
    """The exact set of over-budget versions is the declared one. Both directions."""
    peaks = _peaks()
    over = {n for n, pk in peaks.items() if pk > PEAK_BUDGET}
    declared = set(OVER_BUDGET)

    unexpected = over - declared
    assert not unexpected, (
        f"these exceed PEAK_BUDGET={PEAK_BUDGET} and are not declared: "
        + ", ".join(f"{n} (peak {peaks[n]})" for n in sorted(unexpected))
        + f"\nbudget source: {PEAK_BUDGET_SOURCE}"
        + "\nEither bring the peak down or add it to OVER_BUDGET with a reason."
    )

    stale = declared - over
    assert not stale, (
        "these are declared over-budget but now fit: "
        + ", ".join(f"{n} (peak {peaks.get(n, 'UNREGISTERED')})" for n in sorted(stale))
        + "\nDrop them from OVER_BUDGET."
    )


def test_the_accumulate_multiply_is_the_one_that_fits():
    """The whole argument of docs/decisions/07 in one assertion.

    If this ever fails, either the schedule grew a register or the budget moved -- and
    both are things that should stop the build rather than show up in a table."""
    assert _peak("mul.modmul_acc_trunc.v1") <= PEAK_BUDGET
    assert _peak("mul.modmul_trunc.v1") > PEAK_BUDGET
    # 4U+1 and 3U+1 exactly -- three live registers plus a Gidney carry ladder, and one
    # fewer for the square. A jump here means a scratch register crept in.
    assert _peak("mul.modmul_acc_trunc.v1") == 4 * U + 1
    assert _peak("mul.modsqr_acc_trunc.v1") == 3 * U + 1


@pytest.mark.skipif(FIELD != "25519", reason="1175/1119 state and budget crossing are 25519 snapshots")
def test_the_shorter_tape_is_what_buys_the_budget_not_the_adder():
    """At jump 1 no adder can fit; at jump 2 one can. The difference is 56 qubits of tape.

    `d + a + b` is 665 + 255 + 255 = 1175, which IS `PEAK_BUDGET`. Phase 1 gets away with
    a tape that size by hosting it in the slots `u` and `v` release as they shrink; Phase
    2 cannot, because it READS all 399 entries and the operands stay full width the whole
    time. So a jump-1 replay starts at the budget with nothing left, and every version of
    it is over by exactly its working set -- choosing a cheaper adder moves the overhead,
    never the 1175.

    Jump 2 moves the 1175. Three signals an iteration over 261 iterations pack into 609
    bits against 665, so `d + a + b` is 1119 and the arithmetic has 56 qubits to live in
    for the first time. CDKM wants 50 of them and fits at 1,169; Gidney's ladder wants 509
    and 64-bit blocks want 130, so those stay over. That is the entire qubit case for
    jump 2, and it appears HERE and nowhere else -- Phase 1's peak barely moves (1,112 to
    1,107), because Phase 1 peaks early, at full window width with an almost-empty tape.
    """
    from qsm.inversion import pack, schedule
    from qsm.params import U
    iters, iters_j2 = schedule.iterations(U), schedule.iterations_j2(U)
    assert pack.packed_bits(iters) + 2 * U == PEAK_BUDGET, "the budget is exactly the state"
    assert pack.packed_bits_j2(iters_j2) + 2 * U == PEAK_BUDGET - 56

    j1 = [n for n in _REGISTRY if n.startswith("inv.apply_dialog") and "_j2" not in n]
    assert min(_peak(n) for n in j1) > PEAK_BUDGET, "no jump-1 adder fits, and none can"
    assert _peak("inv.apply_dialog_j2_cdkm_trunc.v1") <= PEAK_BUDGET

    # And the saving is the tape less what the base-5 reader costs to carry, not 56 flat.
    assert _peak("inv.apply_dialog_cdkm_trunc.v1") \
        - _peak("inv.apply_dialog_j2_cdkm_trunc.v1") == 54


def test_the_replays_peak_overhead_is_the_carry_window_not_the_adder():
    """Which is the surprise, and it is what doc 10 records.

    The three replay versions differ ONLY in their controlled adder, so the naive reading
    of the 1223/1309/1689 spread is that the adder decides the peak. It does not decide
    the floor. `CARRY_WINDOW = 45` is needed twice per iteration -- once by `moddouble`
    and once by the constant `+f` inside `modadd_lazy` -- and neither of those is the
    adder. On CDKM the controlled adder itself contributes ONE qubit, so 45 of its 48 is
    the window and the adder is a rounding error.

    The practical consequence: narrowing `CARRY_WINDOW` is the only lever that moves the
    cheapest replay, and `params.replay_fail_prob` prices what that would cost -- the two
    window terms are already 2^-33.2 and 2^-32.4, so there is not much slack in it.
    """
    from qsm.core.circuit import Circuit
    from qsm.field.add import cdkm_add_ctrl
    from qsm.field.modarith import moddouble
    from qsm.params import CARRY_WINDOW, PRIME, U

    plain = Circuit()
    moddouble(plain, plain.input("b", U), PRIME, window=CARRY_WINDOW)
    assert plain.peak_qubits - U == CARRY_WINDOW, "moddouble's window, alone"

    add = Circuit()
    cdkm_add_ctrl(add, add.input("ctrl", 1)[0], add.input("a", U), add.input("b", U))
    assert add.peak_qubits - (2 * U + 1) == 1, "the whole controlled adder, alone"

    # Actual d+a+b state + three reader qubits + the field's carry window.
    # The state equals the historical PEAK_BUDGET only in the 25519 profile.
    from qsm.inversion import pack, schedule
    state = pack.packed_bits(schedule.iterations(U)) + 2 * U
    assert _peak("inv.apply_dialog_cdkm_trunc.v1") == state + 3 + CARRY_WINDOW


@pytest.mark.parametrize("name", sorted(OVER_BUDGET))
def test_every_over_budget_version_is_still_registered(name):
    """A stale exemption for a deleted version would silently weaken the check above."""
    assert name in _REGISTRY, f"{name} is exempted from the peak budget but not registered"
