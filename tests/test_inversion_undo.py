"""Phase 1 backwards: does it really give `x` back, and does it really cost the same?

Two claims are on trial here, and the second one is a correction of something this project
believed for a while.

1. `un_to_dialog` inverts `to_dialog` -- `x` comes back and the tape comes back to |0>.
   The second half is checked for free: `un_to_dialog` *frees* the tape, and
   `Circuit.free` raises `DirtyAncillaError` on a qubit that is not |0>. So a test that
   simulates without raising has already proved the tape was spent.

2. The inverse costs EXACTLY the forward, gate for gate. Reconstructing the values needs
   no comparison (`reference.un_dialog` does it with shifts and adds), and it is tempting
   to conclude the reverse is cheaper. It is not: the composite needs |d> -> |x>, not
   |d> -> |d, x>, so `s` has to be erased, and erasing it means re-deriving
   `b0 & (u > v)` on the restored operands. Same comparator, same width. Toffoli parity
   is the observable form of that argument, so it is asserted rather than described.

The peak is the one place the two directions legitimately differ, and it differs in our
favour -- see `test_the_flat_reverse_fits_without_any_pooling`.
"""

from __future__ import annotations

import random

import pytest

from qsm.bench import registry
from qsm.core.circuit import Circuit, DirtyAncillaError, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl
from qsm.inversion import pack, schedule
from qsm.inversion.gcd import FlatTape, PackedTape, encode_dialog, to_dialog
from qsm.inversion.reference import to_dialog as to_dialog_ref
from qsm.inversion.undo import FlatUnTape, PackedUnTape, un_to_dialog

PRIMES = [29, 61, 127, 251]
LAYOUTS = ["flat", "packed"]


def _iters(p: int) -> int:
    return schedule.iterations(p.bit_length())


def _forward(p: int, layout: str, **kw) -> Circuit:
    """|x> -> |d> at `p`, in either tape layout."""
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    x = c.input("x", n)
    if layout == "flat":
        tape = FlatTape(c.input("d", 2 * it))
    else:
        c.input("d", 0)
        tape = PackedTape(c)
    d = to_dialog(c, x, tape, p, iters=it, **kw)
    if layout != "flat":
        c.rebind("d", d)
    return c


def _reverse(p: int, layout: str, **kw) -> Circuit:
    """|d> -> |x> at `p`. The mirror image of `_forward`, same arguments."""
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    if layout == "flat":
        tape = FlatUnTape(c, c.input("d", 2 * it))
    else:
        tape = PackedUnTape(c, c.input("d", pack.packed_bits(it)), it)
    c.input("x", 0)
    x = un_to_dialog(c, tape, n, p, iters=it, **kw)
    c.rebind("x", x)
    c.rebind("d", Register([], "d"))   # spent, not merely zero
    return c


def _tape(p: int, x: int, layout: str) -> int:
    """The encoded tape the reference writes for `x`, in the layout's own format."""
    tape, _, _ = to_dialog_ref(p, x, _iters(p))
    return encode_dialog(tape) if layout == "flat" else pack.pack_dialog(tape)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("p", PRIMES)
def test_un_to_dialog_gives_x_back_exhaustively(p: int, layout: str):
    """Every x in [1, p): the tape undoes to the input that wrote it, cleanly.

    `d` is not checked against 0 explicitly because it cannot survive being nonzero --
    `un_to_dialog` frees every tape qubit, and a dirty free raises.
    """
    c = _reverse(p, layout)
    for x in range(1, p):
        r = simulate(c, {"d": _tape(p, x, layout), "x": 0}, rng=random.Random(x))
        assert r.outputs["x"] == x, f"p={p} {layout}: got {r.outputs['x']}, want {x}"
        assert r.phase == 0, f"p={p} {layout}: phase leaked on x={x}"


@pytest.mark.parametrize("p", PRIMES)
def test_forward_then_reverse_is_the_identity(p: int):
    """Composed in one circuit, with no classical tape in between."""
    n, it = p.bit_length(), _iters(p)
    c = Circuit()
    x = c.input("x", n)
    d = to_dialog(c, x, FlatTape(c.input("d", 2 * it)), p, iters=it)
    back = un_to_dialog(c, FlatUnTape(c, d), n, p, iters=it)
    c.rebind("x", back)
    c.rebind("d", Register([], "d"))

    for v in range(1, p):
        r = simulate(c, {"x": v, "d": 0}, rng=random.Random(v))
        assert r.outputs["x"] == v
        assert r.phase == 0


@pytest.mark.parametrize("p", PRIMES)
def test_the_narrow_window_undoes_too(p: int):
    """The shrinking schedule run backwards: the window GROWS out of the retired tape."""
    n, it = p.bit_length(), _iters(p)
    widths = schedule.widths(n, it)
    c = _reverse(p, "flat", widths=widths)
    for x in range(1, p):
        r = simulate(c, {"d": _tape(p, x, "flat"), "x": 0}, rng=random.Random(x))
        assert r.outputs["x"] == x
        assert r.phase == 0


@pytest.mark.parametrize("add_ctrl", [gidney_add_ctrl, cdkm_add_ctrl])
@pytest.mark.parametrize("p", [29, 127])
def test_either_controlled_adder_drives_the_reverse(p: int, add_ctrl):
    """The undone subtract is a plain controlled add -- `_csub`'s X gates self-cancel."""
    c = _reverse(p, "flat", add_ctrl=add_ctrl)
    for x in range(1, p):
        r = simulate(c, {"d": _tape(p, x, "flat"), "x": 0}, rng=random.Random(x))
        assert r.outputs["x"] == x


# ---------------------------------------------------------------------------
# The parity claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("p", PRIMES + [1021])
def test_the_inverse_costs_exactly_what_the_forward_costs(p: int, layout: str):
    """The claim `undo.py`'s docstring rests on, and the reason the composite is 2F + R.

    Not "about the same" -- the same integer. If this ever drifts, either the reverse
    stopped erasing `s` with the forward's comparator (wrong, it would dirty the tape) or
    the forward grew a gate the reverse does not mirror.
    """
    widths = schedule.widths(p.bit_length(), _iters(p))
    fwd = count(_forward(p, layout, widths=widths))
    rev = count(_reverse(p, layout, widths=widths))
    assert rev.toffoli == fwd.toffoli, (
        f"p={p} {layout}: forward {fwd.toffoli}, reverse {rev.toffoli}. The inverse of "
        "Phase 1 is not cheaper than Phase 1 -- erasing the tape costs the comparator."
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("p", PRIMES + [1021])
def test_the_reverse_never_peaks_higher_than_the_forward(p: int, layout: str):
    """Free/alloc share one id pool, so the retiring tape feeds the widening window."""
    widths = schedule.widths(p.bit_length(), _iters(p))
    fwd = count(_forward(p, layout, widths=widths))
    rev = count(_reverse(p, layout, widths=widths))
    assert rev.peak_qubits <= fwd.peak_qubits, (
        f"p={p} {layout}: forward peaked at {fwd.peak_qubits}, reverse at "
        f"{rev.peak_qubits}"
    )


@pytest.mark.parametrize("p", PRIMES + [1021])
def test_packing_costs_the_same_in_both_directions(p: int):
    """The 6->5 compression is paid on the way in AND on the way out, at the same rate."""
    widths = schedule.widths(p.bit_length(), _iters(p))
    fwd_flat = count(_forward(p, "flat", widths=widths)).toffoli
    fwd_pack = count(_forward(p, "packed", widths=widths)).toffoli
    rev_flat = count(_reverse(p, "flat", widths=widths)).toffoli
    rev_pack = count(_reverse(p, "packed", widths=widths)).toffoli
    assert fwd_pack - fwd_flat == rev_pack - rev_flat > 0


def test_the_flat_reverse_fits_without_any_pooling():
    """The one asymmetry that is real, and it is in our favour.

    Forward, the flat tape is 798 qubits held alongside a window that only shrinks, so it
    peaks at 1817 and needs `PooledTape` to come down. Backward, the same 798 qubits
    retire two at a time straight into the widening window, so the plain layout lands
    where the *pooled* forward does -- with no pooling logic at all.
    """
    fwd_flat = count(registry.get_builder("inv.to_dialog.v1")().circuit).peak_qubits
    fwd_pooled = count(registry.get_builder("inv.to_dialog_pooled.v1")().circuit).peak_qubits
    rev_flat = count(registry.get_builder("inv.un_to_dialog_flat.v1")().circuit).peak_qubits

    assert fwd_flat > fwd_pooled, "the pooled forward is supposed to be the cheap one"
    assert rev_flat <= fwd_pooled, (
        f"flat reverse peaked at {rev_flat}; the pooled forward manages {fwd_pooled} "
        "and the flat reverse should match it for free"
    )


@pytest.mark.parametrize("name", [
    "inv.un_to_dialog_flat.v1",
    "inv.un_to_dialog.v1",
    "inv.un_to_dialog_trunc.v1",
])
def test_the_registered_reverses_stay_under_the_peak_budget(name: str):
    from qsm import params

    peak = count(registry.get_builder(name)().circuit).peak_qubits
    assert peak <= params.PEAK_BUDGET, f"{name} peaks at {peak}"


# ---------------------------------------------------------------------------
# Truncation, which is not symmetric in its FAILURE MODE even though it is in cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p,k", [(251, 6), (1021, 7)])
def test_a_matched_truncated_pair_still_round_trips(p: int, k: int):
    """Truncate both halves the same way and they agree on every branch they take.

    This is the case `modmul_inplace` is in: the forward's tape need not be the
    reference's tape, but the reverse re-derives the same predicate on the same operands,
    so it undoes whatever the forward actually did -- including the branches where the
    truncation was wrong. The test only means something if truncation *did* change some
    branches, so that is asserted rather than hoped for.

    The x values where the truncated GCD fails to converge inside the iteration budget are
    skipped: `to_dialog` frees a dirty window and raises before the reverse ever runs, so
    they are a Phase 1 budget failure, not an asymmetry between the two directions. Most
    inputs are skipped at these widths, and that is not a defect either -- `k` here is a
    RAW slice width, and `test_the_plain_cmp_bits_width_is_unusable_inside_the_window`
    already records that a raw `k` below `width_pad(n)` wrecks convergence. Every `k` that
    converges for all x at n = 8 or 10 also agrees with the exact comparator everywhere,
    so a `k` that diverges at all is necessarily a `k` that also loses most inputs. The
    claim under test is the conditional one: *given* that Phase 1 converged, the two
    directions agree.
    """
    n, it = p.bit_length(), _iters(p)
    fwd = _forward(p, "flat", cmp_bits=k)

    c = Circuit()
    x = c.input("x", n)
    d = to_dialog(c, x, FlatTape(c.input("d", 2 * it)), p, iters=it, cmp_bits=k)
    back = un_to_dialog(c, FlatUnTape(c, d), n, p, iters=it, cmp_bits=k)
    c.rebind("x", back)
    c.rebind("d", Register([], "d"))

    checked = diverged = 0
    for v in range(1, p):
        try:
            f = simulate(fwd, {"x": v, "d": 0}, rng=random.Random(v))
        except DirtyAncillaError:
            continue                       # the truncated Phase 1 ran out of iterations
        diverged += f.outputs["d"] != _tape(p, v, "flat")
        r = simulate(c, {"x": v, "d": 0}, rng=random.Random(v))
        assert r.outputs["x"] == v, f"p={p}: truncated round trip lost x={v}"
        assert r.phase == 0
        checked += 1

    assert checked > 0, f"p={p}: nothing converged at k={k}; the test saw no round trips"
    assert diverged > 0, (
        f"p={p}: a {k}-bit comparator never took a different branch, so this test is not "
        "exercising the thing it claims to"
    )


def test_a_tape_written_by_a_different_comparator_fails_loudly():
    """Mismatch the two halves and the tape does not come back to |0>.

    Forward, a wrong truncated compare is harmless -- the loop takes the other branch and
    writes a tape that is consistent with itself. Backward the tape is already fixed, so a
    disagreement strands `s` at |1> and the free raises. A loud failure, not a wrong
    answer, which is why the standalone `_trunc` benchmark carries a comparator term its
    forward counterpart does not need.
    """
    p, k = 1021, 2
    c = _reverse(p, "flat", cmp_bits=k)
    raised = 0
    for x in range(1, p):
        try:
            simulate(c, {"d": _tape(p, x, "flat"), "x": 0}, rng=random.Random(x))
        except DirtyAncillaError:
            raised += 1
    assert raised > 0, (
        "a 2-bit comparator must disagree with the exact reference somewhere; if it "
        "never does, the reverse is not re-deriving `s` at all"
    )
