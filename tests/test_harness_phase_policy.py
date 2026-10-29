"""Phase leakage is fatal unless a circuit explicitly declares it as its failure mode.

Being *approximate* must not by itself buy phase tolerance: a truncated constant adder
or comparator truncates by slicing a register and erases with exact `_unand`s, so it can
never legitimately leak phase. Only a vented flag erasure can, and it has to say so via
`Case.meta["tolerates_phase"]`.
"""

import random

import pytest

from qsm.bench import harness, registry
from qsm.core.circuit import Circuit


def _leaky_case(meta):
    """A circuit that deliberately leaks phase on half its inputs."""
    circ = Circuit()
    q = circ.input("q", 1)
    circ.z(q[0])                                  # phase flip iff q == 1

    def ref(v):
        return {"q": v["q"]}                      # value is untouched; only phase moves

    def gen(rng: random.Random):
        return {"q": 1}                           # always trigger the leak

    return registry.Case(circ, ref, gen, meta=meta)


def _register(monkeypatch, name, meta):
    monkeypatch.setitem(registry._REGISTRY, name, lambda: _leaky_case(meta))
    return name


def test_phase_leak_is_fatal_for_an_exact_circuit(monkeypatch):
    v = _register(monkeypatch, "test.phaseleak_exact.v1", {})
    r = harness.benchmark(v, n_trials=4, persist=False)
    assert r.correct is False
    assert "phase" in r.note


def test_phase_leak_is_fatal_for_an_APPROXIMATE_circuit_too(monkeypatch):
    """The regression this file exists for: `fail_analytic` alone must NOT excuse phase."""
    v = _register(monkeypatch, "test.phaseleak_approx.v1", {"fail_analytic": 1e-12})
    r = harness.benchmark(v, n_trials=4, persist=False)
    assert r.correct is False, "approximation must not grant phase tolerance"
    assert "phase" in r.note


def test_phase_leak_is_tolerated_only_when_declared(monkeypatch):
    v = _register(monkeypatch, "test.phaseleak_declared.v1",
                  {"fail_analytic": 0.9, "tolerates_phase": True})
    r = harness.benchmark(v, n_trials=4, persist=False)
    assert r.correct is True
    assert r.phase_seen > 0, "the leak should still be counted, just not fatal"


def test_exact_declaration_cannot_license_phase(monkeypatch):
    v = _register(monkeypatch, "test.phaseleak_exact_tolerant.v1",
                  {"fail_analytic": 0.0, "tolerates_phase": True})
    r = harness.benchmark(v, n_trials=4, persist=False)
    assert not r.correct
    assert r.trials == r.inputs_executed == r.phase_seen == 1
    assert r.aborted and r.fail_bound95 == 1.0
    assert "phase" in r.note


def test_one_shot_declared_phase_regression_rejected(monkeypatch):
    v = _register(monkeypatch, "test.phaseleak_tiny.v1",
                  {"fail_analytic": 1e-12, "tolerates_phase": True})
    r = harness.benchmark(v, n_trials=1, persist=False)
    assert not r.correct
    assert not r.aborted
    assert r.batches_failed == r.batches_completed == r.phase_seen == 1
    assert "phase (vented)" in r.note


#: Versions allowed to declare `tolerates_phase`, each with the reason it must.
#: Adding to this list is a deliberate act -- phase tolerance means the harness stops
#: treating kickback as fatal, which is how a real defect could hide.
PHASE_TOLERANT = {
    # vented flag erasure: a wrong TRUNCATED predicate kicks back phase instead of
    # leaving a dirty ancilla. Rate ~2^-41.
    "field.modadd_lazy.v4",
    # vented erasure against `~a` instead of the unavailable `p - a`. Wrong on a window
    # of width F -> F/(2p) = 2^-251.75. NOT a truncation artefact: it is a property of
    # the predicate, which is why the UNTRUNCATED version declares it too. That is also
    # why it is not named `_exact`: no window setting makes it exact.
    "field.modsub_direct.v2",
    "field.modsub_direct_trunc.v2",
    # NOTE: the sandwich (field.modsub.v1 / _trunc.v1) is deliberately absent. It
    # materialises p - a, so its erasure predicate is exact and nothing kicks back --
    # that is exactly what the extra Toffoli buy.
    #
    # The dialog replay, all four adder variants. It inherits field.modadd_lazy.v4's vent
    # 399 times over, and -- this is the part worth being deliberate about -- the EXACT
    # version needs the tolerance too. That is not truncation: `modadd_lazy`'s vent
    # re-derives its overflow flag as `b < a`, which is only sound for an accumulator
    # below p, and the replay runs lazily by design (doc 05: no fold in the design
    # produces a canonical value). So `params.modadd_lazy_vent_phase_prob` fires whatever
    # the windows are set to, and no window setting removes it -- see
    # `replay_fail_prob`, where it is the fifth term at 2^-242.2.
    #
    # It is 2^-242.2 at 255 bits and NOT negligible at test widths: at p = 29 the lazy
    # band is 3 values of 32, so a small-prime replay leaks phase on most shots. That is
    # why `reference.apply_dialog_faults` exists -- it says which concrete inputs are
    # entitled to the leak, so the tolerance is checked rather than merely granted.
    "inv.apply_dialog.v1",
    "inv.apply_dialog_trunc.v1",
    "inv.apply_dialog_cdkm_trunc.v1",
    "inv.apply_dialog_chunked_trunc.v1",
    #
    # The same four at jump 2, and for the same reason -- but the rate is not the same,
    # and the direction it moves in is worth stating. The vent rides on the ADDITION, of
    # which jump 2 has 261 rather than 399, so a shorter tape is strictly *less* exposed:
    # `replay_j2_fail_prob`'s vent term is 261/399 of `replay_fail_prob`'s. What jump 2
    # adds instead is a second doubling per iteration, and that one cannot vent at all --
    # `moddouble_ctrl_inplace` erases its parity with a Toffoli, not a measurement, so it
    # is exact and contributes no phase. Hence four more entries here and no new *kind*
    # of leak, which is the thing this allowlist is meant to make someone check.
    "inv.apply_dialog_j2.v1",
    "inv.apply_dialog_j2_trunc.v1",
    "inv.apply_dialog_j2_cdkm_trunc.v1",
    "inv.apply_dialog_j2_chunked_trunc.v1",
    #
    # The threaded chunked replays. These leak from the SAME place
    # `inv.apply_dialog_j2_chunked_trunc.v1` does -- the chunked adder's boundary vent,
    # `params.chunked_add_boundary_fail_prob` -- and threading the control does not touch
    # it: the vent's `cmp_lt_phase` runs against the unmasked operand either way, which is
    # why `qsm.audit rates` holds both adders to one bound and measures them at the same
    # ratio. So this is two more entries and, again, no new *kind* of leak.
    #
    # Worth being explicit about what threading DOES change here, since it is the one
    # place a reader might expect a new term and there isn't one. Threading replaces the
    # per-block mask with a gated carry-out, and the gate is a Toffoli -- exact, erased by
    # a Toffoli, no measurement. The mask it replaces was erased by measurement but on a
    # value that is exactly zero, so it never leaked either. The phase budget is entirely
    # the vent's, in both constructions.
    "inv.apply_dialog_j2_threaded_trunc.v1",
    "inv.apply_dialog_j2_threaded_wide_trunc.v1",
    # Same vent again, just deliberately coarser: these two untie `cmp_bits` from `chunk`
    # and spend the vent's precision down to where the composite stops noticing. Fewer
    # bits compared means the leak is *likelier*, not different in kind, so they belong
    # here for exactly the reason the two above do.
    "inv.apply_dialog_j2_threaded_tuned_trunc.v1",
    "inv.apply_dialog_j2_threaded_tuned_narrow_trunc.v1",
    #
    # The two in-place composites. Neither adds a vent of its own: each is
    # `to_dialog_j2` + a replay + a swap + `un_to_dialog_j2`, and every leak in the total
    # belongs to a component already on this list.
    #
    # Which component, though, is NOT the same in the two directions, and getting that
    # wrong is easy: Phase 1 contributes nothing here. `gcd.py` and `undo.py` import
    # `cmp_lt`, not `cmp_lt_phase`, and neither calls `modadd_lazy` -- the tape-builders
    # erase every flag with a Toffoli and measure nothing. So the whole phase budget of
    # both composites is Phase 2's.
    #
    # For the multiply that is `modadd_lazy`'s vent (261 firings) plus the chunked adder's
    # boundary vent. For the division it is the BOUNDARY VENT ALONE: `un_apply_dialog_j2`
    # re-derives the modadd flag with a real `cmp_lt` -- see unreplay.py's header -- so
    # every disagreement the forward would have hidden in a phase arrives instead as a
    # stranded flag and a raise. Measured below at u = 10: with an exact replay comparator
    # the division still leaks phase on 15 shots of 40 (the boundary vent, earning this
    # entry) and still raises `DirtyAncillaError` on 2 (the modadd term, failing closed).
    "inv.modmul_inplace.v1",
    "inv.moddiv_inplace.v1",
    #
    # The point step. Like the two composites it adds no vent of its own -- it is
    # `moddiv_inplace` + `modsqrsub_inplace` + `modmul_inplace` + six `sub_const_mod`, and
    # every one of those already leaks here or in `PHASE_TOLERANT`'s `field.` entries. The
    # division alone puts it on this list; the six constant subtracts are `modsub_direct`
    # with a classical subtrahend, so each carries that version's `F/p` erasure vent.
    #
    # The control changes nothing about the phase budget, which is worth stating because
    # it is the surprising half of this construction: the two expensive regions are
    # UNCONTROLLED (they are exact inverses, so at ctrl=0 the multiply undoes the
    # division), and the gating rides on CNOT layers loading a constant scratch. A CNOT
    # leaks no phase, and `b - 0` never fires the borrow, so the gated version's leak is
    # the plain one's. Both are listed because both are registered, not because they
    # differ.
    "point.add_const_affine_inplace.v1",
    "point.add_const_affine_inplace_ctrl.v1",
    # The same constructions with the F-independent square ripple; identical vents.
    "point.add_const_affine_inplace.v2",
    "point.add_const_affine_inplace_ctrl.v2",
}

#: Of those, the ones that leak phase WITHOUT also promising a clean ancilla.
#: `test_phase_tolerant_versions_are_the_vented_ones` asserts dirty-impossibility for
#: everything else on the list, and that assertion is simply false here: the division's
#: tolerance is earned by one vent (the chunked adder's boundary) while a different,
#: unvented term (`modadd_lazy_vent_phase_prob`, promoted to a value failure by the
#: reverse replay) can still strand a flag. The flag is still right -- drop it and the
#: benchmark calls the boundary vent's kickback fatal -- so the fix is to test the
#: property this version actually has, not to remove it from the list.
FAILS_CLOSED = {
    "inv.moddiv_inplace.v1",
    # Both point versions contain `moddiv_inplace`, so they inherit its one unvented term
    # verbatim -- there is no way for a wrapper to make a stranded flag impossible when the
    # region it calls can strand one. They inherit its third precondition too: a zero
    # operand into either expensive region raises rather than returning garbage, and at
    # u = 10 the random draws below occasionally land on one.
    "point.add_const_affine_inplace.v1",
    "point.add_const_affine_inplace_ctrl.v1",
    # The same constructions with the F-independent square ripple; identical vents.
    "point.add_const_affine_inplace.v2",
    "point.add_const_affine_inplace_ctrl.v2",
}


def test_only_declared_versions_tolerate_phase():
    """Nothing outside PHASE_TOLERANT may claim it. If a new circuit needs it, that
    should be a deliberate decision, not something inherited by copy-paste."""
    import qsm.bench.__main__  # noqa: F401  -- registers every layer

    declaring = set()
    empirical = set()
    for name in registry.all_versions():
        if name.startswith("test."):
            continue
        meta = registry.get_builder(name)().meta
        # An exact circuit declares `{}`, never None. The harness reads `meta.get`
        # unconditionally, so a None here does not degrade -- it takes down every
        # benchmark in the run, including the ones that have nothing to do with it.
        assert isinstance(meta, dict), f"{name}: meta must be a dict, got {meta!r}"
        if meta.get("tolerates_phase"):
            declaring.add(name)
        if meta.get("empirical_only"):
            assert meta.get("fail_analytic") is None
            assert not meta.get("tolerates_phase")
            empirical.add(name)
    assert empirical == {"scalarmul.add_window_affine_w4_inplace.v1",
                         "scalarmul.add_window_affine_w4_inplace.v2"}
    assert declaring == PHASE_TOLERANT, {
        "unexpectedly tolerant": sorted(declaring - PHASE_TOLERANT),
        "declared but no longer tolerant": sorted(PHASE_TOLERANT - declaring),
    }


def test_phase_tolerant_versions_are_the_vented_ones():
    """Sanity-check the *reason* these versions are allowed to leak phase.

    This used to assert only `any(op.kind == MEASURE for op in circ.ops)`, which cannot
    distinguish a vented flag erasure from `gidney_add`'s internal `_unand` measurements
    -- every one of these circuits contains those regardless. Proof it was vacuous: with
    `modadd_lazy`'s vent default flipped off, the old assertion still held.

    The property that actually matters is behavioural, so test that instead: venting
    means a dirty ancilla is IMPOSSIBLE even when the erasure predicate is truncated.
    Checked on the same construction at a width where it is measurable.
    """
    import functools
    import random

    import qsm.bench.__main__  # noqa: F401

    from qsm.core.circuit import Circuit, DirtyAncillaError
    from qsm.core.simulator import simulate
    from qsm.field.add import cdkm_add_ctrl, gidney_add, gidney_add_ctrl
    from qsm.field.add_chunked import (
        gidney_add_ctrl_chunked, gidney_add_ctrl_chunked_threaded,
    )
    from qsm.field.modarith import modadd_lazy, modsub_direct
    from qsm.inversion import pack, replay, schedule
    from qsm.inversion.inplace import moddiv_inplace, modmul_inplace
    from qsm.inversion.reference import to_dialog as to_dialog_ref
    from qsm.inversion.reference import to_dialog_jump as to_dialog_jump_ref
    from qsm.point.affine import add_const_affine_inplace

    u, p = 10, 1019
    builders = {
        "field.modadd_lazy.v4":
            lambda c, a, b: modadd_lazy(c, a, b, p, gidney_add, 4, 3, vent=True),
        "field.modsub_direct.v2":
            lambda c, a, b: modsub_direct(c, a, b, p, gidney_add, None, 3),
        "field.modsub_direct_trunc.v2":
            lambda c, a, b: modsub_direct(c, a, b, p, gidney_add, 4, 3),
    }

    # The replay stand-ins. Same claim, one layer up: the loop inherits `modadd_lazy`'s
    # vent 399 times, so if venting really does make a dirty ancilla impossible then no
    # amount of iterating can produce one. Each registered version gets its own adder
    # here, because the adder is the only thing that differs between them and it is the
    # part that allocates. Chunked is cut at 3 bits so a 10-bit register still crosses a
    # boundary; at 64 it would silently degrade to plain `gidney_add_ctrl`.
    _iters = schedule.iterations(u)
    _replay_adders = {
        "inv.apply_dialog.v1": (gidney_add_ctrl, False),
        "inv.apply_dialog_trunc.v1": (gidney_add_ctrl, True),
        "inv.apply_dialog_cdkm_trunc.v1": (cdkm_add_ctrl, True),
        "inv.apply_dialog_chunked_trunc.v1":
            (functools.partial(gidney_add_ctrl_chunked, chunk=3, cmp_bits=3), True),
    }

    def _make_replay(add_fn, trunc):
        def build(c, a, b):
            d = c.input("d", pack.packed_bits(_iters))
            a2, b2 = replay.apply_dialog(
                c, d, a, b, _iters, p, add_ctrl_fn=add_fn,
                double_window=4 if trunc else None,
                carry_window=4 if trunc else None,
                cmp_bits=3 if trunc else None)
            c.rebind("a", a2)
            c.rebind("b", b2)
        return build

    # The jump-2 replays, same four adders. The stand-in has to be a separate builder
    # rather than a flag on the one above: the tape is a different width, a different
    # radix and a different reference, and the whole point of this test is that the thing
    # being simulated is the thing that was registered.
    _iters_j2 = schedule.iterations_j2(u)

    def _make_replay_j2(add_fn, trunc):
        def build(c, a, b):
            d = c.input("d", pack.packed_bits_j2(_iters_j2))
            a2, b2 = replay.apply_dialog_j2(
                c, d, a, b, _iters_j2, p, add_ctrl_fn=add_fn,
                double_window=4 if trunc else None,
                carry_window=4 if trunc else None,
                cmp_bits=3 if trunc else None)
            c.rebind("a", a2)
            c.rebind("b", b2)
        return build

    # The threaded chunked adder exists only at jump 2 -- Phase 2 is the only place the
    # peak is tight enough to want it -- so it cannot ride the name substitution above,
    # which mints a jump-1 name for every entry. Two cuts rather than one so the pair is
    # not two copies of the same circuit under different names: chunk 3 and chunk 4 stand
    # in for the real 38 and 44, and at u = 10 both still cross a boundary, which is the
    # only property this test needs of them.
    _replay_adders_j2_only = {
        "inv.apply_dialog_j2_threaded_trunc.v1":
            (functools.partial(gidney_add_ctrl_chunked_threaded, chunk=3, cmp_bits=3),
             True),
        "inv.apply_dialog_j2_threaded_wide_trunc.v1":
            (functools.partial(gidney_add_ctrl_chunked_threaded, chunk=4, cmp_bits=4),
             True),
        # The tuned pair vent at fewer bits than they chunk, which is the whole point of
        # them; the stand-ins keep that apart too, or they would be testing the wrong
        # circuit. cmp_bits 2 against chunk 3 and 4 is the small-width version of 32
        # against 43 and 37.
        "inv.apply_dialog_j2_threaded_tuned_trunc.v1":
            (functools.partial(gidney_add_ctrl_chunked_threaded, chunk=4, cmp_bits=2),
             True),
        "inv.apply_dialog_j2_threaded_tuned_narrow_trunc.v1":
            (functools.partial(gidney_add_ctrl_chunked_threaded, chunk=3, cmp_bits=2),
             True),
    }

    builders.update({n: _make_replay(*v) for n, v in _replay_adders.items()})
    builders.update({n.replace("apply_dialog", "apply_dialog_j2"): _make_replay_j2(*v)
                     for n, v in _replay_adders.items()})
    builders.update({n: _make_replay_j2(*v)
                     for n, v in _replay_adders_j2_only.items()})

    # The two composites. They stand apart from everything above in two ways, and both
    # are why they get their own builders rather than a flag: they take no `d` (Phase 1
    # writes the tape from `a`, which is the whole point of them), and they FREE a
    # register whose value they only know classically, which is a way of dirtying an
    # ancilla that no component here has. The threaded chunked adder is the registered
    # tuning in both, cut to 3 bits so a 10-bit register still crosses a boundary; its
    # boundary vent is the leak being tested.
    #
    # `carry_window` stays exact deliberately. A truncated `+F`/`-F` corrupts the VALUE,
    # and the composite's free depends on the value -- `_clear_constant` X-layers away a
    # `p` that would not be there -- so truncating it would make this test measure the
    # window rather than the vent. A truncated `cmp_bits` is the opposite: `modadd_lazy`
    # vents, so a wrong predicate is phase only and the value survives, which is exactly
    # the claim. Hence the multiply truncates it and the division, whose reverse replay
    # turns that same term into a raise, does not.
    _iters_j2_widths = schedule.widths_j2(u, _iters_j2)
    _chunked = functools.partial(gidney_add_ctrl_chunked_threaded, chunk=3, cmp_bits=2)

    def _make_inplace(fn, replay_kw):
        def build(c, a, b):
            a2, b2 = fn(c, a, b, p, _iters_j2, _iters_j2_widths,
                        gcd_kw=dict(cmp_bits=schedule.compare_bits(u)),
                        replay_kw=dict(add_ctrl_fn=_chunked, **replay_kw))
            c.rebind("a", a2)
            c.rebind("b", b2)
        return build

    _composites = {
        "inv.modmul_inplace.v1": _make_inplace(modmul_inplace, dict(cmp_bits=3)),
        "inv.moddiv_inplace.v1": _make_inplace(moddiv_inplace, {}),
    }

    # The point step, on the same two registers: `a` is x, `b` is y, and the constant point
    # is an arbitrary pair below p -- the circuit is a rational map and never reads the
    # curve equation, so an on-curve constant would not exercise anything extra. The inner
    # tunings are the small-width analogues of the registered ones, and `carry_window`
    # stays exact for the reason the composites give above.
    #
    # The gated version forces its control to |1> with an X layer rather than taking an
    # input, because the loop below only supplies `a` and `b`. ctrl = 1 is the right shot
    # to take anyway: at ctrl = 0 the two expensive regions cancel and the circuit is the
    # identity, which would test nothing about the vent.
    def _make_point(ctrl):
        def build(c, a, b):
            if ctrl:
                g = c.alloc(1)
                c.x(g[0])
            a2, b2 = add_const_affine_inplace(
                c, a, b, 3, 7, g[0] if ctrl else None, p,
                const_kw=dict(add_fn=gidney_add, cmp_bits=3),
                sqr_kw=dict(add_fn=gidney_add, cmp_bits=3, window=4, ripple=4),
                inv_kw=dict(iters=_iters_j2, widths=_iters_j2_widths,
                            gcd_kw=dict(cmp_bits=schedule.compare_bits(u)),
                            replay_kw=dict(add_ctrl_fn=_chunked)))
            if ctrl:
                c.x(g[0])
                c.free(g)
            c.rebind("a", a2)
            c.rebind("b", b2)
        return build

    _composites.update({
        "point.add_const_affine_inplace.v1": _make_point(False),
        "point.add_const_affine_inplace_ctrl.v1": _make_point(True),
        # Small-width stand-ins have no square ripple, so v2 shares v1's model.
        "point.add_const_affine_inplace.v2": _make_point(False),
        "point.add_const_affine_inplace_ctrl.v2": _make_point(True),
    })
    builders.update(_composites)

    assert set(builders) == set(PHASE_TOLERANT), (
        "every phase-tolerant version needs a small-width stand-in here; "
        f"missing {sorted(set(PHASE_TOLERANT) - set(builders))}")
    assert FAILS_CLOSED <= set(builders), "FAILS_CLOSED names must be tested, not skipped"

    for name, build in builders.items():
        c = Circuit()
        a, b = c.input("a", u), c.input("b", u)
        build(c, a, b)
        rng = random.Random(23)
        composite = name in _composites
        needs_tape = name.startswith("inv.") and not composite
        dirty = leaked = 0
        for _ in range(24 if composite else 40 if needs_tape else 400):
            # Both composites require both operands non-zero, and the raise on a zero one
            # is a documented feature of them (see `inversion.inplace`'s header), so
            # drawing one here would be testing the precondition rather than the vent.
            lo = 1 if composite else 0
            av, bv = rng.randrange(lo, p), rng.randrange(lo, p)
            vals = {"a": av, "b": bv}
            if needs_tape:
                # An arbitrary 5-bit group could exceed 26 and break `unfold_digit`'s
                # precondition, so `d` has to be a real tape, not random bits. At jump 2
                # the constraint is tighter, not looser: base 5 has no free digit 0, so a
                # group outside [0, 125) is not merely unhandled but unfoldable.
                x = rng.randrange(1, p)
                if "_j2" in name:
                    tape, _, _ = to_dialog_jump_ref(p, x, _iters_j2)
                    vals["d"] = pack.pack_dialog_j2(tape, boundary=True)
                else:
                    tape, _, _ = to_dialog_ref(p, x, _iters)
                    vals["d"] = pack.pack_dialog(tape)
            try:
                leaked += simulate(c, vals, rng=random.Random(rng.random())).phase
            except DirtyAncillaError:
                dirty += 1

        if name in FAILS_CLOSED:
            # Dirty-impossibility is not this one's property, so assert the one that IS:
            # the leak the allowlist entry claims has to be observable, or the entry is
            # granting a tolerance nothing uses.
            assert leaked, (
                f"{name} is on the allowlist but leaked no phase in this construction; "
                "if it no longer vents anywhere, take it off")
        else:
            assert not dirty, (                        # pragma: no cover - the bug
                f"{name} tolerates phase but dirtied an ancilla on {dirty} shots: that "
                "error is entangled garbage, not a phase, so the allowlist entry is "
                "wrong -- or it belongs in FAILS_CLOSED with a reason")


@pytest.mark.parametrize("name", [
    "field.addconst_19_trunc.v1",
    "field.addconst_p_trunc.v1",
    "field.subconst_p_trunc.v1",
    "field.cmp_lt_trunc.v1",
    "field.cmp_ge_p_trunc.v1",
])
def test_truncated_circuits_never_leak_phase(name):
    """Slice-truncation keeps the circuit an exact unitary, so these must stay clean."""
    import qsm.bench.__main__  # noqa: F401

    r = harness.benchmark(name, n_trials=32, persist=False)
    assert r.phase_seen == 0, f"{name} leaked phase {r.phase_seen} times"
    assert r.correct is True
