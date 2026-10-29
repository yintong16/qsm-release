"""Conditional-tag experiments, including inherited failures, not a tuned proof.

Every symbolic audit quantifies over ALL measurement outcomes. The small-prime
censuses keep every canonical pair (including failures), and compare nonzero
lanes against the ORIGINAL tuned composite at identical tuning. Full-width
smoke cases do not license arbitrary structured numerators or divisors.
"""
import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, DirtyAncillaError
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.inversion.config import InversionConfig
from qsm.inversion.inplace import moddiv_inplace, modmul_inplace
from qsm.inversion.schedule import conservative_iterations_j2
from qsm.inversion.zero_safe import (
    _erase_tag,
    _prepare_tag,
    moddiv_zero_safe_inplace,
    modmul_zero_safe_inplace,
)

P255 = (1 << 255) - 19
NEW = (modmul_zero_safe_inplace, moddiv_zero_safe_inplace)
OLD = (modmul_inplace, moddiv_inplace)


def full_tuned(p):
    """Disable truncation, NOT lazy arithmetic: this is still the tuned family."""
    n = p.bit_length()
    iters = conservative_iterations_j2(n)
    return InversionConfig.tuned(
        n, p, iters=iters, widths=[n] * iters, gcd_cmp_bits=None,
        replay_cmp_bits=None, replay_chunk=None, replay_chunk_cmp_bits=None,
        carry_window=None, double_window=None,
    )


def build(p, reverse=False, *, original=False, config=None):
    c = Circuit()
    x, y = c.input("x", p.bit_length()), c.input("y", p.bit_length())
    x, y = (OLD if original else NEW)[reverse](c, x, y, p, config=config)
    c.rebind("x", x)
    c.rebind("y", y)
    assert c._live_qids == set(x.ids + y.ids)
    return c


def audit(c, p, rows, reverse=False):
    expected = [
        {"x": row["x"], "y": row["y"] * (
            pow(row["x"], -1, p) if reverse else row["x"]) % p}
        for row in rows
    ]
    return simulate_batch(compile_symbolic(c), rows, expected,
                          output_moduli={"x": p, "y": p})


@pytest.mark.parametrize("p", [3, 5, 13, 31])
def test_prepare_tag_every_bitstring_and_no_measurement_outcomes(p):
    n = p.bit_length()
    c = Circuit()
    value = c.input("value", n)
    observed = c.input("tag", n)
    tag = _prepare_tag(c, value, p)
    for a, b in zip(tag, observed):
        c.cx(a, b)
    # Reverse the unitary tag preparation via its independent erase helper.
    _erase_tag(c, tag, value, p)
    assert not any(op.kind == ops.MEASURE for op in c.ops)
    for v in range(1 << n):
        for initial in range(1 << n):
            result = simulate(c, {"value": v, "tag": initial}, measurements=[])
            assert result.outputs == {"value": v, "tag": initial ^ (p if v else 0)}
            assert result.phase == 0 and result.measurements == []
    # The helper treats the raw p bitstring as NONZERO, not as canonical zero.
    assert simulate(c, {"value": p, "tag": 0}).outputs["tag"] == p


@pytest.mark.parametrize("p", [3, 5, 13, 31])
def test_erase_tag_checks_the_entire_promise_and_recycles_workspace(p):
    n = p.bit_length()
    c = Circuit()
    value, tag = c.input("value", n), c.input("tag", n)
    _erase_tag(c, tag, value, p)
    assert c._live_qids == set(value.ids)
    assert c._peak == 2 * n
    assert not any(op.kind == ops.MEASURE for op in c.ops)
    # Observe only surviving input; original ALLOC still loads the tag value.
    c.inputs.pop("tag")
    for v in range(1 << n):
        for t in range(1 << n):
            if t == (p if v else 0):
                result = simulate(c, {"value": v, "tag": t}, measurements=[])
                assert result.outputs == {"value": v} and result.phase == 0
            else:
                with pytest.raises(DirtyAncillaError):
                    simulate(c, {"value": v, "tag": t}, measurements=[])


@pytest.mark.parametrize("n,p", [(2, 3), (5, 29), (255, P255)])
def test_prepare_tag_has_no_persistent_extra_flag_or_new_peak(n, p):
    c = Circuit()
    value = c.input("value", n)
    tag = _prepare_tag(c, value, p)
    assert len(tag) == n
    assert c._live_qids == set(value.ids + tag.ids)
    assert c._peak == 2 * n
    _erase_tag(c, tag, value, p)
    assert c._peak == 2 * n and c._live_qids == set(value.ids)


# good, wrong residue, noncanonical encoding, dirty ancilla, any bad phase.
# These counts overlap. Keeping them is a diagnostic, not filtering bad inputs.
CENSUS = {
    (3, False): (6, 0, 0, 0, 0), (3, True): (6, 0, 0, 0, 0),
    (5, False): (11, 0, 5, 0, 7), (5, True): (11, 4, 4, 9, 0),
    (7, False): (42, 0, 0, 0, 0), (7, True): (42, 0, 0, 0, 0),
    (13, False): (83, 7, 21, 16, 38), (13, True): (83, 28, 17, 69, 0),
    (29, False): (542, 41, 53, 74, 138), (29, True): (542, 100, 41, 254, 0),
    (31, False): (930, 0, 0, 0, 0), (31, True): (930, 0, 0, 0, 0),
}


@pytest.mark.parametrize("p", [3, 5, 7, 13, 29, 31])
@pytest.mark.parametrize("reverse", [False, True])
def test_exhaustive_canonical_pairs_all_outcomes_and_inherited_hazard_census(p, reverse):
    cfg = full_tuned(p)
    rows = [{"x": x, "y": y} for x in range(1, p) for y in range(p)]
    c = build(p, reverse, config=cfg)
    result = audit(c, p, rows, reverse)
    assert (result.good_count, result.value_count, result.noncanonical_count,
            result.dirty_count, result.phase_count) == CENSUS[p, reverse]
    zero_mask = sum(1 << i for i, row in enumerate(rows) if row["y"] == 0)
    assert result.bad_mask & zero_mask == 0  # value, representation, frees AND phase
    nonzero = [row for row in rows if row["y"]]
    baseline = build(p, reverse, original=True, config=cfg)
    assert audit(c, p, nonzero, reverse) == audit(baseline, p, nonzero, reverse)


@pytest.mark.parametrize("reverse,x,y,expected,phase", [
    (False, 1, 7, 7, 1),   # correct value, inherited measurement phase
    (False, 2, 7, 14, 1),  # inherited noncanonical result, not canonical 1
    (True, 6, 1, 5, 0),    # inherited wrong residue: 1/6 mod 13 is 11
])
def test_concrete_nonzero_counterexamples_match_original_measurement_stream(
        reverse, x, y, expected, phase):
    results = [simulate(build(13, reverse, original=old, config=full_tuned(13)),
                        {"x": x, "y": y}, rng=random.Random(0))
               for old in (False, True)]
    assert results[0] == results[1]  # outputs, sampled phase AND every measurement
    assert results[0].outputs == {"x": x, "y": expected}
    assert results[0].phase == phase


@pytest.mark.parametrize("reverse_first", [False, True])
def test_closed_domain_uncompute_including_zero_all_measurement_outcomes(reverse_first):
    p, cfg = 31, full_tuned(31)
    c = Circuit()
    x, y = c.input("x", 5), c.input("y", 5)
    for reverse in (reverse_first, not reverse_first):
        x, y = NEW[reverse](c, x, y, p, config=cfg)
    c.rebind("x", x)
    c.rebind("y", y)
    rows = [{"x": x, "y": y} for x in range(1, p) for y in range(p)]
    result = simulate_batch(compile_symbolic(c), rows, rows,
                            output_moduli={"x": p, "y": p})
    assert result.bad_mask == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_zero_divisor_remains_out_of_domain(reverse):
    c = build(13, reverse, config=full_tuned(13))
    for y in (0, 1):
        with pytest.raises(DirtyAncillaError):
            simulate(c, {"x": 0, "y": y})


@pytest.mark.parametrize("fn", NEW)
def test_reject_canonical_config_and_malformed_operands_before_emitting(fn):
    c = Circuit()
    x, y = c.input("x", 4), c.input("y", 4)
    original = len(c.ops)
    for a, b, cfg in [
        (x, y, InversionConfig.conservative(4, 13)),
        (x, x, full_tuned(13)),
        (x, y[:3], full_tuned(13)),
        (x, y, full_tuned(31)),
    ]:
        with pytest.raises(ValueError):
            fn(c, a, b, 13, config=cfg)
        assert len(c.ops) == original


@pytest.mark.parametrize("fn", NEW)
@pytest.mark.parametrize("condition_value", [0, 1])
def test_reject_outer_classical_condition_before_emitting(fn, condition_value):
    c = Circuit()
    x, y = c.input("x", 3), c.input("y", 3)
    bit = c.classical_bit(condition_value)
    original = len(c.ops)
    live = set(c._live_qids)
    with c.condition(bit):
        with pytest.raises(ValueError, match="classical condition"):
            fn(c, x, y, 7, config=full_tuned(7))
    assert len(c.ops) == original
    assert c._live_qids == live


@pytest.mark.slow
@pytest.mark.parametrize("reverse", [False, True])
def test_full255_zero_phase_cleanup_and_structured_diagnostics_match_baseline(reverse):
    rng = random.Random(0x25519)
    generic_x = [rng.randrange(1, P255) for _ in range(3)]
    xs = generic_x + [1, 2, 3, P255 - 1, 1 << 254]
    ys = [0, 1, 19, 1 << 128, P255 - 1, rng.randrange(1, P255)]
    rows = [{"x": x, "y": y} for x in xs for y in ys]
    nonzero = [row for row in rows if row["y"]]
    # Original config=None takes its legacy kwargs path, NOT the settled tuning.
    # Supply the identical resolved object to both sides of this comparison.
    cfg = InversionConfig.tuned(255, P255)
    c = build(P255, reverse, config=cfg)
    resources = count(c)
    result = audit(c, P255, rows, reverse)
    nonzero_result = audit(c, P255, nonzero, reverse)
    # Every FREE and final live ancilla is checked, including the tape uncompute.
    for i, row in enumerate(rows):
        if row["x"] in generic_x + [1, 2] and row["y"] == 0:
            assert not (result.bad_mask >> i) & 1
            assert result.outputs[i] == row
        if row["x"] in generic_x and row["y"] == ys[-1]:
            assert not (result.bad_mask >> i) & 1
        if row["x"] in generic_x and row["y"] == P255 - 1:
            assert (result.phase_mask >> i) & 1  # inherited, NOT repaired
    # Default GCD tuning also fails for some structured DIVISORS, independently
    # of the zero tag. Keep those zero lanes in the audit instead of excluding them.
    for bad_x in (3, P255 - 1, 1 << 254):
        i = rows.index({"x": bad_x, "y": 0})
        assert (result.dirty_mask >> i) & 1
        assert result.outputs[i]["x"] != bad_x
    del c  # do not retain two multi-million-op circuits at once
    baseline = build(P255, reverse, original=True, config=cfg)
    baseline_resources = count(baseline)
    assert nonzero_result == audit(baseline, P255, nonzero, reverse)
    assert resources.toffoli == baseline_resources.toffoli + 507
    assert resources.peak_qubits == baseline_resources.peak_qubits == 1173
    assert resources.measurements == baseline_resources.measurements
