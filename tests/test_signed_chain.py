"""Stage D SMALL gate chains; phase faults are exposed, never postselected away.

These tests establish composed group action and ancilla cleanup on a stated
finite-coordinate domain. They also retain explicit inherited phase witnesses:
passing this file is NOT a claim that the tuned small chains are coherent oracles.
"""
from collections import Counter
import random

import pytest

from qsm.audit.signed_chain import (
    _exact_config, build_small_signed_chain, group_add, group_mul,
)
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.scalarmul.signed_recoding import recode_scalar
from qsm.scalarmul.signed_window import add_signed_window_inplace
from qsm.scalarmul.window import WindowTable
from qsm.scalarmul.window_opt import add_window_affine_optimized_inplace


def _population(chain, multiples=(7, 19, 37)):
    """Enumerate, then reject ONLY ideal-domain exceptions, before any simulation."""
    admitted, exclusions = [], Counter()
    for multiple in multiples:
        point = group_mul(multiple, chain.base, chain.p, chain.a)
        assert point is not None
        for scalar in range(1 << chain.scalar_bits):
            row = dict(scalar=scalar, x=point[0], y=point[1])
            reason = chain.domain_exclusion(row)
            if reason is None:
                admitted.append(row)
            else:
                exclusions[reason] += 1
    assert len(admitted) + sum(exclusions.values()) == len(multiples)*(1 << chain.scalar_bits)
    return admitted, exclusions


def _assert_structure(chain):
    circ, meta = chain.circuit, chain.metadata
    named = {q for register in circ.inputs.values() for q in register}
    assert circ._live == chain.scalar_bits + 2*chain.p.bit_length()
    assert circ._live_qids == named
    assert not named.intersection(meta["recoder_outputs_freed"])
    assert len(circ._free_qids) == len(set(circ._free_qids))
    scalar_alloc = next(op for op in circ.ops if op.kind == "ALLOC" and op.name == "scalar")
    assert tuple(circ.inputs["scalar"]) == scalar_alloc.qubits
    assert count(circ).peak_qubits == circ.peak_qubits
    assert count(circ).peak_qubits >= circ._live + len(meta["recoder_outputs_freed"])
    assert meta["scalar_and_recoder_wires_included"]
    for key in ("recode_range", "cleanup_range"):
        start, stop = meta[key]
        kinds = {op.kind for op in circ.ops[start:stop]}
        assert kinds <= {"X", "CX", "CCX", "ALLOC", "FREE"}
        assert "CX" in kinds
    for start, stop in meta["addition_ranges"]:
        assert any(op.kind == "MEASURE" for op in circ.ops[start:stop])
        assert any(op.kind == "CCX" for op in circ.ops[start:stop])
    assert len(meta["addition_ranges"]) >= 2
    freed = set()
    for op in circ.ops[meta["cleanup_range"][0]:]:
        if op.kind == "FREE":
            freed.update(op.qubits)
    assert set(meta["recoder_outputs_freed"]) <= freed
    inv = meta["inversion"]
    assert inv["iters"] == 2*chain.p.bit_length() + 1
    assert inv["widths"] == (chain.p.bit_length(),)*inv["iters"]
    assert all(inv[key] is None for key in (
        "gcd_cmp_bits", "replay_cmp_bits", "replay_chunk",
        "replay_chunk_cmp_bits", "carry_window", "double_window"))
    assert meta["const_kw"] == meta["sqr_kw"] == {}
    assert not meta["first_addition_omitted"]
    assert not meta["input_preparation_cost_included"]
    assert "SMALL" in meta["scope"] and "no full-size counts" in meta["scope"]
    if chain.variant == "carry":
        assert meta["terminal_constant_slots"] == 1
        start, stop = meta["terminal_range"]
        assert any(op.kind == "CCX" for op in circ.ops[start:stop])
    else:
        assert meta["terminal_constant_slots"] == 0 and meta["terminal_range"] is None
        assert not meta["bias_preparation_emitted"]
        assert not meta["bias_preparation_cost_included"]


def _diagnose(chain, admitted, exclusions, *, random_shots=2):
    expected = [chain.reference(row) for row in admitted]
    # No lane is removed on the basis of any output/phase/dirty-free result.
    symbolic = simulate_batch(compile_symbolic(chain.circuit), admitted, expected,
                              output_moduli={"x": chain.p, "y": chain.p})
    assert symbolic.value_count == 0
    assert symbolic.dirty_count == 0
    assert symbolic.noncanonical_count == 0
    assert symbolic.summary_mismatch_count == 0  # Exact words restored before measured QROM cleanup.
    assert symbolic.constant_phase_count == 0
    observed_phase_mask = 0
    rng = random.Random(2026092901)
    for lane, (row, oracle) in enumerate(zip(admitted, expected)):
        for _ in range(random_shots):
            got = simulate(chain.circuit, row, rng=rng)
            # F=1 plus nonzero boundary domain gives canonical, not just modular, outputs.
            assert got.outputs == oracle
            assert got.outputs["scalar"] == row["scalar"]
            if got.phase:
                observed_phase_mask |= 1 << lane
            if not (symbolic.phase_mask >> lane) & 1:
                assert got.phase == 0
    assert not observed_phase_mask & ~symbolic.phase_mask
    resource = count(chain.circuit)
    print(f"SMALL {chain.variant}/{chain.metadata['sign_strategy']} "
          f"L={chain.scalar_bits},a={chain.address_bits}: "
          f"admitted={len(admitted)}, excluded={sum(exclusions.values())}, "
          f"all-outcome-phase-hazards={symbolic.phase_count}, "
          f"random-phase-witnesses={observed_phase_mask.bit_count()}, "
          f"CCX={resource.toffoli}, actual-global-peak={resource.peak_qubits}; "
          f"exclusions={dict(exclusions)}")
    return symbolic, observed_phase_mask


@pytest.mark.parametrize("variant", ["carry", "bias"])
@pytest.mark.parametrize("sign_strategy", ["conjugate", "selected"])
def test_two_window_all_small_scalars_group_action_cleanup_and_random_phases(variant, sign_strategy):
    chain = build_small_signed_chain(variant=variant, sign_strategy=sign_strategy)
    _assert_structure(chain)
    rows, exclusions = _population(chain)
    assert {row["scalar"] for row in rows} == set(range(16))
    assert len(rows) == {"carry": 41, "bias": 43}[variant]
    assert sum(exclusions.values()) == {"carry": 7, "bias": 5}[variant]
    symbolic, observed = _diagnose(chain, rows, exclusions)
    # The limitation is intentional and visible, not xfail or phase-based resampling.
    assert symbolic.phase_count == (2 if sign_strategy == "conjugate" else 1)
    assert observed != 0
    recodings = [recode_scalar(row["scalar"], 4, 1, variant) for row in rows]
    assert {digit for coding in recodings for digit in coding.digits} == {-2, -1, 0, 1}
    if variant == "carry":
        assert {coding.terminal_carry for coding in recodings} == {0, 1}
        assert chain.metadata["bias"] == 0
    else:
        assert chain.bias == 10
        row = rows[0]
        point = (row["x"], row["y"])
        logical_point = group_add(point, group_mul(-10, chain.base, 127, 3), 127, 3)
        assert logical_point != point
        expected = group_add(logical_point, group_mul(row["scalar"], chain.base, 127, 3), 127, 3)
        assert expected == (chain.reference(row)["x"], chain.reference(row)["y"])


@pytest.mark.parametrize("variant", ["carry", "bias"])
@pytest.mark.parametrize("scalar_bits,address_bits", [(4, 2), (5, 1)])
def test_partial_top_chunks_and_three_window_chains(variant, scalar_bits, address_bits):
    chain = build_small_signed_chain(scalar_bits, address_bits, variant=variant)
    _assert_structure(chain)
    rows, exclusions = _population(chain, multiples=(7, 19))
    assert rows and exclusions
    symbolic, _ = _diagnose(chain, rows, exclusions, random_shots=1)
    assert symbolic.samples == len(rows)
    assert chain.metadata["radix_slots"] == (2 if address_bits == 2 else 3)
    recodings = [recode_scalar(k, scalar_bits, address_bits, variant)
                 for k in range(1 << scalar_bits)]
    if variant == "carry":
        if address_bits == 2:
            assert all(coding.terminal_carry == 0 for coding in recodings)
        else:
            # At a=1 the top bit plus an incoming carry can reach B=2.
            assert {coding.terminal_carry for coding in recodings} == {0, 1}
            assert recodings[-1].terminal_carry == 1
        # Even a known-zero carry still pays a physical constant-addition slot.
        assert chain.metadata["terminal_constant_slots"] == 1
    else:
        assert chain.bias == (36 if address_bits == 2 else 42)


def test_phase_witness_is_also_present_in_matched_positive_constant_arithmetic():
    """Carry k=6,R=(69,67) / bias k=0,R=(69,67) reach this second slot."""
    chain = build_small_signed_chain()
    first, second = chain.tables
    assert group_add((69, 67), first.operand(0, 1), 127, 3) == (70, 106)
    point, sign, address = (70, 106), 1, 0
    operand = second.points[address]
    assert operand == (120, 23)
    signed_output = group_add(point, second.operand(address, sign), 127, 3)
    c = Circuit()
    j, s = c.input("j", 1), c.input("s", 1)
    x, y = c.input("x", 7), c.input("y", 7)
    signed_regions = []
    add_signed_window_inplace(c, x, y, j, s, second, config=_exact_config(127, 1),
                              const_kw={}, sqr_kw={}, regions=signed_regions)
    row = dict(j=address, s=sign, x=point[0], y=point[1])
    oracle = dict(j=address, s=sign, x=signed_output[0], y=signed_output[1])
    signed = simulate_batch(compile_symbolic(c), [row], [oracle])
    assert signed.phase_count == 1 and signed.value_count == signed.dirty_count == 0

    # An actual unsigned one-bit circuit, on the ideal conjugated accumulator.
    positive = Circuit()
    ctrl = positive.input("ctrl", 1)
    xx, yy = positive.input("x", 7), positive.input("y", 7)
    positive_regions = []
    add_window_affine_optimized_inplace(
        positive, xx, yy, ctrl, WindowTable(127, 3, 4, (None, operand)),
        config=_exact_config(127, 1), const_kw={}, sqr_kw={}, regions=positive_regions)
    mapped = (70, 21)
    mapped_output = group_add(mapped, operand, 127, 3)
    positive_row = dict(ctrl=1, x=mapped[0], y=mapped[1])
    positive_oracle = dict(ctrl=1, x=mapped_output[0], y=mapped_output[1])
    unsigned = simulate_batch(compile_symbolic(positive), [positive_row], [positive_oracle])
    assert unsigned.phase_count == 1 and unsigned.value_count == unsigned.dirty_count == 0

    def arithmetic_measurements(circ, regions):
        indices = {index for label, start, stop in regions if label == "arithmetic"
                   for index in range(start, stop)}
        return [mi for mi, oi in enumerate(i for i, op in enumerate(circ.ops)
                                           if op.kind == "MEASURE") if oi in indices]

    signed_positions = arithmetic_measurements(c, signed_regions)
    positive_positions = arithmetic_measurements(positive, positive_regions)
    assert len(signed_positions) == len(positive_positions)
    rng = random.Random(2026092901)
    phases = set()
    for _ in range(12):
        signed_m = [rng.randrange(2) for op in c.ops if op.kind == "MEASURE"]
        positive_m = [0 for op in positive.ops if op.kind == "MEASURE"]
        shared = [rng.randrange(2) for _ in signed_positions]
        for si, pi, bit in zip(signed_positions, positive_positions, shared):
            signed_m[si] = positive_m[pi] = bit
        got_signed = simulate(c, row, measurements=signed_m)
        got_positive = simulate(positive, positive_row, measurements=positive_m)
        assert got_signed.outputs == oracle and got_positive.outputs == positive_oracle
        assert got_signed.phase == got_positive.phase
        phases.add(got_signed.phase)
    assert phases == {0, 1}
    print("Inherited phase witness: second signed slot input=(70,106), sign=1,address=0; "
          "matched positive constant input=(70,21), ctrl=1, operand=(120,23); "
          "symbolic hazard in BOTH, equal random phases with aligned arithmetic outcomes")


@pytest.mark.parametrize("kwargs,match", [
    ({"scalar_bits": 2}, "at least two"),
    ({"scalar_bits": 13}, "SMALL"),
    ({"scalar_bits": True}, "SMALL"),
    ({"address_bits": 4}, "SMALL"),
    ({"p": 2047}, "SMALL"),
    ({"p": 63}, "prime"),
    ({"p": 29}, "F=1"),
    ({"variant": "unsigned"}, "variant"),
    ({"p": 31, "a": 2, "b": 3, "base": (3, 6)}, "terminal carry point is infinity"),
])
def test_small_scope_and_unsupported_domains_rejected(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        build_small_signed_chain(**kwargs)


def test_reference_validates_domain_without_simulation_or_recoding(monkeypatch):
    chain = build_small_signed_chain()
    import qsm.audit.signed_chain as module
    monkeypatch.setattr(module, "recode_scalar", lambda *a, **k: pytest.fail("oracle used recoder"))
    row = {"scalar": 6, "x": 69, "y": 67}
    assert chain.reference(row)["scalar"] == 6
    for changes, message in [({"scalar": 16}, "scalar"), ({"x": 127}, "canonical"),
                             ({"y": 0}, "on the declared curve")]:
        with pytest.raises(ValueError, match=message):
            chain.reference(dict(row, **changes))
