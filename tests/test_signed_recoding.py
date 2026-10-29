"""Integer identities, endpoint bijection and actual reversible op-stream tests."""
from __future__ import annotations

import json

import pytest

from qsm.audit.signed_recoding import (
    build_recoder_circuit, deterministic_scalars, main, recoder_resource_report,
    schedule_counts, schedule_sweep,
)
from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.scalarmul.signed_recoding import (
    decode_digit, encode_digit, recode_scalar, signed_recode_inplace,
    twos_complement_to_sign_address_inplace,
)


@pytest.mark.parametrize("a", range(1, 6))
def test_endpoint_bijection(a):
    bound = 1 << a
    states = set()
    for digit in range(-bound, bound):
        state = encode_digit(digit, a)
        assert decode_digit(*state, a) == digit
        states.add(state)
    assert len(states) == 2 * bound
    assert encode_digit(0, a) == (0, 0)
    assert encode_digit(-bound, a) == (1, 0)
    assert encode_digit(-1, a) == (1, 1)
    assert encode_digit(bound - 1, a) == (0, bound - 1)


@pytest.mark.parametrize("a", range(1, 6))
@pytest.mark.parametrize("variant", ["carry", "bias"])
def test_integer_identity_exhaustive_through_twelve_bits(a, variant):
    bound, w = 1 << a, a + 1
    for length in range(1, 13):
        q = (length + w - 1) // w
        for scalar in range(1 << length):
            recoded = recode_scalar(scalar, length, a, variant)
            assert len(recoded.digits) == q
            assert all(-bound <= digit < bound for digit in recoded.digits)
            # Independent identity, not only the dataclass reconstruction method.
            weighted = sum(digit * (2 * bound)**i for i, digit in enumerate(recoded.digits))
            correction = (recoded.terminal_carry * (2 * bound)**q if variant == "carry"
                          else bound * sum((2 * bound)**i for i in range(q)))
            assert weighted + correction == scalar == recoded.reconstruct()
            assert recoded.bias == (correction if variant == "bias" else 0)
            if variant == "bias":
                assert recoded.terminal_carry == 0


def test_long_carry_chain_overflow_zero_and_partial_top():
    # Base 8: 7 -> -1,c=1; 7+1=8 -> 0,c=1; top 3+1=4 -> -4,c=1.
    recoded = recode_scalar(7 + 7 * 8 + 3 * 64, 8, 2)
    assert recoded.digits == (-1, 0, -4)
    assert recoded.terminal_carry == 1
    assert recoded.encoded == ((1, 1), (0, 0), (1, 0))
    assert recode_scalar(255, 8, 2, "bias").digits == (3, 3, -1)
    assert schedule_counts(8, 2)["carry"]["terminal_constant_addition_slots"] == 1
    assert schedule_counts(2, 2)["carry"]["terminal_constant_addition_slots"] == 0


@pytest.mark.parametrize("a", range(1, 6))
def test_emitted_address_conversion_exhaustive(a):
    circ = Circuit()
    word = circ.input("word", a + 1)
    twos_complement_to_sign_address_inplace(circ, word)
    forward_ops = len(circ.ops)
    for value in range(1 << (a + 1)):
        digit = value if value < 1 << a else value - (1 << (a + 1))
        sign, address = encode_digit(digit, a)
        result = simulate(circ, {"word": value}, measurements=[])
        assert result.outputs == {"word": address | sign << a}
        assert result.phase == 0
    twos_complement_to_sign_address_inplace(circ, word)
    for value in range(1 << (a + 1)):
        result = simulate(circ, {"word": value}, measurements=[])
        assert result.outputs == {"word": value}
        assert result.phase == 0
    assert forward_ops >= 1
    assert count(circ).measurements == 0


def _expected(scalar, length, a, variant, digits=0, terminal=0):
    reference = recode_scalar(scalar, length, a, variant)
    expected = {"scalar": scalar, "digits": digits ^ reference.packed_digits}
    if variant == "carry":
        expected["terminal"] = terminal ^ reference.terminal_carry
    return expected


@pytest.mark.parametrize("a", [1, 2, 3, 4])
@pytest.mark.parametrize("variant", ["carry", "bias"])
def test_emitted_recode_exhaustive_small(a, variant):
    for length in range(1, 9):
        circ = build_recoder_circuit(length, a, variant)
        assert not circ.exact_blocks
        assert all(op.kind in (ops.ALLOC, ops.FREE, ops.X, ops.CX, ops.CCX) for op in circ.ops)
        for scalar in range(1 << length):
            initial = {"scalar": scalar, "digits": 0}
            if variant == "carry":
                initial["terminal"] = 0
            result = simulate(circ, initial, measurements=[])
            assert result.outputs == _expected(scalar, length, a, variant)
            assert result.phase == 0
        signed_recode_inplace(
            circ, circ.inputs["scalar"], circ.inputs["digits"], a, variant=variant,
            terminal=circ.inputs["terminal"][0] if variant == "carry" else None,
            inverse=True,
        )
        for scalar in range(1 << length):
            initial = {"scalar": scalar, "digits": 0}
            if variant == "carry":
                initial["terminal"] = 0
            result = simulate(circ, initial, measurements=[])
            assert result.outputs == initial
            assert result.phase == 0


@pytest.mark.parametrize("variant", ["carry", "bias"])
def test_emitted_xor_oracle_arbitrary_destinations(variant):
    circ = build_recoder_circuit(3, 1, variant)
    for scalar in range(8):
        for digits in range(16):
            for terminal in (range(2) if variant == "carry" else (0,)):
                initial = {"scalar": scalar, "digits": digits}
                if variant == "carry":
                    initial["terminal"] = terminal
                result = simulate(circ, initial, measurements=[])
                assert result.outputs == _expected(scalar, 3, 1, variant, digits, terminal)
                assert result.phase == 0


@pytest.mark.parametrize("variant", ["carry", "bias"])
def test_twelve_bit_emitted_compute_uncompute_frees_all_outputs(variant):
    forward = build_recoder_circuit(12, 2, variant)
    for value in range(1 << 12):
        initial = {"scalar": value, "digits": 0}
        if variant == "carry":
            initial["terminal"] = 0
        result = simulate(forward, initial, measurements=[])
        assert result.outputs == _expected(value, 12, 2, variant)
        assert result.phase == 0
    circ = Circuit()
    scalar = circ.input("scalar", 12)
    digits = circ.alloc(12)
    terminal = circ.alloc(1) if variant == "carry" else None
    for inverse in (False, True):
        signed_recode_inplace(circ, scalar, digits, 2, variant=variant,
                              terminal=terminal[0] if terminal is not None else None,
                              inverse=inverse)
    circ.free(digits)
    if terminal is not None:
        circ.free(terminal)
    for value in range(1 << 12):
        result = simulate(circ, {"scalar": value}, measurements=[])
        assert result.outputs == {"scalar": value}
        assert result.phase == 0


@pytest.mark.parametrize("length", [252, 253, 255, 256])
@pytest.mark.parametrize("a", [15, 16, 17])
def test_full_size_edges_schedules_and_actual_resources(length, a):
    ledger = schedule_counts(length, a)
    w, q = a + 1, (length + a) // (a + 1)
    assert ledger["unsigned"]["radix_slots"] == (length + a - 1) // a
    assert ledger["carry"]["radix_slots"] == ledger["bias"]["radix_slots"] == q
    assert ledger["carry"]["terminal_constant_addition_slots"] == recode_scalar(
        (1 << length) - 1, length, a).terminal_carry
    for variant in ("carry", "bias"):
        for scalar in deterministic_scalars(length, a):
            assert recode_scalar(scalar, length, a, variant).reconstruct() == scalar
        report = recoder_resource_report(length, a, variant)
        # Independent count from the emitted CDKM/conversion decomposition.
        expected_ccx = 4 * a * q + (8 * (w + 1) * (q - 1) if variant == "carry" else 0)
        assert report["forward"]["toffoli"] == expected_ccx
        assert report["compute_uncompute"]["toffoli"] == 2 * expected_ccx
        persistent = length + q * w + int(variant == "carry")
        scratch = q + 2 * (w + 1) if variant == "carry" else w + a
        assert report["persistent_input_and_output_qubits"] == persistent
        assert report["temporary_scratch_peak"] == scratch
        assert report["forward"]["peak_qubits"] == persistent + scratch
        assert report["compute_uncompute"]["peak_qubits"] == persistent + scratch
        assert report["forward"]["measurements"] == 0
        assert report["diagnostic_scalars"] > 8
        assert report["validated"]
        json.dumps(report)


def test_reference_schedule_and_initialization_obligations():
    rows = schedule_sweep()
    assert len(rows) == 12
    row = schedule_counts(256, 16)
    assert row["unsigned"]["radix_slots"] == 16
    assert row["carry"]["radix_slots"] == 16
    assert row["carry"]["top_chunk_bits"] == 1
    assert row["carry"]["terminal_constant_addition_slots"] == 0
    row = schedule_counts(255, 16)
    assert row["carry"]["radix_slots"] == 15
    assert row["carry"]["terminal_constant_addition_slots"] == 1
    assert row["bias"]["terminal_constant_addition_slots"] == 0
    policies = row["bias"]["initialization_policies"]
    assert policies["known_bias_point"]["signed_addition_slots"] == 15
    assert policies["absorbed_first_digit"]["signed_addition_slots"] == 14
    assert policies["absorbed_first_digit"]["initialization_lookup_address_bits"] == 17
    assert policies["arbitrary_accumulator"]["known_point_correction_slots"] == 1


@pytest.mark.parametrize("length", range(1, 9))
@pytest.mark.parametrize("a", range(1, 5))
def test_terminal_slot_is_exact_domain_maximum(length, a):
    possible = max(recode_scalar(scalar, length, a).terminal_carry for scalar in range(1 << length))
    assert schedule_counts(length, a)["carry"]["terminal_constant_addition_slots"] == possible


def test_audit_cli_json(capsys):
    main(["--scalar-bits", "5", "8", "--address-bits", "2", "--resources"])
    report = json.loads(capsys.readouterr().out)
    assert len(report["schedules"]) == 2
    assert len(report["recoders"]) == 4
    assert all(row["validated"] for row in report["recoders"])


@pytest.mark.parametrize("args", [(-1, 3, 2), (8, 3, 2), (0, 0, 2), (0, 3, 0)])
def test_invalid_scalar_domain(args):
    with pytest.raises(ValueError):
        recode_scalar(*args)


def test_invalid_encodings_and_configuration():
    for digit in (-5, 4):
        with pytest.raises(ValueError):
            encode_digit(digit, 2)
    with pytest.raises(ValueError):
        decode_digit(2, 0, 2)
    with pytest.raises(ValueError):
        decode_digit(0, 4, 2)
    with pytest.raises(TypeError):
        recode_scalar(True, 4, 2)
    with pytest.raises(TypeError):
        recode_scalar(0, True, 2)
    with pytest.raises(ValueError):
        recode_scalar(0, 4, 2, "unknown")


def test_invalid_register_contract_is_rejected_before_emission():
    circ = Circuit()
    scalar, digits, terminal = circ.input("scalar", 5), circ.input("digits", 6), circ.input("terminal", 1)
    initial_ops = len(circ.ops)
    invalid_calls = [
        lambda: signed_recode_inplace(circ, scalar, digits, 2),
        lambda: signed_recode_inplace(circ, scalar, digits, 2, variant="bias", terminal=terminal[0]),
        lambda: signed_recode_inplace(circ, scalar, digits[:5], 2, terminal=terminal[0]),
        lambda: signed_recode_inplace(circ, scalar, digits, 2, terminal=scalar[0]),
        lambda: signed_recode_inplace(circ, scalar, Register([99, *digits.ids[1:]]), 2,
                                      terminal=terminal[0]),
    ]
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()
        assert len(circ.ops) == initial_ops
