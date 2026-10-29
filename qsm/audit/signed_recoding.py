"""Stage-A schedules and emitted recoder audits, independent of point windows.

Run ``python -m qsm.audit.signed_recoding`` for pure schedule JSON; add
``--resources`` for physical recoder counts and deterministic simulations.
These are not complete scalar-multiplication estimates or full-window runs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random

from ..core.circuit import Circuit
from ..core.resources import count
from ..core.simulator import simulate
from ..scalarmul.signed_recoding import (
    _parameters, recode_scalar, signed_recode_inplace,
)


SCALAR_LENGTHS = (252, 253, 255, 256)
ADDRESS_WIDTHS = (15, 16, 17)


def schedule_counts(scalar_bits: int, address_bits: int) -> dict:
    """Pure single-scalar slot ledger on the domain [0,2**L), not a gate estimate.

    Terminal slots are worst-case coherent slots, not average nonzero digits.
    A partial top chunk of a bits can still carry when a lower chunk carries.
    Initialization is explicitly separate; the ledger does not subtract a first
    addition while also pretending bias initialization is free.
    """
    _parameters(scalar_bits, address_bits, "carry")
    a, w = address_bits, address_bits + 1
    unsigned_q = (scalar_bits + a - 1) // a
    q = (scalar_bits + w - 1) // w
    maximum = recode_scalar((1 << scalar_bits) - 1, scalar_bits, a, "carry")
    bias = recode_scalar(0, scalar_bits, a, "bias").bias
    common = {"radix_bits": w, "radix_slots": q,
              "top_chunk_bits": scalar_bits - (q - 1) * w}
    return {
        "scalar_bits": scalar_bits, "address_bits": a,
        "scalar_domain": {"minimum": 0, "maximum_exclusive": 1 << scalar_bits},
        "magnitude_table_entries": 1 << a,
        "unsigned": {
            "radix_bits": a, "radix_slots": unsigned_q,
            "top_chunk_bits": scalar_bits - (unsigned_q - 1) * a,
            "terminal_constant_addition_slots": 0,
            "initialization": "not selected; radix slots include the first slot",
        },
        "carry": {
            **common, "terminal_constant_addition_slots": maximum.terminal_carry,
            "initialization": "not selected; radix slots include the first slot",
        },
        "bias": {
            **common, "terminal_constant_addition_slots": 0, "offset": bias,
            "initialization_policies": {
                "known_bias_point": {
                    "signed_addition_slots": q, "initialization_lookup_slots": 0,
                    "obligation": "prepare the public finite bias point; no first-addition saving",
                },
                "absorbed_first_digit": {
                    "signed_addition_slots": q - 1, "initialization_lookup_slots": 1,
                    "initialization_lookup_address_bits": w,
                    "obligation": "budget the full signed first-digit table and validate exceptions",
                },
                "arbitrary_accumulator": {
                    "signed_addition_slots": q, "known_point_correction_slots": 1,
                    "obligation": "add the bias point explicitly; it is not free initialization",
                },
            },
        },
        "scope": "single scalar, no point costs, no omitted additions, no selected initialization",
    }


def schedule_sweep(
    scalar_lengths=SCALAR_LENGTHS, address_widths=ADDRESS_WIDTHS,
) -> list[dict]:
    return [schedule_counts(length, width)
            for length in scalar_lengths for width in address_widths]


def deterministic_scalars(
    scalar_bits: int, address_bits: int, *, samples: int = 8, seed: int = 2026092901,
) -> tuple[int, ...]:
    """Edges, isolated digit boundaries, long carries and fixed random diagnostics."""
    _parameters(scalar_bits, address_bits, "carry")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 0:
        raise ValueError("samples must be a nonnegative integer")
    w, bound = address_bits + 1, 1 << address_bits
    maximum = (1 << scalar_bits) - 1
    values = {0, 1, maximum, maximum - 1, 1 << (scalar_bits - 1)}
    q = (scalar_bits + w - 1) // w
    for i in range(q):
        for digit in (bound - 1, bound, bound + 1, 2 * bound - 1):
            values.add((digit << (i * w)) & maximum)
        # All-one lower chunks create long carry chains and t=2B.
        values.add(((1 << min((i + 1) * w, scalar_bits)) - 1) & maximum)
    rng = random.Random(seed)
    values.update(rng.getrandbits(scalar_bits) for _ in range(samples))
    return tuple(sorted(values))


def build_recoder_circuit(scalar_bits: int, address_bits: int, variant: str = "carry") -> Circuit:
    """Build one actual XOR recoder with named scalar/digits[/terminal] inputs."""
    _parameters(scalar_bits, address_bits, variant)
    w = address_bits + 1
    q = (scalar_bits + w - 1) // w
    circ = Circuit()
    scalar = circ.input("scalar", scalar_bits)
    digits = circ.input("digits", q * w)
    terminal = circ.input("terminal", 1)[0] if variant == "carry" else None
    signed_recode_inplace(circ, scalar, digits, address_bits, variant=variant, terminal=terminal)
    return circ


def recoder_resource_report(
    scalar_bits: int, address_bits: int, variant: str = "carry", *, validate: bool = True,
) -> dict:
    """JSON-ready emitted forward/compute+uncompute resources, including input wires.

    Simulations execute the physical op stream (not semantic blocks), demand clean
    FREEs and zero phase, and cover the deterministic full-width diagnostic set.
    No point consumer intervenes in the compute/uncompute measurement here.
    """
    circ = build_recoder_circuit(scalar_bits, address_bits, variant)
    forward = count(circ)
    persistent = sum(len(reg) for reg in circ.inputs.values())
    values = deterministic_scalars(scalar_bits, address_bits) if validate else ()
    for scalar in values:
        initial = {"scalar": scalar, "digits": 0}
        reference = recode_scalar(scalar, scalar_bits, address_bits, variant)
        expected = {"scalar": scalar, "digits": reference.packed_digits}
        if variant == "carry":
            initial["terminal"] = 0
            expected["terminal"] = reference.terminal_carry
        result = simulate(circ, initial, measurements=[])
        if result.outputs != expected or result.phase != 0 or result.measurements:
            raise AssertionError(f"forward recoder failed at scalar={scalar}")
    signed_recode_inplace(
        circ, circ.inputs["scalar"], circ.inputs["digits"], address_bits,
        variant=variant,
        terminal=circ.inputs["terminal"][0] if variant == "carry" else None,
        inverse=True,
    )
    for scalar in values:
        initial = {"scalar": scalar, "digits": 0}
        if variant == "carry":
            initial["terminal"] = 0
        result = simulate(circ, initial, measurements=[])
        if result.outputs != initial or result.phase != 0 or result.measurements:
            raise AssertionError(f"inverse recoder failed at scalar={scalar}")
    return {
        "scalar_bits": scalar_bits, "address_bits": address_bits, "variant": variant,
        "forward": asdict(forward), "compute_uncompute": asdict(count(circ)),
        "persistent_input_and_output_qubits": persistent,
        "digit_output_qubits": len(circ.inputs["digits"]),
        "terminal_output_qubits": int(variant == "carry"),
        "temporary_scratch_peak": forward.peak_qubits - persistent,
        "diagnostic_scalars": len(values), "validated": validate,
        "measurement_outcomes": "none: X/CX/CCX only, structurally phase-free",
        "scope": "full scalar preserved plus full digit output; no streaming or point arithmetic",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-bits", type=int, nargs="+", default=SCALAR_LENGTHS)
    parser.add_argument("--address-bits", type=int, nargs="+", default=ADDRESS_WIDTHS)
    parser.add_argument("--resources", action="store_true",
                        help="emit, count and simulate recoders; never build point windows")
    args = parser.parse_args(argv)
    report = {"schedules": schedule_sweep(args.scalar_bits, args.address_bits)}
    if args.resources:
        report["recoders"] = [recoder_resource_report(length, width, variant)
                              for length in args.scalar_bits for width in args.address_bits
                              for variant in ("carry", "bias")]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
