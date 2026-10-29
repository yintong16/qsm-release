"""Streamed double-scalar construction with actual gate/lifetime accounting.

Fourier rotations are explicit unsynthesized feedback phase gates. Large-basis
checks execute the deferred arithmetic oracle, not a full Shor statevector.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import time

from ..bench.calibration import source_digest
from ..core.circuit import Circuit, DirtyAncillaError, Register
from ..core.ops import Op
from ..inversion.config import InversionConfig
from ..params import FIELD, PRIME, U
from ..point import active as curve
from ..point.group import scalar_mul
from ..scalarmul.local_signed import bias_recode_inplace
from ..scalarmul.lookup import lookup_xor_unary
from ..scalarmul.odd_window import make_odd_window_table
from ..scalarmul.zero_point import add_zero_point_inplace, zero_point_reference
from ..scalarmul.signed_window import SignedWindowTable, add_signed_window_inplace, signed_window_reference
from ..scalarmul.window import WindowTable, add_window_affine_inplace_ref
from ..scalarmul.window_opt import OptimizedWindowConfig, add_window_affine_optimized_inplace
from .signed_chain import group_add
from .streamed_ir import BasisAccumulator, FeedbackPhaseOp, ResourceAccumulator


SIGNED_VARIANTS = ("signed", "signed_zero_safe")
ODD_VARIANTS = ("odd_torsion", "odd_torsion_add")
VARIANTS = ("unsigned", *SIGNED_VARIANTS, *ODD_VARIANTS)


WEI25519_TORSION = (
    19624287790469256669057814461137428606839005560001276145469620721820115669041,
    25869741026945134960544184956460972567356779614910045322022475500191642319642,
)


@dataclass(frozen=True)
class Group:
    p: int
    a: int
    b: int
    order: int
    P: tuple[int, int]
    Q: tuple[int, int]
    name: str = "explicit"
    torsion: tuple[int, int] | None = None

    @property
    def n(self):
        return self.p.bit_length()

    def validate(self):
        if self.p < 5 or self.order < 3 or self.order % 2 == 0:
            raise ValueError("odd group order and odd-prime field required")
        if (4*self.a**3 + 27*self.b**2) % self.p == 0:
            raise ValueError("singular curve")
        for point in (self.P, self.Q):
            x, y = point
            if not 0 <= x < self.p or not 0 <= y < self.p:
                raise ValueError("noncanonical public point")
            if (y*y-x*x*x-self.a*x-self.b) % self.p:
                raise ValueError("public point not on curve")
            if scalar_mul(self.order, x, y, p=self.p, a=self.a) is not None:
                raise ValueError("public point outside declared subgroup")
        if self.torsion is not None:
            self.validate_torsion()

    def validate_torsion(self):
        if self.torsion is None:
            raise ValueError("odd_torsion requires an explicit order-eight offset")
        x, y = self.torsion
        if not 0 <= x < self.p or not 0 < y < self.p:
            raise ValueError("torsion offset must be canonical with nonzero y")
        if (y*y-x*x*x-self.a*x-self.b) % self.p:
            raise ValueError("torsion offset not on curve")
        # Subgroup-reduced Group.mul cannot establish a torsion point's order.
        mul = lambda k: scalar_mul(k, x, y, p=self.p, a=self.a)
        if mul(8) is not None or mul(4) is None or mul(self.order) is None:
            raise ValueError("torsion offset must have exact order eight outside the subgroup")

    def mul(self, k, point):
        return scalar_mul(k % self.order, *point, p=self.p, a=self.a)

    def add(self, left, right):
        return group_add(left, right, self.p, self.a)


def active_group(*, torsion=False) -> Group:
    if torsion and FIELD != "25519":
        raise ValueError("the active curve has no supported order-eight offset")
    scalar = int.from_bytes(hashlib.sha256(b"qsm public Q fixture 20260929").digest(), "big") % (curve.ORDER-1) + 1
    q = curve.mul(scalar, curve.GX, curve.GY)
    return Group(PRIME, curve.CURVE.A, curve.CURVE.B, curve.ORDER,
                 (curve.GX, curve.GY), q, FIELD, WEI25519_TORSION if torsion else None)


def windows(u_bits: int, v_bits: int, window_bits: int) -> list[dict]:
    if any(type(v) is not int or v < 1 for v in (u_bits, v_bits, window_bits)):
        raise ValueError("positive integer scalar and window widths required")
    if window_bits < 2 or window_bits > 17:
        raise ValueError("window width must be from two through seventeen")
    result = []
    for axis, length in (("u", u_bits), ("v", v_bits)):
        for offset in reversed(range(0, length, window_bits)):
            result.append({"axis": axis, "offset": offset,
                           "bits": min(window_bits, length-offset)})
    return result


def shift_point(group: Group, schedule: list[dict], variant: str):
    if variant not in VARIANTS:
        raise ValueError("unknown scalar variant")
    if variant in ODD_VARIANTS:
        group.validate_torsion()
        shift = group.torsion
        inverse_two = pow(2, -1, group.order)
        for slot in schedule:
            base = group.P if slot["axis"] == "u" else group.Q
            bias = -((1 << slot["bits"])-1)*inverse_two*(1 << slot["offset"])
            shift = group.add(shift, group.mul(bias, base))
        return shift
    shift = group.mul(pow(2, -1, group.order), group.P)
    if variant in SIGNED_VARIANTS:
        for slot in schedule:
            base = group.P if slot["axis"] == "u" else group.Q
            shift = group.add(shift, group.mul(-(1 << (slot["bits"]-1+slot["offset"])), base))
    return shift


def _successive(group: Group, first, step, length):
    points = []
    point = first
    for _ in range(length):
        if point is None:
            raise ValueError("classical table contains infinity")
        points.append(point)
        point = group.add(point, step)
    return tuple(points)


def _initial_points(group, slot, variant):
    weighted = group.mul(1 << slot["offset"], group.P)
    if variant in ODD_VARIANTS:
        group.validate_torsion()
        bias = -((1 << slot["bits"])-1)*pow(2, -1, group.order)
        first = group.add(group.torsion, group.mul(bias, weighted))
        return _successive(group, first, weighted, 1 << slot["bits"])
    half = group.mul(pow(2, -1, group.order), group.P)
    first = half
    if variant in SIGNED_VARIANTS:
        first = group.add(first, group.mul(-(1 << (slot["bits"]-1)), weighted))
    return _successive(group, first, weighted, 1 << slot["bits"])


def _table(group, slot, variant):
    weighted = group.mul(1 << slot["offset"], group.P if slot["axis"] == "u" else group.Q)
    if variant in ODD_VARIANTS:
        return make_odd_window_table(slot["bits"], weighted, order=group.order,
                                     p=group.p, a=group.a, b=group.b)
    if variant == "unsigned":
        pts = (None,) + _successive(group, weighted, weighted, (1 << slot["bits"])-1)
        return WindowTable(group.p, group.a, group.b, pts)
    size = 1 << (slot["bits"]-1)
    pts = _successive(group, weighted, weighted, size)
    if size == 1:
        return pts[0]
    return SignedWindowTable(group.p, group.a, group.b, (pts[-1],)+pts[:-1])


def _config(group, bits, full_width):
    if full_width:
        iterations = 2*group.n+1
        inversion = InversionConfig.tuned(group.n, group.p, iters=iterations,
            widths=(group.n,)*iterations, gcd_cmp_bits=None, replay_cmp_bits=None,
            replay_chunk=None, replay_chunk_cmp_bits=None, carry_window=None, double_window=None)
        return OptimizedWindowConfig.tuned(group.n, group.p, bits, inversion=inversion, adder_chunk=None)
    return OptimizedWindowConfig.tuned(group.n, group.p, bits)


def _fingerprint(points):
    return hashlib.sha256(json.dumps(points, separators=(",", ":")).encode()).hexdigest()


def _drop_log(circ):
    circ.ops.clear()
    circ.exact_blocks.clear()
    circ._exact_block_seals.clear()


def run(*, group: Group | None = None, u_bits=255, v_bits=255, window_bits=16,
        variant="signed", recycled=True, samples=4, seed=2026092905,
        input_pairs=None, full_width=False, track_depth=False, stream=None,
        retain_ops=False, hash_stream=True) -> dict:
    group = active_group(torsion=variant in ODD_VARIANTS) if group is None else group
    group.validate()
    schedule = windows(u_bits, v_bits, window_bits)
    shift = shift_point(group, schedule, variant)
    if (variant in ODD_VARIANTS or variant == "signed_zero_safe") and any(slot["bits"] < 2 for slot in schedule[1:]):
        raise ValueError(f"{variant} requires at least two bits in each arithmetic window")
    projection = (scalar_mul(group.order, *group.torsion, p=group.p, a=group.a)
                  if variant in ODD_VARIANTS else None)
    if group.order <= 1 << max(slot["bits"] for slot in schedule):
        raise ValueError("subgroup too small for the chosen window tables")
    if type(samples) is not int or samples < 0:
        raise ValueError("nonnegative diagnostic sample count required")
    if retain_ops and (group.n > 10 or max(u_bits, v_bits) > 12):
        raise ValueError("retained-IR mode is limited to small tests")
    digest = source_digest()
    started = time.monotonic()
    rng = random.Random(seed)
    pairs = ([(0, 0), ((1 << u_bits)-1, (1 << v_bits)-1)] +
             [(rng.getrandbits(u_bits), rng.getrandbits(v_bits)) for _ in range(samples)]) if input_pairs is None else list(input_pairs)
    if any(type(u) is not int or type(v) is not int or not 0 <= u < 1 << u_bits or not 0 <= v < 1 << v_bits for u, v in pairs):
        raise ValueError("diagnostic scalar outside declared domain")
    trials = []
    for index, (u, v) in enumerate(pairs):
        expected = group.add(group.add(group.mul(u, group.P), group.mul(v, group.Q)), shift)
        trials.append({"u": u, "v": v, "expected": expected,
                       "domain_failure": None, "value_failure": None, "control_failure": None,
                       "dirty_failure": None, "noncanonical_stages": [], "phase_stages": [],
                       "state": BasisAccumulator(inputs={"u": u, "v": v},
                            rng=random.Random(seed+100000+index), record_measurements=False),
                       "ideal": None, "active": True})
        if variant in ODD_VARIANTS:
            trials[-1]["horizontal_stages"] = []
    circ = Circuit()
    retained = []
    resource = ResourceAccumulator(digest=hash_stream, track_depth=track_depth)
    full_registers = {}
    if not recycled:
        full_registers = {"u": circ.input("u", u_bits), "v": circ.input("v", v_bits)}
        resource.consume(circ.ops)
        for trial in trials:
            trial["state"].consume(circ.ops, constants=circ._cbit_const)
        if retain_ops:
            retained.extend(circ.ops)
        _drop_log(circ)
    feedback = {"u": [], "v": []}
    stage_rows = []
    x = y = None
    for index, slot in enumerate(schedule):
        stage_start = time.monotonic()
        axis, offset, bits = slot["axis"], slot["offset"], slot["bits"]
        before = resource.snapshot()
        feedback_before = len(feedback[axis])
        if recycled:
            word = circ.alloc(bits)
            header_alloc = tuple(circ.ops)
            for q in word:
                circ.ops.append(Op("H", (q,)))
            header_end = len(circ.ops)
        else:
            word = full_registers[axis][offset:offset+bits]
            header_alloc = ()
            header_end = 0
        for trial in trials:
            if trial["active"] and recycled:
                trial["state"].consume(header_alloc, constants=circ._cbit_const)
                trial["state"].load(word, (trial[axis] >> offset) & ((1 << bits)-1))
        if index == 0:
            points = _initial_points(group, slot, variant)
            packed = circ.alloc(2*group.n)
            x, y = packed[:group.n], packed[group.n:]
            words = tuple(pt[0] | (pt[1] << group.n) for pt in points)
            lookup_xor_unary(circ, word, packed, words)
            circ.inputs["x"], circ.inputs["y"] = x, y
            table_fingerprint = _fingerprint(points)
        else:
            table = _table(group, slot, variant)
            if variant == "unsigned":
                x, y = add_window_affine_optimized_inplace(circ, x, y, word, table,
                    config=_config(group, bits, full_width),
                    const_kw={} if full_width else None, sqr_kw={} if full_width else None)
                table_fingerprint = table.fingerprint
            elif variant in ODD_VARIANTS:
                x, y = add_zero_point_inplace(circ, x, y, word, table,
                    config=_config(group, bits-1, full_width),
                    phase_correction=variant == "odd_torsion",
                    restoration="add" if variant == "odd_torsion_add" else "subtract",
                    const_kw={} if full_width else None, sqr_kw={} if full_width else None)
                table_fingerprint = table.fingerprint
            elif bits > 1:
                bias_recode_inplace(circ, word)
                x, y = add_signed_window_inplace(circ, x, y, word[:-1], word[-1], table,
                    config=_config(group, bits-1, full_width), sign_strategy="conjugate",
                    const_kw={} if full_width else None, sqr_kw={} if full_width else None,
                    zero_safe=variant == "signed_zero_safe")
                bias_recode_inplace(circ, word, inverse=True)
                table_fingerprint = table.fingerprint
            else:
                # A one-bit biased word selects -P at zero and identity at one.
                from ..point.affine import add_const_affine_inplace, tuned_point_kwargs
                target = (table[0], (-table[1]) % group.p)
                cfg = _config(group, 1, full_width)
                kwargs = tuned_point_kwargs(group.n, group.p, cfg.inversion)
                if full_width:
                    kwargs["const_kw"], kwargs["sqr_kw"] = {}, {}
                circ.x(word[0])
                x, y = add_const_affine_inplace(circ, x, y, *target, ctrl=word[0], p=group.p, **kwargs)
                circ.x(word[0])
                table_fingerprint = _fingerprint((target,))
            circ.rebind("x", x)
            circ.rebind("y", y)
        body_end = len(circ.ops)
        for trial in trials:
            raw = (trial[axis] >> offset) & ((1 << bits)-1)
            if index == 0:
                expected_stage = points[raw]
            else:
                weighted = group.mul(1 << offset, group.P if axis == "u" else group.Q)
                digit = raw - (1 << (bits-1)) if variant in SIGNED_VARIANTS else raw
                if variant in ODD_VARIANTS:
                    digit = table.digit(raw)
                    addend = group.mul(digit*pow(2, -1, group.order), weighted)
                    expected_stage = group.add(trial["ideal"], addend)
                    if trial["ideal"] is not None and trial["ideal"][1] == addend[1]:
                        trial["horizontal_stages"].append(index)
                else:
                    expected_stage = group.add(trial["ideal"], group.mul(digit, weighted))
                if trial["active"]:
                    try:
                        if trial["ideal"] is None:
                            raise AssertionError("identity intermediate accumulator")
                        if variant == "unsigned":
                            ref = add_window_affine_inplace_ref(*trial["ideal"], raw, table)
                        elif variant in ODD_VARIANTS:
                            ref = zero_point_reference(*trial["ideal"], raw, table)
                        elif bits > 1:
                            sign = int(digit < 0)
                            address = abs(digit) % (1 << (bits-1))
                            ref = signed_window_reference(*trial["ideal"], address, sign, table,
                                                          zero_safe=variant == "signed_zero_safe")
                        else:
                            target = (table[0], (-table[1]) % group.p)
                            rx, ry = trial["ideal"]
                            if rx == target[0] or ry == target[1]:
                                raise AssertionError("one-bit signed dummy/operand collision")
                            if raw == 0 and (expected_stage is None or expected_stage[0] == target[0]):
                                raise AssertionError("one-bit signed output collision")
                            ref = expected_stage
                        if ref != expected_stage:
                            raise AssertionError("independent oracle mismatch")
                    except AssertionError as exc:
                        trial["domain_failure"] = {"stage": index, "reason": str(exc)}
                        trial["active"] = False
            if variant in ODD_VARIANTS and (expected_stage is None or
                    scalar_mul(group.order, *expected_stage, p=group.p, a=group.a) != projection):
                raise AssertionError("ideal accumulator escaped its fixed torsion coset")
            trial["ideal"] = expected_stage
            if not trial["active"]:
                continue
            try:
                trial["state"].consume(circ.ops[header_end:body_end], constants=circ._cbit_const)
            except DirtyAncillaError as exc:
                trial["dirty_failure"] = {"stage": index, "reason": str(exc)}
                trial["active"] = False
                continue
            actual = (trial["state"].read(x), trial["state"].read(y))
            if expected_stage is None or tuple(v % group.p for v in actual) != expected_stage:
                if trial["value_failure"] is None:
                    trial["value_failure"] = {"stage": index, "actual": actual, "expected": expected_stage}
            if max(actual) >= group.p:
                trial["noncanonical_stages"].append(index)
            if trial["state"].read(word) != raw and trial["control_failure"] is None:
                trial["control_failure"] = index
            if trial["state"].phase:
                trial["phase_stages"].append(index)
        if recycled:
            for q in reversed(word.ids):
                prior = feedback[axis]
                if prior:
                    circ.ops.append(FeedbackPhaseOp((q,), tuple(prior), len(prior)+1))
                prior.append(circ.measure(q))
            for trial in trials:
                if trial["active"]:
                    trial["state"].discard(word)
        resource.consume(circ.ops)
        after = resource.snapshot()
        stage_rows.append({**slot, "index": index, "kind": "initial_lookup" if index == 0 else "point_addition",
                           "table_fingerprint": table_fingerprint,
                           "control_qubits": list(word.ids),
                           "op_nodes_begin": before["op_nodes"],
                           "arithmetic_nodes_begin": before["op_nodes"] + header_end,
                           "arithmetic_nodes_end": before["op_nodes"] + body_end,
                           "op_nodes_end": after["op_nodes"],
                           "fourier_output_bit_offset": feedback_before,
                           "fourier_output_cbits": feedback[axis][feedback_before:] if recycled else [],
                           "toffoli": after["toffoli"]-before["toffoli"],
                           "clifford": after["clifford"]-before["clifford"],
                           "measurements": after["measurements"]-before["measurements"],
                           "global_peak_so_far": after["peak_qubits"],
                           "seconds": time.monotonic()-stage_start})
        if stream is not None:
            print(f"{variant} {index+1}/{len(schedule)} {axis}[{offset}:{offset+bits}] "
                  f"{stage_rows[-1]['kind']}: CCX={stage_rows[-1]['toffoli']}, peak={after['peak_qubits']}",
                  file=stream, flush=True)
        if retain_ops:
            retained.extend(circ.ops)
        _drop_log(circ)
        for trial in trials:
            if trial["active"]:
                trial["state"].reset_classical(circ._cbit_const)
        if index == 0:
            del points, words
        else:
            del table
        gc.collect()
    diagnostics = []
    for trial in trials:
        if trial["ideal"] != trial["expected"]:
            raise AssertionError("shifted scalar schedule does not match independent integer-scalar oracle")
        state = trial.pop("state")
        trial.pop("ideal")
        trial["phase"] = state.phase if trial["active"] else None
        trial["output"] = (state.read(x), state.read(y)) if trial["active"] else None
        trial["observed_pass"] = bool(trial["active"] and not trial["value_failure"] and
                                       trial["control_failure"] is None and state.phase == 0)
        if variant in ODD_VARIANTS or variant == "signed_zero_safe":
            trial["observed_pass"] &= not trial["noncanonical_stages"]
        diagnostics.append(trial)
    if resource.snapshot()["peak_qubits"] != circ.peak_qubits or resource.snapshot()["live_qubits"] != circ._live:
        raise AssertionError("streamed resources disagree with persistent circuit allocator")
    expected_live = 2*group.n + (0 if recycled else u_bits+v_bits)
    if circ._live != expected_live:
        raise AssertionError("unexpected persistent workspace at schedule end")
    if digest != source_digest():
        raise RuntimeError("QSM sources changed during full schedule construction")
    result = {"version": "full-scalar-stream-v1", "variant": variant, "group": asdict(group),
              "u_bits": u_bits, "v_bits": v_bits, "window_bits": window_bits, "source_digest": digest,
              "recycled": recycled, "full_width_test_settings": full_width,
              "stream_hash_enabled": hash_stream,
              "feedback_phase_convention": "diag(1,exp(-2*pi*i*sum(bit(source[j])*2**j)/2**denominator_power))",
              "scalar_qft_signs": {"u": -1, "v": -1},
              "initialization": "full first-word QROM with global P/2 offset; no identity entries",
              "output_shift": shift, "stages": stage_rows, "resources": resource.snapshot(),
              "point_additions": len(schedule)-1, "diagnostics": diagnostics,
              "seed": seed, "seconds": time.monotonic()-started,
              "scope": "all scheduled arithmetic operations actually emitted and counted with persistent allocator; "
                       "not a point-count multiplication; H/feedback/X-measurement recycling schedule explicit; "
                       "arbitrary feedback phase synthesis unpriced; basis diagnostics are the deferred arithmetic "
                       "oracle, not full Shor statevector or statistical success certification; incomplete/lazy "
                       "arithmetic hazards retained; no postprocessing omission enabled by default"}
    if group.torsion is None:
        result["group"].pop("torsion")
    if variant in ODD_VARIANTS:
        result.update({
            "version": "full-scalar-odd-torsion-v1",
            "initialization": "full first-word odd QROM with fixed order-eight T0; no sign conjugation",
            "torsion_offset": group.torsion, "torsion_projection": projection,
            "phase_correction": variant == "odd_torsion",
            "coordinate_restoration": "add" if variant == "odd_torsion_add" else "subtract",
            "zero_contract": "literal canonical zero; lazy p-as-zero is not supported",
            "domain_contract": "ideal addition-only trajectory in T0+<P>, nonzero subgroup addends; horizontal chords allowed",
            "diagnostic_contract": "seeded measurement trajectories, not all-outcome analysis; observed_pass also requires canonical stage boundaries",
        })
    if variant == "signed_zero_safe":
        result.update({
            "zero_safe": True,
            "coordinate_restoration": "add",
            "phase_correction": False,
            "zero_contract": "literal canonical zero; lazy p-as-zero is not supported",
            "diagnostic_contract": "seeded measurement trajectories, not all-outcome analysis; observed_pass also requires canonical stage boundaries",
        })
    if variant in ("odd_torsion_add", "signed_zero_safe"):
        result.update({
            "version": ("full-scalar-signed-zero-safe-v1" if variant == "signed_zero_safe" else
                        "full-scalar-odd-torsion-positive-add-v1"),
            "restoration_adder": "modadd.borrowed_pseudo_mersenne",
            "restoration_unlookup": "paired",
            "restoration_exact_reduction": True,
            "restoration_carry_window": None,
            "restoration_comparator_bits": None,
            "restoration_contract": "canonical internal payloads required; inherited replay representation/phase hazards remain",
        })
    if retain_ops:
        result["_retained_ops"] = retained
        result["_circuit"] = circ
    return result


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=["unsigned", "signed"])
    parser.add_argument("--u-bits", type=int, default=U)
    parser.add_argument("--v-bits", type=int, default=U)
    parser.add_argument("--window-bits", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026092905)
    parser.add_argument("--coherent", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--stream-digest", action="store_true",
                        help="hash every emitted op; source and table fingerprints are always recorded")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.json.exists():
        parser.error("JSON target exists; use a fresh path")
    rows = []
    for variant in dict.fromkeys(args.variants):
        rows.append(run(u_bits=args.u_bits, v_bits=args.v_bits, window_bits=args.window_bits,
                        variant=variant, recycled=not args.coherent, samples=args.samples,
                        seed=args.seed, track_depth=args.depth, stream=sys.stdout,
                        hash_stream=args.stream_digest))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x") as f:
        json.dump({"version": "full-scalar-study-v1", "field": FIELD, "rows": rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved {args.json}")
    return int(any(not d["observed_pass"] for r in rows for d in r["diagnostics"]))


if __name__ == "__main__":
    raise SystemExit(main())
