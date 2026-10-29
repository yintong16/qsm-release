"""The field profile: that selecting secp256k1 changes the prime and NOTHING else.

The port is only worth anything if the two columns are honestly comparable, and there are
exactly two ways for that to go wrong: the accuracy silently moves with the field, or some
circuit quietly keeps a 25519 constant. Both are checked here, and the secp256k1 half runs
in a subprocess because a field is bound at import time.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from qsm import params

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def in_field(field: str | None, src: str) -> dict:
    """Run with the chosen field (None unsets QSM_FIELD); read the JSON result."""
    env = dict(os.environ, PYTHONPATH=REPO)
    if field is None:
        env.pop("QSM_FIELD", None)
    else:
        env["QSM_FIELD"] = field
    p = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, cwd=REPO,
        env=env,
    )
    assert p.returncode == 0, p.stderr[-3000:]
    for line in p.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    raise AssertionError(f"no JSON line:\n{p.stdout[-2000:]}")


# --- the profile itself ----------------------------------------------------
def test_default_is_25519_and_reproduces_the_old_hardcoded_constants():
    """The derivation must give back exactly what was written down before it existed."""
    # Test the actual unset default, even when pytest itself selects secp256k1.
    got = in_field(None, _PROBE)
    assert got["field"] == "25519"
    assert (got["u"], got["f"]) == (255, 19)
    assert int(got["prime"]) == (1 << 255) - 19
    assert got["L"] == 45
    assert got["cmp"] == 40


def test_every_field_is_pseudo_mersenne_with_odd_f():
    """Odd F is assumed by `modhalve`, `moddouble` and the whole lazy contract."""
    for name, cfg in params.FIELDS.items():
        p = (1 << cfg["u"]) - cfg["f"]
        assert cfg["f"] & 1, name
        assert p > 0 and p & 1, name
        assert pow(2, p - 1, p) == 1, f"{name}: p is not prime"


def test_carry_window_is_derived_so_accuracy_cannot_drift_with_the_field():
    """The point of deriving L: both fields sit at the same per-operation failure rate.

    If L were per-field literal, a port could change the accuracy and the gate counts at
    once, and the comparison would mean nothing.
    """
    rates = {}
    for name, cfg in params.FIELDS.items():
        L = math.ceil(params.ERROR_BUDGET + math.log2(cfg["f"]))
        rates[name] = math.log2(params.carry_fail_prob(L, cfg["f"]))
    assert rates["25519"] == pytest.approx(-params.ERROR_BUDGET, abs=1.0)
    assert rates["secp256k1"] == pytest.approx(-params.ERROR_BUDGET, abs=1.0)
    # ...and the window gap is exactly the log2(F) gap, which is the claim in the docs.
    ls = {n: math.ceil(params.ERROR_BUDGET + math.log2(c["f"]))
          for n, c in params.FIELDS.items()}
    assert ls["secp256k1"] - ls["25519"] == 28


def test_an_unknown_field_is_refused_rather_than_silently_ignored():
    p = subprocess.run([sys.executable, "-c", "import qsm.params"],
                       capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ, QSM_FIELD="p256", PYTHONPATH=REPO))
    assert p.returncode != 0
    assert "QSM_FIELD" in p.stderr


# --- the port actually takes ------------------------------------------------
_PROBE = """
import json
from qsm import params as P
from qsm.point import active
print("JSON" + json.dumps({
    "field": P.FIELD, "u": P.U, "f": P.F, "prime": str(P.PRIME),
    "L": P.CARRY_WINDOW, "cmp": P.CMP_BITS, "gx": str(active.GX), "gy": str(active.GY),
}))
"""


def test_selecting_secp256k1_moves_the_prime_the_window_and_the_curve():
    got = in_field("secp256k1", _PROBE)
    assert got["field"] == "secp256k1"
    assert got["u"] == 256
    assert got["f"] == (1 << 32) + 977
    assert int(got["prime"]) == (1 << 256) - (1 << 32) - 977
    assert got["L"] == 73                       # 40 + 33, not 45
    assert got["cmp"] == params.CMP_BITS        # F-independent, must NOT move
    # the curve followed the field
    from qsm.point import secp256k1
    assert int(got["gx"]) == secp256k1.GX
    assert int(got["gy"]) == secp256k1.GY


# Representative registered operands from every module that used a fixed _N.
# Check built registers, not just the exported constants or reference metadata.
_WIDTHS = """
import json
import qsm.field
from qsm.params import U
from qsm.bench.registry import get_builder
specs = {
    "field.add_cdkm.v1": {"a": U, "b": U},
    "field.add_gidney_ctrl.v1": {"ctrl": 1, "a": U, "b": U},
    "field.sub_gidney.v1": {"a": U, "b": U},
    "field.cswap.v1": {"ctrl": 1, "a": U, "b": U},
    "field.rshift1_ctrl.v1": {"ctrl": 1, "r": U},
    "field.rotate1_ctrl.v1": {"ctrl": 1, "r": U},
    "field.add_gidney_chunked_64.v1": {"a": U, "b": U},
    "field.modadd.v2": {"a": U, "b": U},
    "field.modadd.v3": {"a": U, "b": U},
    "field.cmp_ge_p.v1": {"b": U + 1, "flag": 1},
    "field.cmp_lt.v1": {"a": U, "b": U + 1, "flag": 1},
    "field.cmp_lt_ctrl.v1": {"ctrl": 1, "a": U, "b": U, "flag": 1},
    "field.addconst_p.v2": {"b": U},
    "field.subconst_p_decomp.v1": {"b": U + 1},
    "field.addconst_19.v1": {"b": U},
    "field.addconst_19.v2": {"b": U},
    "field.addconst_19_trunc.v1": {"b": U},
}
for name, want in specs.items():
    got = {key: len(reg) for key, reg in get_builder(name)().circuit.inputs.items()}
    assert got == want, (name, got, want)
print("JSON" + json.dumps({"checked": len(specs)}))
"""


@pytest.mark.parametrize("field", ["25519", "secp256k1"])
def test_registered_operand_widths_follow_the_selected_field(field):
    assert in_field(field, _WIDTHS)["checked"] == 17


_DEFAULT_ARITHMETIC = """
import inspect
import itertools
import json
import random
from qsm.params import PRIME, U, F
from qsm.core.circuit import Circuit
from qsm.core.simulator import simulate
from qsm.field import modarith as M
from qsm.bench.registry import get_builder

# Every public default modulus must be selected at import time, not only the
# handful of functions simulated below.
for name, fn in vars(M).items():
    if inspect.isfunction(fn) and fn.__module__ == M.__name__:
        arg = inspect.signature(fn).parameters.get("p")
        if arg is not None and arg.default is not inspect.Parameter.empty:
            assert arg.default == PRIME, (name, arg.default, PRIME)

values = [0, 1, F - 1, F, F + 1, 1 << (U - 1), PRIME - 1]
rng = random.Random(20260916)
checked = 0
for fn in [M.modadd, M.modadd_cmp]:
    c = Circuit()
    a, b = c.input("a", U), c.input("b", U)
    fn(c, a, b)                         # Deliberately omit p and tuning kwargs.
    for av, bv in itertools.product(values, repeat=2):
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs == {"a": av, "b": (bv + av) % PRIME}, (fn.__name__, av, bv)
        assert r.phase == 0, (fn.__name__, av, bv)
        checked += 1

# The exact registered comparator must also compare against the selected p.
case = get_builder("field.cmp_ge_p.v1")()
for b in [0, (1 << 255) - 19, PRIME - 1, PRIME, PRIME + 1, 1 << U]:
    for flag in (0, 1):
        v = {"b": b, "flag": flag}
        want = {"b": b, "flag": flag ^ int(b >= PRIME)}
        assert case.reference(v) == want
        r = simulate(case.circuit, v, rng=rng)
        assert r.outputs == want and r.phase == 0
        checked += 1
print("JSON" + json.dumps({"checked": checked}))
"""


@pytest.mark.parametrize("field", ["25519", "secp256k1"])
def test_default_arithmetic_uses_the_selected_modulus(field):
    assert in_field(field, _DEFAULT_ARITHMETIC)["checked"] == 110


_DECOMPOSITION = """
import json
import random
from qsm.params import PRIME, U, F, CARRY_WINDOW
from qsm.core.simulator import simulate
from qsm.field.constadd import _p_case
from qsm.bench.registry import get_builder
rng = random.Random(20260916)
checked = 0
mod = 1 << (U + 1)
values = [0, 1, F - 1, F, F + 1, PRIME - 1, PRIME, PRIME + 1,
          (1 << U) - 1, 1 << U, mod - 1]
for adding in (False, True):
    case = _p_case(adding=adding, window=None)
    assert len(case.circuit.inputs["b"]) == U + 1
    for b in values:
        want = {"b": (b + (PRIME if adding else -PRIME)) % mod}
        assert case.reference({"b": b}) == want
        r = simulate(case.circuit, {"b": b}, rng=rng)
        assert r.outputs == want and r.phase == 0, (adding, b, r.outputs, want)
        checked += 1

# A version named 19 must still add 19 when the selected deficit is not 19.
# For the truncated version use inputs away from its documented carry boundary.
for name in ["field.addconst_19.v1", "field.addconst_19.v2", "field.addconst_19_trunc.v1"]:
    case = get_builder(name)()
    if "trunc" in name:
        assert case.meta["fail_analytic"] == 19 / (1 << CARRY_WINDOW)
    for b in [0, 1, 19, 1 << (U - 1), 1 << (CARRY_WINDOW - 1)]:
        want = {"b": b + 19}
        assert case.reference({"b": b}) == want, name
        r = simulate(case.circuit, {"b": b}, rng=rng)
        assert r.outputs == want and r.phase == 0, (name, b)
        checked += 1
print("JSON" + json.dumps({"checked": checked}))
"""


@pytest.mark.parametrize("field", ["25519", "secp256k1"])
def test_decomposed_modulus_and_literal_19_match_their_references(field):
    assert in_field(field, _DECOMPOSITION)["checked"] == 37


_SQUARE_RIPPLE = """
import inspect
import json
import random
from unittest.mock import patch
from qsm import params as P
from qsm.mul import accumulate as A
from qsm.bench.registry import get_builder
from qsm.core.simulator import simulate

assert P.SQUARE_RIPPLE == P.ERROR_BUDGET == 40
assert P.SQUARE_RIPPLE != P.CARRY_WINDOW
full, narrow, doubles = A.tri_schedule_counts(P.U, P.SQUARE_RIPPLE)
assert (full, narrow, doubles) == (40, P.U - 41, P.U)
assert P.narrow_row_carry_fail_prob(P.SQUARE_RIPPLE) == 2.0 ** -41
assert A.TRI_TRUNC_R == A.TRI_OVERFLOW + A.tri_truncation_prob(P.U, 40)
assert A.TRISUB_TRUNC_R == (A.TRISUB_OVERFLOW + A.tri_truncation_prob(P.U, 40)
                           + P.U * P.modhalve_borrow_fail_prob(P.CARRY_WINDOW, P.F))

# Spy on the real emitters, not metadata alone: v1 retains L and v2 must pass R.
# Mock only the factory for v1 to avoid rebuilding historical full-width circuits.
for factory, version in [('_tri_case', 'mul.modsqr_tri_trunc_flat.v1'),
                         ('_trisub_case', 'mul.modsqrsub_inplace_trunc_flat.v1'),
                         ('_trisub_case', 'mul.modsqrsub_inplace_ctrl_trunc_flat.v1')]:
    with patch.object(A, factory) as spy:
        get_builder(version)()
        assert spy.call_args.args[0] == P.CARRY_WINDOW, version

rng = random.Random(20261011)
checked = 0
for version, emitter, rate in [
    ('mul.modsqr_tri_trunc_flat.v2', 'modsqr_tri', A.TRI_TRUNC_R),
    ('mul.modsqrsub_inplace_trunc_flat.v2', 'modsqrsub_inplace', A.TRISUB_TRUNC_R),
    ('mul.modsqrsub_inplace_ctrl_trunc_flat.v2', 'modsqrsub_inplace', A.TRISUB_TRUNC_R),
]:
    fn = getattr(A, emitter)
    with patch.object(A, emitter, wraps=fn) as spy:
        case = get_builder(version)()
        bound = inspect.signature(fn).bind(*spy.call_args.args, **spy.call_args.kwargs)
        assert bound.arguments['ripple'] == P.SQUARE_RIPPLE, version
        assert bound.arguments['carry_window'] == P.CARRY_WINDOW, version
    assert case.meta['fail_analytic'] == rate, version
    assert len(case.circuit.inputs['x']) == P.U
    for x in [0, 19, rng.randrange(P.PRIME)]:
        values = {'x': x}
        if 'out' in case.circuit.inputs:
            values['out'] = 0
            want = {'x': x, 'out': x * x % P.PRIME}
        else:
            values['acc'] = rng.randrange(P.PRIME)
            want = {'x': x, 'acc': (values['acc'] - x * x) % P.PRIME}
        if 'c' in case.circuit.inputs:
            values['c'] = want['c'] = 1
        assert case.reference(values) == want, (version, values)
        got = simulate(case.circuit, values, rng=rng)
        assert got.outputs == want and got.phase == 0, (version, values)
        checked += 1
        if 'c' in values:
            values['c'] = 0
            assert case.reference(values) == values
            got = simulate(case.circuit, values, rng=rng)
            assert got.outputs == values and got.phase == 0
            checked += 1
print('JSON' + json.dumps({'checked': checked, 'ripple': P.SQUARE_RIPPLE}))
"""


@pytest.mark.parametrize("field", ["25519", "secp256k1"])
def test_square_ripple_is_field_independent_and_registered_circuits_use_it(field):
    assert in_field(field, _SQUARE_RIPPLE) == {"checked": 12, "ripple": 40}


_BUILD = """
import json
import qsm.point
from qsm.bench.registry import get_builder
from qsm.core.resources import count
from qsm import params as P
c = get_builder("point.add_const_affine_inplace.v1")().circuit
r = count(c)
print("JSON" + json.dumps({"toffoli": r.toffoli, "peak": r.peak_qubits,
                           "width": len(c.inputs["x"])}))
"""


@pytest.mark.slow
def test_the_point_addition_builds_at_the_other_prime_and_costs_a_little_more():
    """The headline of docs/decisions/15-prime-port.md, as a regression test.

    Loose bounds deliberately: the claim is "a few percent, not a factor", and pinning the
    exact count here would make every retune of the square or the GCD fail this test for
    no reason.
    """
    here = in_field("25519", _BUILD)
    there = in_field("secp256k1", _BUILD)
    assert here["width"] == 255 and there["width"] == 256
    ratio = there["toffoli"] / here["toffoli"]
    assert 1.0 < ratio < 1.15, ratio
    assert 0 < there["peak"] - here["peak"] < 60, (here["peak"], there["peak"])
