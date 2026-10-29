import random

import pytest

from qsm.audit.guarded_full_window import guarded_gcd_emission, run
from qsm.core.circuit import Circuit
from qsm.core.simulator import simulate
from qsm.inversion import gcd, undo


def test_both_emitters_are_temporarily_replaced_and_restored():
    originals = gcd.cmp_lt, undo.cmp_lt
    with guarded_gcd_emission(2):
        assert (gcd.cmp_lt, undo.cmp_lt) != originals
        for emitter in (gcd.cmp_lt, undo.cmp_lt):
            c = Circuit()
            a, b = c.input("a", 8), c.input("b", 8)
            flag, ctrl = c.input("flag", 1), c.input("ctrl", 1)
            emitter(c, a, b, flag[0], k=4, ctrl=ctrl[0])
            for av, bv in [(167, 163), (3, 1), (15, 16), (255, 0)]:
                for cv in (0, 1):
                    r = simulate(c, {"a": av, "b": bv, "flag": 0, "ctrl": cv}, rng=random.Random(41))
                    assert r.outputs["flag"] == (cv & int((bv >> 2) < (av >> 2)))
                    assert r.phase == 0
    assert (gcd.cmp_lt, undo.cmp_lt) == originals


def test_emitters_restore_after_construction_exception():
    originals = gcd.cmp_lt, undo.cmp_lt
    with pytest.raises(RuntimeError):
        with guarded_gcd_emission(8):
            raise RuntimeError("construction interrupted")
    assert (gcd.cmp_lt, undo.cmp_lt) == originals


def test_full_width_exact_comparator_is_not_replaced_by_approximation():
    with guarded_gcd_emission(8):
        c = Circuit()
        a, b = c.input("a", 8), c.input("b", 8)
        flag = c.input("flag", 1)
        gcd.cmp_lt(c, a, b, flag[0], k=None)
        r = simulate(c, {"a": 3, "b": 1, "flag": 0}, rng=random.Random(1))
        assert r.outputs["flag"] == 1
        assert r.phase == 0


@pytest.mark.parametrize("kwargs", [
    {"guard_bits": []}, {"guard_bits": [0]}, {"guard_bits": [101]},
    {"window_bits": 0}, {"window_bits": 17}, {"trials": 0}, {"meas_reps": 0},
])
def test_bad_full_window_requests_fail_before_building(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)
