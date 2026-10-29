import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add, gidney_add

ADDERS = [cdkm_add, gidney_add]


def _add_circuit(add_fn, n: int) -> Circuit:
    c = Circuit()
    a = c.input("a", n)
    b = c.input("b", n)
    add_fn(c, a, b)
    return c


def _n_measurements(c: Circuit) -> int:
    from qsm.core import ops
    return sum(1 for op in c.ops if op.kind == ops.MEASURE)


@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_add_exhaustive(add_fn, n: int):
    c = _add_circuit(add_fn, n)
    mask = (1 << n) - 1
    # Force *every* measurement-outcome pattern so phase-cleanliness is checked on all
    # branches (not just the random ones the harness happens to draw).
    n_meas = _n_measurements(c)
    meas_vectors = (
        [[(k >> j) & 1 for j in range(n_meas)] for k in range(1 << n_meas)]
        if n_meas <= 8 else [None]
    )
    for av in range(1 << n):
        for bv in range(1 << n):
            for mv in meas_vectors:
                r = simulate(c, {"a": av, "b": bv}, measurements=mv)
                assert r.outputs["a"] == av, f"a changed ({add_fn.__name__}, n={n}, {av},{bv})"
                assert r.outputs["b"] == (av + bv) & mask, \
                    f"wrong sum ({add_fn.__name__}, n={n}, {av},{bv})"
                assert r.phase == 0, \
                    f"dirty phase ({add_fn.__name__}, n={n}, {av},{bv}, meas={mv})"
                # (a clean-ancilla failure would raise DirtyAncillaError above)


@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
def test_add_random_255bit(add_fn):
    n = 255
    c = _add_circuit(add_fn, n)
    rng = random.Random(0)
    mask = (1 << n) - 1
    for _ in range(64):
        av = rng.getrandbits(n)
        bv = rng.getrandbits(n)
        r = simulate(c, {"a": av, "b": bv}, rng=rng)
        assert r.outputs["a"] == av
        assert r.outputs["b"] == (av + bv) & mask
        assert r.phase == 0


@pytest.mark.parametrize("add_fn", ADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_add_narrow_operand_exhaustive(add_fn, n: int):
    """len(a) < len(b): the missing high bits of a must behave as |0>."""
    for na in range(1, n + 1):
        c = Circuit()
        a = c.input("a", na)
        b = c.input("b", n)
        add_fn(c, a, b)
        mask = (1 << n) - 1
        n_meas = _n_measurements(c)
        mvs = ([[(k >> j) & 1 for j in range(n_meas)] for k in range(1 << n_meas)]
               if n_meas <= 8 else [None])
        for mv in mvs:
            for av in range(1 << na):
                for bv in range(1 << n):
                    r = simulate(c, {"a": av, "b": bv}, measurements=mv)
                    assert r.outputs["a"] == av, f"a changed (na={na},n={n})"
                    assert r.outputs["b"] == (av + bv) & mask, \
                        f"{add_fn.__name__} na={na} n={n}: {av}+{bv}"
                    assert r.phase == 0, f"dirty phase na={na} n={n} {av},{bv}"


def test_narrow_operand_is_free_for_gidney_not_cdkm():
    """Gidney's carries are separate ancillas -> a short `a` needs no padding.
    CDKM parks carries inside `a` -> the padding must be materialised."""
    n, na = 255, 5

    def peak(fn, width):
        c = Circuit()
        a = c.input("a", width)
        b = c.input("b", n)
        fn(c, a, b)
        return count(c).peak_qubits

    # Gidney: the narrow operand is saved outright, qubit for qubit.
    assert peak(gidney_add, na) == na + n + (n - 1)        # 514
    assert peak(gidney_add, n) == n + n + (n - 1)          # 764
    assert peak(gidney_add, n) - peak(gidney_add, na) == n - na

    # CDKM: pads back to full width, so narrowing buys nothing.
    assert peak(cdkm_add, na) == peak(cdkm_add, n) == 2 * n + 1   # 511

    # ...but CDKM is STILL the lower peak overall, because 1 carry beats n-1 carries
    # even after paying for the padding. Narrowing does not flip which adder is
    # more space-efficient -- it only shrinks Gidney's disadvantage.
    assert peak(cdkm_add, na) < peak(gidney_add, na)


def test_carry_still_ripples_full_width():
    """A 1-bit operand must still carry all the way: 0b111..1 + 1 == 0."""
    n = 32
    c = Circuit()
    a = c.input("a", 1)
    b = c.input("b", n)
    gidney_add(c, a, b)
    r = simulate(c, {"a": 1, "b": (1 << n) - 1})
    assert r.outputs["b"] == 0
    assert r.phase == 0


def test_adder_resource_signatures():
    """The whole point of having both adders: opposite Toffoli<->space corners."""
    n = 255
    cd = count(_add_circuit(cdkm_add, n))
    gd = count(_add_circuit(gidney_add, n))

    assert cd.toffoli == 2 * n            # 510
    assert cd.peak_qubits == 2 * n + 1    # 511, one borrowed carry
    assert cd.measurements == 0

    assert gd.toffoli == n - 1            # 254
    assert gd.peak_qubits == 3 * n - 1    # 764, all carries co-resident
    assert gd.measurements == n - 1       # each carry erased by one measurement
