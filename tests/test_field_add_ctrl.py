import random

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.field.add import cdkm_add_ctrl, gidney_add_ctrl, gidney_add_ctrl_threaded

CADDERS = [cdkm_add_ctrl, gidney_add_ctrl, gidney_add_ctrl_threaded]

# All three zero-extend a short `a`. It just is not free for the threaded one -- see the
# cost test at the bottom, and its docstring for why.
NARROW_CADDERS = CADDERS


def _cadd_circuit(add_fn, n: int, na: int | None = None) -> Circuit:
    c = Circuit()
    ctrl = c.input("ctrl", 1)
    a = c.input("a", n if na is None else na)
    b = c.input("b", n)
    add_fn(c, ctrl[0], a, b)
    return c


def _n_measurements(c: Circuit) -> int:
    return sum(1 for op in c.ops if op.kind == ops.MEASURE)


@pytest.mark.parametrize("add_fn", CADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_cadd_exhaustive(add_fn, n: int):
    c = _cadd_circuit(add_fn, n)
    mask = (1 << n) - 1
    n_meas = _n_measurements(c)
    meas_vectors = (
        [[(k >> j) & 1 for j in range(n_meas)] for k in range(1 << n_meas)]
        if n_meas <= 8 else [None]
    )
    for ctrl in (0, 1):
        for av in range(1 << n):
            for bv in range(1 << n):
                want = (av + bv) & mask if ctrl else bv
                for mv in meas_vectors:
                    r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv}, measurements=mv)
                    assert r.outputs["a"] == av, f"a changed ({add_fn.__name__}, {ctrl},{av},{bv})"
                    assert r.outputs["ctrl"] == ctrl, "ctrl not preserved"
                    assert r.outputs["b"] == want, \
                        f"wrong sum ({add_fn.__name__}, {ctrl},{av},{bv})"
                    assert r.phase == 0, \
                        f"dirty phase ({add_fn.__name__}, {ctrl},{av},{bv}, meas={mv})"


@pytest.mark.parametrize("add_fn", NARROW_CADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_cadd_narrow_operand_exhaustive(add_fn, n: int):
    """len(a) < len(b): the missing high bits of a must behave as |0>, control or not."""
    mask = (1 << n) - 1
    for na in range(1, n + 1):
        c = _cadd_circuit(add_fn, n, na)
        for ctrl in (0, 1):
            for av in range(1 << na):
                for bv in range(1 << n):
                    r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv})
                    want = (av + bv) & mask if ctrl else bv
                    assert r.outputs["a"] == av
                    assert r.outputs["ctrl"] == ctrl
                    assert r.outputs["b"] == want, (add_fn.__name__, n, na, ctrl, av, bv)
                    assert r.phase == 0


@pytest.mark.parametrize("add_fn", CADDERS, ids=lambda f: f.__name__)
def test_cadd_random_255bit(add_fn):
    n = 255
    c = _cadd_circuit(add_fn, n)
    rng = random.Random(0)
    mask = (1 << n) - 1
    for _ in range(32):
        ctrl, av, bv = rng.getrandbits(1), rng.getrandbits(n), rng.getrandbits(n)
        r = simulate(c, {"ctrl": ctrl, "a": av, "b": bv}, rng=rng)
        assert r.outputs["a"] == av
        assert r.outputs["ctrl"] == ctrl
        assert r.outputs["b"] == ((av + bv) & mask if ctrl else bv)
        assert r.phase == 0


@pytest.mark.parametrize("add_fn", CADDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_cadd_ctrl_zero_is_exactly_identity(add_fn, n: int):
    """With ctrl=0 the circuit must be the identity on (a,b) -- no residue anywhere."""
    c = _cadd_circuit(add_fn, n)
    for av in range(1 << n):
        for bv in range(1 << n):
            r = simulate(c, {"ctrl": 0, "a": av, "b": bv})
            assert (r.outputs["a"], r.outputs["b"]) == (av, bv)
            assert r.phase == 0


@pytest.mark.parametrize("n", [8, 64, 255])
def test_cadd_costs_match_the_documented_model(n: int):
    """The three flavours sit at the corners the inversion needs; pin every number.

    cdkm_add_ctrl:            4n   Toffoli, 1     ancilla   (2n+2 peak: a, b, ctrl, carry)
    gidney_add_ctrl:          2n-1 Toffoli, 2n-1  ancillas  (mask + carry ladder)
    gidney_add_ctrl_threaded: 2n-1 Toffoli, n-1   ancillas  (carry ladder only)
    """
    cd = count(_cadd_circuit(cdkm_add_ctrl, n))
    assert cd.toffoli == 4 * n
    assert cd.peak_qubits == 2 * n + 2

    gd = count(_cadd_circuit(gidney_add_ctrl, n))
    assert gd.toffoli == 2 * n - 1
    assert gd.peak_qubits == 2 * n + 1 + (2 * n - 1)

    th = count(_cadd_circuit(gidney_add_ctrl_threaded, n))
    assert th.toffoli == 2 * n - 1
    assert th.peak_qubits == 2 * n + 1 + (n - 1)


def test_cadd_gidney_ctrl_beats_cdkm_ctrl_on_toffoli_and_loses_on_space():
    """The tradeoff has to actually be a tradeoff, or one of them is dead weight."""
    n = 255
    cd = count(_cadd_circuit(cdkm_add_ctrl, n))
    gd = count(_cadd_circuit(gidney_add_ctrl, n))
    assert gd.toffoli < cd.toffoli
    assert gd.peak_qubits > cd.peak_qubits


def test_cadd_threaded_strictly_dominates_the_masked_gidney():
    """Not a tradeoff: same Toffoli, exactly n fewer qubits, at every width.

    Doc 11 predicted `2n` Toffoli for the threaded form -- one more than the mask, for
    half the space. The measurement says the extra Toffoli is not there either: the top
    bit has no carry out, so its MAJ never happens and the count lands on `2n-1`. The
    masked `gidney_add_ctrl` is kept only because it accepts len(a) < len(b).
    """
    for n in (2, 8, 64, 255):
        gd = count(_cadd_circuit(gidney_add_ctrl, n))
        th = count(_cadd_circuit(gidney_add_ctrl_threaded, n))
        assert th.toffoli == gd.toffoli, n
        assert gd.peak_qubits - th.peak_qubits == n, n
