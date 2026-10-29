from qsm.core.circuit import Circuit
from qsm.core.resources import count


def test_gate_counts():
    c = Circuit()
    x = c.input("x", 3)
    y = c.input("y", 3)
    for i in range(3):
        c.cx(x[i], y[i])          # 3 clifford
    a = c.alloc(1)
    c.ccx(x[0], y[0], a[0])       # 1 toffoli
    c.free(a)
    r = count(c)
    assert r.toffoli == 1
    assert r.clifford == 3
    assert r.n_gates == 4


def test_peak_qubits():
    c = Circuit()
    c.alloc(5)
    b = c.alloc(3)      # 8 live
    c.free(b)           # back to 5 live
    c.alloc(4)          # live = 5 + 4 = 9
    r = count(c)
    assert r.peak_qubits == 9


def test_measurement_consumes_qubit_in_peak():
    c = Circuit()
    a = c.alloc(1)      # live 1
    m = c.measure(a[0])  # consumed -> live 0
    r = count(c)
    assert r.peak_qubits == 1
    assert r.measurements == 1


def test_volumes_hand_checked():
    """Hand-computed spacetime metrics on a circuit small enough to verify by eye."""
    c = Circuit()
    x = c.input("x", 2)      # live 2
    y = c.input("y", 2)      # live 4
    c.cx(x[0], y[0])         # gate @ live 4
    a = c.alloc(2)           # live 6  (peak)
    c.ccx(x[0], y[0], a[0])  # CCX  @ live 6
    c.ccx(x[1], y[1], a[1])  # CCX  @ live 6
    c.free(a)                # live 4
    c.cx(x[1], y[1])         # gate @ live 4

    r = count(c)
    assert r.toffoli == 2
    assert r.peak_qubits == 6
    # volume = every gate weighted by live width: 4 + 6 + 6 + 4
    assert r.volume == 20
    # toffoli_volume = only the CCXs: 6 + 6
    assert r.toffoli_volume == 12
    # circuit_volume = peak * toffoli = 6 * 2  (the conventional 2D-surface-code cost)
    assert r.circuit_volume == 12
    # refinement is never worse than the peak-based bound
    assert r.toffoli_volume <= r.circuit_volume


def test_toffoli_depth_ignores_clifford():
    c = Circuit()
    q = c.input("q", 3)
    for _ in range(5):
        c.cx(q[0], q[1])          # Cliffords must not advance the magic-state path
    c.ccx(q[0], q[1], q[2])
    c.ccx(q[0], q[1], q[2])
    r = count(c)
    assert r.toffoli_depth == 2
    assert r.depth == 7           # all gates share qubits -> fully serial


def test_transient_peak_penalised_less_by_toffoli_volume():
    """A brief wide spike costs full freight under circuit_volume, little under
    toffoli_volume — the exact effect that decides CDKM vs Gidney."""
    c = Circuit()
    q = c.input("q", 2)
    c.ccx(q[0], q[1], q[0])       # CCX at live 2
    spike = c.alloc(100)          # transient
    c.cx(q[0], q[1])              # no CCX while the spike is live
    c.free(spike)
    c.ccx(q[0], q[1], q[0])       # CCX at live 2

    r = count(c)
    assert r.peak_qubits == 102
    assert r.circuit_volume == 102 * 2      # charges the spike against every Toffoli
    assert r.toffoli_volume == 2 + 2        # no Toffoli ever ran while it was live
    assert r.toffoli_volume < r.circuit_volume
