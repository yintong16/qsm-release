from qsm.core.circuit import Circuit, Register


def test_register_indexing_and_slicing():
    c = Circuit()
    r = c.input("x", 4)
    assert len(r) == 4
    assert isinstance(r[0], int)
    sub = r[1:3]
    assert isinstance(sub, Register)
    assert sub.ids == r.ids[1:3]


def test_peak_qubits_high_water():
    c = Circuit()
    c.alloc(5)
    assert c.peak_qubits == 5
    b = c.alloc(3)
    assert c.peak_qubits == 8
    c.free(b)
    c.alloc(4)  # reuses freed ids; live = 5 + 4 = 9
    assert c.peak_qubits == 9


def test_qid_reuse_after_free():
    c = Circuit()
    a = c.alloc(2)
    ids = set(a.ids)
    c.free(a)
    b = c.alloc(2)
    assert set(b.ids) == ids


def test_condition_stack_attaches_to_ops():
    c = Circuit()
    cb = c.classical_bit(1)
    with c.condition(cb, 1):
        c.x(0)
    assert c.ops[-1].cond == ((cb, 1),)
    c.x(1)
    assert c.ops[-1].cond == ()
