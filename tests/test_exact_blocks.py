"""Exact annotations are sealed sidecars, not changes to the physical circuit."""

from copy import copy
from dataclasses import FrozenInstanceError, fields, replace
import gc
import weakref

import pytest

from qsm.core import ops
from qsm.core.circuit import Circuit, Register
from qsm.core.resources import count
from qsm.core.semantics import ExactLookupBlock, validate_exact_blocks
from qsm.scalarmul import lookup


def _pair(width=2, low_bits=None):
    c = Circuit()
    c.classical_bit(1)  # Classical IDs are deliberately not measurement indices.
    address = c.input("address", width)
    bystander = c.input("bystander", 1)
    target = c.alloc(2)
    values = tuple((3 * row + (row >> 1)) % 4 for row in range(1 << width))
    lookup.lookup_xor_unary(c, address, target, values)
    lookup.unlookup_measure(c, address, target, values, low_bits=low_bits)
    return c, address, target, bystander


@pytest.mark.parametrize("width,split", [(w, s) for w in (1, 2, 3)
                                         for s in range(w + 1)])
def test_capture_boundaries_all_splits_unchanged_ops_and_ten_resources(monkeypatch, width, split):
    c, address, target, bystander = _pair(width, split)
    blocks = validate_exact_blocks(c)
    assert len(blocks) == 2
    forward, inverse = blocks
    assert all(isinstance(block, ExactLookupBlock) for block in blocks)
    assert (forward.kind, inverse.kind) == ("xor", "unlookup")
    assert forward.address == inverse.address == tuple(address)
    assert forward.target == inverse.target == tuple(target)
    assert forward.stop == inverse.start
    assert forward.entry_live == forward.exit_live == inverse.entry_live
    assert inverse.exit_live == frozenset(tuple(address) + tuple(bystander))
    assert forward.cbit_start == 1
    assert forward.cbit_stop == inverse.cbit_start
    assert inverse.cbit_stop == c._next_cbit
    parity, = (op for op in inverse.op_refs if op.kind == ops.CPARITY)
    assert inverse.values is parity.values
    assert inverse.cbit_stop - inverse.cbit_start == (
        len(inverse.values) + sum(op.kind == ops.MEASURE for op in inverse.op_refs)
    )
    for block in blocks:
        assert isinstance(block.op_refs, tuple)
        assert all(saved is c.ops[i] for i, saved in enumerate(block.op_refs, block.start))
    before_ops = tuple(c.ops)
    before_resources = count(c)
    assert len(fields(before_resources)) == 10
    # Disable ONLY annotation recording; independent construction must emit the
    # identical complete IR and all ten resource metrics (including depths).
    monkeypatch.setattr(lookup, "_record_exact_lookup", lambda *a, **kw: None)
    raw, *_ = _pair(width, split)
    assert raw.exact_blocks == []
    assert tuple(raw.ops) == before_ops == tuple(c.ops)
    assert count(raw) == before_resources == count(c)


def test_only_approved_helpers_annotated_and_blocks_sorted():
    c = Circuit()
    address, target = c.input("address", 2), c.alloc(2)
    lookup.lookup_xor(c, address, target, (0, 1, 2, 3))
    lookup.xor_nonzero(c, address, target[0])
    assert c.exact_blocks == []
    assert validate_exact_blocks(c) == ()
    lookup.lookup_xor_unary(c, address, target, (0, 1, 2, 3))
    lookup.unlookup_measure(c, address, target, (0, 1, 2, 3))
    expected = tuple(c.exact_blocks)
    c.exact_blocks.reverse()
    assert validate_exact_blocks(c) == expected


def test_immutable_snapshots_shared_ops_and_no_circuit_strong_cycle():
    c = Circuit()
    address, target = c.input("address", 2), c.alloc(2)
    values = [0, 1, 2, 3]
    lookup.lookup_xor_unary(c, address, target, values)
    block, = validate_exact_blocks(c)
    values[0] = 3
    address.ids.reverse()
    target.ids.reverse()
    assert block.values == (0, 1, 2, 3)
    assert block.address == (0, 1) and block.target == (2, 3)
    with pytest.raises(FrozenInstanceError):
        block.start = 0
    with pytest.raises(TypeError):
        block.values[0] = 3
    assert validate_exact_blocks(c) == (block,)
    owner = weakref.ref(c)
    del c
    gc.collect()
    assert owner() is None  # Keeping a block never keeps its circuit alive.


@pytest.mark.parametrize("change", [
    {}, {"kind": "unlookup"}, {"start": 0}, {"stop": 99999},
    {"address": (1, 0)}, {"target": (3, 2)}, {"values": (3, 2, 1, 0)},
    {"semantic_version": "unsupported"}, {"cbit_start": 0}, {"cbit_stop": 99999},
    {"entry_live": frozenset()}, {"exit_live": frozenset()}, {"op_refs": ()},
    {"_seal": None}, {"_seal": []},
])
def test_replace_descriptor_never_reuses_seal(change):
    c, *_ = _pair()
    c.exact_blocks[0] = replace(c.exact_blocks[0], **change)
    with pytest.raises(ValueError, match="stale|unsealed|foreign"):
        validate_exact_blocks(c)


def test_capture_metadata_separately_from_descriptor():
    c, *_ = _pair()
    # Frozen records disallow normal mutation; even bypassing that on metadata
    # does not rewrite the seal's immutable capture. No hostile-Python claim.
    object.__setattr__(c.exact_blocks[0], "values", (3, 2, 1, 0))
    with pytest.raises(ValueError, match="stale"):
        validate_exact_blocks(c)


@pytest.mark.parametrize("mutation", ["equal", "argument", "insert", "delete", "reverse"])
def test_operation_replacement_and_range_tampering_rejected(mutation):
    c, *_ = _pair()
    block = c.exact_blocks[0]
    index = next(i for i in range(block.start, block.stop)
                 if c.ops[i].kind == ops.XOR_WORD)
    if mutation == "equal":
        c.ops[index] = replace(c.ops[index])
    elif mutation == "argument":
        c.ops[index] = replace(c.ops[index], mask=0)
    elif mutation == "insert":
        c.ops.insert(index, ops.Op(ops.X, (block.address[0],)))
    elif mutation == "delete":
        del c.ops[index]
    else:
        c.ops[block.start:block.stop] = reversed(c.ops[block.start:block.stop])
    with pytest.raises(ValueError, match="identities|range"):
        validate_exact_blocks(c)


def test_foreign_owner_copied_circuit_missing_seal_and_overlap():
    c, *_ = _pair()
    other, *_ = _pair()
    other.exact_blocks = c.exact_blocks[:]
    with pytest.raises(ValueError, match="foreign|unsealed"):
        validate_exact_blocks(other)
    with pytest.raises(ValueError, match="foreign|unsealed"):
        validate_exact_blocks(copy(c))
    original = c.exact_blocks[0]
    c.exact_blocks.append(original)
    with pytest.raises(ValueError, match="overlapping"):
        validate_exact_blocks(c)
    c.exact_blocks.pop()
    c._exact_block_seals.clear()
    with pytest.raises(ValueError, match="unsealed"):
        validate_exact_blocks(c)


@pytest.mark.parametrize("source", ["measurement", "parity_first", "parity_last"])
@pytest.mark.parametrize("escape", ["condition", "parity_source", "measure_producer", "parity_producer", "constant"])
def test_full_ir_preflight_private_classical_escapes_and_overwrites(source, escape):
    c, address, *_ = _pair()
    block = c.exact_blocks[1]
    parity, = (op for op in block.op_refs if op.kind == ops.CPARITY)
    cb = {"measurement": block.cbit_start, "parity_first": parity.base,
          "parity_last": parity.base + len(parity.values) - 1}[source]
    if escape == "condition":
        with c.condition(cb):
            c.z(address[0])
    elif escape == "parity_source":
        c.classical_parities((cb,), (1,))
    elif escape == "measure_producer":
        scratch = c.alloc(1)
        c.measure(scratch[0])
        c.ops[-1] = replace(c.ops[-1], cbit=cb)
    elif escape == "parity_producer":
        # Interval overlap even when base itself is outside the private range.
        c.ops.append(ops.ClassicalParityOp((), (0,) * (cb + 1), 0))
    else:
        c._cbit_const[cb] = 0
    with pytest.raises(ValueError, match="private"):
        validate_exact_blocks(c)


def test_private_classical_reference_before_block_is_also_rejected():
    c = Circuit()
    address, target = c.input("address", 2), c.alloc(2)
    c.z(address[0])
    placeholder = len(c.ops) - 1
    lookup.lookup_xor_unary(c, address, target, (0, 1, 2, 3))
    c.ops[placeholder] = replace(c.ops[placeholder], cond=((c.exact_blocks[0].cbit_start, 1),))
    with pytest.raises(ValueError, match="private"):
        validate_exact_blocks(c)


@pytest.mark.parametrize("fault", [
    "bystander", "word_control", "word_target", "external_condition",
    "external_parity_source", "input_alloc", "extra_output", "consume_address",
])
def test_structural_closure_independent_of_identity_check(monkeypatch, fault):
    c = Circuit()
    cb = c.classical_bit(1)
    address, target = c.input("address", 2), c.alloc(2)
    bystander = c.input("bystander", 1)[0]
    original = lookup._record_exact_lookup

    def inject_then_seal(circ, **kwargs):
        # Sabotage the trusted builder hook to exercise structural checks with
        # current identity references, not just the earlier stale-op rejection.
        if fault == "bystander":
            circ.x(bystander)
        elif fault == "word_control":
            circ.xor_word(target, 1, ctrl=bystander)
        elif fault == "word_target":
            circ.xor_word((target[0], bystander), 1, ctrl=address[0])
        elif fault == "external_condition":
            with circ.condition(cb):
                circ.z(address[0])
        elif fault == "external_parity_source":
            circ.classical_parities((cb,), (1,))
        elif fault == "input_alloc":
            temporary = circ.input("illegal", 1)
            circ.free(temporary)
        elif fault == "extra_output":
            circ.alloc(1)
        else:
            circ.measure(address[0])
            circ.alloc(1)  # Same numeric ID, but an illegally consumed address.
        original(circ, **kwargs)

    monkeypatch.setattr(lookup, "_record_exact_lookup", inject_then_seal)
    lookup.lookup_xor_unary(c, address, target, (0, 1, 2, 3))
    with pytest.raises(ValueError, match="bystander|external|input|exit|preserved"):
        validate_exact_blocks(c)


def test_entry_lifetimes_recomputed_from_original_allocs_not_final_inputs():
    c, address, *_ = _pair()
    replacement = c.alloc(len(address))
    c.rebind("address", replacement)
    assert len(validate_exact_blocks(c)) == 2
    initial = next(i for i, op in enumerate(c.ops) if op.kind == ops.ALLOC)
    c.ops[initial] = replace(c.ops[initial], qubits=c.ops[initial].qubits + (999,))
    with pytest.raises(ValueError, match="entry live"):
        validate_exact_blocks(c)


def test_numeric_qubit_reuse_before_inside_and_after_blocks_is_valid():
    c = Circuit()
    address = c.input("address", 3)
    old = c.alloc(2)
    c.x(old[0])
    c.x(old[0])
    c.free(old)
    target = c.alloc(2)
    assert set(target) == set(old)
    rows = (3, 0, 1, 2, 1, 3, 0, 2)
    for _ in range(2):
        lookup.lookup_xor_unary(c, address, target, rows)
        lookup.unlookup_measure(c, address, target, rows, low_bits=2)
        recycled = c.alloc(len(c._free_qids))
        assert set(target) <= set(recycled)
        c.xor_word(recycled, (1 << len(recycled)) - 1)
        c.xor_word(recycled, (1 << len(recycled)) - 1)
        c.free(recycled)
        target = c.alloc(2)
    assert len(validate_exact_blocks(c)) == 4


def test_noncontiguous_views_and_external_classical_work_remain_valid():
    c = Circuit()
    storage = c.input("storage", 6)
    address = Register([storage[4], storage[1]])
    scratch = c.alloc(3)
    target = Register([scratch[2], scratch[0], scratch[1]])
    c.classical_parities((c.classical_bit(1),), (0, 1, 1))
    lookup.lookup_xor_unary(c, address, target, (0, 3, 6, 1))
    lookup.unlookup_measure(c, address, target, (0, 3, 6, 1))
    c.classical_parities((c.classical_bit(0),), (1, 0))
    forward, inverse = validate_exact_blocks(c)
    assert forward.address == inverse.address == (4, 1)
    assert forward.target == inverse.target == tuple(target)
    assert inverse.exit_live == frozenset(storage)


def test_cross_block_classical_dependency_rejected_even_with_fresh_op_seal(monkeypatch):
    c = Circuit()
    address, target = c.input("address", 2), c.alloc(2)
    lookup.lookup_xor_unary(c, address, target, (0, 1, 2, 3))
    private_cb = c.exact_blocks[0].cbit_start
    original = lookup._record_exact_lookup

    def inject_then_seal(circ, **kwargs):
        circ.classical_parities((private_cb,), (0, 1))
        original(circ, **kwargs)

    monkeypatch.setattr(lookup, "_record_exact_lookup", inject_then_seal)
    lookup.unlookup_measure(c, address, target, (0, 1, 2, 3))
    with pytest.raises(ValueError, match="external"):
        validate_exact_blocks(c)


@pytest.mark.parametrize("fault", ["gap", "conditional_producer", "outside_producer"])
def test_classical_producer_closure_with_fresh_op_seal(monkeypatch, fault):
    c = Circuit()
    address, target = c.input("address", 2), c.alloc(2)
    original = lookup._record_exact_lookup

    def inject_then_seal(circ, **kwargs):
        if fault == "gap":
            circ._next_cbit += 1  # No producer, even though the interval expanded.
        elif fault == "conditional_producer":
            scratch = circ.alloc(1)
            with circ.condition(kwargs["cbit_start"]):
                circ.measure(scratch[0])
        else:
            i = next(i for i in range(kwargs["start"], len(circ.ops))
                     if circ.ops[i].kind == ops.MEASURE)
            circ.ops[i] = replace(circ.ops[i], cbit=circ._next_cbit + 1)
        original(circ, **kwargs)

    monkeypatch.setattr(lookup, "_record_exact_lookup", inject_then_seal)
    lookup.lookup_xor_unary(c, address, target, (0, 1, 2, 3))
    with pytest.raises(ValueError, match="unproduced|producers"):
        validate_exact_blocks(c)


@pytest.mark.parametrize("api", [lookup.lookup_xor_unary, lookup.unlookup_measure])
@pytest.mark.parametrize("fault", ["table", "condition", "split", "named_target"])
def test_invalid_calls_never_record_partial_block(api, fault):
    c = Circuit()
    address = c.input("address", 2)
    target = c.input("target", 2) if fault == "named_target" else c.alloc(2)
    cb = c.classical_bit(1)
    if fault in ("split", "named_target") and api is lookup.lookup_xor_unary:
        return
    before = (tuple(c.ops), count(c), c._next_cbit, frozenset(c._live_qids))
    with pytest.raises(ValueError):
        if fault == "condition":
            with c.condition(cb):
                api(c, address, target, (0, 1, 2, 3))
        else:
            api(c, address, target, () if fault == "table" else (0, 1, 2, 3),
                **({"low_bits": -1} if fault == "split" else {}))
    assert c.exact_blocks == [] and c._exact_block_seals == {}
    assert before == (tuple(c.ops), count(c), c._next_cbit, frozenset(c._live_qids))
