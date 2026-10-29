"""Positive restoration is canonical on its interface, not a cure for upstream faults."""
from dataclasses import replace
from functools import partial
import itertools
import random

import pytest

from qsm.audit import zero_point as audit
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.symbolic_batch import compile_symbolic, simulate_batch
from qsm.field.add import gidney_add
from qsm.field.add_chunked import gidney_add_chunked
from qsm.field.modarith import modadd, modadd_lazy
from qsm.point.affine import sub_const_mod
from qsm.scalarmul import zero_point as point
from test_zero_point import _table, _kwargs, _rows, _audit, _live


P255 = (1 << 255)-19
LEGACY_DIGESTS = {
    (7, False): "32854cce0383a51d14e8a1809d78e1ff1a3aab658eb6e9e011c67d0842ea236b",
    (7, True): "dc07a98cb4698cfda339e52126a8bdc62958e4eb88a9e768bc0022ab4d6c4523",
    (31, False): "cd8b1650b043fde224b84a764e7f779c5c8105770c837272c0a9a05086ccacbe",
    (31, True): "7bc1f959bca2cc8810ffb0f76a8c113a72173b8a656e9fc175f919e055d5863c",
}


def adder(n, chunked):
    return partial(gidney_add_chunked, chunk=(n+2)//2, cmp_bits=None) if chunked else gidney_add


def negate(c, b, p, add_fn, truncated=False):
    n = len(b)
    for q in b:
        c.x(q)
    opts = dict(carry_window=45, cmp_bits=40) if truncated else {}
    sub_const_mod(c, b, ((1 << n)-p-1) % p, p, add_fn=add_fn, **opts)


@pytest.mark.parametrize("p", [13, 31, 61, P255])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("canonical", [False, True])
def test_zero_payload_local_check_all_measurement_outcomes(p, chunked, canonical):
    n = p.bit_length()
    rng = random.Random(20261014)
    values = (list(range(1, p)) if p < 100 else sorted(set(
        [1, 2, 18, 19, 20, p//2, p-20, p-19, p-2, p-1]
        + [rng.randrange(1, p) for _ in range(128)])))
    if p == P255:
        assert len(values) == 138
    c = Circuit()
    a, y = c.input("a", n), c.input("y", n)
    fn = adder(n, chunked)
    if canonical:
        modadd(c, a, y, p, add_fn=fn, inplace_const=False, pseudo_mersenne=True)
    else:
        opts = dict(carry_window=45, cmp_bits=40) if p == P255 else {}
        modadd_lazy(c, a, y, p, add_fn=fn, **opts)
    negate(c, y, p, fn, truncated=p == P255)
    result = simulate_batch(compile_symbolic(c),
        [{"a": v, "y": 0} for v in values],
        [{"a": v, "y": p-v} for v in values], output_moduli={"a": p, "y": p})
    assert result.bad_count == 0
    _live(c)


@pytest.mark.parametrize("p", [13, 31, 61])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("cleanup", ["paired", "split"])
def test_paired_positive_restoration_exhaustive_canonical_payloads(p, chunked, cleanup):
    n = p.bit_length()
    c = Circuit()
    address, x, y = c.input("j", 1), c.input("x", n), c.input("y", n)
    xs, ys = (1, p-1), (p-1, 2)
    fn = adder(n, chunked)
    point._restore_positive(c, address, x, y, xs, ys, p, fn, 0, [], unlookup=cleanup)
    negate(c, y, p, fn)
    rows = [{"j": j, "x": xx, "y": yy}
            for j, xx, yy in itertools.product(range(2), range(p), range(p))]
    expected = [{"j": v["j"], "x": (v["x"]+xs[v["j"]]) % p,
                 "y": (-v["y"]-ys[v["j"]]) % p} for v in rows]
    result = simulate_batch(compile_symbolic(c), rows, expected, output_moduli={"x": p, "y": p})
    zero_y = sum(1 << i for i, v in enumerate(expected) if v["y"] == 0)
    # The unchanged negation maps 0 to raw p; the supported torsion coset excludes y_out=0.
    assert result.noncanonical_mask == result.bad_mask == zero_y
    assert result.value_count == result.phase_count == result.dirty_count == result.summary_mismatch_count == 0
    for i, v in enumerate(result.outputs):
        assert v["x"] == expected[i]["x"]
        assert v["y"] == (p if zero_y >> i & 1 else expected[i]["y"])
    _live(c)


@pytest.mark.parametrize("p", [13, 61, P255])
@pytest.mark.parametrize("chunked", [False, True])
def test_lazy_replacement_has_a_nonzero_payload_phase_witness(p, chunked):
    n = p.bit_length()
    c = Circuit()
    a, b = c.input("a", n), c.input("b", n)
    fn = adder(n, chunked)
    opts = dict(carry_window=45, cmp_bits=40) if p == P255 else {}
    modadd_lazy(c, a, b, p, add_fn=fn, **opts)
    negate(c, b, p, fn, truncated=p == P255)
    got = simulate_batch(compile_symbolic(c), [{"a": 2, "b": p-1}],
                         [{"a": 2, "b": p-1}], output_moduli={"b": p})
    assert got.outputs == [{"a": 2, "b": p-1}]
    assert got.phase_count == got.bad_count == 1
    assert got.value_count == got.dirty_count == got.noncanonical_count == 0


@pytest.mark.parametrize("p", [13, 31, P255])
def test_canonical_restoration_does_not_claim_to_clean_arbitrary_lazy_inputs(p):
    n = p.bit_length()
    c = Circuit()
    a, b = c.input("a", n), c.input("b", n)
    modadd(c, a, b, p, add_fn=adder(n, True), inplace_const=False)
    result = simulate_batch(compile_symbolic(c), [{"a": 1, "b": p}],
                            [{"a": 1, "b": 1}], output_moduli={"b": p})
    assert result.dirty_count == result.bad_count == 1


@pytest.mark.parametrize("p,corrected", LEGACY_DIGESTS)
def test_legacy_modes_keep_their_pre_change_ir_identity(p, corrected):
    case = point.build_zero_point_case(2, phase_correction=corrected, **_kwargs(_table(p, 2)))
    assert compile_symbolic(case.circuit).ir_digest == LEGACY_DIGESTS[p, corrected]


@pytest.mark.parametrize("p,w", [(7, 2), (7, 3), (31, 2), (31, 3), (31, 4)])
@pytest.mark.parametrize("chunked", [False, True])
def test_point_addition_all_inputs_both_signs_and_old_faults(p, w, chunked):
    table = _table(p, w)
    kw = _kwargs(table)
    if chunked:
        kw["config"] = replace(kw["config"], adder_chunk=(p.bit_length()+2)//2)
    old = point.build_zero_point_case(w, **kw)
    new = point.build_zero_point_case(w, restoration="add", phase_correction=False, **kw)
    split = point.build_zero_point_case(w, restoration="add", phase_correction=False,
                                       addition_unlookup="split", **kw)
    rows, expected = _rows(old)
    before, after, split_result = [_audit(case, rows, expected) for case in (old, new, split)]
    assert after == split_result
    assert after.outputs == before.outputs == expected
    assert after.bad_mask == before.bad_mask == before.phase_mask
    assert after.bad_count > 0
    assert after.value_count == after.dirty_count == after.noncanonical_count == after.summary_mismatch_count == 0
    horizontal = sum(1 << i for i, v in enumerate(rows)
                     if v["y"] == table.point_for_word(v["word"])[1])
    assert horizontal and not after.bad_mask & horizontal
    assert {table.digit(v["word"]) < 0 for i, v in enumerate(rows) if horizontal >> i & 1} == {False, True}
    for case in (old, new, split):
        _live(case.circuit)
    assert new.meta["phase_correction"] is False
    assert new.meta["coordinate_restoration"] == "add"
    assert new.meta["restoration_exact_reduction"] is True
    assert new.meta["circuit_config"]["lookup_stages"] == 3
    assert new.meta["circuit_config"]["unlookup_stages"] == 3
    assert split.meta["circuit_config"]["unlookup_stages"] == 4


def test_prefix_and_first_two_tables_are_identical_and_compilation_restores_the_circuit():
    table = _table(31, 3)
    old = point.build_zero_point_case(3, **_kwargs(table))
    new = point.build_zero_point_case(3, restoration="add", phase_correction=False, **_kwargs(table))
    blocks = [[b for b in c.circuit.exact_blocks if b.kind == "xor"] for c in (old, new)]
    assert blocks[0][0].values == blocks[1][0].values
    assert blocks[0][1].values == blocks[1][1].values
    assert blocks[1][0].values == blocks[1][2].values
    assert blocks[0][2].values != blocks[1][2].values
    cuts = [[a for label, a, b in c.meta["resource_regions"] if label == "lookup"][2]
            for c in (old, new)]
    assert old.circuit.ops[:cuts[0]] == new.circuit.ops[:cuts[1]]
    prefixes = []
    for case in (old, new):
        c = case.circuit
        saved = c.ops, c.exact_blocks, c._cbit_const, c._next_cbit
        digest = compile_symbolic(c).ir_digest
        prefixes.append(audit._restoration_prefix(case))
        assert c.ops is saved[0] and c.exact_blocks is saved[1] and c._cbit_const is saved[2]
        assert c._next_cbit == saved[3]
        assert compile_symbolic(c).ir_digest == digest
    assert prefixes[0].ir_digest == prefixes[1].ir_digest


def test_non_f1_upstream_encoding_failures_remain_visible_without_reference_filtering():
    table = _table(13, 2)
    old = point.build_zero_point_case(2, **_kwargs(table))
    new = point.build_zero_point_case(2, restoration="add", phase_correction=False, **_kwargs(table))
    rows, expected = _rows(old)
    a, b = _audit(old, rows, expected), _audit(new, rows, expected)
    assert (a.bad_count, a.value_count, a.dirty_count, a.phase_count) == (38, 28, 22, 33)
    assert (b.bad_count, b.value_count, b.dirty_count, b.phase_count) == (36, 28, 24, 29)
    assert not b.bad_mask & ~a.bad_mask
    diag, analysis = audit._diagnostics(new, table, [{"kind": "ordinary", "input": row} for row in rows])
    assert len(diag) == 40 and analysis["restoration_prefix"]
    assert sum(not d["restoration_input"]["canonical_payloads"] for d in diag) == 10
    assert sum(d["all_outcome_clean"] for d in diag) == 4
    bad = next(row for i, row in enumerate(rows) if b.bad_mask >> i & 1)
    class Fixed:
        def __init__(self):
            self.values = iter((bad["x"], bad["y"], bad["word"]))
        def randrange(self, stop):
            value = next(self.values)
            assert 0 <= value < stop
            return value
    assert new.gen_input(Fixed()) == bad


@pytest.mark.parametrize("kwargs", [
    {"restoration": "unknown"},
    {"restoration": "add"},
    {"restoration": "add", "phase_correction": False, "addition_unlookup": "unknown"},
    {"addition_unlookup": "split"},
])
def test_invalid_policy_combinations_do_not_silently_change_legacy_flags(kwargs):
    with pytest.raises(ValueError):
        point.build_zero_point_case(3, **_kwargs(_table(31, 3)), **kwargs)


def test_old_false_still_means_negative_subtraction_without_patch():
    table = _table(31, 3)
    case = point.build_zero_point_case(3, phase_correction=False, **_kwargs(table))
    third = [b for b in case.circuit.exact_blocks if b.kind == "xor"][2]
    n = table.p.bit_length()
    assert third.values == tuple((-x % table.p) | ((-y % table.p) << n) for x, y in table.points)
    assert case.meta["coordinate_restoration"] == "subtract"


def test_selected_exact_adder_fits_the_existing_workspace_budget():
    c = Circuit()
    a, b = c.input("a", 255), c.input("b", 255)
    modadd(c, a, b, P255, add_fn=adder(255, True), inplace_const=False, pseudo_mersenne=True)
    resources = count(c)
    assert resources.toffoli == 1915
    assert resources.peak_qubits == 645
    assert resources.peak_qubits - 510 == 135


@pytest.mark.parametrize("chunked", [False, True])
def test_full255_nonzero_payloads_near_prime_and_word_boundaries(chunked):
    p, n, window = P255, 255, 1 << 45
    values = [(0, 0), (1, p-1), (2, p-1), (18, p-1), (19, p-1),
              (20, p-1), (21, p-1), (window-1, p-1), (window+1, p-1),
              (p-1, p-1), (p-20, 19), (p-19, 19), (p-18, 19)]
    c = Circuit()
    a, b = c.input("a", n), c.input("b", n)
    modadd(c, a, b, p, add_fn=adder(n, chunked), inplace_const=False, pseudo_mersenne=True)
    result = simulate_batch(compile_symbolic(c),
        [{"a": a, "b": b} for a, b in values],
        [{"a": a, "b": (a+b) % p} for a, b in values], output_moduli={"b": p})
    assert result.bad_count == 0
    assert all(v["b"] < p for v in result.outputs)


def test_rejected_lazy_policy_retains_its_production_tie_and_carry_witnesses():
    p, n = P255, 255
    c = Circuit()
    a, b = c.input("a", n), c.input("b", n)
    modadd_lazy(c, a, b, p, add_fn=adder(n, True), carry_window=45, cmp_bits=40)
    inputs = [{"a": 20, "b": p-1}, {"a": (1 << 45)+1, "b": p-1}]
    expected = [{"a": v["a"], "b": (v["a"]+v["b"]) % p} for v in inputs]
    result = simulate_batch(compile_symbolic(c), inputs, expected, output_moduli={"b": p})
    assert result.outputs[0] == expected[0] and result.phase_mask & 1
    assert result.outputs[1]["b"] == 0 and result.value_mask & 2
