"""Exact population/count audits only: no circuit construction or fitted rates."""

from collections import defaultdict
from math import gcd
import random

import pytest

from qsm.audit.history_math import Prefix, affine_floor_sum, trace_history
from qsm.inversion.reference import to_dialog_jump


PRIMES = [p for p in range(3, 252, 2)
          if all(p % d for d in range(3, int(p**0.5) + 1, 2))]


def _reference_rows(p, x):
    """Independent state reconstruction from the established reference's tape."""
    tape, final_u, final_v = to_dialog_jump(p, x, 4 * p.bit_length(), 2)
    assert (final_u, final_v) == (1, 0)
    u, v = p, x
    past = []
    for i, (shifts, odd, swap) in enumerate(tape):
        if not v:
            assert all(t == (2, 0, 0) for t in tape[i:])
            break
        entry_bits = max(u, v).bit_length()
        v //= 1 << shifts
        key = (tuple(past), shifts, odd)
        yield i, key, u, v, entry_bits, (shifts, odd, swap)
        if swap:
            u, v = v, u
        if odd:
            v -= u
        past.append((shifts, odd, swap))


def _brute_bad(prefix, k, w):
    B = 1 << max(0, w - k)
    return sum(
        0 <= v < u < (1 << w) and u // B == v // B
        for t in range(prefix.count)
        for u, v in [(prefix.u0 + prefix.du * t, prefix.v0 + prefix.dv * t)]
    )


@pytest.mark.parametrize("p", PRIMES)
def test_populations_against_exhaustive_reference_history_groups(p):
    groups = defaultdict(list)
    rows = {}
    keys = {}
    for x in range(1, p):
        actual = list(trace_history(p, x))
        reference = list(_reference_rows(p, x))
        assert len(actual) == len(reference)
        assert not actual[0].active
        for rec, (i, key, u, v, entry_bits, tape) in zip(actual, reference):
            assert (rec.i, rec.u, rec.v, rec.entry_bits, rec.tape) == (
                i, u, v, entry_bits, tape)
            assert rec.active == bool(i and tape[1])
            assert keys.setdefault(rec.history_key, key) == key
            groups[key].append(x)
            if key in rows:
                assert rows[key].prefix == rec.prefix
                assert rows[key].history_key == rec.history_key
            else:
                rows[key] = rec
            q = rec.prefix
            assert q.x0 + q.step * rec.rank == x
            assert 0 <= rec.rank < q.count
            assert (q.u0 + q.du * rec.rank, q.v0 + q.dv * rec.rank) == (u, v)
            assert rec.S == sum(t[0] for t in key[0]) + tape[0]
            assert rec.a * u + rec.b * v == p
            assert q.a * q.du + q.b * q.dv == 0
            assert rec.m == rec.a + rec.b
    for key, xs in groups.items():
        rec = rows[key]
        q = rec.prefix
        assert q.count == len(xs)
        assert [q.x0 + q.step * t for t in range(q.count)] == xs
        if not rec.active:
            continue
        assert q.a & 1 and not q.b & 1 and gcd(q.a, 2 * q.m) == 1
        r = q.residue_first_positive
        assert 1 <= r <= 2 * q.m and r % 2 == 0
        assert (q.a * r - (p - q.m)) % (2 * q.m) == 0
        for k in range(p.bit_length() + 2):
            w = p.bit_length()
            exact = q.bad_count(k, w)
            assert exact == _brute_bad(q, k, w)
            assert exact <= q.residue_bound(k, w) <= q.count
            B = 1 << max(0, w - k)
            candidates = sum((q.a * d - (p - q.m)) % (2 * q.m) == 0
                             for d in range(2, B, 2))
            assert q.residue_candidates(k, w) == candidates
            assert q.continuous_model(k, w) == B / (2 * q.m)
            if k >= w - 1:
                assert exact == 0
        # This checks cuts of negative as well as positive affine slopes.
        for w in range(1, p.bit_length()):
            assert q.bad_count(1, w, fit_only=True) == _brute_bad(q, 1, w)
            endpoints = (q.u0, q.v0, q.u0 + q.du * (q.count - 1),
                         q.v0 + q.dv * (q.count - 1))
            if max(endpoints) >= 1 << w:
                with pytest.raises(ValueError, match="fit"):
                    q.bad_count(1, w)
            else:
                assert q.bad_count(1, w) == _brute_bad(q, 1, w)


def test_floor_sum_signed_coefficients_and_empty_range():
    for count in range(6):
        for modulus in range(1, 6):
            for slope in range(-5, 6):
                for intercept in range(-5, 6):
                    assert affine_floor_sum(count, modulus, slope, intercept) == sum(
                        (intercept + slope * t) // modulus for t in range(count))
    rng = random.Random(718)
    for _ in range(100):
        count = rng.randrange(100)
        modulus = rng.randrange(1, 1 << 200)
        slope = rng.randrange(-(1 << 255), 1 << 255)
        intercept = rng.randrange(-(1 << 255), 1 << 255)
        assert affine_floor_sum(count, modulus, slope, intercept) == sum(
            (intercept + slope * t) // modulus for t in range(count))
    for n, m in [(-1, 2), (1, 0), (0, -1)]:
        with pytest.raises(ValueError):
            affine_floor_sum(n, m, 0, 0)


def test_distinct_histories_survive_coincident_states_and_current_order_is_unseen():
    by_state = defaultdict(set)
    by_key = defaultdict(set)
    even_to_odd = singleton = False
    for x in range(1, 252):
        if x == 251:
            continue
        previous = None
        for rec in trace_history(251, x):
            if rec.active:
                by_state[(rec.i, rec.u, rec.v)].add(rec.history_key)
                by_key[rec.history_key].add(rec.tape[2])
                if previous is not None and previous.tape[1] == 0:
                    even_to_odd = True
                if rec.prefix.count == 1:
                    singleton = True
                    assert rec.prefix.du == rec.prefix.dv == rec.rank == 0
            previous = rec
    assert any(len(keys) > 1 for keys in by_state.values())
    assert any(orders == {0, 1} for orders in by_key.values())
    assert even_to_odd and singleton


def test_selected_4093_and_full_width_tapes_and_inverse_reconstruction():
    rng = random.Random(4093)
    for p in [4093, (1 << 255) - 19]:
        xs = [1, 2, 3, 4, 8, p - 1, p - 2] + [rng.randrange(1, p) for _ in range(24)]
        for x in xs:
            rows = list(trace_history(p, x))
            ref = list(_reference_rows(p, x))
            assert len(rows) == len(ref)
            past = []
            for rec, expected in zip(rows, ref):
                assert (rec.u, rec.v, rec.tape, rec.entry_bits) == (
                    expected[2], expected[3], expected[5], expected[4])
                assert rec.a * rec.u + rec.b * rec.v == p
                # Reverse the current partial iteration, then all previous tapes.
                u, v = rec.u, rec.v << rec.tape[0]
                for shifts, odd, swap in reversed(past):
                    if odd:
                        v += u
                    if swap:
                        u, v = v, u
                    v <<= shifts
                assert (u, v) == (p, x)
                past.append(rec.tape)


def test_x3_late_singleton_and_model_is_not_an_upper_bound():
    p = (1 << 255) - 19
    rows = list(trace_history(p, 3))
    assert len(rows) == 131
    rec = rows[129]
    assert (rec.active, rec.u, rec.v, rec.entry_bits, rec.S) == (True, 3, 1, 3, 254)
    assert (rec.prefix.x0, rec.prefix.count, rec.rank) == (3, 1, 0)
    # A mathematical bucket wider than the actual values still has an exact error.
    assert rec.prefix.bad_count(100, 255) == 1
    assert rec.prefix.residue_bound(100, 255) == 1
    assert rec.prefix.continuous_model(100, 255) < 1
    assert rec.prefix.residue_first_positive == 2
    assert rec.prefix.bad_count(2, 3) == 0
    assert rows[-1].u == rows[-1].v == 1
    assert rows[-1].tape[2] == 0


@pytest.mark.parametrize("p,x", [(2, 1), (4, 1), (1, 1), (-3, 1),
                                  (29, 0), (29, 29), (29, -1), (15, 3)])
def test_trace_rejects_invalid_inputs(p, x):
    with pytest.raises(ValueError):
        list(trace_history(p, x))


def test_odd_composite_coprime_trace_and_helper_preconditions():
    rows = list(trace_history(15, 2))
    tape, u, v = to_dialog_jump(15, 2, len(rows), 2)
    assert [r.tape for r in rows] == tape
    assert (u, v) == (1, 0)
    inactive = next(r.prefix for r in trace_history(29, 8) if not r.tape[1])
    with pytest.raises(ValueError, match="odd"):
        inactive.bad_count(2, 5)
    with pytest.raises(ValueError, match="odd"):
        _ = inactive.residue_first_positive
    q = next(r.prefix for r in trace_history(29, 3) if r.active)
    for k, w in [(-1, 5), (1, 0)]:
        with pytest.raises(ValueError):
            q.bad_count(k, w)
        with pytest.raises(ValueError):
            q.residue_bound(k, w)
    # A constant-parity affine progression must not spuriously split in half.
    q = Prefix(1, 2, 3, 29, 0, 1, 2)
    assert q._parity(1) is q
    with pytest.raises(ValueError):
        q._parity(0)
