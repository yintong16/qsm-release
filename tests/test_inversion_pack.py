"""The 5-per-3 dialog packing, classically and as a circuit.

The packing is the thing that makes the replay possible at all -- 665 + 255 + 255 is
`PEAK_BUDGET` and 798 + 510 is 133 over -- so its two halves have to be exactly inverse and
the accumulator has to be exactly base-3. Both are small enough to check exhaustively, and
this file does, at every digit position and every reachable accumulator value.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from qsm.core.circuit import Circuit
from qsm.core.resources import count
from qsm.core.simulator import simulate
from qsm.inversion import pack

#: The three reachable (b0, s) patterns. (0, 1) needs b1 without b0, and s = b0 & b1.
PATTERNS = [(0, 0), (1, 0), (1, 1)]


# --- the classical map -----------------------------------------------------
@pytest.mark.parametrize("length", range(1, 8))
def test_pack_round_trips_exhaustively(length: int):
    """Every tape of this length survives, including the partial trailing group."""
    for tape in itertools.product(PATTERNS, repeat=length):
        tape = list(tape)
        d = pack.pack_dialog(tape)
        assert pack.unpack_dialog(d, length) == tape
        assert d < 1 << pack.packed_bits(length), "packed value overflows its own width"


def test_pack_is_injective_on_a_full_group():
    """27 states into 5 qubits, with no collisions -- the whole claim, on one group."""
    seen = {pack.pack_dialog(list(t)) for t in itertools.product(PATTERNS, repeat=3)}
    assert len(seen) == 27
    assert max(seen) < 1 << pack.GROUP_BITS


def test_the_packed_width_is_665_and_not_much_above_the_floor():
    """665 at 399 iterations, against 798 unpacked and a 633-qubit information floor.

    The floor is log2(3) per iteration; 5/3 spends 5.2% over it. Recorded rather than
    chased: the denser 3^5-in-8-bits packing would reach 640 and is deferred, because the
    saving only matters if the replay turns out not to fit at 665.
    """
    assert pack.packed_bits(399) == 665
    assert pack.packed_bits(399) < 2 * 399
    floor = math.ceil(399 * math.log2(3))
    assert floor == 633
    assert pack.packed_bits(399) / floor < 1.06


# --- the fold circuit ------------------------------------------------------
def _run(j: int, r_in: int, p_in: int = 0, invert: bool = False):
    """Run fold_digit (or its inverse) on one concrete state and read everything back.

    At digit 0 the pair *is* the accumulator, so there is no second register; `p_in` is
    ignored there and the caller puts the pattern into `r_in`.
    """
    c = Circuit()
    r = c.input("r", pack.DIGIT_WIDTH[j])
    if j == 0:
        pair, inputs = (r[0], r[1]), {"r": r_in}
    else:
        p = c.input("p", 2)
        pair, inputs = (p[0], p[1]), {"r": r_in, "p": p_in}
    (pack.unfold_digit if invert else pack.fold_digit)(c, r, j, *pair)
    return simulate(c, inputs, rng=random.Random(j * 97 + r_in * 7 + p_in))


@pytest.mark.parametrize("j", range(pack.GROUP))
def test_fold_digit_is_base_3_and_clears_the_pair(j: int):
    """R += 3^j * t, and the two qubits the digit arrived on come back |0>.

    Exhaustive over every accumulator value reachable at this digit position, which is
    what licenses reading `t = R div 3^j` back out -- the property `unfold_digit` and
    therefore the whole replay depends on.
    """
    for prev in range(3 ** j):
        for b0, s in PATTERNS:
            pat = b0 | (s << 1)
            r = _run(j, pat if j == 0 else prev, pat)
            assert r.phase == 0
            expected = prev + (3 ** j) * (b0 + s)
            assert r.outputs["r"] == expected, (j, prev, b0, s, r.outputs["r"])
            if j:
                assert r.outputs["p"] == 0, f"digit {j} left {r.outputs['p']} behind"


@pytest.mark.parametrize("j", range(pack.GROUP))
def test_unfold_digit_inverts_fold_digit(j: int):
    """The replay reads the tape with this, so 'inverse' has to mean bit-for-bit."""
    for prev in range(3 ** j):
        for b0, s in PATTERNS:
            # Feed back exactly what the fold produced: the accumulator holding the digit,
            # and (above digit 0) a pair the fold left in |0>.
            r = _run(j, prev + (3 ** j) * (b0 + s), 0, invert=True)
            assert r.phase == 0
            if j == 0:
                assert r.outputs["r"] == b0 | (s << 1), (j, prev, b0, s)
            else:
                assert r.outputs["r"] == prev, (j, prev, b0, s, r.outputs["r"])
                assert r.outputs["p"] == b0 | (s << 1), (j, prev, b0, s)


def test_digit_zero_is_free_and_the_others_are_nearly_so():
    """One CNOT, then two small constant adds and two constant compares.

    The thermometer-is-already-binary observation is what keeps this cheap: paying an
    arithmetic fold on all three digits would cost half again as much for nothing.
    """
    tof = []
    for j in range(pack.GROUP):
        c = Circuit()
        r = c.input("r", pack.DIGIT_WIDTH[j])
        pair = (r[0], r[1]) if j == 0 else tuple(c.input("p", 2).ids)
        pack.fold_digit(c, r, j, *pair)
        tof.append(count(c).toffoli)
    assert tof[0] == 0, "digit 0 is a relabelling"
    assert sum(tof) <= 32, f"a group costs {sum(tof)} Toffoli; the loop it rides is ~1,600"
