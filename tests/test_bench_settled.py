"""Two separate concerns, deliberately kept apart.

`select`/`matches_any` are pure filtering LOGIC and are tested against SYNTHETIC version
names. They used to be tested against real ones, which meant every time a circuit was
settled these tests failed for no reason -- the logic had not changed, the example had.
Names under `zzz.` match nothing in the real SETTLED list, so they stay valid forever.

The CONTENTS of the real list get their own tests at the bottom, which check coherence
(no typos, no stale entries) rather than pinning any particular circuit's status.
"""

import fnmatch

import pytest

from qsm.bench import registry
from qsm.bench.settled import SETTLED, is_settled, matches_any, select

# Synthetic. `zzz.*` is matched by nothing in SETTLED, `demo.*` by the real glob.
ALL = [
    "demo.and.v1",
    "demo.xor.v1",
    "zzz.alpha.v1",
    "zzz.alpha.v2",
    "zzz.beta.v1",
]
SETTLED_IN_ALL = ["demo.and.v1", "demo.xor.v1"]
ACTIVE_IN_ALL = ["zzz.alpha.v1", "zzz.alpha.v2", "zzz.beta.v1"]


# ---------------------------------------------------------------------------
# filtering logic
# ---------------------------------------------------------------------------
def test_default_hides_settled():
    got = select(ALL)
    for v in SETTLED_IN_ALL:
        assert v not in got, f"{v} is settled and should be hidden"
    for v in ACTIVE_IN_ALL:
        assert v in got, f"{v} is not settled and should be shown"


def test_all_includes_settled():
    assert select(ALL, include_settled=True) == ALL


def test_naming_a_settled_version_still_runs_it():
    """Explicit request overrides the settled list."""
    assert select(ALL, requested=["demo.and.v1"]) == ["demo.and.v1"]


def test_glob_in_requested():
    assert select(ALL, requested=["zzz.alpha.*"]) == ["zzz.alpha.v1", "zzz.alpha.v2"]


def test_exclude_applied_last_and_wins():
    # even an explicitly-named version can be excluded
    assert select(ALL, requested=["zzz.alpha.*"], exclude=["zzz.alpha.v1"]) == [
        "zzz.alpha.v2"
    ]
    # and exclude works against --all
    got = select(ALL, exclude=["demo.*"], include_settled=True)
    assert not any(v.startswith("demo.") for v in got)
    assert "zzz.beta.v1" in got


def test_unknown_exact_name_is_kept_so_the_harness_can_report_it():
    assert select(ALL, requested=["zzz.nope.v9"]) == ["zzz.nope.v9"]


def test_unknown_glob_matches_nothing_silently():
    assert select(ALL, requested=["nosuch.*"]) == []


def test_matcher_basics():
    assert is_settled("demo.and.v1")
    assert not is_settled("zzz.alpha.v1")
    assert matches_any("field.modadd.v2", ["field.*"])
    assert not matches_any("field.modadd.v2", ["mul.*"])


# ---------------------------------------------------------------------------
# contents of the real list: coherence, not status
# ---------------------------------------------------------------------------
def test_every_settled_entry_matches_something_registered():
    """Catches typos and entries left behind after a rename. An entry matching nothing
    is silently useless, which is exactly the kind of rot a settled list accumulates."""
    import qsm.bench.__main__  # noqa: F401  -- registers every layer

    versions = registry.all_versions()
    unmatched = [pat for pat in SETTLED
                 if not any(fnmatch.fnmatch(v, pat) for v in versions)]
    assert not unmatched, f"settled entries matching no registered version: {unmatched}"


def test_something_is_still_active():
    """A fully-settled registry means the default table is empty, which is almost
    certainly a mistake rather than a milestone."""
    import qsm.bench.__main__  # noqa: F401

    active = select(registry.all_versions())
    assert active, "every registered version is settled -- the default table would be empty"


@pytest.mark.slow
def test_versions_named_exact_really_are_exact():
    """A name is a claim. `field.modsub_direct_exact.v2` once shipped with
    `log2_fail = 2^-251.8` beside it, because the `_exact` suffix was copied from
    `modadd_lazy_exact.v4` -- where it is true -- to a circuit whose erasure predicate
    is approximate no matter how the truncation windows are set.

    The benchmark table caught it, but only because someone read the row. This makes it
    fail loudly instead.
    """
    import qsm.bench.__main__  # noqa: F401  -- registers every layer

    liars = []
    for name in registry.all_versions():
        if "exact" not in name:
            continue
        meta = registry.get_builder(name)().meta
        fail = meta.get("fail_analytic", 0.0)
        if fail or meta.get("tolerates_phase"):
            liars.append((name, fail, meta.get("tolerates_phase", False)))
    assert not liars, (
        "versions named *exact* that carry an error term "
        "(name, fail_analytic, tolerates_phase): " + repr(liars)
    )


@pytest.mark.slow
def test_trunc_naming_convention_holds():
    """Convention: the BARE name is untruncated, `_trunc` is the truncated variant.

    Every family follows it -- addconst_19 / addconst_19_trunc, cmp_lt / cmp_lt_trunc,
    modneg / modneg_trunc. modsub was briefly registered the other way round, with the
    bare name meaning *truncated* and the untruncated one carrying a `_notrunc` suffix
    invented to compensate. Two circuits in a family can only differ in which is the
    default, so getting it backwards silently inverts what `python -m qsm.bench modsub.v1`
    runs.

    Enforced structurally: wherever both `X.vN` and `X_trunc.vM` exist, the `_trunc` one
    must be the cheaper (that is what truncation is for) and must carry an error term.
    """
    import qsm.bench.__main__  # noqa: F401

    from qsm.core.resources import count

    versions = registry.all_versions()
    checked = 0
    for name in versions:
        if "_trunc." not in name:
            continue
        base_stem = name.split("_trunc.")[0]          # field.addconst_19_trunc.v1 -> field.addconst_19
        siblings = [v for v in versions if v.rsplit(".", 1)[0] == base_stem]
        if not siblings:
            continue          # a _trunc with no untruncated sibling is allowed
        for base in siblings:
            checked += 1
            t = registry.get_builder(name)()
            b = registry.get_builder(base)()
            assert count(t.circuit).toffoli < count(b.circuit).toffoli, (
                f"{name} should be cheaper than {base}; is the naming inverted?")
            assert t.meta.get("fail_analytic"), \
                f"{name} is named _trunc but declares no error term"
    assert checked, "expected at least one _trunc/base pair to check"


@pytest.mark.slow
def test_every_registered_version_actually_emits_gates():
    """Registration is a claim that there is a circuit here to measure.

    `field.double_wide.v1` was registered once. Doubling-without-reducing is a
    relabelling of wires -- bit i becomes bit i+1 -- so it emits nothing: zero Toffoli,
    zero Cliffords, zero depth. The benchmark row was therefore all zeros except
    peak_qubits, which reported the input register's own width back as if it had been
    measured. A reader scanning the table sees a circuit that costs nothing, rather than
    the truth, which is that it is not a circuit.

    The line this draws: a circuit version emits gates; a change of REPRESENTATION does
    not, and belongs to the scheduler's bookkeeping. Zero-gate helpers still get tested
    (see `test_double_wide_is_literally_free`) -- they just do not get a table row.
    """
    import qsm.bench.__main__  # noqa: F401  -- registers every layer

    from qsm.core.resources import count

    empty = [v for v in registry.all_versions()
             if count(registry.get_builder(v)().circuit).n_gates == 0]
    assert not empty, (
        "registered versions that emit no gates -- these are relabellings or type "
        f"changes, not circuits: {empty}")


def test_settled_list_has_no_duplicates():
    assert len(SETTLED) == len(set(SETTLED)), \
        [p for p in SETTLED if SETTLED.count(p) > 1]
