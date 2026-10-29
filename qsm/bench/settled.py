"""The settled list: versions that have been scrutinised and are no longer iterated on.

`python -m qsm.bench` hides these by default, so the table shows only what is under
active work. They are still registered, still benchmarkable, and their history stays in
`benchmarks/results.json` — this only affects the default view.

To see them anyway:
    python -m qsm.bench --all                 # everything
    python -m qsm.bench field.sub_cdkm.v1     # naming a version always runs it

EDIT THIS LIST. Add a version once you are satisfied with it; remove it the moment you
start iterating again. Entries are exact version keys or fnmatch globs.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable

SETTLED: list[str] = [
    "demo.*",              # framework smoke tests, not real circuits
    "field.sub_cdkm.v1",   # verified: same Toffoli/peak as its adder, +2n Cliffords
    "field.sub_gidney.v1",
    # Plain adders: the CDKM/Gidney tradeoff is measured and documented in
    # docs/decisions/02-carry-storage-vs-carry-flow.md. Nothing left to tune here until
    # the scalar-mult layer tells us where the global peak lands.
    "field.add_cdkm.v1",
    "field.add_gidney.v1",
    # Exact modular addition, v1..v3. Superseded by the approximate line of work; kept
    # registered as the exact baseline the approximate versions are measured against.
    "field.modadd.v1",
    "field.modadd.v2",
    "field.modadd.v3",
    # Non-truncated constant adders and comparators. Their truncated counterparts
    # (*_trunc) are ~6x cheaper and are what the design actually uses; these stay
    # registered as the exact baselines those are measured against.
    "field.addconst_19.v1",
    "field.addconst_19.v2",
    "field.addconst_p.v1",
    "field.addconst_p.v2",
    "field.cmp_ge_p.v1",
    "field.cmp_lt.v1",
    # Truncated constant adders and comparators. The rates are exact closed forms
    # (F/2^L and 2^-(k+1)), validated exhaustively at small widths, and the danger sets
    # are characterised by predicates in qsm/params.py. Retuning is a matter of changing
    # CARRY_WINDOW / CMP_BITS, not of changing these circuits.
    "field.addconst_19_trunc.v1",   # 43 Toffoli, 2^-40.8
    "field.addconst_p_trunc.v1",    # 43 Toffoli, 2^-40.8
    "field.cmp_ge_p_trunc.v1",      # 39 Toffoli, 2^-40.0
    "field.cmp_lt_trunc.v1",        # 40 Toffoli, 2^-40.0
    # b -= p, both rungs. The decomposition (flip bit U, then +19) is what makes the
    # truncation possible at all -- p does not fit in any small window -- and both are
    # measured in 03-approximate-arithmetic.md under "three rungs".
    "field.subconst_p_decomp.v1",   # 254 Toffoli, exact
    "field.subconst_p_trunc.v1",    # 43 Toffoli, 2^-40.8
    # Lazy modular addition. The headline Layer-0 result (339 Toffoli, 3.0x below
    # Litinski's 4n) and the most heavily scrutinised circuit here: the lazy contract,
    # the venting, the composition behaviour, the F(F-1)/2 lazy+lazy hole and the
    # laziness-vs-truncation accounting are all measured and written up. Settled not
    # because it is unimportant but because there is nothing left to learn from it at
    # this layer -- the next move on it is the multiplication layer's to make.
    "field.modadd_lazy.v4",
    "field.modadd_lazy_exact.v4",
    # ---------------------------------------------------------------------
    # Layer 0 CLOSED (2026-08-29). Everything below was reviewed across the
    # five audit dimensions, its precondition hazards characterised by exact
    # predicates in qsm/params.py, and its failure rates measured against
    # closed forms rather than assumed. `python -m qsm.audit all` re-checks the
    # lot. Nothing here is expected to change again unless Layer 1 or 2 asks
    # for a different contract -- at which point it gets a new vN, not an edit.
    #
    # THE ONLY UNSETTLED LAYER IS `mul.*`. That is deliberate: the default
    # `python -m qsm.bench` view is now exactly the multiplication work.
    "field.modadd_vented.v3",       # 1021 Toffoli; venting moves the failure to phase
    "field.moddouble.v1",           # 254 exact / 45 truncated -- the cheapest in the layer
    "field.moddouble_trunc.v1",
    "field.moddouble_inplace.v2",   # same arithmetic, +2n CNOTs to honour in-place
    "field.moddouble_inplace_trunc.v2",
    "field.moddouble_lazy_wide.v1",  # the only circuit that CLOSES the doubling non-closure
    "field.modneg.v1",              # 42 Toffoli, and corrupts above p -- a scheduling rule
    "field.modneg_trunc.v1",
    "field.modsub.v1",              # the sandwich; inherits modneg's single-operand hazard
    "field.modsub_trunc.v1",
    "field.modsub_direct.v2",       # 84 cheaper AND 250 orders safer; the one to use
    "field.modsub_direct_trunc.v2",
]


def matches_any(version: str, patterns: Iterable[str]) -> bool:
    """True if `version` matches any exact key or fnmatch glob in `patterns`."""
    return any(fnmatch.fnmatch(version, p) for p in patterns)


def is_settled(version: str) -> bool:
    return matches_any(version, SETTLED)


def select(
    all_versions: list[str],
    requested: list[str] | None = None,
    exclude: list[str] | None = None,
    include_settled: bool = False,
) -> list[str]:
    """Resolve which versions to benchmark.

    `requested` may contain exact keys or globs; naming something explicitly overrides
    the settled list (you asked for it, you get it). `exclude` is applied last and always
    wins, so it can drop explicitly-requested versions too.
    """
    if requested:
        chosen = [v for v in all_versions if matches_any(v, requested)]
        # keep exact names the registry doesn't know, so the harness reports them missing
        chosen += [r for r in requested if r not in all_versions and not _is_glob(r)]
    else:
        chosen = list(all_versions)
        if not include_settled:
            chosen = [v for v in chosen if not is_settled(v)]

    if exclude:
        chosen = [v for v in chosen if not matches_any(v, exclude)]
    return chosen


def _is_glob(s: str) -> bool:
    return any(ch in s for ch in "*?[")
