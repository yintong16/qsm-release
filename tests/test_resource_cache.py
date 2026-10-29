"""Registry caching bounds live IR while retaining counts for stable builders."""

import gc
import weakref

import conftest

from qsm.bench import registry
from qsm.bench.registry import Case
from qsm.core.circuit import Circuit
from qsm.core import resources


def test_registry_cache_bounds_live_circuits_and_reuses_counts(monkeypatch):
    counted = []
    built = []
    raw_count = conftest._count

    def count(circ):
        counted.append(id(circ))
        return raw_count(circ)

    def builder():
        c = Circuit()
        q = c.input("x", 1)
        c.x(q[0])
        built.append(weakref.ref(c))
        return Case(c, lambda v: {"x": v["x"] ^ 1}, lambda rng: {"x": 0})

    # Distinct builder identities, even though the tiny circuits happen to agree.
    builders = {str(i): (lambda: builder()) for i in range(3)}
    monkeypatch.setattr(registry, "_REGISTRY", builders)
    monkeypatch.setattr(conftest, "_count", count)
    conftest.pytest_collection_finish(None)
    for _ in range(2):
        for build in registry._REGISTRY.values():
            case = build()
            assert resources.count(case.circuit).n_gates == 1
    del case
    gc.collect()
    assert len(built) == 6
    assert len(counted) == 3
    assert sum(ref() is not None for ref in built) == 2


def test_ad_hoc_counts_are_not_frozen():
    c = Circuit()
    q = c.input("x", 1)
    c.x(q[0])
    first = resources.count(c).n_gates
    c.x(q[0])
    assert resources.count(c).n_gates == first + 1
