"""Cache registered resource counts without retaining every million-gate circuit.

Registry-wide tests revisit finished builders. Resource records are small and can remain
cached for the session; emitted circuits must be bounded because conservative variants
contain tens of millions of operations. Ad-hoc circuits remain mutable and uncached.
"""

from __future__ import annotations

import functools

_ATTR = "_qsm_test_resources"
_count = None


def pytest_configure(config) -> None:
    global _count
    from qsm.core import resources

    _count = resources.count

    @functools.wraps(_count)
    def count(circ):
        got = getattr(circ, _ATTR, None)
        return _count(circ) if got is None else got

    resources.count = count


def pytest_collection_finish(session) -> None:
    from qsm.bench import registry

    resource_cache = {}

    @functools.lru_cache(maxsize=2)
    def cached(build):
        case = build()
        if build not in resource_cache:
            resource_cache[build] = _count(case.circuit)
        setattr(case.circuit, _ATTR, resource_cache[build])
        return case

    def memoise(build):
        @functools.wraps(build)
        def wrapped():
            return cached(build)
        return wrapped

    for name, build in list(registry._REGISTRY.items()):
        registry._REGISTRY[name] = memoise(build)
