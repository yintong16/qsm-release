"""Version registry: every circuit variant is registered under a unique name.

A `Case` bundles everything needed to benchmark a circuit version: the built circuit,
a classical reference (input values -> expected output values), and a random-input
generator. Registering a duplicate version name raises — so variants must be named
distinctly (`layer.name.vN`), old versions stay runnable, and nothing is silently
overwritten.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.circuit import Circuit

RefFn = Callable[[dict[str, int]], dict[str, int]]
GenFn = Callable[[random.Random], dict[str, int]]


@dataclass
class Case:
    circuit: Circuit
    reference: RefFn
    gen_input: GenFn
    meta: dict = field(default_factory=dict)
    # Analysis-only objects are excluded from persisted metadata and circuit identity.
    analysis_context: dict = field(default_factory=dict)


_REGISTRY: dict[str, Callable[[], Case]] = {}


def circuit_version(version: str):
    """Decorator registering a zero-arg builder that returns a `Case`."""
    def deco(builder: Callable[[], Case]) -> Callable[[], Case]:
        if version in _REGISTRY:
            raise ValueError(
                f"duplicate circuit version {version!r} — pick a new name so old "
                f"versions stay comparable"
            )
        _REGISTRY[version] = builder
        return builder
    return deco


def get_builder(version: str) -> Callable[[], Case]:
    if version not in _REGISTRY:
        raise KeyError(f"no circuit version {version!r} registered")
    return _REGISTRY[version]


def all_versions() -> list[str]:
    return sorted(_REGISTRY)
