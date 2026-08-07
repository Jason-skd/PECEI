"""Capability aggregation: reduce per-component attributes to entity-level.

Pluggable via a policy mapping attribute -> :class:`Aggregation`. Defaults are
placeholders (per-type buoyancy/weight numbers are illustrative, to be tuned
when survival dynamics land in the environment tick).
"""
from __future__ import annotations

from enum import Enum

from .entity import Entity


class Aggregation(str, Enum):
    SUM = "sum"
    ALL = "all"
    ANY = "any"


DEFAULT_POLICY: dict[str, Aggregation] = {
    "weight": Aggregation.SUM,
    "buoyancy": Aggregation.SUM,
    "fireproof": Aggregation.ALL,  # entity fireproof iff every component is
    "burn": Aggregation.ANY,       # entity can burn iff any component can
}


def capability(
    entity: Entity, name: str, policy: dict[str, Aggregation] | None = None
) -> float | bool:
    """Aggregate component attribute ``name`` across the whole entity."""
    mode = (policy or DEFAULT_POLICY).get(name, Aggregation.SUM)
    vals = [c.get(name) for c in entity.components.values() if c.get(name) is not None]
    if mode is Aggregation.SUM:
        return float(sum(vals))
    if mode is Aggregation.ALL:
        return bool(vals) and all(bool(v) for v in vals)
    if mode is Aggregation.ANY:
        return any(bool(v) for v in vals)
    raise ValueError(f"unknown aggregation {mode!r}")  # pragma: no cover


def floats(entity: Entity, policy: dict[str, Aggregation] | None = None) -> bool:
    """Placeholder buoyancy heuristic: entity floats iff buoyancy >= weight."""
    return capability(entity, "buoyancy", policy) >= capability(entity, "weight", policy)
