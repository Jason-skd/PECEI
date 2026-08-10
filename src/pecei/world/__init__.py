"""World layer: ground-truth spatial model + primitive actions.

Imports nothing above itself (no observation/engine/llm). The World is the only
mutable placement state; everything else reads or drives it.
"""
from .actions import ActionType, ActionResult, apply_action
from .capability import DEFAULT_POLICY, Aggregation, capability, floats
from .component import Component, ComponentType
from .effects import (
    BURN_DURATION,
    BRITTLE_DURATION,
    EXTRA_STEP_BRITTLE,
    EXTRA_STEP_BURNING,
    EXTRA_STEP_SOAKED,
    EgoStatus,
    SOAK_DURATION,
)
from .entity import Direction, Entity
from .grid import Grid, Occupant
from .map_parser import MapSpec, load_map, load_map_spec, load_world, parse_map
from .world import World

__all__ = [
    "ActionType",
    "ActionResult",
    "Aggregation",
    "BURN_DURATION",
    "BRITTLE_DURATION",
    "Component",
    "ComponentType",
    "DEFAULT_POLICY",
    "Direction",
    "EXTRA_STEP_BRITTLE",
    "EXTRA_STEP_BURNING",
    "EXTRA_STEP_SOAKED",
    "EgoStatus",
    "Entity",
    "Grid",
    "MapSpec",
    "Occupant",
    "SOAK_DURATION",
    "World",
    "apply_action",
    "capability",
    "floats",
    "load_map",
    "load_map_spec",
    "load_world",
    "parse_map",
]
