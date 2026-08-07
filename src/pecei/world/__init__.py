"""World layer: ground-truth spatial model. Imports nothing above itself."""
from .component import Component, ComponentType
from .entity import Direction, Entity
from .grid import Grid, Occupant
from .map_parser import MapSpec, load_map, parse_map

__all__ = [
    "Component",
    "ComponentType",
    "Direction",
    "Entity",
    "Grid",
    "Occupant",
    "MapSpec",
    "load_map",
    "parse_map",
]
