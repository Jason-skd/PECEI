"""Map parsing: JSON/YAML -> validated MapSpec -> Grid.

The map is a list of Entity specs placed at anchors. Validation via pydantic is
the "compile-time" check that the map itself is well-formed (consistent with
the project's typed-spec discipline). Everything occupying a cell is an Entity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .component import Component, ComponentType
from .entity import Direction, Entity
from .grid import Grid


class ComponentSpec(BaseModel):
    offset: tuple[int, int]
    type: ComponentType
    attrs: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> Component:
        return Component.of(self.type, **self.attrs)


class EntitySpec(BaseModel):
    name: str
    anchor: tuple[int, int]
    orientation: Direction = Direction.NORTH
    is_ego: bool = False
    components: list[ComponentSpec]

    def build(self) -> Entity:
        return Entity(
            eid=self.name,
            components={tuple(c.offset): c.build() for c in self.components},
            anchor=tuple(self.anchor),
            orientation=self.orientation,
            is_ego=self.is_ego,
        )


class MapSpec(BaseModel):
    width: int
    height: int
    entities: list[EntitySpec]

    def build_grid(self) -> Grid:
        grid = Grid(self.width, self.height)
        for spec in self.entities:
            grid.place(spec.build())
        return grid


def parse_map(data: dict[str, Any]) -> Grid:
    """Validate a parsed map dict and build its Grid."""
    return MapSpec.model_validate(data).build_grid()


def load_map(path: str | Path) -> Grid:
    """Read a JSON or YAML map file and build its Grid."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        # YAML is a JSON superset; accept .yaml/.yml and fall back to YAML.
        data = yaml.safe_load(text)
    return parse_map(data)
