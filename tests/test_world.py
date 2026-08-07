from pathlib import Path

import pytest

from pecei.world import (
    Component,
    ComponentType,
    Direction,
    Entity,
    Grid,
    load_map,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "src" / "pecei" / "maps" / "example01.yaml"


def test_component_defaults_and_overrides():
    wood = Component.of("wood")
    assert wood.get("burn") is True
    assert wood.get("fireproof") is False

    stone = Component.of("stone")
    assert stone.get("fireproof") is True

    wet_wood = Component.of("wood", burn=False, wet=True)
    assert wet_wood.get("burn") is False
    assert wet_wood.get("wet") is True


def test_entity_placement_north():
    e = Entity(
        eid="robot",
        components={
            (0, 0): Component.of("wheel"),
            (0, 1): Component.of("metal"),
            (0, 2): Component.of("brain"),
        },
        anchor=(2, 2),
        orientation=Direction.NORTH,
    )
    cells = e.abs_cells()
    assert set(cells) == {(2, 2), (2, 3), (2, 4)}
    assert cells[(2, 2)].ctype is ComponentType.WHEEL
    assert cells[(2, 4)].ctype is ComponentType.BRAIN
    assert e.has_brain is True
    assert e.brain_local() == (0, 2)


def test_entity_orientation_east_rotates_footprint():
    # local offsets (0,0),(0,1),(0,2); EAST = one CW turn: (x,y) -> (-y,x)
    e = Entity(
        eid="robot",
        components={
            (0, 0): Component.of("wheel"),
            (0, 1): Component.of("metal"),
            (0, 2): Component.of("brain"),
        },
        anchor=(2, 2),
        orientation=Direction.EAST,
    )
    cells = e.abs_cells()
    assert set(cells) == {(2, 2), (1, 2), (0, 2)}
    assert cells[(2, 2)].ctype is ComponentType.WHEEL
    assert cells[(0, 2)].ctype is ComponentType.BRAIN


def test_unplaced_entity_cannot_place():
    e = Entity(eid="ghost", components={(0, 0): Component.of("stone")})
    with pytest.raises(ValueError):
        e.placements()


def test_grid_blocking_and_bounds():
    grid = Grid(3, 3)
    grid.place(Entity(eid="rock", components={(0, 0): Component.of("stone")}, anchor=(1, 1)))
    grid.place(Entity(eid="river", components={(0, 0): Component.of("water")}, anchor=(2, 0)))
    assert grid.is_blocked(1, 1) is True
    assert grid.is_blocked(2, 0) is False  # water is non-blocking terrain
    assert grid.is_blocked(0, 0) is False
    with pytest.raises(IndexError):
        grid.occupants(5, 5)


def test_load_example_map_structure_and_ascii():
    grid = load_map(EXAMPLE)
    assert grid.width == 6
    assert grid.height == 5

    def ctype(x: int, y: int) -> ComponentType:
        occs = grid.occupants(x, y)
        assert occs, f"expected occupant at ({x},{y})"
        return occs[-1].component.ctype

    # ego body (NORTH from anchor (1,1))
    assert ctype(1, 1) is ComponentType.WHEEL
    assert ctype(1, 2) is ComponentType.METAL
    assert ctype(1, 3) is ComponentType.BRAIN
    # river
    assert ctype(3, 1) is ComponentType.WATER
    assert ctype(3, 2) is ComponentType.WATER
    # fire
    assert ctype(4, 3) is ComponentType.FIRE

    expected = "\n".join([
        ". . . . . .",
        ". o . ~ . .",
        ". m . ~ . .",
        ". @ . . * .",
        ". . . . . .",
    ])
    assert grid.ascii() == expected
