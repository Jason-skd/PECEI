from pathlib import Path

from pecei.infra import complexity
from pecei.world import Component, Entity, World, load_world

REPO = Path(__file__).resolve().parents[1]
EXAMPLE02 = REPO / "src" / "pecei" / "maps" / "example02.yaml"


def _brain(eid, anchor, **attrs):
    return Entity(eid=eid, components={(0, 0): Component.of("brain", **attrs)},
                  anchor=anchor, is_ego=True)


def test_clear_path_complexity():
    w = World.empty(5, 1, goal=(4, 0))
    w.add(_brain("ego", (0, 0)))
    assert complexity(w, "ego", (4, 0)) == 4.0


def test_fire_penalty_for_non_fireproof():
    w = World.empty(3, 1, goal=(2, 0))
    w.add(_brain("ego", (0, 0)))
    w.add(Entity(eid="fire", components={(0, 0): Component.of("fire")}, anchor=(1, 0)))
    assert complexity(w, "ego", (2, 0)) == 10.0  # (1+8) + 1


def test_fireproof_entity_ignores_fire():
    w = World.empty(3, 1, goal=(2, 0))
    w.add(_brain("ego", (0, 0), fireproof=True))
    w.add(Entity(eid="fire", components={(0, 0): Component.of("fire")}, anchor=(1, 0)))
    assert complexity(w, "ego", (2, 0)) == 2.0


def test_wall_blocks_returns_none():
    w = World.empty(3, 1, goal=(2, 0))
    w.add(_brain("ego", (0, 0)))
    w.add(Entity(eid="wall", components={(0, 0): Component.of("stone")}, anchor=(1, 0)))
    assert complexity(w, "ego", (2, 0)) is None


def test_example02_complexity_is_three_steps():
    w = load_world(EXAMPLE02)
    ego = w.ego
    assert ego.anchor == (3, 2)
    assert complexity(w, ego.eid, w.goal) == 3.0
