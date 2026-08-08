"""Navigable wrappers over an Observation for DSL attribute access.

The interpreter evaluates ``ob.front.is_blocked`` etc. against these. ``NavObs``
exposes direction-relative cells (front/left/right/back/here, relative to the
observer's facing); ``NavCell`` exposes type predicates + ``ctype``.
"""
from __future__ import annotations

from pecei.observation.observation import CellView, Observation
from pecei.world.component import ComponentType

_SOLID = {
    ComponentType.STONE,
    ComponentType.METAL,
    ComponentType.WOOD,
    ComponentType.WHEEL,
    ComponentType.BRAIN,
}


class NavCell:
    def __init__(self, cv: CellView | None, is_goal_cell: bool = False) -> None:
        self._cv = cv
        self._is_goal = is_goal_cell

    def _types(self) -> set[ComponentType]:
        return set(self._cv.ctypes) if self._cv else set()

    @property
    def is_empty(self) -> bool:
        return not self._types()

    @property
    def is_blocked(self) -> bool:
        return any(t in _SOLID for t in self._types())

    @property
    def is_goal(self) -> bool:
        return self._is_goal

    @property
    def ctype(self) -> str:
        if self._cv and self._cv.ctypes:
            return self._cv.ctypes[-1].value
        return "empty"

    def _has(self, t: ComponentType) -> bool:
        return t in self._types()

    is_fire = property(lambda self: self._has(ComponentType.FIRE))
    is_water = property(lambda self: self._has(ComponentType.WATER))
    is_stone = property(lambda self: self._has(ComponentType.STONE))
    is_wood = property(lambda self: self._has(ComponentType.WOOD))
    is_metal = property(lambda self: self._has(ComponentType.METAL))
    is_wheel = property(lambda self: self._has(ComponentType.WHEEL))
    is_brain = property(lambda self: self._has(ComponentType.BRAIN))


class NavObs:
    def __init__(self, obs: Observation, goal: tuple[int, int] | None) -> None:
        self._obs = obs
        self._goal = tuple(goal) if goal else None

    def _cell(self, name: str) -> NavCell:
        d = self._obs.orientation
        deltas = {
            "front": d.delta,
            "right": d.rotate(1).delta,
            "back": (-d.delta[0], -d.delta[1]),
            "left": d.rotate(-1).delta,
            "here": (0, 0),
        }
        dx, dy = deltas[name]
        cv = self._obs.at(dx, dy)
        wx = self._obs.anchor[0] + dx
        wy = self._obs.anchor[1] + dy
        is_goal = self._goal == (wx, wy)
        return NavCell(cv, is_goal)

    front = property(lambda self: self._cell("front"))
    right = property(lambda self: self._cell("right"))
    back = property(lambda self: self._cell("back"))
    left = property(lambda self: self._cell("left"))
    here = property(lambda self: self._cell("here"))
