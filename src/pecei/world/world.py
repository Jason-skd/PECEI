"""World: the canonical mutable game state.

Owns the Entity objects (single source of truth for anchors/orientation) and a
Grid (spatial index kept in sync). The World is the only layer that holds
mutable placement state; primitive actions mutate it, observation/engine read or
drive it. Environment dynamics (fire/water/wood/metal effects) tick here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .component import ComponentType
from .effects import BURN_DURATION, BRITTLE_DURATION, EgoStatus, SOAK_DURATION
from .entity import Entity
from .grid import Grid


@dataclass
class World:
    grid: Grid
    entities: dict[str, Entity] = field(default_factory=dict)
    goal: tuple[int, int] | None = None
    ego_status: EgoStatus = field(default_factory=EgoStatus)

    @classmethod
    def empty(cls, width: int, height: int, goal: tuple[int, int] | None = None) -> "World":
        return cls(grid=Grid(width, height), goal=goal)

    def add(self, entity: Entity) -> None:
        if entity.anchor is None:
            raise ValueError(f"entity {entity.eid!r} has no anchor; cannot add to world")
        self.entities[entity.eid] = entity
        self.grid.place(entity)

    def entity(self, eid: str) -> Entity:
        return self.entities[eid]

    @property
    def ego(self) -> Entity | None:
        for e in self.entities.values():
            if e.is_ego:
                return e
        return None

    def occupies(self, eid: str, x: int, y: int) -> bool:
        return any(o.eid == eid for o in self.grid.occupants(x, y))

    def tick_environment(self) -> int:
        """Advance non-agent dynamics by one round; return extra steps lost.

        Terrain drives the ego's status:
          - standing on ``fire`` (and not soaked)  -> burning (lasts BURN_DURATION)
          - standing on ``water``                  -> soaked; quenches burning
          - soaked + metal component               -> brittle (lasts BRITTLE_DURATION)
          - burning body destroys any wood in its own cells (wood becomes rubble)
        Returns the number of extra steps imposed by currently active statuses
        (the round engine adds them to the round counter).
        """
        ego = self.ego
        if ego is None:
            return 0
        st = self.ego_status
        cells = list(ego.placements())

        if self._cell_has(cells, ComponentType.WATER):
            st.soaked = True
            st.soaked_left = SOAK_DURATION
            st.burning = False          # water quenches fire immediately
            st.burning_left = 0

        if self._cell_has(cells, ComponentType.FIRE) and not st.soaked:
            st.burning = True
            st.burning_left = BURN_DURATION

        if st.soaked and any(
            c.ctype is ComponentType.METAL for c in ego.components.values()
        ):
            st.brittle = True
            st.brittle_left = BRITTLE_DURATION

        if st.burning:
            self._burn_wood(cells)

        self._decay_status()
        return st.extra_steps()

    def _cell_has(self, cells: list[tuple[int, int]], ctype: ComponentType) -> bool:
        return any(
            self.grid.in_bounds(x, y)
            and any(o.component.ctype is ctype for o in self.grid.occupants(x, y))
            for (x, y) in cells
        )

    def _burn_wood(self, cells: list[tuple[int, int]]) -> None:
        """A burning ego destroys every wood component in ``cells``.

        Each destroyed component is removed from its Entity; an entity that ends
        up with no components is removed from the world entirely (wood rubble
        stops blocking movement).
        """
        for (x, y) in cells:
            if not self.grid.in_bounds(x, y):
                continue
            for occ in list(self.grid.occupants(x, y)):
                if occ.component.ctype is not ComponentType.WOOD:
                    continue
                ent = self.entities.get(occ.eid)
                if ent is None:
                    continue  # fixture occupant (e.g. GOAL marker); skip
                del ent.components[occ.local]
                self.grid.remove(ent.eid)
                if ent.components:
                    self.grid.place(ent)
                else:
                    self.entities.pop(ent.eid, None)

    def _decay_status(self) -> None:
        st = self.ego_status
        if st.burning:
            st.burning_left -= 1
            if st.burning_left <= 0:
                st.burning = False
        if st.soaked:
            st.soaked_left -= 1
            if st.soaked_left <= 0:
                st.soaked = False
        if st.brittle:
            st.brittle_left -= 1
            if st.brittle_left <= 0:
                st.brittle = False
