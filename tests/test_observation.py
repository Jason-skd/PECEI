from pecei.observation import observe
from pecei.world import Component, ComponentType, Direction, Entity, World

# Canonical frame note: observe() returns an egocentric view rotated so the
# observer's gaze is always +x. "front" is therefore +x, "behind" is -x, and
# at(0,0) is the observer's own anchor cell. at() returns an empty CellView for
# unseen offsets (never None), so visibility assertions use `cells` membership.


def _eye_world(extra=None, orientation=Direction.NORTH, anchor=(2, 2)):
    world = World.empty(5, 5)
    world.add(
        Entity(
            eid="eye",
            components={(0, 0): Component.of("brain")},
            anchor=anchor,
            orientation=orientation,
            is_ego=True,
        )
    )
    for e in extra or []:
        world.add(e)
    return world


def test_cone_front_visible_sides_and_behind_hidden():
    world = _eye_world()
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    # front (canonical +x): (1,0),(2,0) in cone
    assert (1, 0) in obs.cells
    assert (2, 0) in obs.cells
    # sides (0, ±1): 90° off-axis, outside cone
    assert (0, 1) not in obs.cells
    assert (0, -1) not in obs.cells
    # behind (-1, 0): outside cone
    assert (-1, 0) not in obs.cells


def test_occlusion_blocks_cell_behind_wall():
    # observer faces NORTH; the stone sits one step ahead in world terms
    wall = Entity(eid="w", components={(0, 0): Component.of("stone")}, anchor=(2, 1))
    world = _eye_world(extra=[wall])
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    # canonical: the wall lies one step forward -> (1, 0)
    front = obs.at(1, 0)
    assert ComponentType.STONE in front.ctypes  # wall cell seen
    assert (2, 0) not in obs.cells              # cell behind the wall occluded


def test_no_leak_of_hidden_entity():
    hidden = Entity(eid="h", components={(0, 0): Component.of("fire")}, anchor=(2, 4))
    world = _eye_world(extra=[hidden])
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    for view in obs.cells.values():
        assert ComponentType.FIRE not in view.ctypes


def test_observer_always_sees_own_body():
    world = World.empty(5, 5)
    world.add(
        Entity(
            eid="rob",
            components={
                (0, 0): Component.of("wheel"),
                (0, 1): Component.of("metal"),
                (0, 2): Component.of("brain"),
            },
            anchor=(2, 1),
            orientation=Direction.NORTH,
            is_ego=True,
        )
    )
    obs = observe(world, "rob", vision_range=3, half_angle=45)
    # body extends behind the gaze (canonical -x); always self-perceived
    assert (0, 0) in obs.cells
    assert (-1, 0) in obs.cells
    assert (-2, 0) in obs.cells


def test_goal_visible_as_component():
    # goal marker stamped into the cell one step ahead of a NORTH-facing eye
    world = World.empty(5, 5, goal=(2, 1))
    world.add(
        Entity(
            eid="eye",
            components={(0, 0): Component.of("brain")},
            anchor=(2, 2),
            orientation=Direction.NORTH,
            is_ego=True,
        )
    )
    world.grid.stamp(2, 1, "goal", Component.of(ComponentType.GOAL))
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    front = obs.at(1, 0)   # canonical: one step ahead
    assert front.is_goal and front.ctype == "goal"


def test_at_unseen_offset_returns_empty_cell():
    world = _eye_world()
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    # (0, -1) is off-axis (outside the cone) -> safe empty cell, not None
    unseen = obs.at(0, -1)
    assert unseen is not None
    assert unseen.is_empty and not unseen.is_blocked and not unseen.is_goal


def test_map_edge_visible_as_boundary():
    # An eye at the bottom edge facing SOUTH must SEE the edge as a boundary
    # cell (is_blocked=True), not as empty — otherwise it walks into the edge
    # forever. Regression for the 02_one_wall boundary-stuck loops.
    from pecei.world.component import ComponentType
    world = World.empty(5, 5)
    world.add(
        Entity(
            eid="eye",
            components={(0, 0): Component.of("brain")},
            anchor=(2, 4),                 # bottom row
            orientation=Direction.SOUTH,   # gaze points off the map
            is_ego=True,
        )
    )
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    ahead = obs.at(1, 0)                   # canonical forward = one step south = OOB
    assert ComponentType.BOUNDARY in ahead.ctypes
    assert ahead.is_blocked                # the agent can see the edge blocks it
    assert ahead.ctype == "boundary"
