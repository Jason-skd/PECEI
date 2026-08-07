from pecei.observation import observe
from pecei.world import Component, ComponentType, Direction, Entity, World


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
    # front (north): relative (0,-1),(0,-2) in cone
    assert obs.at(0, -1) is not None
    assert obs.at(0, -2) is not None
    # sides (±1, 0): 90° off-axis, outside cone
    assert obs.at(1, 0) is None
    assert obs.at(-1, 0) is None
    # behind (0, +1): outside cone
    assert obs.at(0, 1) is None


def test_occlusion_blocks_cell_behind_wall():
    wall = Entity(eid="w", components={(0, 0): Component.of("stone")}, anchor=(2, 1))
    world = _eye_world(extra=[wall])
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    front = obs.at(0, -1)
    assert front is not None and ComponentType.STONE in front.ctypes  # wall cell seen
    assert obs.at(0, -2) is None  # cell behind the wall occluded


def test_no_leak_of_hidden_entity():
    hidden = Entity(eid="h", components={(0, 0): Component.of("fire")}, anchor=(2, 4))
    world = _eye_world(extra=[hidden])
    obs = observe(world, "eye", vision_range=3, half_angle=45)
    assert obs.at(0, 2) is None  # behind observer, outside cone
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
    # body extends to relative (0,1),(0,2) — behind the north cone, but self-perceived
    assert obs.at(0, 1) is not None
    assert obs.at(0, 2) is not None
