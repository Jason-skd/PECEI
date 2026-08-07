from pecei.world import Component, Entity, Aggregation, capability, floats


def _entity(*comps):
    return Entity(eid="x", components={(i, 0): c for i, c in enumerate(comps)})


def test_sum_aggregation():
    e = _entity(Component.of("wood"), Component.of("wood"))
    assert capability(e, "weight") == 2.0
    assert capability(e, "buoyancy") == 1.0  # 0.5 per wood


def test_all_and_any_modes():
    assert capability(_entity(Component.of("stone")), "fireproof") is True
    assert capability(_entity(Component.of("wood")), "fireproof") is False
    # ALL: mixed -> False
    assert capability(_entity(Component.of("stone"), Component.of("wood")), "fireproof") is False
    # ANY: wood burns
    assert capability(_entity(Component.of("wood")), "burn") is True
    assert capability(_entity(Component.of("stone")), "burn") is False


def test_floats_heuristic_and_override():
    assert floats(_entity(Component.of("wood"), Component.of("wood"))) is False  # 1.0 < 2.0
    assert floats(_entity(Component.of("wood", buoyancy=5.0))) is True


def test_custom_policy_overrides_default():
    e = _entity(Component.of("stone"), Component.of("wood"))
    custom = {"fireproof": Aggregation.ANY}  # any fireproof component => entity fireproof
    assert capability(e, "fireproof", policy=custom) is True
