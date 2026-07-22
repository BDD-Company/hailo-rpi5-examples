"""Host test for the switchable N-branch detection-section string builder.
Pure string assembly — no GStreamer instantiation, so it runs off-device."""
from tiling_pipeline import build_switchable_detection_section


def _fake_branch(name, tx, ty):
    return f"[{name}:{tx}x{ty}]"


def _fake_queue(name):
    return f"queue name={name}"


def test_builds_one_valve_and_selector_pad_per_tier():
    grids = [(3, 2), (2, 1), (1, 1)]          # index 0 = most tiles, last = whole
    s = build_switchable_detection_section(grids, _fake_branch, _fake_queue)
    for i in range(3):
        assert f"valve_tier{i}" in s
        assert f"branch_selector.sink_{i}" in s
    assert "input-selector name=branch_selector" in s


def test_only_whole_valve_open_at_startup():
    grids = [(3, 2), (2, 1), (1, 1)]
    s = build_switchable_detection_section(grids, _fake_branch, _fake_queue)
    # whole is the LAST tier (index 2): its valve starts open, the tile valves closed.
    assert "valve name=valve_tier2 drop=false" in s
    assert "valve name=valve_tier0 drop=true" in s
    assert "valve name=valve_tier1 drop=true" in s


def test_two_tier_ladder_still_builds():
    s = build_switchable_detection_section([(2, 1), (1, 1)], _fake_branch, _fake_queue)
    assert "valve_tier0" in s and "valve_tier1" in s
    assert "valve name=valve_tier1 drop=false" in s   # whole open


def test_names_what_switch_to_tier_looks_up():
    """app_base.switch_to_tier() resolves `valve_tier{i}` and `branch_selector` by name,
    and maps tier i onto pad sink_i. Nothing else pins those strings together, so a
    rename on one side fails silently at runtime."""
    grids = [(3, 2), (2, 1), (1, 1)]
    s = build_switchable_detection_section(grids, _fake_branch, _fake_queue)
    assert "name=branch_selector" in s
    for i, (tx, ty) in enumerate(grids):
        assert f"name=valve_tier{i}" in s
        # tier i's branch must terminate on sink_i — that mapping IS the switch.
        assert f"[tier{i}:{tx}x{ty}] ! branch_selector.sink_{i}" in s


def test_grids_reach_the_branch_factory_in_ladder_order():
    seen = []
    build_switchable_detection_section(
        [(3, 2), (2, 1), (1, 1)],
        lambda name, tx, ty: seen.append((name, tx, ty)) or f"[{name}]",
        _fake_queue)
    assert seen == [("tier0", 3, 2), ("tier1", 2, 1), ("tier2", 1, 1)]
