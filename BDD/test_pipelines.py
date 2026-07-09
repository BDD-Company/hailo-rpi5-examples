"""Pipeline-string tests for the inference wrapper.

These are pure string assertions — no GStreamer pipeline is built — but pipelines.py
imports hailo_apps, which only exists on the Pi. So the whole module skips on a dev
host and runs on-device.

Why bother: the tiled/whole-frame choice and the element NAMES are load-bearing.
app_base.switch_tiling() looks up `valve_whole` / `valve_tile` / `branch_selector` by
name, and _install_detection_start_probe() looks up `<branch>_wrapper_input_q`. Nothing
else pins those strings together, so a rename in one place fails silently at runtime.
"""

import pytest

pytest.importorskip("hailo_apps", reason="pipelines.py imports hailo_apps (Pi only)")

from pipelines import (  # noqa: E402
    INFERENCE_PIPELINE_WRAPPER,
    SWITCHABLE_DETECTION_SECTION,
)


INNER = 'fakeinner'


def test_whole_frame_uses_whole_buffer_cropper_and_no_tile_queue():
    s = INFERENCE_PIPELINE_WRAPPER(INNER, tiles_x=1, tiles_y=1)
    assert 'hailocropper' in s and 'hailotilecropper' not in s
    assert 'hailoaggregator' in s and 'hailotileaggregator' not in s
    # The whole-frame inference branch must link straight to the aggregator: an extra
    # queue there is a thread hand-off the lowest-latency path does not need.
    assert 'tile_q' not in s


def test_tiled_uses_tile_cropper_and_flattens_detections():
    s = INFERENCE_PIPELINE_WRAPPER(INNER, tiles_x=2, tiles_y=1)
    assert 'hailotilecropper' in s
    assert 'tiles-along-x-axis=2' in s and 'tiles-along-y-axis=1' in s
    # Without flatten-detections the callback's get_objects_typed() silently finds zero.
    assert 'flatten-detections=true' in s


def test_tile_iou_threshold_is_configurable():
    s = INFERENCE_PIPELINE_WRAPPER(INNER, tiles_x=2, tiles_y=2, tile_iou_threshold=0.15)
    assert 'iou-threshold=0.15' in s


def test_zero_overlap_omits_the_overlap_properties():
    """0.0 means abutting tiles; emitting overlap-*-axis=0.0 is noise, not behaviour."""
    s = INFERENCE_PIPELINE_WRAPPER(INNER, tiles_x=2, tiles_y=2, tiling_overlap=0.0)
    assert 'overlap-x-axis' not in s

    s = INFERENCE_PIPELINE_WRAPPER(INNER, tiles_x=2, tiles_y=2, tiling_overlap=0.1)
    assert 'overlap-x-axis=0.1' in s and 'overlap-y-axis=0.1' in s


def test_bypass_queue_is_never_leaky():
    """hailoaggregator pairs sink_0 (bypass) with sink_1 (inference) by buffer. A leaky
    bypass queue drops the buffer it is waiting for and the aggregator stops emitting
    forever — so this must stay leaky=no on both the whole-frame and tiled paths."""
    for tiles in ((1, 1), (2, 2)):
        s = INFERENCE_PIPELINE_WRAPPER(INNER, tiles_x=tiles[0], tiles_y=tiles[1])
        bypass = next(part for part in s.split('!') if 'bypass_q' in part)
        assert 'leaky=no' in bypass, (tiles, bypass)


def test_branch_names_follow_the_wrapper_name():
    """switch_tiling() and the detection-start probe look these up by name."""
    s = INFERENCE_PIPELINE_WRAPPER(INNER, name='tile_wrapper', tiles_x=2, tiles_y=1)
    assert 'name=tile_wrapper_input_q' in s
    assert 'name=tile_wrapper_crop' in s
    assert 'name=tile_wrapper_agg' in s


# ---------------------------------------------------------------------------
# SWITCHABLE_DETECTION_SECTION: which branch boots live.
# ---------------------------------------------------------------------------
def _valve_drop(section, valve_name):
    """Extract `drop=<x>` from the `valve name=<valve_name> drop=<x>` element."""
    frag = section.split(f'valve name={valve_name} ')[1]
    return frag.split()[0].split('=')[1]


def test_switchable_section_boots_on_whole_frame_by_default():
    s = SWITCHABLE_DETECTION_SECTION('WHOLE', 'TILE')
    assert _valve_drop(s, 'valve_whole') == 'false'   # whole-frame feeds
    assert _valve_drop(s, 'valve_tile') == 'true'     # tile branch idle


def test_switchable_section_boots_on_tiling_when_asked():
    s = SWITCHABLE_DETECTION_SECTION('WHOLE', 'TILE', start_on_tiling=True)
    assert _valve_drop(s, 'valve_whole') == 'true'    # whole-frame idle
    assert _valve_drop(s, 'valve_tile') == 'false'    # tile branch feeds


def test_switchable_section_always_names_what_switch_tiling_looks_up():
    """switch_tiling() resolves these three by name; nothing else pins them together."""
    for start in (False, True):
        s = SWITCHABLE_DETECTION_SECTION('WHOLE', 'TILE', start_on_tiling=start)
        for name in ('valve_whole', 'valve_tile', 'branch_selector'):
            assert f'name={name}' in s, (start, name)
        # whole -> sink_0, tile -> sink_1: switch_tiling maps to_tiling onto these pads.
        assert 'WHOLE ! branch_selector.sink_0' in s
        assert 'TILE ! branch_selector.sink_1' in s
