#!/usr/bin/env python3
"""Unit tests for the pre-flight detection / in-flight pursuit flight-mode machinery.

Pure-Python (no hailo/picamera/gi), so it runs on the dev machine and on the Pi alike:
    python -m pytest BDD/test_flight_mode.py -q     # or: python BDD/test_flight_mode.py
"""

import threading

from helpers import (
    FlightMode,
    FlightModeController,
    Tile,
    XY,
    Rect,
    tile_layout,
    remap_tile_bbox,
    _axis_positions,
    DEFAULT_CAMERA_ID,
)


def _approx(a, b, eps=1e-6):
    return abs(a - b) <= eps


# --------------------------------------------------------------------------- #
# FlightModeController
# --------------------------------------------------------------------------- #

def test_controller_starts_in_detection_when_enabled():
    fmc = FlightModeController(enabled=True, detection_camera_id=2,
                               tiles_x=2, tiles_y=2, tile_size=(640, 640),
                               switch_after_consecutive_detections=7)
    assert fmc.enabled is True
    assert fmc.mode() is FlightMode.DETECTION
    assert fmc.is_detection() and not fmc.is_pursuit()
    assert fmc.detection_camera_id == 2
    assert fmc.tiles_x == 2 and fmc.tiles_y == 2
    assert fmc.tile_size == (640, 640)
    assert fmc.tile_count == 4
    assert fmc.switch_after_consecutive_detections == 7


def test_controller_starts_in_pursuit_when_disabled():
    fmc = FlightModeController(enabled=False)
    assert fmc.enabled is False
    assert fmc.mode() is FlightMode.PURSUIT
    assert fmc.is_pursuit() and not fmc.is_detection()
    assert fmc.detection_camera_id == DEFAULT_CAMERA_ID


def test_switch_to_pursuit_is_one_way_and_idempotent():
    fmc = FlightModeController(enabled=True)
    assert fmc.is_detection()
    assert fmc.switch_to_pursuit() is True     # transitioned
    assert fmc.is_pursuit()
    assert fmc.switch_to_pursuit() is False     # already there
    assert fmc.is_pursuit()


def test_switch_is_thread_safe_single_winner():
    fmc = FlightModeController(enabled=True)
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(fmc.switch_to_pursuit())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1, results   # exactly one thread sees the transition
    assert fmc.is_pursuit()


# --------------------------------------------------------------------------- #
# Tile geometry
# --------------------------------------------------------------------------- #

def test_axis_positions_basic_and_overlap():
    # 2 tiles of 640 over 1280 -> edge-to-edge, no overlap.
    assert _axis_positions(1280, 640, 2) == [0, 640]
    # 2 tiles of 640 over 720 -> forced overlap, last tile clamped to frame-tile.
    assert _axis_positions(720, 640, 2) == [0, 80]
    # Single tile -> origin only.
    assert _axis_positions(1280, 640, 1) == [0]


def test_axis_positions_rejects_gaps_and_oversize():
    for bad in (
        lambda: _axis_positions(2000, 640, 2),  # 2*640 < 2000 -> gaps
        lambda: _axis_positions(640, 720, 2),   # tile bigger than frame
        lambda: _axis_positions(1280, 640, 0),  # n < 1
    ):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_tile_layout_2x2_640_over_1280x720():
    tiles = tile_layout(1280, 720, 640, 640, 2, 2)
    assert len(tiles) == 4
    assert all(isinstance(t, Tile) for t in tiles)

    # Pixel rects (row-major: TL, TR, BL, BR).
    expected_px = [(0, 0, 640, 640), (640, 0, 1280, 640),
                   (0, 80, 640, 720), (640, 80, 1280, 720)]
    for t, exp in zip(tiles, expected_px):
        assert tuple(int(v) for v in t.pixel_rect.to_xyxy()) == exp

    # Every tile is exactly the HEF input size (no scaling needed downstream).
    for t in tiles:
        assert _approx(t.pixel_rect.width, 640)
        assert _approx(t.pixel_rect.height, 640)

    # Normalized extents: 640/1280 = 0.5 in x, 640/720 ~= 0.8889 in y.
    for t in tiles:
        assert _approx(t.extent_norm.x, 0.5)
        assert _approx(t.extent_norm.y, 640 / 720)

    # Origins.
    origins = [(round(t.origin_norm.x, 4), round(t.origin_norm.y, 4)) for t in tiles]
    assert origins == [(0.0, 0.0), (0.5, 0.0),
                       (0.0, round(80 / 720, 4)), (0.5, round(80 / 720, 4))]


def test_tile_layout_covers_whole_frame():
    # The union of the 4 tiles must cover [0,1280]x[0,720] with no gaps.
    tiles = tile_layout(1280, 720, 640, 640, 2, 2)
    xs = sorted({int(t.pixel_rect.to_xyxy()[0]) for t in tiles})
    x_rights = sorted({int(t.pixel_rect.to_xyxy()[2]) for t in tiles})
    y_bottoms = sorted({int(t.pixel_rect.to_xyxy()[3]) for t in tiles})
    assert min(xs) == 0 and max(x_rights) == 1280
    assert max(y_bottoms) == 720


# --------------------------------------------------------------------------- #
# Tile-local -> full-frame remap
# --------------------------------------------------------------------------- #

def test_remap_tile_bbox_each_tile():
    tiles = tile_layout(1280, 720, 640, 640, 2, 2)
    # A centered unit-ish box in tile-local coords.
    local = Rect.from_xyxy(0.0, 0.0, 1.0, 1.0)  # the whole tile
    for t in tiles:
        full = remap_tile_bbox(local, t.origin_norm, t.extent_norm)
        x1, y1, x2, y2 = full.to_xyxy()
        # Full-frame box must equal the tile's normalized placement.
        assert _approx(x1, t.origin_norm.x)
        assert _approx(y1, t.origin_norm.y)
        assert _approx(x2, t.origin_norm.x + t.extent_norm.x)
        assert _approx(y2, t.origin_norm.y + t.extent_norm.y)


def test_remap_tile_bbox_point_in_br_tile():
    tiles = tile_layout(1280, 720, 640, 640, 2, 2)
    br = tiles[3]  # origin (0.5, 80/720), extent (0.5, 640/720)
    local = Rect.from_xyxy(0.0, 0.0, 0.1, 0.1)
    full = remap_tile_bbox(local, br.origin_norm, br.extent_norm)
    x1, y1, x2, y2 = full.to_xyxy()
    assert _approx(x1, 0.5)
    assert _approx(y1, 80 / 720)
    assert _approx(x2, 0.5 + 0.1 * 0.5)
    assert _approx(y2, 80 / 720 + 0.1 * (640 / 720))


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all flight-mode tests passed")


if __name__ == "__main__":
    main()
