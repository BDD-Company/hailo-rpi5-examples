#!/usr/bin/env python3
"""Pre-flight detection / in-flight pursuit flight-mode machinery.

Split out of helpers.py: the thread-safe FlightModeController (which the producer and the
drone controller share to switch between grounded tiled-search and airborne pursuit) plus the
pure tile-geometry helpers it relies on. Kept dependency-light (only XY/Rect/DEFAULT_CAMERA_ID
from helpers) so it stays unit-testable off-Pi.
"""

from dataclasses import dataclass
from enum import Enum
import threading

from helpers import XY, Rect, DEFAULT_CAMERA_ID


class FlightMode(Enum):
    DETECTION = 'detection'   # armed but grounded, scanning a tiled frame for a target
    PURSUIT   = 'pursuit'     # airborne, intercepting (whole-frame, latency-optimized)


@dataclass(slots=True, frozen=True)
class Tile:
    """One sub-frame of the detection-mode tiling grid.

    `pixel_rect` is the integer pixel region to crop out of the full capture frame (a NATIVE
    crop — no scaling). `origin_norm`/`extent_norm` place the tile in normalized full-frame
    coords so a tile-local detection bbox ([0..1] within the tile) can be remapped back to
    full-frame [0..1] via `remap_tile_bbox`.
    """
    index : int
    pixel_rect : Rect
    origin_norm : XY
    extent_norm : XY


def _axis_positions(frame_size : int, tile_size : int, n : int) -> list[int]:
    """Top-left pixel positions of `n` tiles of `tile_size` evenly covering `frame_size`.

    First tile at 0, last at frame-tile, evenly spaced; adjacent tiles overlap by
    `tile_size - (frame_size - tile_size)/(n-1)` px. The overlap is therefore COMPUTED from
    the frame size, tile (HEF input) size and tile count — never configured directly.
    Requires `tile_size <= frame_size` and `n*tile_size >= frame_size` so the union covers
    the whole axis with no gaps.
    """
    if n < 1:
        raise ValueError("tile count must be >= 1")
    if tile_size > frame_size:
        raise ValueError(f"tile_size {tile_size} exceeds frame_size {frame_size}")
    if n == 1:
        return [0]
    if n * tile_size < frame_size:
        raise ValueError(f"{n} tiles of {tile_size}px cannot cover {frame_size}px (gaps)")
    span = frame_size - tile_size
    return [round(k * span / (n - 1)) for k in range(n)]


def tile_layout(frame_w : int, frame_h : int, tile_w : int, tile_h : int,
                tiles_x : int, tiles_y : int) -> list[Tile]:
    """2-D grid of `tiles_x`*`tiles_y` native tiles of `tile_w`x`tile_h` covering the frame.

    Row-major index (0 = top-left). Overlap on each axis is auto-computed by `_axis_positions`.
    """
    xs = _axis_positions(frame_w, tile_w, tiles_x)
    ys = _axis_positions(frame_h, tile_h, tiles_y)
    tiles : list[Tile] = []
    idx = 0
    for py in ys:
        for px in xs:
            tiles.append(Tile(
                index=idx,
                pixel_rect=Rect.from_xyxy(px, py, px + tile_w, py + tile_h),
                origin_norm=XY(px / frame_w, py / frame_h),
                extent_norm=XY(tile_w / frame_w, tile_h / frame_h),
            ))
            idx += 1
    return tiles


def remap_tile_bbox(local : Rect, origin : XY, extent : XY) -> Rect:
    """Map a tile-local normalized bbox ([0..1] within the tile) to full-frame normalized coords:
    full = origin + local * extent (per-corner)."""
    return Rect(
        origin + local.p1.multiplied_by_XY(extent),
        origin + local.p2.multiplied_by_XY(extent),
    )


class FlightModeController:
    """Shared, thread-safe selector for the current flight mode (DETECTION vs PURSUIT).

    Mirrors `CameraSwitcher`: the producer (picamera_thread) reads `is_detection()` to decide
    whether to split each captured frame into native 640x640 tiles (detection) or push the whole
    frame (pursuit); the drone controller reads it to gate takeoff and calls `switch_to_pursuit()`
    once a target trajectory is established. The transition is ONE-WAY within a mission.

    When `enabled` is False the mode is permanently PURSUIT, so every code path behaves exactly as
    it did before this feature existed.
    """
    def __init__(self, *,
                 enabled : bool = False,
                 detection_camera_id : int = DEFAULT_CAMERA_ID,
                 engine : str = 'producer',
                 tiles_x : int = 2,
                 tiles_y : int = 2,
                 overlap : float = 0.1,
                 tile_size : tuple[int, int] = (640, 640),
                 capture_fps : int = 10,
                 frame_size : tuple[int, int] = (1332, 990),
                 batch_size : int | None = None,
                 switch_after_consecutive_detections : int = 10):
        self._enabled = bool(enabled)
        self._detection_camera_id = detection_camera_id
        # Detection engine:
        #   'producer' — the picamera thread crops square HEF-input tiles and pushes them as
        #     separate buffers, reassembled in the callback (batch-1). Needs the capture_fps
        #     throttle (tile-shedding) and a caps flip at the switch.
        #   'cropper'  — the picamera thread pushes WHOLE `frame_size` frames; the pipeline's
        #     hailotilecropper tiles them and a BATCHED hailonet (batch_size = tiles) infers them,
        #     hailotileaggregator reassembles. Runs unthrottled; the switch is a pipeline valve flip.
        self._engine = engine
        self._tiles_x = int(tiles_x)
        self._tiles_y = int(tiles_y)
        self._overlap = float(overlap)
        self._tile_size = (int(tile_size[0]), int(tile_size[1]))
        self._capture_fps = int(capture_fps)
        self._frame_size = (int(frame_size[0]), int(frame_size[1]))
        self._batch_size = int(batch_size) if batch_size else self._tiles_x * self._tiles_y
        self._switch_after = int(switch_after_consecutive_detections)
        self._mode = FlightMode.DETECTION if self._enabled else FlightMode.PURSUIT
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def mode(self) -> FlightMode:
        with self._lock:
            return self._mode

    def is_detection(self) -> bool:
        with self._lock:
            return self._mode is FlightMode.DETECTION

    def is_pursuit(self) -> bool:
        with self._lock:
            return self._mode is FlightMode.PURSUIT

    def switch_to_pursuit(self) -> bool:
        """One-way DETECTION -> PURSUIT. Returns True only on the transition (idempotent after)."""
        with self._lock:
            if self._mode is FlightMode.PURSUIT:
                return False
            self._mode = FlightMode.PURSUIT
            return True

    # Immutable config — no lock needed.
    @property
    def detection_camera_id(self) -> int:
        return self._detection_camera_id

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def overlap(self) -> float:
        return self._overlap

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def tiles_x(self) -> int:
        return self._tiles_x

    @property
    def tiles_y(self) -> int:
        return self._tiles_y

    @property
    def tile_size(self) -> tuple[int, int]:
        return self._tile_size

    @property
    def tile_count(self) -> int:
        return self._tiles_x * self._tiles_y

    @property
    def capture_fps(self) -> int:
        return self._capture_fps

    @property
    def switch_after_consecutive_detections(self) -> int:
        return self._switch_after

    def tiles_for_frame(self, frame_w : int, frame_h : int) -> list[Tile]:
        """Concrete tile layout for a given capture frame size (HEF-input-sized tiles)."""
        tw, th = self._tile_size
        return tile_layout(frame_w, frame_h, tw, th, self._tiles_x, self._tiles_y)
