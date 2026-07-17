#!/usr/bin/env python3

from __future__ import annotations


class FrameOrderGuard:
    """Enforce strict, per-camera, monotonic frame ordering on a consumer.

    Hard invariant of this project: the capture -> inference -> callback ->
    control chain must deliver frames to every consumer in strictly increasing
    per-camera frame-id order. It is fine to SKIP frames (drop some to keep
    latency low) — a frame id that jumps AHEAD of the last accepted one is
    accepted — but a frame whose id is <= the last one accepted for that camera
    (i.e. a reorder or a duplicate) is REJECTED, so no consumer ever steps
    backward in time or double-counts a frame (which would corrupt ByteTrack,
    the velocity estimator and the PD history).

    Ids are tracked per camera because each camera's producer runs its own
    independent frame counter, so two cameras legitimately emit overlapping ids
    and a camera switch resets to that camera's own (possibly lower) id.

    Note: this guards ORDER only. Bounding how MANY consecutive frames may be
    skipped (no big batch drops) is the job of the leaky=downstream GStreamer
    queues, which shed the oldest buffer one at a time.
    """

    __slots__ = ("_last_by_camera",)

    def __init__(self) -> None:
        self._last_by_camera: dict[int, int] = {}

    def accept(self, camera_id: int, frame_id: int) -> bool:
        """Return True and record the frame if it is strictly newer than the
        last accepted frame for ``camera_id``; return False (drop it) if it is a
        reorder or duplicate. Forward skips are accepted."""
        last = self._last_by_camera.get(camera_id)
        if last is not None and frame_id <= last:
            return False
        self._last_by_camera[camera_id] = frame_id
        return True

    def reset(self) -> None:
        """Forget every camera's high-water mark, starting a new stream generation.

        Only for a DELIBERATE stream discontinuity, where ids legitimately restart
        from a lower value and nothing upstream is stale — today that is the file
        source's loop rewind (``GStreamerApp.on_eos``): a file's frame ids come from
        ``buffer.offset``, which restarts at 0 after the seek. Without this the mark
        stays at the end of the previous pass and every later pass is rejected in
        full, leaving the app running but permanently blind.

        Safe only because the rewind uses a FLUSH seek, which drops in-flight
        buffers: no frame from the previous generation can still arrive and be
        mistaken for an in-order frame of the new one. Call it AFTER the flush.
        This does not weaken the invariant — order is enforced as strictly as ever
        within each generation.
        """
        self._last_by_camera.clear()

    def last(self, camera_id: int) -> int | None:
        """Last accepted frame id for ``camera_id`` (None if none yet)."""
        return self._last_by_camera.get(camera_id)
