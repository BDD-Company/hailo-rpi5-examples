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
        source's loop rewind (``GStreamerApp.on_eos``).

        LOAD-BEARING on the file path. ``normalized_frame_id`` derives the file-input
        id from ``buffer.pts`` (as a frame index) because that is the only
        branch-independent id that survives the tile aggregator (``buffer.offset`` is
        stamped per branch and restarts low on every tiling switch). ``buffer.pts``
        restarts at 0 on the loop's flush seek, so without this reset the mark would
        stay at the end of the previous pass, every later pass would be rejected in
        full, and the app would run on permanently blind.

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


def resolve_frame_id(frame_meta_ts, pts, offset, frame_duration_ns):
    """Pick the per-frame id fed to BOTH :class:`FrameOrderGuard` and ByteTracker.

    That id has two hard requirements:

    * **branch-independent** -- the same source frame must resolve to the same id no
      matter which tiling rung produced the buffer, or a branch switch re-blinds the
      guard (measured 53% of frames dropped: the per-branch ``buffer.offset`` restarts
      low whenever a valve reopens, so the shared guard rejects the whole catch-up
      window as reorders);
    * **~1 per frame** -- ByteTracker's ``track_buffer`` expiry counts frame_id
      DIFFERENCES, not wall time (see ``BYTETracker.update``), so the id must step by
      ~1 per frame.

    The only buffer property that is both is ``buffer.pts``: it survives the hailo tile
    aggregator (which rebuilds tiled frames into a fresh buffer that DROPS custom
    reference metas) and is identical for a given source frame on every rung. But raw
    pts is nanoseconds -- it jumps ~3e7 per frame and would break ByteTracker -- so we
    divide by the frame duration to get a real frame index.

    Priority: producer frame-id meta (survives only a 1x1/whole-frame branch) ->
    pts-as-frame-index -> raw offset (per-branch; last resort when pts can't be indexed)
    -> None (caller substitutes a wallclock id). All arguments are plain ints or None
    (the app.py wrapper maps the GStreamer NONE sentinels to None), keeping this pure
    and host-testable.

    Note the pts index restarts at 0 on a file-loop flush seek; ``on_stream_rewound``
    handles that by resetting the guard.
    """
    if frame_meta_ts is not None:
        return frame_meta_ts
    if pts is not None and frame_duration_ns:
        return round(pts / frame_duration_ns)
    if offset is not None:
        return offset
    return None
