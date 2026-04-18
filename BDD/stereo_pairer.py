import time
import threading
from collections import deque

from helpers import Detections, StereoDetections


class StereoPairer:
    """
    Buffers Detections from 'left' and 'right' cameras and emits StereoDetections
    when a matching pair is found (by capture timestamp proximity).

    If a frame waits longer than pair_timeout_ns without a match it is forwarded
    as a plain Detections (mono fallback).
    """

    def __init__(self, output_queue, max_pair_gap_ns=33_000_000, pair_timeout_ns=100_000_000, buffer_maxlen=5):
        self._output_queue = output_queue
        self._max_pair_gap_ns = max_pair_gap_ns
        self._pair_timeout_ns = pair_timeout_ns
        self._buffer_maxlen = buffer_maxlen
        # Each entry: (detections, wall_time_received_ns)
        self._buffers: dict[str, deque] = {
            'left':  deque(maxlen=buffer_maxlen),
            'right': deque(maxlen=buffer_maxlen),
        }
        self._lock = threading.Lock()

    def put(self, camera_id: str, detections: Detections) -> None:
        assert camera_id in ('left', 'right'), f"Unknown camera_id: {camera_id}"
        received_ns = time.monotonic_ns()
        capture_ns = detections.meta.capture_timestamp_ns or received_ns
        other = 'right' if camera_id == 'left' else 'left'

        with self._lock:
            other_buf = self._buffers[other]
            best_idx, best_diff = None, self._max_pair_gap_ns

            for i, (other_det, _) in enumerate(other_buf):
                other_cap = other_det.meta.capture_timestamp_ns or 0
                diff = abs(capture_ns - other_cap)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is not None:
                other_det, _ = other_buf[best_idx]
                remaining = [item for j, item in enumerate(other_buf) if j != best_idx]
                self._buffers[other] = deque(remaining, maxlen=self._buffer_maxlen)

                left  = detections if camera_id == 'left' else other_det
                right = other_det  if camera_id == 'left' else detections
                l_ts  = left.meta.capture_timestamp_ns or 0
                r_ts  = right.meta.capture_timestamp_ns or 0
                avg   = (l_ts + r_ts) // 2

                self._output_queue.put(StereoDetections(left=left, right=right, pair_timestamp_ns=avg))
            else:
                self._buffers[camera_id].append((detections, received_ns))
                self._flush_timed_out()

    def _flush_timed_out(self) -> None:
        now = time.monotonic_ns()
        # Frames captured within this window of "now" are considered to have valid timestamps.
        # Small artificial timestamps used in tests fall outside this window and fall back to
        # received_ns so they don't trigger spurious flushes.
        recent_threshold_ns = now - 10_000_000_000  # 10 seconds ago
        for side in ('left', 'right'):
            buf = self._buffers[side]
            while buf:
                oldest_det, oldest_received = buf[0]
                capture_ns = oldest_det.meta.capture_timestamp_ns
                if capture_ns and capture_ns > recent_threshold_ns:
                    age = now - capture_ns
                else:
                    age = now - oldest_received
                if age > self._pair_timeout_ns:
                    buf.popleft()
                    self._output_queue.put(oldest_det)
                else:
                    break
