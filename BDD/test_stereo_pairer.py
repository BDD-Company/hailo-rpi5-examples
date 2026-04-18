# BDD/test_stereo_pairer.py
import time

import pytest
from helpers import Detections, StereoDetections, FrameMetadata
from stereo_pairer import StereoPairer


def _make_detections(frame_id=1, ts=1_000_000_000):
    return Detections(
        frame_id=frame_id,
        frame=None,
        detections=[],
        meta=FrameMetadata(capture_timestamp_ns=ts),
    )


def test_stereo_detections_fields():
    left = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=1_000_500_000)
    sd = StereoDetections(left=left, right=right, pair_timestamp_ns=1_000_250_000)
    assert sd.left is left
    assert sd.right is right
    assert sd.pair_timestamp_ns == 1_000_250_000


class _CollectQueue:
    """Simple list-backed queue stub for tests."""
    def __init__(self):
        self.items = []
    def put(self, item):
        self.items.append(item)


def test_pair_emitted_when_timestamps_close():
    q = _CollectQueue()
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000)
    left  = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=1_010_000_000)  # 10 ms apart
    pairer.put('left', left)
    pairer.put('right', right)
    assert len(q.items) == 1
    pair = q.items[0]
    assert isinstance(pair, StereoDetections)
    assert pair.left is left
    assert pair.right is right


def test_no_pair_when_timestamps_too_far():
    q = _CollectQueue()
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000)
    left  = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=2_000_000_000)  # 1 second apart
    pairer.put('left', left)
    pairer.put('right', right)
    assert len(q.items) == 0


def test_mono_fallback_after_timeout():
    q = _CollectQueue()
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000, pair_timeout_ns=50_000_000)
    old_ts = time.monotonic_ns() - 100_000_000  # 100 ms ago
    left = _make_detections(1, ts=old_ts)
    pairer.put('left', left)
    right = _make_detections(2, ts=old_ts + 500_000_000)
    pairer.put('right', right)
    assert len(q.items) == 1
    assert q.items[0] is left
    assert isinstance(q.items[0], Detections)
    assert not isinstance(q.items[0], StereoDetections)


def test_pair_timestamp_is_average():
    q = _CollectQueue()
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000)
    left  = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=1_020_000_000)
    pairer.put('left', left)
    pairer.put('right', right)
    assert q.items[0].pair_timestamp_ns == 1_010_000_000
