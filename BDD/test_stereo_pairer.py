# BDD/test_stereo_pairer.py
import pytest
from helpers import Detections, StereoDetections, FrameMetadata


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
