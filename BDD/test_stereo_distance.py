import numpy as np
import pytest
import tempfile, os
from helpers import Detection, Rect, XY
from stereo_distance import (
    StereoCalibration,
    stereo_distance,
    match_stereo_detections,
)


def _make_identity_calib(w=640, h=480, focal_px=500.0, baseline_m=0.12):
    """Identity rectification maps (no distortion) for testing."""
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return StereoCalibration(
        map_left_x=xs, map_left_y=ys,
        map_right_x=xs, map_right_y=ys,
        focal_px=focal_px,
        baseline_m=baseline_m,
        image_size=(w, h),
    )


def _det(cx, cy, w=0.1, h=0.1):
    """Make a Detection with center at (cx, cy) in normalized coords."""
    return Detection(bbox=Rect.from_xywh(cx - w/2, cy - h/2, w, h))


def test_stereo_distance_basic():
    calib = _make_identity_calib(w=640, h=480, focal_px=500.0, baseline_m=0.12)
    # Left center at pixel 320, right center at pixel 260 → disparity = 60px
    # Expected: 0.12 * 500 / 60 = 1.0 m
    left_det  = _det(320/640, 240/480)
    right_det = _det(260/640, 240/480)
    d = stereo_distance(left_det, right_det, calib, frame_width=640, frame_height=480)
    assert d is not None
    assert abs(d - 1.0) < 0.01


def test_stereo_distance_zero_disparity_returns_none():
    calib = _make_identity_calib()
    det = _det(0.5, 0.5)
    result = stereo_distance(det, det, calib, frame_width=640, frame_height=480)
    assert result is None


def test_stereo_distance_negative_disparity_returns_none():
    calib = _make_identity_calib(w=640)
    # right center is to the LEFT of left center → negative disparity
    left_det  = _det(200/640, 0.5)
    right_det = _det(300/640, 0.5)
    result = stereo_distance(left_det, right_det, calib, frame_width=640, frame_height=480)
    assert result is None


def test_match_stereo_detections_matches_by_iou():
    left  = [_det(0.3, 0.3), _det(0.7, 0.7)]
    right = [_det(0.71, 0.71), _det(0.31, 0.31)]
    pairs = match_stereo_detections(left, right, iou_threshold=0.1)
    assert len(pairs) == 2
    centers_l = {(round(p[0].bbox.center.x, 1), round(p[0].bbox.center.y, 1)) for p in pairs}
    assert (0.3, 0.3) in centers_l
    assert (0.7, 0.7) in centers_l


def test_match_stereo_detections_no_match_below_threshold():
    left  = [_det(0.1, 0.1)]
    right = [_det(0.9, 0.9)]
    pairs = match_stereo_detections(left, right, iou_threshold=0.1)
    assert pairs == []


def test_stereo_calibration_save_load_roundtrip():
    calib = _make_identity_calib(w=320, h=240, focal_px=400.0, baseline_m=0.10)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name
    try:
        calib.save(path)
        loaded = StereoCalibration.load(path)
        assert loaded.focal_px == pytest.approx(400.0)
        assert loaded.baseline_m == pytest.approx(0.10)
        assert loaded.image_size == (320, 240)
        assert np.allclose(loaded.map_left_x, calib.map_left_x)
    finally:
        os.unlink(path)
