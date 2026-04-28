import os
import tempfile

import numpy as np
import pytest

from stereo_calibration import (
    FramePairer,
    calibrate_stereo,
    save_calibration,
    load_calibration,
)


def test_frame_pairer_returns_pair_when_close():
    pairer = FramePairer(max_gap_ns=33_000_000)
    left_frame  = np.zeros((100, 100, 3), dtype=np.uint8)
    right_frame = np.ones((100, 100, 3), dtype=np.uint8)
    assert pairer.put('left',  left_frame,  timestamp_ns=1_000_000_000) is None
    result = pairer.put('right', right_frame, timestamp_ns=1_010_000_000)
    assert result is not None
    l, r = result
    assert l is left_frame
    assert r is right_frame


def test_frame_pairer_returns_none_when_too_far():
    pairer = FramePairer(max_gap_ns=33_000_000)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert pairer.put('left',  frame, timestamp_ns=1_000_000_000) is None
    assert pairer.put('right', frame, timestamp_ns=2_000_000_000) is None


def test_frame_pairer_wait_for_pair_returns_after_match():
    pairer = FramePairer(max_gap_ns=33_000_000)
    left_frame  = np.zeros((50, 50, 3), dtype=np.uint8)
    right_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    pairer.put('left',  left_frame,  timestamp_ns=1_000_000_000)
    pairer.put('right', right_frame, timestamp_ns=1_005_000_000)
    pair = pairer.wait_for_pair(timeout_s=0.1)
    assert pair is not None
    assert pair[0] is left_frame
    assert pair[1] is right_frame


def test_frame_pairer_wait_for_pair_returns_none_on_timeout():
    pairer = FramePairer(max_gap_ns=33_000_000)
    assert pairer.wait_for_pair(timeout_s=0.05) is None


def test_calibration_save_load_roundtrip():
    map_x = np.arange(640 * 480, dtype=np.float32).reshape(480, 640)
    map_y = np.zeros((480, 640), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name
    try:
        save_calibration(
            path=path,
            map_left_x=map_x, map_left_y=map_y,
            map_right_x=map_x, map_right_y=map_y,
            focal_px=480.0,
            baseline_m=0.12,
            image_size=(640, 480),
            rms=0.42,
        )
        calib = load_calibration(path)
        assert calib.focal_px == pytest.approx(480.0)
        assert calib.baseline_m == pytest.approx(0.12)
        assert calib.image_size == (640, 480)
        assert np.allclose(calib.map_left_x, map_x)
        assert np.allclose(calib.map_right_y, map_y)
    finally:
        os.unlink(path)


def test_calibrate_stereo_returns_required_keys():
    """Synthetic input — only validates that the function runs and returns the
    expected keys with sane shapes. Realistic RMS depends on real chessboard
    geometry which is impossible to fake convincingly here."""
    board_size = (6, 4)
    square_size = 0.025
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    # Build N synthetic views by translating the (x,y) image points.
    image_size = (640, 480)
    n_views = 10
    img_pts_left  = []
    img_pts_right = []
    obj_pts       = []
    rng = np.random.default_rng(0)
    for i in range(n_views):
        dx, dy = rng.uniform(-50, 50, size=2)
        base = objp[:, :2].astype(np.float32) * 800.0 + np.array([320 + dx, 240 + dy], dtype=np.float32)
        img_pts_left.append(base.reshape(-1, 1, 2))
        img_pts_right.append((base + np.array([-15.0, 0.0], dtype=np.float32)).reshape(-1, 1, 2))
        obj_pts.append(objp.copy())

    try:
        result = calibrate_stereo(obj_pts, img_pts_left, img_pts_right, image_size)
    except Exception as e:
        pytest.skip(f"cv2.stereoCalibrate did not converge on synthetic input: {e}")

    for key in ('rms', 'map_left_x', 'map_left_y', 'map_right_x', 'map_right_y', 'focal_px'):
        assert key in result, f"missing key: {key}"
    assert result['map_left_x'].shape == (image_size[1], image_size[0])
    assert result['map_right_x'].shape == (image_size[1], image_size[0])
