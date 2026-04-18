#!/usr/bin/env python3
"""Tests for bytetrack.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from bytetrack import KalmanFilter


def test_initiate_shape():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)


def test_initiate_mean_values():
    kf = KalmanFilter()
    bbox = np.array([0.3, 0.4, 0.08, 0.12])
    mean, _ = kf.initiate(bbox)
    assert np.allclose(mean[:4], bbox)
    assert np.allclose(mean[4:], 0.0)


def test_predict_advances_position_with_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    mean[4] = 0.05  # inject vcx
    mean_pred, _ = kf.predict(mean, cov)
    assert mean_pred[0] > 0.5        # cx moved right
    assert np.isclose(mean_pred[0], 0.55, atol=1e-9)


def test_predict_covariance_grows():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    _, cov2 = kf.predict(mean, cov)
    assert np.trace(cov2) > np.trace(cov)


def test_update_moves_mean_toward_measurement():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    measurement = np.array([0.6, 0.5, 0.1, 0.1])
    mean_upd, _ = kf.update(mean, cov, measurement)
    assert 0.5 < mean_upd[0] < 0.6   # pulled toward measurement


def test_update_reduces_uncertainty():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    _, cov_pred = kf.predict(mean, cov)
    _, cov_upd = kf.update(mean, cov_pred, np.array([0.5, 0.5, 0.1, 0.1]))
    assert np.trace(cov_upd) < np.trace(cov_pred)


from bytetrack import STrack, TrackState


def _make_strack(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=0.8):
    kf = KalmanFilter()
    STrack.reset_counter()
    return STrack(np.array([x1, y1, x2, y2]), score, kf)


def test_strack_initial_state():
    t = _make_strack()
    assert t.state == TrackState.New
    assert t.track_id is None


def test_strack_activate_assigns_id_and_tracked():
    t = _make_strack()
    t.activate(frame_id=0)
    assert t.state == TrackState.Tracked
    assert t.track_id == 1


def test_strack_ids_increment():
    STrack.reset_counter()
    kf = KalmanFilter()
    t1 = STrack(np.array([0.1, 0.1, 0.3, 0.3]), 0.9, kf)
    t2 = STrack(np.array([0.5, 0.5, 0.7, 0.7]), 0.9, kf)
    t1.activate(0)
    t2.activate(0)
    assert t2.track_id == t1.track_id + 1


def test_strack_bbox_before_activate_returns_raw():
    t = _make_strack(0.1, 0.1, 0.3, 0.3)
    assert np.allclose(t.bbox, [0.1, 0.1, 0.3, 0.3])


def test_strack_bbox_after_activate_is_kalman():
    t = _make_strack(0.1, 0.1, 0.3, 0.3)
    t.activate(0)
    # Kalman bbox should be very close to initial detection
    assert np.allclose(t.bbox, [0.1, 0.1, 0.3, 0.3], atol=1e-6)


def test_strack_predict_changes_bbox():
    t = _make_strack()
    t.activate(0)
    bbox_before = t.bbox.copy()
    t.mean[4] = 0.05  # inject vcx
    t.predict()
    assert t.bbox[0] > bbox_before[0]


def test_strack_update_sets_det_bbox_and_tracked():
    t = _make_strack()
    t.activate(0)
    new_bbox = np.array([0.2, 0.2, 0.4, 0.4])
    t.update(new_bbox, 0.9, frame_id=1)
    assert t.state == TrackState.Tracked
    assert np.allclose(t.det_bbox, new_bbox)
    assert t.frame_id == 1


def test_strack_mark_lost_and_removed():
    t = _make_strack()
    t.activate(0)
    t.mark_lost()
    assert t.state == TrackState.Lost
    t.mark_removed()
    assert t.state == TrackState.Removed


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq", __file__]))
