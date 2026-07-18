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


# ── Phase-2 measurement-noise floor (noise reduction design §3.3) ────────
# ByteTrack scales measurement noise by box height, so a tiny (far) box is
# assumed sub-pixel accurate and barely filtered. A floor makes the filter
# distrust exactly those small far targets. Default 0.0 => today's behaviour.

def _small_box_update(floor):
    """Filter one noisy cx jump on a tiny box; return the updated cx."""
    kf = KalmanFilter(measurement_noise_floor=floor)
    bbox = np.array([0.5, 0.5, 0.02, 0.02])   # ~8px target: h=0.02 normalised
    mean, cov = kf.initiate(bbox)
    mean, cov = kf.predict(mean, cov)
    measurement = np.array([0.6, 0.5, 0.02, 0.02])  # cx jumps +0.1 (noise)
    mean_upd, _ = kf.update(mean, cov, measurement)
    return mean_upd[0]


def test_noise_floor_makes_small_box_filter_more():
    # With the floor the filtered cx moves LESS toward the noisy measurement.
    cx_no_floor = _small_box_update(0.0)
    cx_floored  = _small_box_update(0.02)
    assert abs(cx_floored - 0.5) < abs(cx_no_floor - 0.5)


def test_noise_floor_default_zero_is_unchanged():
    # Default constructor (no floor) behaves exactly like an explicit 0.0 floor.
    kf_default = KalmanFilter()
    kf_zero    = KalmanFilter(measurement_noise_floor=0.0)
    bbox = np.array([0.5, 0.5, 0.02, 0.02])
    m_d, c_d = kf_default.initiate(bbox)
    m_z, c_z = kf_zero.initiate(bbox)
    meas = np.array([0.6, 0.5, 0.02, 0.02])
    upd_d, _ = kf_default.update(*kf_default.predict(m_d, c_d), meas)
    upd_z, _ = kf_zero.update(*kf_zero.predict(m_z, c_z), meas)
    assert np.allclose(upd_d, upd_z)


def test_noise_floor_inert_for_large_box():
    # A large box's height-scaled std already exceeds a small floor, so the floor
    # changes nothing: the leverage is on small far targets only.
    kf_no_floor = KalmanFilter()
    kf_floor    = KalmanFilter(measurement_noise_floor=0.001)
    bbox = np.array([0.5, 0.5, 0.4, 0.4])   # big near target
    meas = np.array([0.55, 0.5, 0.4, 0.4])
    m0, c0 = kf_no_floor.initiate(bbox)
    m1, c1 = kf_floor.initiate(bbox)
    upd0, _ = kf_no_floor.update(*kf_no_floor.predict(m0, c0), meas)
    upd1, _ = kf_floor.update(*kf_floor.predict(m1, c1), meas)
    assert np.allclose(upd0, upd1)


from bytetrack import STrack, TrackState


@pytest.fixture(autouse=True)
def reset_strack_counter():
    STrack.reset_counter()
    yield


def _make_strack(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=0.8):
    kf = KalmanFilter()
    return STrack(np.array([x1, y1, x2, y2]), score, kf)


from bytetrack import BYTETracker


def test_bytetracker_forwards_noise_floor_to_kalman():
    t = BYTETracker(measurement_noise_floor=0.02)
    assert t._kf._meas_floor == 0.02


def test_bytetracker_noise_floor_defaults_to_zero():
    assert BYTETracker()._kf._meas_floor == 0.0


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


from bytetrack import BYTETracker, _nms


def _det(x1, y1, x2, y2, score):
    return [x1, y1, x2, y2, score]


def _tracker():
    STrack.reset_counter()
    return BYTETracker(track_thresh=0.5, det_thresh=0.6, match_thresh=0.8,
                       track_buffer=5, frame_rate=30)


def test_nms_iou_suppresses_duplicate():
    # Two nearly-identical boxes: IoU > 0.9
    dets = np.array([
        _det(0.10, 0.90, 0.15, 0.99, 0.63),
        _det(0.11, 0.90, 0.15, 0.99, 0.35),
    ])
    result = _nms(dets, iou_thresh=0.3, dist_thresh=None)
    assert len(result) == 1
    assert result[0, 4] == pytest.approx(0.63)  # higher-confidence kept


def test_nms_dist_suppresses_close_centres():
    # Two boxes with low IoU (~0.15) but close centres (dist ≈ 0.05)
    dets = np.array([
        _det(0.561, 0.349, 0.597, 0.412, 0.63),
        _det(0.577, 0.357, 0.623, 0.494, 0.50),
    ])
    result_iou_only = _nms(dets, iou_thresh=0.3, dist_thresh=None)
    assert len(result_iou_only) == 2          # IoU alone doesn't suppress
    result_both = _nms(dets, iou_thresh=0.3, dist_thresh=0.06)
    assert len(result_both) == 1              # centre distance suppresses
    assert result_both[0, 4] == pytest.approx(0.63)


def test_nms_keeps_distant_boxes():
    # Two detections far apart: must not be suppressed
    dets = np.array([
        _det(0.1, 0.1, 0.2, 0.2, 0.9),
        _det(0.7, 0.7, 0.9, 0.9, 0.8),
    ])
    result = _nms(dets, iou_thresh=0.3, dist_thresh=0.06)
    assert len(result) == 2


def test_tracker_nms_prevents_spurious_track():
    # Without NMS: duplicate detection creates 2 tracks.
    # With NMS: only 1 track created.
    STrack.reset_counter()
    t_no_nms = BYTETracker(track_thresh=0.3, det_thresh=0.35, match_thresh=0.3,
                            track_buffer=5, frame_rate=30, match_max_dist=0.2)
    dets = np.array([
        _det(0.084, 0.928, 0.116, 0.990, 0.63),
        _det(0.086, 0.928, 0.115, 0.990, 0.35),  # near-identical duplicate
    ])
    tracks_no_nms = t_no_nms.update(dets, frame_id=0)
    assert len(tracks_no_nms) == 2  # spurious second track

    STrack.reset_counter()
    t_nms = BYTETracker(track_thresh=0.3, det_thresh=0.35, match_thresh=0.3,
                        track_buffer=5, frame_rate=30, match_max_dist=0.2,
                        nms_thresh=0.3, nms_dist_thresh=0.06)
    tracks_nms = t_nms.update(dets, frame_id=0)
    assert len(tracks_nms) == 1    # duplicate suppressed


def test_tracker_no_dets_returns_empty():
    t = _tracker()
    result = t.update(np.empty((0, 5)), frame_id=0)
    assert result == []


def test_tracker_single_detection_creates_track():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    tracks = t.update(dets, frame_id=0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].state == TrackState.Tracked


def test_tracker_same_detection_keeps_track_id():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    tracks = t.update(dets, frame_id=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_tracker_lost_track_remembered():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    # Frame with no detections
    tracks = t.update(np.empty((0, 5)), frame_id=1)
    assert tracks == []
    assert len(t.lost_stracks) == 1


def test_tracker_lost_track_recovered():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    t.update(np.empty((0, 5)), frame_id=1)  # goes lost
    # Reappears
    tracks = t.update(dets, frame_id=2)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1          # same ID recovered


def test_tracker_lost_track_removed_after_buffer():
    # track_buffer=5 → max_lost_age = 5 frames
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    # 6 empty frames
    for fid in range(1, 7):
        t.update(np.empty((0, 5)), frame_id=fid)
    assert len(t.lost_stracks) == 0


def test_tracker_low_confidence_below_det_thresh_not_tracked():
    # score=0.4 < det_thresh=0.6 → should not create a new track
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.4)])
    tracks = t.update(dets, frame_id=0)
    assert tracks == []


def test_tracker_low_conf_rescues_lost_track():
    t = _tracker()
    # Establish a track
    high_det = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(high_det, frame_id=0)
    # Next frame: only low-confidence detection at same position
    low_det = np.array([_det(0.11, 0.11, 0.31, 0.31, 0.35)])
    tracks = t.update(low_det, frame_id=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1          # rescued via Stage 2


def test_tracker_two_detections_two_tracks():
    t = _tracker()
    dets = np.array([
        _det(0.1, 0.1, 0.3, 0.3, 0.9),
        _det(0.6, 0.6, 0.8, 0.8, 0.9),
    ])
    tracks = t.update(dets, frame_id=0)
    assert len(tracks) == 2
    ids = {tr.track_id for tr in tracks}
    assert len(ids) == 2                    # distinct IDs


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq", __file__]))
