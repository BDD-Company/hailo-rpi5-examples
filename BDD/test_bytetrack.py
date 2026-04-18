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


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq", __file__]))
