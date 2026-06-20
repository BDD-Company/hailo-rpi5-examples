"""Constant-velocity Kalman filter for the 3-D NED target state.

Smooths the per-frame camera->NED position measurements into a stable position
AND velocity estimate. Replaces the noisy cluster / 2-point velocity that made the
ProNav lead jitter on a fast crossing target (the ~1.5 m, high-variance result on
the 70/17/150 scenario).

State x = [pn, pe, pd, vn, ve, vd] (NED position + velocity). Constant-velocity
model with white-noise-acceleration process noise. Tunable by two scales:
  q_scale : process noise (higher -> trusts the CV model less, more responsive)
  r_scale : measurement noise (higher -> trusts the camera measurement less, smoother)
"""
import numpy as np


class TargetKalman:
    def __init__(self, q_scale: float = 1.0, r_scale: float = 1.0):
        self.q_scale = float(q_scale)
        self.r_scale = float(r_scale)
        self.x = None        # (6,) state
        self.P = None        # (6,6) covariance
        self._last_t = None

    def reset(self):
        self.x = None
        self.P = None
        self._last_t = None

    def update(self, pn: float, pe: float, pd: float, t_s: float, meas_range: float = None):
        """Ingest a NED position measurement at time t_s (seconds).
        meas_range (m): estimated range to target; inflates measurement noise at
        distance for stable long-range tracking (~150 m). Returns
        (pos=(n,e,d), vel=(n,e,d)) — the smoothed estimate."""
        z = np.array([pn, pe, pd], dtype=float)
        if self.x is None:
            self.x = np.array([pn, pe, pd, 0.0, 0.0, 0.0], dtype=float)
            self.P = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
            self._last_t = t_s
            return (pn, pe, pd), (0.0, 0.0, 0.0)

        dt = t_s - self._last_t
        if dt <= 0:
            dt = 1e-3
        self._last_t = t_s

        # Predict (constant velocity)
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = self.q_scale
        dt2, dt3 = dt * dt, dt * dt * dt
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt3 / 3.0
            Q[i, i + 3] = Q[i + 3, i] = q * dt2 / 2.0
            Q[i + 3, i + 3] = q * dt
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Update with the position measurement
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        # Range-adaptive measurement noise: monocular NED-position error grows ~ with
        # range^2 (a few-px bbox at 150 m => +/-tens of metres). Inflate R when the
        # target is far so the filter leans on the constant-velocity model (stable
        # tracking out to ~150 m); ~1x at/under the reference range.
        _rf = 1.0
        if meas_range is not None and meas_range > 0:
            _rf = (meas_range / 60.0) ** 2
            if _rf > 4.0:
                _rf = 4.0
            if _rf < 1.0:
                _rf = 1.0
        R = self.r_scale * _rf * np.eye(3)
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ H) @ P_pred

        return (float(self.x[0]), float(self.x[1]), float(self.x[2])), \
               (float(self.x[3]), float(self.x[4]), float(self.x[5]))
