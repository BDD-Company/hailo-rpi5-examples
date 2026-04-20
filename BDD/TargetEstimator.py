#!/usr/bin/env python3

import math
from collections import deque
from enum import Enum

import numpy as np

from helpers import XY, Rect
from telemetry_position import PositionNED, VelocityNED
from interfaces import TargetEstimatorInterface


class VelocityMethod(Enum):
    """Velocity estimation strategy for TargetEstimator3D."""
    LINEAR = "linear"           # simple 2-point (last two samples)
    WLS = "wls"                 # weighted least-squares (existing)
    NUMPY_REGRESSION = "numpy"  # numpy linear regression over all samples
    CLUSTER = "cluster"         # average first-N vs last-M cluster endpoints


def _exp_fast(x: float) -> float:
    """math.exp with clamping to avoid overflow on large negative values."""
    if x < -50.0:
        return 0.0
    return math.exp(x)


class TargetEstimator(TargetEstimatorInterface):
    def __init__(self, max_positions : int = 10, max_age_ns : int = 500_000_000):
        assert(max_positions > 1)
        assert(max_age_ns > 1)

        self.max_target_positions = int(max_positions)
        self.max_target_position_age_nanoseconds = int(max_age_ns)
        self.target_positions : deque[tuple[int, XY]] = deque(maxlen=self.max_target_positions)


    def history_size(self):
        return len(self.target_positions)

    def max_history_size(self):
        return self.max_target_positions

    def clear_history(self):
        return self.target_positions.clear()

    def name(self):
        return "2D"

    def describe_prev_estimation(self):
        return self.name() + " linear"

    def max_age_ns(self) -> int:
        return self.max_target_position_age_nanoseconds


    def _forget_old_positions(self, reference_timestamp_nanoseconds : int):
        if len(self.target_positions) == 0:
            return

        oldest_allowed_timestamp = (
            int(reference_timestamp_nanoseconds)
            - self.max_target_position_age_nanoseconds
        )
        while (
            self.target_positions
            and self.target_positions[0][0] < oldest_allowed_timestamp
        ):
            self.target_positions.popleft()


    def add_target_pos(self, current_target_pos: XY, current_timestamp_nanoseconds):
        assert(isinstance(current_target_pos, XY))

        # current_timestamp_nanoseconds = int(current_timestamp_nanoseconds)
        # self._forget_old_positions(current_timestamp_nanoseconds)

        # # Keep timestamps monotonic so the velocity estimate stays stable.
        # while (
        #     self.target_positions
        #     and self.target_positions[-1][0] >= current_timestamp_nanoseconds
        # ):
        #     self.target_positions.pop()

        self.target_positions.append(
            (
                current_timestamp_nanoseconds,
                current_target_pos.clone(),
            )
        )


    def _estimate_target_velocity(self):
        if len(self.target_positions) < 2:
            return XY()

        # TODO: perhaps use newest and one before that for estimating speed to be more accurate
        oldest_timestamp, oldest_pos = self.target_positions[-2]
        newest_timestamp, newest_pos = self.target_positions[-1]
        delta_t_nanoseconds = newest_timestamp - oldest_timestamp
        if delta_t_nanoseconds <= 0:
            return XY()

        return (newest_pos - oldest_pos) / delta_t_nanoseconds


    def estimate_target_pos(self, at_timestamp_nanoseconds, fallback=None):
        if not self.target_positions:
            return fallback

        at_timestamp_nanoseconds = int(at_timestamp_nanoseconds)
        self._forget_old_positions(at_timestamp_nanoseconds)
        if not self.target_positions:
            return fallback

        newest_timestamp, newest_pos = self.target_positions[-1]
        if len(self.target_positions) == 1:
            return newest_pos.clone()

        target_velocity = self._estimate_target_velocity()
        delta_t_nanoseconds = at_timestamp_nanoseconds - newest_timestamp
        return newest_pos + (target_velocity * delta_t_nanoseconds)


class TargetEstimator3D(TargetEstimatorInterface):
    """Track target position in the NED world frame (3-D).

    Stores absolute NED positions computed from detection + distance + telemetry.
    Because positions are in a fixed world frame, drone movement is implicitly
    accounted for - successive positions directly reveal target velocity.
    """

    def __init__(self, max_positions: int = 20, max_age_ns: int = 500_000_000):
        assert max_positions > 1
        assert max_age_ns > 1
        self.max_positions = int(max_positions)
        self._max_age_ns = int(max_age_ns)
        self._positions: deque[tuple[int, PositionNED]] = deque(maxlen=self.max_positions)
        self._last_method = None

    def history_size(self) -> int:
        return len(self._positions)

    def max_history_size(self):
        return self.max_positions

    def clear_history(self) -> None:
        self._positions.clear()

    def name(self):
        return "3D"

    def max_age_ns(self) -> int:
        return self._max_age_ns


    def describe_prev_estimation(self):
        return f"{self.name()} {self._last_method.name if self._last_method else '--'}"


    def _forget_old(self, ref_ts_ns: int) -> None:
        cutoff = int(ref_ts_ns) - self._max_age_ns
        while self._positions and self._positions[0][0] < cutoff:
            self._positions.popleft()


    def add(self, pos: PositionNED, timestamp_ns: int) -> None:
        self._positions.append((int(timestamp_ns), pos))


    # Minimum samples for the weighted least-squares fit.
    # Below this threshold, fall back to simple 2-point linear estimate.
    _MIN_POINTS_FOR_WLS = 3

    # Exponential half-life for sample weighting (in nanoseconds).
    # Samples older than this get half the weight of the newest sample.
    # ~150 ms ≈ 4-5 frames at 30 fps.
    _WEIGHT_HALFLIFE_NS = 150_000_000

    def _estimate_velocity_linear(self) -> tuple[float, float, float]:
        """Simple 2-point velocity: (vN, vE, vD) in m/ns."""
        ts0, p0 = self._positions[-2]
        ts1, p1 = self._positions[-1]
        dt = ts1 - ts0
        if dt <= 0:
            return (0.0, 0.0, 0.0)
        return (
            (p1.north_m - p0.north_m) / dt,
            (p1.east_m - p0.east_m) / dt,
            (p1.down_m - p0.down_m) / dt,
        )

    def _estimate_velocity_wls(self) -> tuple[float, float, float]:
        """Weighted least-squares velocity over all stored samples.

        Fits pos = offset + velocity * t for each NED axis independently,
        with exponential weights that favour recent samples.
        Returns (vN, vE, vD) in m/ns.
        """
        ts_ref = self._positions[-1][0]  # newest timestamp as reference
        ln2 = 0.6931471805599453  # math.log(2)
        decay = ln2 / self._WEIGHT_HALFLIFE_NS

        # Accumulate WLS sums in a single pass.
        sw = 0.0   # Σ w_i
        swt = 0.0  # Σ w_i * t_i
        swtt = 0.0 # Σ w_i * t_i²
        sw_n = 0.0; sw_e = 0.0; sw_d = 0.0       # Σ w_i * val_i
        swt_n = 0.0; swt_e = 0.0; swt_d = 0.0    # Σ w_i * t_i * val_i

        for ts, p in self._positions:
            age = ts_ref - ts  # >= 0
            w = _exp_fast(-decay * age)
            t = ts - ts_ref    # <= 0 for past samples, 0 for newest

            sw += w
            wt = w * t
            swt += wt
            swtt += wt * t
            sw_n += w * p.north_m;  swt_n += wt * p.north_m
            sw_e += w * p.east_m;   swt_e += wt * p.east_m
            sw_d += w * p.down_m;   swt_d += wt * p.down_m

        denom = sw * swtt - swt * swt
        if abs(denom) < 1e-30:
            return self._estimate_velocity_linear()

        # WLS slope = (Σw · Σwt·v  -  Σwt · Σw·v) / denom
        vn = (sw * swt_n - swt * sw_n) / denom
        ve = (sw * swt_e - swt * sw_e) / denom
        vd = (sw * swt_d - swt * sw_d) / denom
        return (vn, ve, vd)

    def _estimate_velocity_numpy(self) -> tuple[float, float, float]:
        """Numpy least-squares linear regression over all stored samples.

        Fits pos = a + b*t for each NED axis using np.linalg.lstsq.
        Returns (vN, vE, vD) in m/ns.
        """
        n = len(self._positions)
        ts = np.empty(n)
        north = np.empty(n)
        east = np.empty(n)
        down = np.empty(n)

        ts_ref = self._positions[-1][0]
        for i, (t, p) in enumerate(self._positions):
            ts[i] = t - ts_ref
            north[i] = p.north_m
            east[i] = p.east_m
            down[i] = p.down_m

        # Design matrix [ones, t] for intercept + slope
        A = np.column_stack([np.ones(n), ts])
        # lstsq returns (solution, residuals, rank, sv); solution is [intercept, slope]
        vn = np.linalg.lstsq(A, north, rcond=None)[0][1]
        ve = np.linalg.lstsq(A, east, rcond=None)[0][1]
        vd = np.linalg.lstsq(A, down, rcond=None)[0][1]
        return (float(vn), float(ve), float(vd))

    # Fraction of buffer used for each cluster in the cluster-averaging method.
    _CLUSTER_FRACTION = 1 / 3

    def _estimate_velocity_cluster(self) -> tuple[float, float, float]:
        """Cluster-averaging velocity: average first-N and last-M positions.

        Splits the buffer into a "beginning" cluster (first N samples) and
        an "end" cluster (last M samples), averages each, then computes
        velocity from the two averaged points and their mean timestamps.
        This smooths out high-frequency noise and reveals the long-term trend.
        Returns (vN, vE, vD) in m/ns.
        """
        total = len(self._positions)
        k = max(1, int(total * self._CLUSTER_FRACTION))

        # Average the first-k samples (beginning cluster)
        t0 = 0.0; n0 = 0.0; e0 = 0.0; d0 = 0.0
        for i in range(k):
            ts, p = self._positions[i]
            t0 += ts; n0 += p.north_m; e0 += p.east_m; d0 += p.down_m
        t0 /= k; n0 /= k; e0 /= k; d0 /= k

        # Average the last-k samples (end cluster)
        t1 = 0.0; n1 = 0.0; e1 = 0.0; d1 = 0.0
        for i in range(total - k, total):
            ts, p = self._positions[i]
            t1 += ts; n1 += p.north_m; e1 += p.east_m; d1 += p.down_m
        t1 /= k; n1 /= k; e1 /= k; d1 /= k

        dt = t1 - t0
        if abs(dt) < 1e-6:
            return self._estimate_velocity_linear()

        return ((n1 - n0) / dt, (e1 - e0) / dt, (d1 - d0) / dt)

    def _estimate_velocity(self, method: VelocityMethod = VelocityMethod.WLS) -> tuple[float, float, float]:
        """Return (vN, vE, vD) in m/ns using the chosen method.

        Falls back to simpler methods when there aren't enough samples.
        """
        if len(self._positions) < 2:
            return (0.0, 0.0, 0.0)
        if len(self._positions) < self._MIN_POINTS_FOR_WLS:
            return self._estimate_velocity_linear()

        if method == VelocityMethod.LINEAR:
            return self._estimate_velocity_linear()
        elif method == VelocityMethod.NUMPY_REGRESSION:
            return self._estimate_velocity_numpy()
        elif method == VelocityMethod.CLUSTER:
            return self._estimate_velocity_cluster()
        else:
            return self._estimate_velocity_wls()

    def estimate_velocity(self, at_timestamp_ns: int, fallback: VelocityNED | None = None,
                 method: VelocityMethod = VelocityMethod.WLS) -> VelocityNED | None:
        if not self._positions:
            return fallback

        at_timestamp_ns = int(at_timestamp_ns)
        self._forget_old(at_timestamp_ns)
        if not self._positions:
            return fallback

        if len(self._positions) == 1:
            return VelocityNED(0, 0, 0)

        vn, ve, vd = self._estimate_velocity(method)
        return VelocityNED(vn, ve, vd)

    def estimate(self, at_timestamp_ns: int, fallback: PositionNED | None = None,
                 method: VelocityMethod = VelocityMethod.WLS) -> PositionNED | None:
        """Estimate target NED position at *at_timestamp_ns*.

        With >=3 points: weighted least-squares fit filters noise and
        uses all stored history. With 2 points: simple linear extrapolation.
        """
        if not self._positions:
            return fallback

        at_timestamp_ns = int(at_timestamp_ns)
        self._forget_old(at_timestamp_ns)
        if not self._positions:
            return fallback

        ts, newest = self._positions[-1]
        if len(self._positions) == 1:
            return newest

        vn, ve, vd = self._estimate_velocity(method)
        dt = at_timestamp_ns - ts
        self._last_method = method
        return PositionNED(
            north_m=newest.north_m + vn * dt,
            east_m=newest.east_m + ve * dt,
            down_m=newest.down_m + vd * dt,
        )

    def estimate_based_on_velocity(self, at_timestamp_ns: int, fallback: PositionNED | None = None,
                velocity : VelocityNED = VelocityNED(0, 0, 0)):
        if not self._positions:
            return fallback

        at_timestamp_ns = int(at_timestamp_ns)
        self._forget_old(at_timestamp_ns)
        if not self._positions:
            return fallback

        ts, newest = self._positions[-1]
        if len(self._positions) == 1:
            return newest

        dt = at_timestamp_ns - ts
        return PositionNED(
            north_m=newest.north_m + velocity.north_m_s * dt,
            east_m=newest.east_m + velocity.east_m_s * dt,
            down_m=newest.down_m + velocity.down_m_s * dt,
        )

    @property
    def latest(self) -> PositionNED | None:
        """Most recent stored position, or None."""
        if self._positions:
            return self._positions[-1][1]
        return None


def test():
    t = TargetEstimator()
    t.add_target_pos(XY(1, 2), 1)
    t.add_target_pos(XY(2, 3), 2)
    t.add_target_pos(XY(3, 4), 3)
    t.add_target_pos(XY(4, 5), 4)
    estimation = t.estimate_target_pos(5)
    assert(estimation == XY(5, 6))


    t = TargetEstimator()
    t.add_target_pos(XY(-0.1, -0.2), 1)
    t.add_target_pos(XY(-0.2, -0.3), 2)
    estimation = t.estimate_target_pos(3)
    print(estimation)
    estimation = t.estimate_target_pos(5)
    print(estimation)

    t = TargetEstimator()
    bbox=Rect.from_xywh(x=0.489, y=0.211, w=0.181, h=0.019),

def test2():
    t = TargetEstimator()
    rects = [
        Rect.from_xywh(x=0.317, y=0.383, w=0.629, h=0.429),
        Rect.from_xywh(x=0.318, y=0.382, w=0.631, h=0.431),
        # Rect(x=0.319, y=0.381, w=0.632, h=0.432),
        # Rect(x=0.320, y=0.380, w=0.634, h=0.434),
        # Rect(x=0.322, y=0.378, w=0.635, h=0.435),
        # Rect(x=0.323, y=0.377, w=0.637, h=0.437),
        # Rect(x=0.324, y=0.376, w=0.639, h=0.439),
        # Rect(x=0.325, y=0.375, w=0.640, h=0.440),
        # Rect(x=0.326, y=0.374, w=0.642, h=0.442),
        # Rect(x=0.328, y=0.372, w=0.643, h=0.443),
        # Rect(x=0.329, y=0.371, w=0.644, h=0.444),
        # Rect(x=0.330, y=0.370, w=0.646, h=0.446),
        # Rect(x=0.331, y=0.369, w=0.647, h=0.447),
        # Rect(x=0.333, y=0.367, w=0.649, h=0.449),
        # Rect(x=0.334, y=0.366, w=0.650, h=0.450),
    ]

    for i, r in enumerate(rects):
        t.add_target_pos(r.center, i)

    estimation = t.estimate_target_pos(len(rects) + 1, None)
    print(estimation)


if __name__ == "__main__":
    test2()