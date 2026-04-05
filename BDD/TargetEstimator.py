#!/usr/bin/env python3

from collections import deque

from helpers import XY, Rect


class TargetEstimator:
    def __init__(self, max_target_positions : int = 10, max_target_position_age_nanoseconds : int = 500):
        assert(max_target_positions > 1)
        assert(max_target_position_age_nanoseconds > 1)

        self.max_target_positions = int(max_target_positions)
        self.max_target_position_age_nanoseconds = int(max_target_position_age_nanoseconds)
        self.target_positions : deque[tuple[int, XY]] = deque(maxlen=self.max_target_positions)


    def history_size(self):
        return len(self.target_positions)


    def clear_history(self):
        return self.target_positions.clear()


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
    bbox=Rect(x=0.489, y=0.211, w=0.181, h=0.019),

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