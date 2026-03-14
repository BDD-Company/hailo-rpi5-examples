#!/usr/bin/env python3

from collections import deque

from helpers import XY


class TargetEstimator:
    def __init__(self, max_target_positions : int = 5, max_target_position_age_nanoseconds : int = 500):
        assert(max_target_positions > 1)
        assert(max_target_position_age_nanoseconds > 1)

        self.max_target_positions = int(max_target_positions)
        self.max_target_position_age_nanoseconds = int(max_target_position_age_nanoseconds)
        self.target_positions : deque[tuple[int, XY]] = deque(maxlen=self.max_target_positions)


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
        oldest_timestamp, oldest_pos = self.target_positions[0]
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
