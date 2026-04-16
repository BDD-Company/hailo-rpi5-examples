#! /usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np


class FrameSinkInterface(ABC):

    @abstractmethod
    def start(self, frame_size : tuple[int, int]):
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray):
        pass

    @abstractmethod
    def stop(self):
        pass

class TargetEstimatorInterface(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def describe_prev_estimation(self) -> str:
        pass

    @abstractmethod
    def history_size() -> int:
        pass

    @abstractmethod
    def max_history_size() -> int:
        pass

    @abstractmethod
    def clear_history():
        pass

    @abstractmethod
    def max_age_ns() -> int:
        pass