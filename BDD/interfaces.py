#! /usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np


class FrameSinkInterface(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def submit(self, frame: np.ndarray):
        pass

    @abstractmethod
    def stop(self):
        pass
