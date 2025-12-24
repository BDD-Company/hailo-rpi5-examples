#!/usr/bin/env python

from interfaces import FrameSinkInterface

class MultiSink(FrameSinkInterface):
    def __init__(self, sinks : list[FrameSinkInterface]):
        self.sinks = sinks
        # [ assert isinstance(s, FrameSinkInterface) for s in sinks]

    def __del__(self):
        try:
            self.stop()
        except:
            pass

    def start(self, frame_size):
        for s in self.sinks:
            s.start(frame_size)

    def stop(self):
        for s in self.sinks:
            s.stop()

    def process_frame(self, frame):
        for s in self.sinks:
            s.process_frame(frame)
