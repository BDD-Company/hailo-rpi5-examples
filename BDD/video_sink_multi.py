#!/usr/bin/env python

from interfaces import FrameSinkInterface

class MultiSink(FrameSinkInterface):
    def __init__(self, sinks : list[FrameSinkInterface] | None = None):
        # Accept no args so callers can build it empty and append() conditionally
        # (e.g. append a recorder only when recording). Copy so the caller's list
        # isn't aliased.
        self.sinks = list(sinks) if sinks is not None else []

    def append(self, sink : FrameSinkInterface):
        assert isinstance(sink, FrameSinkInterface)
        if sink in self.sinks:
            return

        self.sinks.append(sink)

    def __len__(self):
        return len(self.sinks)

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
