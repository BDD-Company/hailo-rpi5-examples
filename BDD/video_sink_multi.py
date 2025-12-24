#!/usr/bin/env python

from interfaces import FrameSink, CapturedFrame

class MultiSink(FrameSink):
    def __init__(self, sinks : list[FrameSink]):
        self.sinks = sinks

    def __del__(self):
        try:
            self.stop()
        except:
            pass

    def start(self):
        for s in self.sinks:
            s.start()

    def stop(self):
        for s in self.sinks:
            s.stop()

    def process_frame(self, frame : CapturedFrame):
        for s in self.sinks:
            s.process_frame(frame)
