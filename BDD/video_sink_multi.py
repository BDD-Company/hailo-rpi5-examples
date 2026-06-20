#!/usr/bin/env python

import logging

from interfaces import FrameSinkInterface

logger = logging.getLogger(__name__)


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
        # Isolate failures: if one sink can't start (e.g. RecorderSink missing
        # an H.264 encoder), drop it and keep the others. Otherwise a single
        # broken sink takes the whole --display window down with it.
        survivors = []
        for s in self.sinks:
            try:
                s.start(frame_size)
                survivors.append(s)
            except Exception:
                logger.exception("MultiSink: %s.start() failed — dropping this sink", type(s).__name__)
        self.sinks = survivors

    def stop(self):
        for s in self.sinks:
            try:
                s.stop()
            except Exception:
                logger.exception("MultiSink: %s.stop() failed", type(s).__name__)

    def process_frame(self, frame):
        for s in self.sinks:
            try:
                s.process_frame(frame)
            except Exception:
                logger.exception("MultiSink: %s.process_frame() failed", type(s).__name__)
