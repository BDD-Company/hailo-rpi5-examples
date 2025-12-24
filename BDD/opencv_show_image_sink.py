#!/usr/bin/env python3

import cv2

from interfaces import FrameSinkinterfaces

class OpenCVShowImageSink():
    def __init__(self, input_handler = None, window_title = '', fps_hint = 60):
        self._window_name = str(id(self))
        self._window_title = window_title
        self._input_handler = input_handler
        self._fps_counter = FPSCounter()
        self._fps_hint = fps_hint

    def start(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle(self._window_name, self._window_name)

    def stop(self):
        cv2.destroyWindow(self._window_name)

    def set_input_handler(self, input_handler):
        self._input_handler = input_handler

    def process_frame(self, frame : interfaces.CapturedFrame):
        current_fps = self._fps_counter.on_frame()
        cv2.setWindowTitle(self._window_name, f"{self._window_title} zoom: {frame.metadata.zoom_factor} FPS: {current_fps}")

        cv2.imshow(self._window_name, frame.data)
        key = cv2.waitKeyEx(int(1000 / (self._fps_hint * 2)))
        if key == -1:
            return

        if self._input_handler:
            ok_to_continue = self._input_handler(key & 0xFF)
            if not ok_to_continue:
                raise Exception("User requested to terminate by pressing: %s", key)
