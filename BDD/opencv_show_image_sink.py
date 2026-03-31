#!/usr/bin/env python3

import cv2
import queue
import threading

from interfaces import FrameSinkInterface

import logging

logger = logging.getLogger(__name__)

class OpenCVShowImageSink(FrameSinkInterface):
    def __init__(self, input_handler = None, window_title = '', fps_hint = 60):
        self._window_name = str(id(self))
        self._window_title = window_title
        self._input_handler = input_handler
        # self._fps_counter = FPSCounter()
        self._fps_hint = fps_hint
        self._frame_queue = queue.Queue()
        self._display_thread = None


    def start(self, frame_size):
        self._display_thread = threading.Thread(
            target=self.__display_thread_func,
            args=(frame_size, ),
            name="DISLAY"
        )
        self._display_thread.start()


    def stop(self):
        self._frame_queue.put_nowait(None)
        if self._display_thread is not None:
            self._display_thread.join()


    def set_input_handler(self, input_handler):
        self._input_handler = input_handler


    def process_frame(self, frame):
        # current_fps = self._fps_counter.on_frame()
        if frame is None:
            return

        self._frame_queue.put(frame)


    def __display_thread_func(self, frame_size):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle(self._window_name, self._window_title)
        cv2.resizeWindow(self._window_name, frame_size[0], frame_size[0])
        # cv2.setWindowProperty(self._window_name, cv2.CV_WND_PROP_FULLSCREEN, cv2.CV_WINDOW_FULLSCREEN)
        frame_id = -1
        while True:
            try:
                frame = self._frame_queue.get()
                if frame is None:
                    break

                frame_id += 1
                cv2.imshow(self._window_name, frame)
                # cv2.setWindowTitle(self._window_name, f"{self._window_title} zoom: {frame.metadata.zoom_factor} FPS: {current_fps}")
                key = cv2.waitKeyEx(int(1000 / (self._fps_hint * 2)))
                # if key == -1:
                #     return

                if self._input_handler:
                    ok_to_continue = self._input_handler(key & 0xFF)
                    if not ok_to_continue:
                        raise Exception("User requested to terminate by pressing: %s", key)

            except:
                logger.exception("exception on frame %s", frame_id, exc_info=True)
        cv2.destroyWindow(self._window_name)

