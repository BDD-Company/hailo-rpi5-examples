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
        # Bounded on purpose: a live display shows the NEWEST frame; if cv2.imshow can't
        # keep up with the producer (e.g. slow over VNC) we DROP stale frames rather than
        # buffer them. The old unbounded queue.Queue() grew ~2.6MB/frame and exhausted RAM
        # in ~90s -> OOM -> the whole pipeline (camera included) stalled and the app aborted.
        self._frame_queue = queue.Queue(maxsize=2)
        self._display_thread = None


    def start(self, frame_size):
        self._display_thread = threading.Thread(
            target=self.__display_thread_func,
            args=(frame_size, ),
            name="DISLAY"
        )
        self._display_thread.start()


    def stop(self):
        # Drain first so the None sentinel always fits in the bounded queue even when the
        # display thread is behind (a full queue would make put_nowait(None) raise).
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        self._frame_queue.put(None)
        if self._display_thread is not None:
            self._display_thread.join()


    def set_input_handler(self, input_handler):
        self._input_handler = input_handler


    def process_frame(self, frame):
        # current_fps = self._fps_counter.on_frame()
        if frame is None:
            return

        # Latest-frame-wins: never block or grow unbounded. If the display is behind, drop
        # the stale queued frame and enqueue the newest. (Prevents the OOM described in
        # __init__.)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass


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
                # Frames arrive RGB (annotate_frame_with_detection_info builds the canvas
                # via to_rgb; the recorder sink declares caps=RGB). cv2.imshow's convention
                # is BGR, so without this swap red<->blue are exchanged (a pink ball reads
                # blue). GStreamer sinks respect the declared caps and need no swap.
                cv2.imshow(self._window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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

