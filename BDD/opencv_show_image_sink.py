#!/usr/bin/env python3

import cv2
import multiprocessing
import queue

from interfaces import FrameSinkInterface

import logging

logger = logging.getLogger(__name__)


# Top-level so it's picklable for both fork and spawn start methods.
# Runs in a separate process — required because OpenCV here is built against
# Qt5, whose highgui is not thread-safe off the *main thread of the process*:
# calling namedWindow/imshow/waitKey from a worker thread silently renders
# nothing. The main thread of this app is owned by GLib.MainLoop (Hailo
# pipeline), so we offload the display to its own process where it gets its
# own main thread. Same pattern as app_base.display_user_data_frame.
def _display_process_func(frame_queue, frame_size, window_title, fps_hint):
    # Subprocess runs with inherited stdout/stderr; print directly so failures
    # surface even if Python logging isn't reconfigured in the child.
    import os, sys, traceback
    try:
        print(f"[display] pid={os.getpid()} DISPLAY={os.environ.get('DISPLAY')} "
              f"XAUTHORITY={os.environ.get('XAUTHORITY')} "
              f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '(unset)')}",
              file=sys.stderr, flush=True)
        window_name = window_title or 'BDD'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle(window_name, window_title)
        cv2.resizeWindow(window_name, frame_size[0], frame_size[1])
        print(f"[display] window '{window_name}' created at {frame_size}", file=sys.stderr, flush=True)
        wait_ms = max(1, int(1000 / max(1, fps_hint * 2)))
        frame_id = -1
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            frame_id += 1
            if frame_id == 0:
                print(f"[display] first frame received, shape={frame.shape}",
                      file=sys.stderr, flush=True)
            cv2.imshow(window_name, frame)
            cv2.waitKey(wait_ms)
        cv2.destroyAllWindows()
    except Exception:
        print("[display] FATAL:", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        raise


class OpenCVShowImageSink(FrameSinkInterface):
    def __init__(self, input_handler=None, window_title='', fps_hint=60):
        self._window_title = window_title
        self._input_handler = input_handler
        self._fps_hint = fps_hint
        # multiprocessing.Queue can survive across fork/spawn; bounded so a stuck
        # display process won't grow memory without limit.
        self._frame_queue = multiprocessing.Queue(maxsize=4)
        self._display_process = None

    def start(self, frame_size):
        self._display_process = multiprocessing.Process(
            target=_display_process_func,
            args=(self._frame_queue, frame_size, self._window_title, self._fps_hint),
            name="DISPLAY",
            daemon=True,
        )
        self._display_process.start()

    def stop(self):
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._display_process is not None:
            self._display_process.join(timeout=2.0)
            if self._display_process.is_alive():
                self._display_process.terminate()
                self._display_process.join(timeout=1.0)

    def set_input_handler(self, input_handler):
        self._input_handler = input_handler

    def process_frame(self, frame):
        if frame is None:
            return
        # Drop on backpressure: keeps the upstream debug_output_thread snappy and
        # avoids piling up frames if the display process can't keep up.
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass
