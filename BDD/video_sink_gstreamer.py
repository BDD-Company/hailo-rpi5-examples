# pip install numpy opencv-python
# sudo apt-get install -y python3-gi python3-gst-1.0 gir1.2-gst-rtsp-server-1.0 \
#   gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly}

import time, threading, queue, pathlib
import numpy as np
import cv2
import datetime
import os

import interfaces
from OverwriteQueue import OverwriteQueue
import logging

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst
from gi.repository import GLib
from gi.repository import GstRtspServer

Gst.init(None)

logger = logging.getLogger(__name__)

# -------------------------- Common helpers -----------------------------------

class _FrameQueuePusher:
    """Small helper to push frames from a Queue to a given appsrc."""
    def __init__(self, appsrc_getter, drop_if_error = True, overwriting_queue = False, queue_size = 120):
        self._get_appsrc = appsrc_getter
        # self._q = queue.Queue(maxsize=120)  # ~4s at 30 FPS
        self._q = OverwriteQueue(maxsize=queue_size)  if overwriting_queue else queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._drop_if_error = drop_if_error

    def start(self):
        self._thr.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        self._thr.join(timeout=timeout)

    def submit(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame.copy())
        except queue.Full:
            # Drop (real-time bias)
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.01)
            except queue.Empty:
                continue
            appsrc = self._get_appsrc()
            if appsrc is not None:
                try:
                    data = frame.tobytes()
                    buf = Gst.Buffer.new_allocate(None, len(data), None)
                    buf.fill(0, data)
                    ret = appsrc.emit("push-buffer", buf)
                    # optional: handle backpressure; here we just drop on error
                    if ret != Gst.FlowReturn.OK and not self._drop_if_error:
                        # simple yield
                        time.sleep(0.001)
                    # logger.debug("!!! emitted frame for : %s", appsrc.name)
                except Exception:
                    logger.exception("Got exception for src: %s", appsrc.name)
                    # swallow errors to keep realtime flow
                    pass
            else:
                # Not a problem, means no client was conneted
                # logger.warning("appsrc is None")
                pass
            # self._q.task_done()


def _validate_frame(frame: np.ndarray, expect_w: int, expect_h: int):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be BGR uint8 HxWx3 numpy array")
    h, w = frame.shape[:2]
    if w != expect_w or h != expect_h:
        raise ValueError(f"unexpected frame size: got {w}x{h}, expected {expect_w}x{expect_h}")


# --------------------------- Class 1: RTSP -----------------------------------

class RtspStreamerSink(interfaces.FrameSinkInterface):
    """
    Serve frames over RTSP:
      - URL: rtsp://<host>:<port><path>

    Interface: start(), submit(frame), stop()
    """
    def __init__(self,
                 fps_hint: float = 30.0,
                 port: int = 8554,
                 path: str = "/stream",
                 bitrate_kbps: int = 4000,
                 keyint_seconds: int = 2):
        self.w, self.h = map(int, (0, 0))
        self.fps = float(fps_hint) if fps_hint and fps_hint > 0 else 30.0
        self.port = int(port)
        self.path = path if path.startswith("/") else "/" + path
        self.bitrate = int(bitrate_kbps)
        self.keyint = max(1, int(self.fps * keyint_seconds))

        self._mainctx = None
        self._mainloop = None
        self._mainloop_thr = None
        self._server = None
        self._factory = None
        self._rtsp_appsrc = None
        self._address = "0.0.0.0"  # bind explicitly on IPv4 to avoid IPv6-only binds

        self._pusher = _FrameQueuePusher(lambda: self._rtsp_appsrc, drop_if_error=True, overwriting_queue = True, queue_size = max(1, int(self.fps / 2)))

    # public API
    def start(self, frame_size):
        self.w, self.h = map(int, frame_size)
        # GLib loop
        self._mainctx = GLib.MainContext.new()
        self._mainloop = GLib.MainLoop(context=self._mainctx)
        self._mainloop_thr = threading.Thread(target=self._mainloop.run, daemon=True)

        # RTSP server
        self._server = GstRtspServer.RTSPServer.new()
        self._server.props.service = str(self.port)
        # bind explicitly; some envs default to IPv6-only and VLC with 127.0.0.1 won’t reach it
        try:
            self._server.set_address(self._address)
        except Exception:
            pass  # API exists on all recent builds; ignore if older
        # attach to the same context the mainloop will run in
        self._server.attach(self._mainctx)

        mounts = self._server.get_mount_points()
        self._factory = GstRtspServer.RTSPMediaFactory.new()
        self._factory.set_shared(True)

        launch = (
            f'appsrc name=rtsp_src is-live=true block=false format=time do-timestamp=true '
            f'caps=video/x-raw,format=RGB,width={self.w},height={self.h},framerate=0/1 '
            f'! queue max-size-buffers=10 leaky=downstream '
            f'! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast '
            f'bitrate={self.bitrate} key-int-max={self.keyint} '
            f'! h264parse config-interval=1 '
            f'! rtph264pay config-interval=1 name=pay0 pt=96 '
        )
        # IMPORTANT: add spaces inside the parentheses so the parser
        # doesn't “stick” the ) to the previous token.
        self._factory.set_launch(f'( {launch} )')
        self._factory.connect("media-configure", self._on_media_configure)

        mounts.add_factory(self.path, self._factory)

        self._pusher.start()
        self._mainloop_thr.start()

        logger.debug(f"[RtspStreamerSink] RTSP at rtsp://0.0.0.0:{self.port}{self.path}")

    def process_frame(self, frame):
        if not hasattr(self, "frame_id"):
            self.frame_id = 0

        _validate_frame(frame, self.w, self.h)
        self._pusher.submit(frame)

        # logger.debug("[RtspStreamerSink] %s%s\tpushed frame %s", self.port, self.path, self.frame_id)
        self.frame_id += 1


    def stop(self):
        self._pusher.stop()
        # No persistent pipeline to EOS — created per-client; just stop mainloop.
        if self._mainloop:
            try:
                self._mainloop.quit()
            except Exception:
                pass
        if self._mainloop_thr:
            self._mainloop_thr.join(timeout=2.0)

    # internals
    def _on_media_configure(self, factory, media):
        # logger.debug("[RtspStreamerSink] media-configure: building pipeline for client")
        element = media.get_element()
        self._rtsp_appsrc = element.get_child_by_name("rtsp_src")
        caps = Gst.Caps.from_string(f"video/x-raw,format=RGB,width={self.w},height={self.h},framerate=0/1")
        self._rtsp_appsrc.set_property("caps", caps)
        self._rtsp_appsrc.set_property("is-live", True)
        self._rtsp_appsrc.set_property("format", Gst.Format.TIME)
        self._rtsp_appsrc.set_property("block", False)
        self._rtsp_appsrc.set_property("do-timestamp", True)

        # logging.debug("Got media_configure for %s", self._rtsp_appsrc.name)


# ------------------------ Class 2: Segment Recorder --------------------------

class RecorderSink(interfaces.FrameSinkInterface):
    """
    Save frames as rolling MP4 clips with max duration (e.g., 30 s each),
    using splitmuxsink.

    Interface: start(), submit(frame), stop()
    """
    def __init__(self,
                 fps_hint: float = 30.0,
                 out_dir: str = "./recordings",
                 segment_seconds: int = 30,
                 bitrate_kbps: int = 4000,
                 keyint_seconds: int = 2,
                 filename_base : str = 'record',
                 filename_pattern: str = "%%s-%Y%m%d-%H%M%S-%%05d.mp4"):
        self.w, self.h = map(int, (0, 0))
        self.fps = float(fps_hint) if fps_hint and fps_hint > 0 else 30.0
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.segment_ns = int(segment_seconds * 1e9)
        self.bitrate = int(bitrate_kbps)
        self.keyint = max(1, int(self.fps * keyint_seconds))
        self.filename_base = filename_base
        self.filename_pattern = filename_pattern

        # test filename_pattern, allow to crash early
        assert self._on_format_location(None, 0)

        self._pipeline = None
        self._appsrc = None
        self._splitmux = None
        self._pusher = _FrameQueuePusher(lambda: self._appsrc, drop_if_error=False, overwriting_queue=True)

    # public API
    def start(self, frame_size):
        self.w, self.h = map(int, frame_size)
        # Build pipeline
        # NOTE: appsrc block=true to naturally backpressure if disk/encoder is slow
        pattern_path = str(self.out_dir / "clip-%05d.mp4")  # initial; we override via format-location
        launch = (
            f'appsrc name=rec_src is-live=true block=true format=time do-timestamp=true '
            f'caps=video/x-raw,format=RGB,width={self.w},height={self.h} '
            f'! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast '
            f'bitrate={self.bitrate} key-int-max={self.keyint} '
            f'! h264parse config-interval=1 '
            f'! splitmuxsink name=smx max-size-time={self.segment_ns} muxer-factory=mp4mux async-finalize=true '
            f'location="{pattern_path}"'
        )
        self._pipeline = Gst.parse_launch(launch)
        self._appsrc = self._pipeline.get_by_name("rec_src")
        self._splitmux = self._pipeline.get_by_name("smx")
        self._splitmux.connect("format-location", self._on_format_location)

        self._pipeline.set_state(Gst.State.PLAYING)
        self._pusher.start()
        logger.info(f"[RecorderSink] Writing ~{self.segment_ns/1e9:.0f}s MP4 segments to {self.out_dir}")

    def process_frame(self, frame):
        _validate_frame(frame, self.w, self.h)
        self._pusher.submit(frame)

        # logger.debug("[RecorderSink] %s / %s\tpushed frame %s", self.out_dir, self.filename_base, frame.id)

    def stop(self):
        # Drain gracefully
        self._pusher.stop()
        if self._pipeline:
            try:
                self._pipeline.send_event(Gst.Event.new_eos())
            except Exception:
                pass
            self._pipeline.set_state(Gst.State.NULL)

    # internals
    def _on_format_location(self, splitmux, fragment_id: int):
        # Build timestamped filename with fragment counter
        filename = datetime.datetime.now().strftime(self.filename_pattern) % (self.filename_base, fragment_id)
        return str(self.out_dir / filename)


# ----------------------------- Example usage ---------------------------------
if __name__ == "__main__":
    from video_sink_multi import MultiSink
    from helpers import configure_logging

    configure_logging(level = logging.DEBUG)

    w, h = 640, 480

    rtsp = RtspStreamerSink(fps_hint=30, port=8554, path="/stream", bitrate_kbps=3000)
    rec  = RecorderSink(fps_hint=30, out_dir="./recordings", segment_seconds=30, bitrate_kbps=3000)
    sink = MultiSink([
        rtsp,
        rec
        # OpenCVShowImageSink(window_title='DEBUG IMAGE')
    ])

    # rtsp.start((w, h))
    # # rec.start((w, h))
    sink.start((w, h))

    print("Watch with:")
    print("  vlc rtsp://127.0.0.1:8554/stream")
    print("Stop: Ctrl+C")

    try:
        frame_id = 0
        while True:

            # demo frame
            img = np.zeros((h, w, 3), dtype=np.uint8)
            x = int((time.time() * 100) % w)
            cv2.circle(img, (x, h//2), 40, (0, 255, 255), -1)
            cv2.putText(img, time.strftime("%H:%M:%S"), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            # feed both (in your app, just call .submit on the subset you use)

            frame = img
            sink.process_frame(frame)
            # rtsp.process_frame(frame)
            # rec.process_frame(frame)

            time.sleep(0.03)
            frame_id += 1

    except KeyboardInterrupt:
        pass
    finally:
        # shutdown order: stop sources first, then pipelines/loop
        # rec.stop()
        # rtsp.stop()
        sink.stop()
