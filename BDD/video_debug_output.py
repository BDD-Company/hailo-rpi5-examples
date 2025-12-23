import os, time, threading, queue, pathlib, signal
import numpy as np
import cv2

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GLib, GstRtspServer

Gst.init(None)

from interfaces import FrameSinkInterface

import logging
logger = logging.getLogger(__name__)

class FrameStreamer(FrameSinkInterface):
    """
    Push BGR np.ndarray frames in, get:
      - RTSP at rtsp://<host>:<port><path>
      - Rolling MP4 clips (<= segment_seconds each) in record_dir
    """

    def __init__(self,
                 frame_size,                 # (width, height)
                 fps_hint: float = 30.0,
                 rtsp_port: int = 8554,
                 rtsp_path: str = "/stream",
                 record_dir: str = "/tmp/recordings",
                 segment_seconds: int = 30,
                 bitrate_kbps: int = 4000,
                 keyint_seconds: int = 2):
        self.w, self.h = int(frame_size[0]), int(frame_size[1])
        assert self.w > 0 and self.h > 0
        self.fps_hint = float(fps_hint) if fps_hint and fps_hint > 0 else None
        self.segment_ns = int(segment_seconds * 1e9)
        self.bitrate_kbps = int(bitrate_kbps)
        self.keyint = max(1, int((self.fps_hint or 30.0) * keyint_seconds))
        self.rtsp_port = rtsp_port
        self.rtsp_path = rtsp_path
        self.record_dir = pathlib.Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)

        # Queued frames (producer -> pusher thread)
        self.q = queue.Queue(maxsize=120)  # ~4s at 30 fps

        # GLib mainloop for RTSP server
        self.mainloop = GLib.MainLoop()
        self.mainloop_thread = threading.Thread(target=self.mainloop.run, daemon=True)

        # RTSP server setup (lazy: stream starts when a client connects)
        self.server = GstRtspServer.RTSPServer.new()
        self.server.props.service = str(rtsp_port)
        self.server.attach(None)

        mounts = self.server.get_mount_points()
        self.factory = GstRtspServer.RTSPMediaFactory.new()
        self.factory.set_shared(True)
        # appsrc -> x264 -> h264parse -> rtph264pay (config-interval=1) -> pay0
        launch = (
            f'appsrc name=rtsp_src is-live=true block=false format=time do-timestamp=true '
            f'caps=video/x-raw,format=BGR,width={self.w},height={self.h} '
            f'! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate={self.bitrate_kbps} key-int-max={self.keyint} '
            f'! h264parse config-interval=1 '
            f'! rtph264pay config-interval=1 name=pay0'
        )
        self.factory.set_launch(f'({launch})')
        mounts.add_factory(self.rtsp_path, self.factory)

        # The RTSP pipeline (and its appsrc) are created only on first client connect.
        self.factory.connect("media-configure", self._on_media_configure)
        self.rtsp_appsrc = None  # set on client connect

        # Recording pipeline (always running)
        record_launch = (
            f'appsrc name=rec_src is-live=true block=true format=time do-timestamp=true '
            f'caps=video/x-raw,format=BGR,width={self.w},height={self.h} '
            f'! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate={self.bitrate_kbps} key-int-max={self.keyint} '
            f'! h264parse config-interval=1 '
            f'! splitmuxsink name=smx max-size-time={self.segment_ns} muxer-factory=mp4mux async-finalize=true '
            f'location="{str(self.record_dir / "clip-%05d.mp4")}"'
        )
        self.rec_pipeline = Gst.parse_launch(record_launch)
        self.rec_appsrc = self.rec_pipeline.get_by_name("rec_src")
        self.splitmux = self.rec_pipeline.get_by_name("smx")

        # optional: timestamped filenames via "format-location" signal
        self.splitmux.connect("format-location", self._on_format_location)

        self.rec_pipeline.set_state(Gst.State.PLAYING)

        # Start GLib loop (RTSP server) and pushing thread
        self.push_thread = threading.Thread(target=self._pusher, daemon=True)
        self._stop = threading.Event()


    def start(self):
        self.mainloop_thread.start()
        self.push_thread.start()

        logger.debug(f"RTSP ready at rtsp://0.0.0.0:{self.rtsp_port}{self.rtsp_path}")
        logger.debug(f"Recording to {self.record_dir} in ~{self.segment_ns/1e9:.0f}s chunks")


    # --- public API -----------------------------------------------------------
    def submit(self, frame: np.ndarray):
        """Enqueue one BGR frame (H×W×3, uint8). Non-blocking: drops if queue full."""
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be BGR uint8 HxWx3 numpy array")
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            raise ValueError(f"unexpected frame size {frame.shape[1]}x{frame.shape[0]}, expected {self.w}x{self.h}")
        ts_ns = time.monotonic_ns()
        try:
            self.q.put_nowait((frame.copy(), ts_ns))
        except queue.Full:
            # Drop (real-time bias). Recording appsrc is blocking and will naturally back-pressure.
            pass

    def stop(self):
        self._stop.set()
        self.push_thread.join(timeout=2.0)
        # Gracefully finalize last MP4 fragment
        self.rec_pipeline.send_event(Gst.Event.new_eos())
        self.rec_pipeline.set_state(Gst.State.NULL)
        # Stop RTSP mainloop
        try:
            self.mainloop.quit()
        except Exception:
            pass
        self.mainloop_thread.join(timeout=2.0)

    # --- internals ------------------------------------------------------------
    def _on_media_configure(self, factory, media):
        # Called when a client connects: grab the appsrc inside the RTSP media bin
        element = media.get_element()
        self.rtsp_appsrc = element.get_child_by_name("rtsp_src")
        # Ensure caps/properties are applied (some distros need explicit set after creation)
        caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={self.w},height={self.h}")
        self.rtsp_appsrc.set_property("caps", caps)
        self.rtsp_appsrc.set_property("is-live", True)
        self.rtsp_appsrc.set_property("format", Gst.Format.TIME)
        self.rtsp_appsrc.set_property("block", False)
        self.rtsp_appsrc.set_property("do-timestamp", True)

    def _on_format_location(self, splitmux, fragment_id: int):
        # Timestamped filenames: clip-YYYYmmdd-HHMMSS-<id>.mp4
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return str(self.record_dir / f"clip-{stamp}-{fragment_id:05d}.mp4")

    @staticmethod
    def _np_to_gst_buffer(frame: np.ndarray) -> Gst.Buffer:
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        # Rely on appsrc do-timestamp=True for PTS/DTS; only fill payload here.
        buf.fill(0, data)
        return buf

    def _push_to_appsrc(self, appsrc, frame: np.ndarray) -> bool:
        if appsrc is None:
            return False
        buf = self._np_to_gst_buffer(frame)
        ret = appsrc.emit("push-buffer", buf)
        return ret == Gst.FlowReturn.OK

    def _pusher(self):
        """Pull frames from queue and push:
           1) synchronously to recording appsrc (block=True, so we don't lose frames)
           2) best-effort to RTSP appsrc (drop if back-pressured / no client)
        """
        while not self._stop.is_set():
            try:
                frame, _ts = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            # 1) Recording (blocking)
            self._push_to_appsrc(self.rec_appsrc, frame)
            # 2) RTSP (best-effort)
            try:
                self._push_to_appsrc(self.rtsp_appsrc, frame)
            except Exception:
                pass
            self.q.task_done()

# --- Example usage ------------------------------------------------------------
if __name__ == "__main__":
    w, h = 640, 480
    fs = FrameStreamer((w, h), fps_hint=30.0, rtsp_port=8554, rtsp_path="/stream",
                       record_dir="./recordings", segment_seconds=30, bitrate_kbps=3000)
    fs.start()

    print("Submit test frames, view with e.g.:")
    print("  VLC:    rtsp://127.0.0.1:8554/stream")
    print("  ffplay: ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/stream")

    try:
        t0 = time.time()
        while True:
            # demo: moving gradient
            img = np.zeros((h, w, 3), dtype=np.uint8)
            x = int((time.time() * 100) % w)
            cv2.circle(img, (x, h//2), 40, (0, 255, 255), -1)
            cv2.putText(img, time.strftime("%H:%M:%S"), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            fs.submit(img)
            # pace roughly ~fps_hint, but the system works with variable input rates too
            time.sleep(1.0/30.0)
    except KeyboardInterrupt:
        pass
    finally:
        fs.stop()
