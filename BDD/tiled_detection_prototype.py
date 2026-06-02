#!/usr/bin/env python3
"""Standalone prototype: parallel detection-only pipeline using hailotilecropper.

Goal: maximize detection FPS by (a) splitting a LARGE pre-tiling frame into a configurable
NxM grid of tiles with hailotilecropper (native Hailo tiling), (b) BATCHING all tiles of a
frame through hailonet (batch-size = N*M), and (c) reassembling + cross-tile NMS with
hailotileaggregator — instead of the app's producer-side, one-buffer-per-tile, batch-1 path.

This is intentionally separate from app.py / app_base.py: it's a benchmark/exploration harness
to find the best (frame size, tile grid, batch size) operating point before integrating the
chosen config into detection mode.

Examples (run on the Pi inside the hailo venv, with setup_env.sh sourced):
    # pure-compute ceiling with a synthetic source:
    python BDD/tiled_detection_prototype.py --source test --width 1332 --height 990 --tiles 2x2
    # 3x3 tiling off a bigger sensor frame:
    python BDD/tiled_detection_prototype.py --source test --width 2028 --height 1520 --tiles 3x3
    # end-to-end off the real camera (libcamerasrc), fastest imx477 mode:
    python BDD/tiled_detection_prototype.py --source camera --width 1332 --height 990 --tiles 2x2 --fps 120

The HEF defaults to scripts/bdd.sh's HAILO_MODEL; override with --hef. Reports rendered FPS
(full frames/s out of the aggregator), the implied tile-inference rate, and avg detections/frame.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad

DEFAULT_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"


def default_hef() -> str:
    """Resolve HAILO_MODEL from scripts/bdd.sh (the project's source of truth)."""
    bdd_sh = Path(__file__).resolve().parent.parent / "scripts" / "bdd.sh"
    try:
        for line in bdd_sh.read_text().splitlines():
            m = re.match(r'\s*export\s+HAILO_MODEL="([^"]+)"', line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return os.environ.get("HAILO_MODEL", "")


def parse_grid(s: str) -> tuple[int, int]:
    m = re.fullmatch(r'(\d+)x(\d+)', s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"--tiles must look like '2x2', got {s!r}")
    return int(m.group(1)), int(m.group(2))


def build_source(source: str, width: int, height: int, fps: int) -> str:
    """RGB source at width x height @ fps. 'test' = synthetic (compute ceiling), 'camera' =
    libcamerasrc (real imx477; simplest camera path for a prototype — not the app's picamera2
    appsrc low-latency path)."""
    caps = f"video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1"
    if source == "test":
        return f"videotestsrc is-live=true ! {caps}"
    if source == "camera":
        # Pin format=NV12: without an explicit processed format libcamerasrc negotiates raw
        # Bayer (SBGGR16) at an exact sensor-mode size and videoconvert can't debayer it. NV12 is
        # the ISP-native processed output; convert to RGB for the cropper/model.
        return (f"libcamerasrc ! video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
                f"videoconvert n-threads=4 ! {caps}")
    raise ValueError(f"unknown source {source!r}")


def build_pipeline(hef: str, so: str, source_str: str,
                   tiles_x: int, tiles_y: int, overlap: float,
                   model_size: int, batch_size: int, lean: bool = False) -> str:
    """Tiling detection pipeline. cropper.src_0 = bypass -> agg.sink_0; cropper.src_1 = tiles ->
    scale/convert -> batched hailonet -> hailofilter -> agg.sink_1. The tile-branch queues are
    NON-leaky: hailotileaggregator pairs the bypass with all N*M tiles by buffer, so dropping a
    tile would desync it. The input queue is leaky to shed whole frames under backpressure.

    lean=True drops the explicit `videoscale ! caps(model)` from the inference branch: hailonet's
    fixed input caps propagate back through videoconvert so the hailotilecropper scales each tile
    straight to the model size (one resize instead of cropper-native + videoscale). The per-tile
    scale/convert is the pipeline bottleneck, so this is the main throughput lever."""
    nms = ("nms-score-threshold=0.3 nms-iou-threshold=0.45 "
           "output-format-type=HAILO_FORMAT_TYPE_FLOAT32")
    if lean:
        tile_to_net = (
            f"videoconvert n-threads=2 ! "
            f"queue name=net_q leaky=no max-size-buffers={max(8, batch_size * 2)} max-size-bytes=0 max-size-time=0 ! "
        )
    else:
        tile_to_net = (
            f"videoscale n-threads=4 ! video/x-raw,width={model_size},height={model_size} ! "
            f"videoconvert n-threads=2 ! "
            f"queue name=net_q leaky=no max-size-buffers={max(8, batch_size * 2)} max-size-bytes=0 max-size-time=0 ! "
        )
    return (
        f"{source_str} ! "
        f"queue name=in_q leaky=downstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
        f"hailotilecropper internal-offset=true name=cropper "
        f"tiles-along-x-axis={tiles_x} tiles-along-y-axis={tiles_y} "
        f"overlap-x-axis={overlap} overlap-y-axis={overlap} "
        f"hailotileaggregator name=agg flatten-detections=true "
        # bypass branch
        f"cropper. ! queue name=bypass_q leaky=no max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! agg.sink_0 "
        # inference branch (batched tiles)
        f"cropper. ! queue name=tile_q leaky=no max-size-buffers={max(8, batch_size * 2)} max-size-bytes=0 max-size-time=0 ! "
        f"{tile_to_net}"
        f"hailonet hef-path={hef} batch-size={batch_size} vdevice-group-id=1 {nms} force-writable=true ! "
        f"queue name=filt_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! "
        f"hailofilter so-path={so} qos=false ! "
        f"queue name=agg_in_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! agg.sink_1 "
        # output: count detections, then a tee'd fps sink
        f"agg. ! queue name=out_q leaky=no max-size-buffers=4 ! "
        f"identity name=probe ! "
        f"fpsdisplaysink name=fps video-sink=fakesink text-overlay=false sync=false signal-fps-measurements=true"
    )


class Bench:
    def __init__(self, args):
        self.args = args
        self.frames = 0
        self.det_total = 0
        self.fps_samples: list[float] = []
        self.t0 = None

    def on_probe(self, pad, info, _u):
        buf = info.get_buffer()
        if buf is not None:
            roi = hailo.get_roi_from_buffer(buf)
            self.det_total += len(roi.get_objects_typed(hailo.HAILO_DETECTION))
            self.frames += 1
            if self.t0 is None:
                self.t0 = time.monotonic()
        return Gst.PadProbeReturn.OK

    def on_fps(self, _sink, fps, _drop, _avg):
        self.fps_samples.append(fps)
        tiles = self.args.tiles_x * self.args.tiles_y
        print(f"  fps={fps:6.2f}  tile_inferences/s={fps * tiles:7.1f}  "
              f"frames={self.frames}  dets/frame={self.det_total / max(1, self.frames):.2f}",
              flush=True)
        return True

    def run(self):
        a = self.args
        Gst.init(None)
        source_str = build_source(a.source, a.width, a.height, a.fps)
        pstr = build_pipeline(a.hef, a.so, source_str, a.tiles_x, a.tiles_y,
                              a.overlap, a.model_size, a.batch_size, a.lean)
        tiles = a.tiles_x * a.tiles_y
        print(f"# {a.tiles_x}x{a.tiles_y} tiles ({tiles}) | frame {a.width}x{a.height} -> tile {a.model_size}x{a.model_size}"
              f" | batch={a.batch_size} | source={a.source} | overlap={a.overlap}")
        if a.print_pipeline:
            print(pstr.replace(" ! ", " !\n    "))
        pipeline = Gst.parse_launch(pstr)
        pipeline.get_by_name("probe").get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self.on_probe, None)
        pipeline.get_by_name("fps").connect("fps-measurements", self.on_fps)

        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_msg(_b, msg):
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"ERROR: {err} :: {dbg}", file=sys.stderr)
                loop.quit()
            elif msg.type == Gst.MessageType.EOS:
                loop.quit()
            return True
        bus.connect("message", on_msg)

        GLib.timeout_add_seconds(a.duration, loop.quit)
        pipeline.set_state(Gst.State.PLAYING)
        start = time.monotonic()
        try:
            loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)
        elapsed = time.monotonic() - (self.t0 or start)
        wall_fps = self.frames / elapsed if elapsed > 0 else 0.0
        steady = self.fps_samples[1:] or self.fps_samples  # drop first warmup sample
        avg_fps = sum(steady) / len(steady) if steady else wall_fps
        print("# ---- summary ----")
        print(f"# frames={self.frames}  wall_fps={wall_fps:.2f}  avg_measured_fps={avg_fps:.2f}  "
              f"tile_inf/s={avg_fps * tiles:.1f}  dets/frame={self.det_total / max(1, self.frames):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["test", "camera"], default="test")
    ap.add_argument("--width", type=int, default=1332, help="pre-tiling frame width")
    ap.add_argument("--height", type=int, default=990, help="pre-tiling frame height")
    ap.add_argument("--tiles", type=parse_grid, default=(2, 2), help="grid like 2x2, 3x3")
    ap.add_argument("--overlap", type=float, default=0.1, help="tile overlap fraction (0..1)")
    ap.add_argument("--model-size", type=int, default=640, help="HEF input size (square)")
    ap.add_argument("--batch-size", type=int, default=None, help="hailonet batch-size (default = tiles)")
    ap.add_argument("--fps", type=int, default=60, help="source framerate cap")
    ap.add_argument("--duration", type=int, default=15, help="measure seconds")
    ap.add_argument("--hef", default=default_hef())
    ap.add_argument("--so", default=DEFAULT_SO)
    ap.add_argument("--lean", action="store_true",
                    help="drop explicit videoscale; let the cropper scale tiles to the model size")
    ap.add_argument("--print-pipeline", action="store_true")
    args = ap.parse_args()
    args.tiles_x, args.tiles_y = args.tiles
    if args.batch_size is None:
        args.batch_size = args.tiles_x * args.tiles_y
    if not args.hef or not Path(args.hef).exists():
        ap.error(f"HEF not found: {args.hef!r} (pass --hef)")
    Bench(args).run()


if __name__ == "__main__":
    main()
