#!/usr/bin/env python3
"""NV12 tiling benchmark for the Hailo GStreamer plugin.

Builds: source(NV12) -> tee -> [ tiling branch: NxM hailotilecropper -> hailonet -> aggregator ]
                              -> [ whole-frame branch: 1x1 cropper   -> hailonet -> aggregator ]
Both hailonets share one VDevice (same vdevice-group-id, round-robin scheduler), so this
models the real "NxM tiles + 1 whole frame" per camera frame on a single Hailo-8L.

Measures, per branch, over --duration seconds:
  - throughput (aggregated frames/s out of the branch)
  - per-frame in->out latency (cropper-sink -> aggregator-src, FIFO matched; the inner
    path is non-leaky so every timed input yields exactly one output)
  - tiles/s actually pushed through hailonet (sanity: ~= fps * NxM)

The source is synthetic (videotestsrc NV12) because the Pi camera is the SAME across all
tiling modes; tiling cost lives entirely in the inference path. Add the camera Stage-A
capture latency (~135-170 ms, see latency-budget memory) to get true sensor->command e2e.
"""
import argparse
import collections
import time
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


def make_pipeline(args):
    common_net = (
        f"hef-path={args.hef} batch-size=1 vdevice-group-id=bench_shared "
        f"scheduling-algorithm=1 force-writable=true"
    )
    # is-live=true + framerate caps -> operating-point test (camera-like).
    # is-live=false, no framerate -> throughput-ceiling test (push as fast as NPU accepts).
    if args.picam:
        # picamera2 -> appsrc: buffers are SYSTEM MEMORY NV12 (no dmabuf), so the cropper
        # maps them directly with NO conversion and NO copy hop — the production-correct
        # zero-(format)-copy path for tiling. A producer thread (start_picam) pushes frames.
        src = ("appsrc name=app_source is-live=true do-timestamp=true format=time "
               "max-buffers=4 leaky-type=downstream")
        src_caps = (f"video/x-raw,format=NV12,width={args.width},height={args.height},"
                    f"framerate={args.fps}/1")
    elif args.camera:
        # real sensor, NV12 out via ISP — matches BDD app libcamerasrc caps (exposure pinned).
        # ROOT CAUSE of the tiling segfault is MEMORY TYPE, not format: libcamerasrc hands
        # DMABUF that hailotilecropper can't map. The format conversion below is ONLY a trick
        # to force a system-memory copy (NV12->NV12 passthrough does not copy and still crashes).
        # We stay in YUV (NV12->I420->NV12) to avoid the RGB color-matrix cost. This copy is a
        # WORKAROUND for the benchmark, NOT a production pattern:
        #   * whole-frame needs NO copy  -> feed hailonet directly (it maps dmabuf NV12 fine).
        #   * tiling in production        -> feed system-memory NV12 from picamera2/appsrc,
        #                                    which never produces dmabuf (zero-copy, no hop).
        src = ("libcamerasrc name=src exposure-time-mode=manual exposure-time=8000")
        src_caps = (
            f"video/x-raw,format=NV12,width={args.width},height={args.height},"
            f"framerate={args.fps}/1 ! videoconvert ! video/x-raw,format=I420 ! "
            f"videoconvert ! video/x-raw,format=NV12"
        )
    elif args.uncapped:
        src_caps = f"video/x-raw,format=NV12,width={args.width},height={args.height}"
        src = f"videotestsrc name=src is-live=false num-buffers={args.num_buffers} pattern=ball"
    else:
        src_caps = (
            f"video/x-raw,format=NV12,width={args.width},height={args.height},"
            f"framerate={args.fps}/1"
        )
        src = "videotestsrc name=src is-live=true pattern=ball"

    # Cropper is a two-pad fork: src_0 = bypass (original full frame) -> agg.sink_0,
    # src_1 = crops -> hailonet -> agg.sink_1. agg stitches them back to one frame.
    def branch(name, nx, ny):
        # multi-scale scale-level S emits a pyramid (1x1)+(2x2)+...+(SxS) of crops
        # through ONE hailonet (single net-group) -> models "+1 whole frame" with no
        # round-robin scheduler tax. preq must hold all crops of one frame.
        if args.multiscale:
            geom = (f"tiling-mode=multi-scale scale-level={args.multiscale} "
                    f"tiles-along-x-axis={args.multiscale} tiles-along-y-axis={args.multiscale}")
            ncrops = sum(i * i for i in range(1, args.multiscale + 1))
        else:
            geom = (f"tiling-mode=single-scale tiles-along-x-axis={nx} "
                    f"tiles-along-y-axis={ny}")
            ncrops = nx * ny
        return (
            f"t. ! queue name={name}_q leaky=downstream max-size-buffers=2 ! "
            f"hailotilecropper name={name}_crop {geom} internal-offset=true "
            f"hailotileaggregator name={name}_agg "
            f"{name}_crop.src_0 ! queue name={name}_bypass leaky=no max-size-buffers={args.maxq} ! {name}_agg.sink_0 "
            f"{name}_crop.src_1 ! queue name={name}_preq leaky=no max-size-buffers={ncrops + 1} ! "
            f"hailonet name={name}_net {common_net} ! "
            f"queue name={name}_postq leaky=no max-size-buffers={args.maxq} ! {name}_agg.sink_1 "
            f"{name}_agg. ! identity name={name}_out ! fakesink name={name}_sink sync=false async=false "
        )

    s = f"{src} ! {src_caps} ! tee name=t " + branch("tile", args.nx, args.ny)
    if not args.single:
        s += branch("full", 1, 1)
    return s


class BranchStat:
    def __init__(self):
        self.t_in = collections.deque()
        self.latencies = []   # ms, cropper-sink -> aggregator-src (inference path)
        self.cap_lat = []     # ms, buffer-PTS(capture) -> aggregator-src (incl. camera Stage-A)
        self.out_count = 0
        self.tile_count = 0
        self.in_count = 0


def attach_probes(pipe, name, stat):
    crop = pipe.get_by_name(f"{name}_crop")
    net = pipe.get_by_name(f"{name}_net")
    agg = pipe.get_by_name(f"{name}_agg")

    def on_crop_in(pad, info):
        stat.in_count += 1
        stat.t_in.append(time.monotonic())
        return Gst.PadProbeReturn.OK

    def on_net_in(pad, info):
        stat.tile_count += 1
        return Gst.PadProbeReturn.OK

    def on_agg_out(pad, info):
        if stat.t_in:
            stat.latencies.append((time.monotonic() - stat.t_in.popleft()) * 1000.0)
        # full capture->here latency via running-time vs buffer PTS (real-camera e2e)
        buf = info.get_buffer()
        clk = pipe.get_pipeline_clock()
        if buf is not None and clk is not None and buf.pts != Gst.CLOCK_TIME_NONE:
            now_rt = clk.get_time() - pipe.get_base_time()
            if now_rt >= buf.pts:
                stat.cap_lat.append((now_rt - buf.pts) / 1e6)
        stat.out_count += 1
        return Gst.PadProbeReturn.OK

    crop.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_crop_in)
    net.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_net_in)
    agg.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_agg_out)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def start_picam(pipe, args):
    """Push picamera2 NV12 frames into appsrc as SYSTEM-MEMORY buffers (no dmabuf).
    Returns a stop() callable. The producer copies the sensor buffer into a Gst buffer
    once (same as the BDD app's new_wrapped_full) — no format conversion, no dmabuf."""
    import threading
    from picamera2 import Picamera2

    appsrc = pipe.get_by_name("app_source")
    picam = Picamera2()
    cfg = picam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "NV12"},
        buffer_count=4,
        controls={"FrameRate": float(args.fps), "ExposureTime": 8000, "AeEnable": False},
    )
    picam.configure(cfg)
    picam.start()
    stop = threading.Event()

    def run():
        while not stop.is_set():
            arr = picam.capture_buffer("main")          # flat NV12 bytes, system memory
            gstbuf = Gst.Buffer.new_wrapped(arr.tobytes())
            if appsrc.emit("push-buffer", gstbuf) != Gst.FlowReturn.OK:
                break
        try:
            appsrc.emit("end-of-stream")
            picam.stop()
            picam.close()
        except Exception:
            pass

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return lambda: (stop.set(), th.join(timeout=2))


def read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="/home/bdd/models/yolov11n_nv12.hef")
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--uncapped", action="store_true")
    ap.add_argument("--single", action="store_true", help="tiling branch only, no whole-frame +1")
    ap.add_argument("--maxq", type=int, default=4, help="bypass/post queue depth (latency knob)")
    ap.add_argument("--multiscale", type=int, default=0,
                    help="scale-level S: ONE cropper emits (1x1)+..+(SxS) crops via one net-group")
    ap.add_argument("--camera", action="store_true", help="use real libcamerasrc NV12 source")
    ap.add_argument("--picam", action="store_true",
                    help="picamera2 -> appsrc system-memory NV12 (zero-copy tiling, no dmabuf)")
    ap.add_argument("--num-buffers", type=int, default=100000)
    args = ap.parse_args()

    Gst.init(None)
    desc = make_pipeline(args)
    pipe = Gst.parse_launch(desc)

    tile, full = BranchStat(), BranchStat()
    attach_probes(pipe, "tile", tile)
    if not args.single:
        attach_probes(pipe, "full", full)

    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()

    errors = []

    def on_msg(_bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            errors.append(f"{err}: {dbg}")
            loop.quit()
        elif t == Gst.MessageType.EOS:
            loop.quit()
        return True

    bus.connect("message", on_msg)

    temp0 = read_temp()
    stop_producer = None
    if args.picam:
        stop_producer = start_picam(pipe, args)
    pipe.set_state(Gst.State.PLAYING)
    # warm up (first inference re-streams weights ~42 ms; let scheduler settle)
    GLib.timeout_add(int(args.duration * 1000) + 1500, lambda: (loop.quit(), False)[1])
    t_start = time.monotonic()
    loop.run()
    if stop_producer:
        stop_producer()
    wall = time.monotonic() - t_start
    temp1 = read_temp()

    if errors:
        print(f"ERROR nx={args.nx} ny={args.ny}: {errors[0][:300]}")
        _hard_teardown(pipe)
        return

    ntiles = args.nx * args.ny

    def row(label, st, tiles_per_frame):
        fps = st.out_count / wall if wall else 0
        return {
            "label": label,
            "fps": fps,
            "tiles_per_frame": tiles_per_frame,
            "inf_s": st.tile_count / wall if wall else 0,
            "lat_p50": pct(st.latencies, 50),
            "lat_p90": pct(st.latencies, 90),
            "lat_p99": pct(st.latencies, 99),
            "in": st.in_count,
            "out": st.out_count,
        }

    tr = row(f"tiling {args.nx}x{args.ny}", tile, ntiles)
    fr = row("whole-frame", full, 1)

    if args.single:
        sys_fps, total_inf = tr["fps"], tr["inf_s"]
        e2e_pipe_p50, e2e_pipe_p90 = tr["lat_p50"], tr["lat_p90"]
    else:
        sys_fps = min(tr["fps"], fr["fps"])
        total_inf = tr["inf_s"] + fr["inf_s"]
        # branch e2e contribution = the slower of the two branches per frame
        e2e_pipe_p50 = max(tr["lat_p50"], fr["lat_p50"])
        e2e_pipe_p90 = max(tr["lat_p90"], fr["lat_p90"])

    print("RESULT " + "|".join(str(x) for x in [
        f"{args.nx}x{args.ny}+1",
        ntiles + 1,
        round(sys_fps, 1),
        round(total_inf, 1),
        round(tr["fps"], 1), round(tr["lat_p50"], 1), round(tr["lat_p90"], 1), round(tr["lat_p99"], 1),
        round(fr["fps"], 1), round(fr["lat_p50"], 1), round(fr["lat_p90"], 1),
        round(e2e_pipe_p50, 1), round(e2e_pipe_p90, 1),
        round(pct(tile.cap_lat, 50), 1), round(pct(tile.cap_lat, 90), 1),  # capture->agg (real e2e)
        round(temp0, 1), round(temp1, 1),
        f"{tr['out']}/{tr['in']}",
    ]))
    import sys as _sys
    _sys.stdout.flush()
    _hard_teardown(pipe)


def _hard_teardown(pipe):
    """Set NULL with a watchdog; hailonet/VDevice release can deadlock mid-inference.
    If NULL doesn't complete in time, force process exit so the kernel frees /dev/hailo0."""
    import os
    import threading
    done = threading.Event()

    def _null():
        try:
            pipe.set_state(Gst.State.NULL)
            pipe.get_state(3 * Gst.SECOND)
        except Exception:
            pass
        done.set()

    threading.Thread(target=_null, daemon=True).start()
    if not done.wait(4.0):
        os._exit(0)   # kernel releases the device fd on exit; next run gets a clean device


if __name__ == "__main__":
    main()
