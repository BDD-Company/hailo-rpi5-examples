#!/usr/bin/env python3
"""Validate the detection<->pursuit runtime switch for the dual-branch tiling design, and measure
the switch latency.

Topology (built once, both branches live, gated by valves):

    src(big frame) ! tee
      tee ! valve(det)     ! hailotilecropper(NxM) + batched hailonet + hailotileaggregator  -> funnel
      tee ! valve(pursuit) ! videoscale->model ! hailonet(batch1)                            -> funnel
    funnel ! probe ! fakesink

Detection branch emits the full-frame (bypass) buffer (width = frame width); pursuit branch emits a
model-sized buffer (width = model size) — so the probe tells the branches apart by caps width. We run
detection, flip the valves at T, and time how long until the first pursuit-branch buffer reaches the
probe (= switch latency, the camera-gap-free cost of changing modes). Also reports per-mode
capture->detection latency + fps before and after the switch. Test source only (no camera needed).
"""

import argparse
import re
import time
from pathlib import Path

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo

DEFAULT_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"


def default_hef():
    bdd_sh = Path(__file__).resolve().parent.parent / "scripts" / "bdd.sh"
    for line in bdd_sh.read_text().splitlines():
        m = re.match(r'\s*export\s+HAILO_MODEL="([^"]+)"', line)
        if m:
            return m.group(1)
    return ""


def build(hef, so, width, height, fps, tiles_x, tiles_y, overlap, model, batch):
    nms = "nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true"
    return (
        f"videotestsrc is-live=true ! video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 ! "
        f"tee name=t "
        # detection branch (tiling, batched)
        f"t. ! queue ! valve name=det_valve drop=false ! "
        f"hailotilecropper internal-offset=true name=dc tiles-along-x-axis={tiles_x} tiles-along-y-axis={tiles_y} "
        f"overlap-x-axis={overlap} overlap-y-axis={overlap} "
        f"hailotileaggregator name=da flatten-detections=true "
        f"dc. ! queue leaky=no max-size-buffers=4 ! da.sink_0 "
        f"dc. ! queue leaky=no max-size-buffers={max(8, batch*2)} ! videoconvert ! "
        f"hailonet hef-path={hef} batch-size={batch} vdevice-group-id=1 {nms} ! "
        f"queue leaky=no max-size-buffers=8 ! hailofilter so-path={so} qos=false ! queue leaky=no ! da.sink_1 "
        f"da. ! queue ! f. "
        # pursuit branch (whole frame, batch-1)
        f"t. ! queue ! valve name=pursuit_valve drop=true ! "
        f"videoscale ! video/x-raw,width={model},height={model} ! videoconvert ! "
        f"hailonet hef-path={hef} batch-size=1 vdevice-group-id=1 {nms} ! "
        f"queue leaky=no max-size-buffers=8 ! hailofilter so-path={so} qos=false ! queue leaky=no ! f. "
        # merge + probe
        f"funnel name=f ! queue ! identity name=probe ! fakesink sync=false async=false"
    )


class Switch:
    def __init__(self, a):
        self.a = a
        self.det_lat, self.pur_lat = [], []
        self.det_n = self.pur_n = 0
        self.flip_mono = None
        self.switch_latency_ms = None
        self.phase = "detection"

    def on_probe(self, pad, info, _u):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        caps = pad.get_current_caps()
        w = 0
        if caps and caps.get_size():
            ok, w = caps.get_structure(0).get_int("width")
            if not ok:
                w = 0
        is_pursuit = (w == self.a.model)  # pursuit branch emits model-sized buffers
        # switch latency: first pursuit buffer after the flip
        if self.flip_mono is not None and is_pursuit and self.switch_latency_ms is None:
            self.switch_latency_ms = (time.monotonic() - self.flip_mono) * 1000.0
            self.phase = "pursuit"
        # per-mode pipeline latency from buffer age
        if buf.pts != Gst.CLOCK_TIME_NONE:
            elem = pad.get_parent_element()
            clk = elem.get_clock()
            if clk is not None:
                now_rt = clk.get_time() - elem.get_base_time()
                if now_rt != Gst.CLOCK_TIME_NONE and now_rt >= buf.pts:
                    age = (now_rt - buf.pts) / 1e6
                    if is_pursuit:
                        self.pur_lat.append(age); self.pur_n += 1
                    else:
                        self.det_lat.append(age); self.det_n += 1
        return Gst.PadProbeReturn.OK

    def run(self):
        a = self.a
        Gst.init(None)
        pstr = build(a.hef, a.so, a.width, a.height, a.fps, a.tiles_x, a.tiles_y, a.overlap, a.model, a.batch)
        pipe = Gst.parse_launch(pstr)
        pipe.get_by_name("probe").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self.on_probe, None)
        det_valve = pipe.get_by_name("det_valve")
        pursuit_valve = pipe.get_by_name("pursuit_valve")
        loop = GLib.MainLoop()
        bus = pipe.get_bus(); bus.add_signal_watch()

        def on_msg(_b, m):
            if m.type == Gst.MessageType.ERROR:
                err, dbg = m.parse_error(); print(f"ERROR: {err} :: {dbg}"); loop.quit()
            elif m.type == Gst.MessageType.EOS:
                loop.quit()
            return True
        bus.connect("message", on_msg)

        def do_switch():
            # The flip itself: close detection, open pursuit. This is the whole mode change.
            t = time.monotonic()
            pursuit_valve.set_property("drop", False)
            det_valve.set_property("drop", True)
            self.flip_mono = t
            print(f"# --- FLIP at t={a.detection_s}s: det->pursuit (det_n={self.det_n} so far) ---")
            return False

        GLib.timeout_add_seconds(a.detection_s, do_switch)
        GLib.timeout_add_seconds(a.detection_s + a.pursuit_s, loop.quit)
        pipe.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        finally:
            pipe.set_state(Gst.State.NULL)

        def stats(xs):
            xs = xs[5:] or xs
            if not xs:
                return "n/a"
            s = sorted(xs)
            return f"mean={sum(s)/len(s):.1f} p50={s[len(s)//2]:.1f} p95={s[min(len(s)-1,int(0.95*len(s)))]:.1f} n={len(s)}"
        print(f"# DETECTION ({a.tiles_x}x{a.tiles_y} batch{a.batch} @ {a.width}x{a.height}) latency ms: {stats(self.det_lat)}  frames={self.det_n}")
        print(f"# PURSUIT   (whole-frame batch1 -> {a.model}) latency ms: {stats(self.pur_lat)}  frames={self.pur_n}")
        print(f"# SWITCH LATENCY (flip -> first pursuit result): "
              f"{self.switch_latency_ms:.1f} ms" if self.switch_latency_ms is not None else "# SWITCH LATENCY: NOT OBSERVED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1332)
    ap.add_argument("--height", type=int, default=990)
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--overlap", type=float, default=0.1)
    ap.add_argument("--model", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--detection-s", type=int, default=6)
    ap.add_argument("--pursuit-s", type=int, default=6)
    ap.add_argument("--hef", default=default_hef())
    ap.add_argument("--so", default=DEFAULT_SO)
    a = ap.parse_args()
    a.tiles_x, a.tiles_y = (int(x) for x in a.tiles.split("x"))
    Switch(a).run()


if __name__ == "__main__":
    main()
