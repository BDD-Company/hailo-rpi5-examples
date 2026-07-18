"""Per-stage jitter decomposition — where is the aim 'jitter' actually born?

Companion to 2026-07-12-target-estimate-noise-reduction-design.md (see §3.1.1).
Computes RMS 2nd-difference (the design's jitter metric) at each stage on a flight
log, to separate *detection noise* from *platform motion* and *dropout*:

  - detection bbox center (cx,cy)  -> bearing-noise source (× FOV -> degrees)
    reported BOTH naively and WITHIN contiguous-frame tracks (the latter excludes
    dropout/reacquisition jumps, giving true frame-to-frame detector noise)
  - detection bbox size (w,h)       -> range-noise source (§1.2)
  - platform attitude (roll,pitch)  -> real airframe motion, from telemetry
  - commanded aim (roll,pitch)      -> from a controller replay

2026-07-18 result on the UAE corpus: detector bbox is stable (~1.6° bearing on the
dive), while the aim buzzes 26.7° — dominated by platform roll oscillation (17.7°)
and detection dropout (target seen 41% of frames). The design's "aim buzz = bbox
shake" premise does NOT hold on this corpus. See the design doc §3.1.1.

Run:  cd BDD && PYTHONPATH=. python experiments/stage_jitter.py [LOG ...]
Default logs: the UAE dive + cruise.
"""
import sys

import debug_drone_controller as ddc
import aim_metrics as am

FOV_H, FOV_V = 107.0, 85.0  # config.yaml camera frame_angular_size_deg
_UAE = "/media/enmk/BDD/_BACKUPS/UAE/2026-04/2026-04-27/_DEBUG_13/BDD_20260427-%s.log"
DEFAULT_LOGS = [_UAE % "160029", _UAE % "155753"]

jit = am.rms_second_difference


def analyze(log):
    _cfg, frames, base = ddc.parse_log(log)
    ids = sorted(frames)

    cx, cy = [], []
    att_roll_all, att_pitch_all, att_roll_det, att_pitch_det = [], [], [], []
    for fid in ids:
        fd = frames[fid]
        ae = (fd.get("telemetry", {}) or {}).get("attitude_euler", {}) or {}
        has_att = "roll_deg" in ae
        if has_att:
            att_roll_all.append(ae["roll_deg"]); att_pitch_all.append(ae["pitch_deg"])
        dets = fd.get("detections", [])
        if not dets:
            continue
        b = dets[0].bbox
        cx.append(b.left_edge + b.width / 2); cy.append(b.top_edge + b.height / 2)
        if has_att:
            att_roll_det.append(ae["roll_deg"]); att_pitch_det.append(ae["pitch_deg"])

    # detection noise WITHIN contiguous frame-id runs (excludes dropout jumps)
    det_ids = [fid for fid in ids if frames[fid].get("detections")]
    runs, cur = [], []
    for fid in det_ids:
        if cur and fid == cur[-1] + 1:
            cur.append(fid)
        else:
            if len(cur) >= 3:
                runs.append(cur)
            cur = [fid]
    if len(cur) >= 3:
        runs.append(cur)

    def run_jit(get):
        num = den = 0.0
        for run in runs:
            j = jit([get(frames[f]["detections"][0].bbox) for f in run])
            if j == j:  # not nan
                num += j * len(run); den += len(run)
        return (num / den) if den else float("nan")

    cxj = run_jit(lambda b: b.left_edge + b.width / 2)
    cyj = run_jit(lambda b: b.top_edge + b.height / 2)
    wj  = run_jit(lambda b: b.width)
    hj  = run_jit(lambda b: b.height)

    print(f"\n===== {log.split('/')[-1]} =====")
    print(f"frames={len(ids)}  detected={len(cx)} ({100*len(cx)/max(len(ids),1):.0f}%)  "
          f"contiguous_runs(>=3)={len(runs)}")
    print("-- detection bbox jitter WITHIN contiguous tracks (true detector noise) --")
    print(f"  center x {cxj:.5f} = {cxj*FOV_H:6.3f} deg   y {cyj:.5f} = {cyj*FOV_V:6.3f} deg")
    print(f"  size w {wj:.5f}   h {hj:.5f}")
    print("-- detection bbox center jitter, naive (dropout-inflated) --")
    print(f"  x {jit(cx)*FOV_H:6.3f} deg   y {jit(cy)*FOV_V:6.3f} deg")
    print("-- platform attitude jitter (telemetry, deg) --")
    print(f"  roll {jit(att_roll_all):6.3f} (all) {jit(att_roll_det):6.3f} (detected)   "
          f"pitch {jit(att_pitch_all):6.3f} (all) {jit(att_pitch_det):6.3f} (detected)")

    # commanded aim via a controller replay (blank frames)
    import logging
    from debug_drone_controller import (ReplayQueue, run_replay, build_detections_list,
                                         VideoReader, load_replay_config)
    logging.disable(logging.CRITICAL)
    config = load_replay_config(ddc.REPLAY_CONFIG_DEFAULT, None)
    q = ReplayQueue(build_detections_list(frames, VideoReader(None)), auto_advance=True)
    drone = run_replay(config, q, frames, base)
    logging.disable(logging.NOTSET)
    jb = am.jitter_by_axis(am.aim_series_from_commands(drone.commands))
    print(f"-- commanded aim jitter (replay, deg) --  roll {jb['roll']:6.3f}   pitch {jb['pitch']:6.3f}")


if __name__ == "__main__":
    for lg in (sys.argv[1:] or DEFAULT_LOGS):
        analyze(lg)
