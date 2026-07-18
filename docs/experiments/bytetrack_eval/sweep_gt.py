"""Param sweep against exact ground truth (see groundtruth.py).

Supersedes sweep.py, which ranked params using the lifespan heuristic on camera
runs. Here the true ball is known per frame, so "did the lock stay on the true
ball" is measured, not inferred.

    python3 sweep_gt.py <clean.log.gz> <noisy.log.gz> [nb_frames]
"""
import sys

from harness import parse_raw, ONBOX
from groundtruth import truth_from_clean, analyze_labeled, find_shift

# FAST-target sweep (2026-07-18). A fast ball outruns its Kalman prediction, so the
# levers that bite are (a) nms_dist_thresh (don't let a bigger noise ball delete the
# small true one), (b) recovery_max_dist (Stage-1.5 re-acquire by last raw position
# when prediction error > match_max_dist), (c) match_max_dist (association gate vs
# large per-frame jumps), (d) track_buffer (keep the true id alive across blackouts).
# ONBOX baseline = main config (nms_dist=0.06, mmd=0.2, rec=None, buf=30).
CASES = [
    ("baseline main (nms.06 mmd.2 buf30)", dict(), True),
    ("lock OFF baseline",                  dict(), False),
    # --- nms_dist_thresh: confirm the 0.06->0.02 fix on the FAST clip ---
    ("nms_dist=0.04",                      dict(nms_dist_thresh=0.04), True),
    ("nms_dist=0.02",                      dict(nms_dist_thresh=0.02), True),
    ("nms_dist=0.015",                     dict(nms_dist_thresh=0.015), True),
    # --- track_buffer (bridge blackouts) on top of the nms fix ---
    ("nms.02 buf60",                       dict(nms_dist_thresh=0.02, track_buffer=60), True),
    ("nms.02 buf90",                       dict(nms_dist_thresh=0.02, track_buffer=90), True),
    # --- recovery_max_dist: re-acquire by last raw position (fast-motion knob) ---
    ("nms.02 rec0.05",                     dict(nms_dist_thresh=0.02, recovery_max_dist=0.05), True),
    ("nms.02 rec0.10",                     dict(nms_dist_thresh=0.02, recovery_max_dist=0.10), True),
    ("nms.02 rec0.15",                     dict(nms_dist_thresh=0.02, recovery_max_dist=0.15), True),
    ("nms.02 buf90 rec0.10",               dict(nms_dist_thresh=0.02, track_buffer=90, recovery_max_dist=0.10), True),
    # --- match_max_dist: wider association gate for large per-frame jumps ---
    ("nms.02 mmd0.15",                     dict(nms_dist_thresh=0.02, match_max_dist=0.15), True),
    ("nms.02 mmd0.30",                     dict(nms_dist_thresh=0.02, match_max_dist=0.30), True),
    ("nms.02 buf90 rec0.10 mmd0.30",       dict(nms_dist_thresh=0.02, track_buffer=90, recovery_max_dist=0.10, match_max_dist=0.30), True),
    # --- confirm the other knobs stay inert (don't tune blind) ---
    ("nms.02 track_thr0.5",                dict(nms_dist_thresh=0.02, track_thresh=0.5), True),
    ("nms.02 match_thr0.5",                dict(nms_dist_thresh=0.02, match_thresh=0.5), True),
]

if __name__ == '__main__':
    clean_p, noisy_p = sys.argv[1], sys.argv[2]
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
    clean_frames, _ = parse_raw(clean_p)
    noisy_frames, _ = parse_raw(noisy_p)
    truth, _ = truth_from_clean(clean_frames, nb)
    shift, med = find_shift(truth, noisy_frames, nb)
    print(f"truth for {len(truth)}/{nb} frames; clean/noisy shift={shift:+d} "
          f"(median nearest-det dist {med:.4f})\n")

    hdr = (f"{'case':<38}{'on_true%':>9}{'on_noise%':>10}{'lost%':>7}"
           f"{'FALSE':>6}{'frag':>5}{'switch':>7}{'relockMx':>9}{'>15fr':>6}{'ids':>6}")
    print(hdr); print("-" * len(hdr))
    for label, over, use_lock in CASES:
        p = dict(ONBOX); p.update(over)
        r = analyze_labeled(noisy_frames, truth, p, label, nb_frames=nb,
                            use_lock=use_lock, verbose=False, shift=shift)
        lost = 100 - r['true_pct'] - r['noise_pct']
        print(f"{label:<38}{r['true_pct']:>9.1f}{r['noise_pct']:>10.1f}{lost:>7.1f}"
              f"{r['n_false']:>6}{r['n_frag']:>5}{r['n_switch']:>7}"
              f"{r['max_relock']:>9}{r['n_slow_relock']:>6}{r['nids']:>6}")
