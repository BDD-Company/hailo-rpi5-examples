"""Definitive offline harness: re-run the real BYTETracker over captured RAW
detections, replay the controller lock, count/classify target switches, and
sweep params. Validated against the on-box logged track_ids.

Data source: '!!! RAWDETS n=K [(x1,y1,x2,y2,conf), ...]' lines (all detections,
pre-tracker) added on feat/bytetrack-locking-eval.
"""
import sys, re, math, gzip
from collections import defaultdict
import numpy as np

import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "BDD"))
from bytetrack import BYTETracker, STrack

RAW_RE = re.compile(r"frame=#(\d+) !!! RAWDETS n=(\d+) \[(.*)\]")
TUP_RE = re.compile(r"\(([-\d.]+), ([-\d.]+), ([-\d.]+), ([-\d.]+), ([-\d.]+)\)")
# on-box logged ids for validation
OUT_RE = re.compile(r"frame=#(\d+) ByteTracker output: \d+ active tracks \[(.*)\]")
TRK_RE = re.compile(r"\((\d+), <TrackState")

CONFIDENCE_MIN = 0.4
CLEAR_AFTER = 3
ONBOX = dict(track_thresh=0.3, det_thresh=0.35, match_thresh=0.3, track_buffer=30,
             frame_rate=30.0, match_max_dist=0.2, recovery_max_dist=None,
             nms_thresh=0.3, nms_dist_thresh=0.06)

def parse_raw(path):
    frames = []
    onbox_ids = {}
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rb') as f:
        for raw in f:
            line = raw.decode('utf-8', 'replace')
            if 'RAWDETS' in line:
                m = RAW_RE.search(line)
                if not m:
                    continue
                fid = int(m.group(1))
                dets = [tuple(map(float, t)) for t in TUP_RE.findall(m.group(3))]
                frames.append((fid, dets))
            elif 'ByteTracker output' in line:
                m = OUT_RE.search(line)
                if m:
                    onbox_ids[int(m.group(1))] = sorted(int(t) for t in TRK_RE.findall(m.group(2)))
    frames.sort(key=lambda x: x[0])
    return frames, onbox_ids

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def run_tracker(frames, params):
    """Return per-frame [(track_id|None, conf, cx, cy, w, h)] reproducing app.py:
    run BYTETracker, then map each active track to the raw det by best IoU."""
    STrack.reset_counter()
    tk = BYTETracker(**params)
    out = []
    track_life = defaultdict(list)   # tid -> [fids]
    for fid, dets in frames:
        if dets:
            arr = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in dets])
        else:
            arr = np.empty((0, 5))
        tracks = tk.update(arr, fid)
        # map track -> raw det index by IoU (>= 0.3, like app._MIN_MATCH_IOU default)
        tid_by_det = {}
        for tr in tracks:
            best_i, best_iou = None, 0.30
            for i, d in enumerate(dets):
                iou = _iou(tr.det_bbox, (d[0], d[1], d[2], d[3]))
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i is not None:
                tid_by_det[best_i] = tr.track_id
                track_life[tr.track_id].append(fid)
        fr = []
        for i, d in enumerate(dets):
            cx, cy = (d[0]+d[2])/2, (d[1]+d[3])/2
            w, h = d[2]-d[0], d[3]-d[1]
            fr.append((tid_by_det.get(i), d[4], cx, cy, w, h))
        out.append((fid, fr))
    return out, track_life

def replay_lock(frames):
    locked = None; last_seen = None; have_history = False
    lpf = {}; switches = []
    for fid, dets in frames:
        pool = [d for d in dets if d[1] >= CONFIDENCE_MIN]
        picked = None
        if pool:
            if locked is not None:
                cand = [d for d in pool if d[0] == locked]
                picked = max(cand, key=lambda d: d[1]) if cand else None
            else:
                picked = max(pool, key=lambda d: d[1])
        if picked is not None:
            prev = locked
            if picked[0] is not None:
                locked = picked[0]
            last_seen = fid; have_history = True
            if prev is not None and locked is not None and locked != prev:
                switches.append((fid, prev, locked))
        else:
            if last_seen is not None and (fid - last_seen) > CLEAR_AFTER and have_history:
                locked = None; have_history = False
        lpf[fid] = locked
    return lpf, switches

def analyze(frames_raw, params, label, validate_ids=None, verbose=True):
    tframes, life = run_tracker(frames_raw, params)
    nfr = len(tframes)
    fids = [f[0] for f in tframes]
    span = max(fids)-min(fids)+1
    lpf, switches = replay_lock(tframes)
    # id validation vs on-box
    valid = ""
    if validate_ids is not None:
        match = tot = 0
        derived = {}
        for fid, fr in tframes:
            derived[fid] = sorted(t for (t, *_ ) in fr if t is not None)
        for fid, ob in validate_ids.items():
            if fid in derived:
                tot += 1
                if derived[fid] == ob:
                    match += 1
        valid = f"  [id-repro vs on-box: {match}/{tot} = {match/tot*100:.1f}%]" if tot else ""
    # true ball = track with max coverage (always-present ball dominates)
    true_id = max(life, key=lambda t: len(life[t])) if life else None
    true_cov = len(life[true_id])/nfr if true_id is not None else 0
    # persistence threshold: "true-ish" tracks live in a large fraction; noise transient
    lock_frames = sum(1 for f in fids if lpf[f] is not None)
    # locked-id segments
    seq = []
    for fid, _ in tframes:
        lid = lpf[fid]
        if not seq or seq[-1][0] != lid:
            seq.append([lid, fid, fid])
        else:
            seq[-1][2] = fid
    locked_ids = sorted({s[0] for s in seq if s[0] is not None}, key=lambda t: -len(life.get(t, [])))
    # Reacquisition transitions: consecutive DISTINCT non-None locked ids (the lock
    # moved from one id to another, whether directly or through a None loss gap).
    nonnull = [(lid, a, b) for lid, a, b in seq if lid is not None]
    PERSIST = nfr * 0.10   # lifespan (frames present) above which an id is the true ball
    reacq = []             # (lost_frame, reacq_frame, idA, idB, gap, A_true, B_true)
    for (idA, aA, bA), (idB, aB, bB) in zip(nonnull, nonnull[1:]):
        if idA == idB:
            continue
        reacq.append((bA, aB, idA, idB, aB-bA,
                      len(life.get(idA, [])) > PERSIST, len(life.get(idB, [])) > PERSIST))
    n_false = sum(1 for *_a, At, Bt in reacq if At and not Bt)      # true -> noise (BAD)
    n_frag  = sum(1 for *_a, At, Bt in reacq if At and Bt)          # true -> true (same ball re-id)
    n_recov = sum(1 for *_a, At, Bt in reacq if (not At) and Bt)    # noise -> true (benign)
    n_n2n   = sum(1 for *_a, At, Bt in reacq if (not At) and not Bt)

    if verbose:
        print(f"===== {label} ====={valid}")
        print(f"frames={nfr} span={span} (~{span/30:.0f}s)  distinct track ids={len(life)}")
        det_hist = defaultdict(int)
        for _, d in frames_raw:
            det_hist[len(d)] += 1
        print(f"raw dets/frame: " + ", ".join(f"{k}:{v}" for k,v in sorted(det_hist.items())))
        print(f"true ball track id = {true_id} (coverage {true_cov*100:.1f}% of frames, {len(life[true_id])} fr)")
        print(f"locked {lock_frames}/{nfr} ({lock_frames/nfr*100:.1f}%)")
        # lifespans of every id that was EVER locked -> shows true(persistent) vs noise(transient)
        print("ids that were ever LOCKED, with lifespan (frames present / span):")
        for lid in locked_ids:
            fl = life.get(lid, [])
            sp = (max(fl)-min(fl)+1) if fl else 0
            tag = "TRUE-ball (persistent)" if len(fl) > nfr*0.10 else "NOISE (transient)"
            print(f"    id={lid}: present {len(fl)} fr, span {sp} fr  -> {tag}")
        # reacquisition transitions (the real switch events; most go through a None gap)
        print(f"\nTOTAL lock switches (id A->B, direct or via loss gap): {len(reacq)}")
        print(f"    FALSE true->noise (locked a transient noise ball) : {n_false}")
        print(f"    true->true  (same true ball, ByteTrack re-id'd)   : {n_frag}")
        print(f"    noise->true (benign recovery/acquisition)         : {n_recov}")
        print(f"    noise->noise                                      : {n_n2n}")
        for lostF, reacqF, a, b, gap, At, Bt in reacq:
            la, lb = len(life.get(a, [])), len(life.get(b, []))
            kind = ("FALSE true->NOISE" if (At and not Bt) else
                    "frag same-true" if (At and Bt) else
                    "recover noise->true" if (not At and Bt) else "noise->noise")
            print(f"    lost@{lostF} reacq@{reacqF} gap={gap:>3}  {a}(life {la}) -> {b}(life {lb})   {kind}")
    return dict(nfr=nfr, span=span, nids=len(life), true_id=true_id, true_cov=true_cov,
                lock_frames=lock_frames, life=life, reacq=reacq,
                n_switch=len(reacq), n_false=n_false, n_frag=n_frag,
                n_recov=n_recov, n_n2n=n_n2n,
                false_rate_min=n_false/(span/30/60) if span else 0)

if __name__ == '__main__':
    path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path
    frames_raw, onbox = parse_raw(path)
    analyze(frames_raw, ONBOX, label, validate_ids=onbox)
