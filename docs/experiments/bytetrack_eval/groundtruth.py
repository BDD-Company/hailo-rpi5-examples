"""Exact per-frame ground truth for the ByteTrack locking eval.

Replaces the track-lifespan persistence heuristic used for the camera runs. The
clean and noisy clips are the same source frames at the same frame ids (see
file-input-handoff-2026-07-15.md), so the clean run's detection at frame id F IS
the true ball's position at frame id F. A detection in the noisy run is TRUE if
it sits on that position, NOISE otherwise -- no guessing from lifespans.

FIRST PASS ONLY: buffer.offset keeps counting across the loop rewind, so ids
above the clip's frame count are a second pass over the same footage and would
be aligned against the wrong truth. Everything here filters to id < nb_frames.

    python3 groundtruth.py <clean.log.gz> <noisy.log.gz> [nb_frames]
"""
import sys
from collections import defaultdict

from harness import parse_raw, run_tracker, replay_lock, ONBOX, CONFIDENCE_MIN, _iou

# A noisy det is the true ball if its centre is within this much of the clean
# ball's centre, in normalised frame units. The ball spans ~10-200px of a 1280px
# frame (0.008-0.16 wide), and it moves fast, so a fixed radius either misses the
# big ball or swallows noise next to the small one: scale with the ball, floor at
# 3% of frame for the smallest.
def _tol(w, h):
    return max(0.03, 0.75 * max(w, h))


def truth_from_clean(clean_frames, nb_frames=None):
    """{fid: (cx, cy, w, h)} for the true ball, from the noise-free clip."""
    truth = {}
    multi = 0
    for fid, dets in clean_frames:
        if nb_frames is not None and fid >= nb_frames:
            continue
        if not dets:
            continue
        if len(dets) > 1:
            multi += 1
        # clean clip has exactly one ball; if the detector fires twice, trust conf
        d = max(dets, key=lambda d: d[4])
        truth[fid] = ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2, d[2] - d[0], d[3] - d[1])
    return truth, multi


def label_det(truth, fid, cx, cy):
    """True/False/None -- None when the clean run has no truth for this frame."""
    t = truth.get(fid)
    if t is None:
        return None
    tx, ty, tw, th = t
    return ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 <= _tol(tw, th)


def find_shift(truth, noisy_frames, nb_frames, span=3):
    """Frame offset between the clean and noisy runs, or None if they don't align.

    Do not assume the two runs are frame-aligned just because both clips were cut
    from the same timeline: measured, the noisy run sat +1 frame off the clean one
    (median nearest-detection distance 0.0037 at shift 0 vs exactly 0.0000 at +1 --
    identical detections, so it really is the same source frame). The ball moves
    ~0.004/frame, so a small offset hides inside any sane tolerance and quietly
    biases every label instead of failing loudly. Pick the shift that actually
    lines the ball up.
    """
    best = None
    for s in range(-span, span + 1):
        ds = []
        for fid, dets in noisy_frames:
            if fid >= nb_frames or not dets:
                continue
            t = truth.get(fid + s)
            if not t:
                continue
            ds.append(min((((d[0] + d[2]) / 2 - t[0]) ** 2 +
                           ((d[1] + d[3]) / 2 - t[1]) ** 2) ** 0.5 for d in dets))
        if not ds:
            continue
        ds.sort()
        med = ds[len(ds) // 2]
        if best is None or med < best[1]:
            best = (s, med)
    return best


def replay_nolock(tframes):
    """target_lock: false -- every frame independently takes the highest-confidence
    detection. Baseline for what the lock is worth."""
    lpf = {}
    for fid, dets in tframes:
        pool = [d for d in dets if d[1] >= CONFIDENCE_MIN]
        lpf[fid] = max(pool, key=lambda d: d[1])[0] if pool else None
    return lpf


def analyze_labeled(noisy_frames, truth, params, label, nb_frames=None,
                    use_lock=True, verbose=True, shift=0):
    frames = [(f, d) for f, d in noisy_frames
              if nb_frames is None or f < nb_frames]
    tframes, life = run_tracker(frames, params)
    lpf = replay_lock(tframes)[0] if use_lock else replay_nolock(tframes)
    nfr = len(tframes)
    truth = {f - shift: t for f, t in truth.items()} if shift else truth

    # Label every detection, then a track is TRUE if most of the detections it
    # was matched to were the true ball. A track that drifts off the ball onto a
    # noise blob is judged by where it spent its life, which is what "the lock is
    # on the true ball" has to mean.
    votes = defaultdict(lambda: [0, 0])   # tid -> [n_true, n_noise]
    n_true_det = n_noise_det = n_unlabelled = 0
    for fid, fr in tframes:
        for tid, conf, cx, cy, w, h in fr:
            lab = label_det(truth, fid, cx, cy)
            if lab is None:
                n_unlabelled += 1
            elif lab:
                n_true_det += 1
            else:
                n_noise_det += 1
            if tid is not None and lab is not None:
                votes[tid][0 if lab else 1] += 1
    is_true = {t: v[0] > v[1] for t, v in votes.items()}

    # Frames where the lock was actually on the true ball
    on_true = on_noise = 0
    for fid, fr in tframes:
        lid = lpf[fid]
        if lid is None:
            continue
        if is_true.get(lid, False):
            on_true += 1
        else:
            on_noise += 1

    # Lock segments -> switches, classified by exact labels
    seq = []
    for fid, _ in tframes:
        lid = lpf[fid]
        if not seq or seq[-1][0] != lid:
            seq.append([lid, fid, fid])
        else:
            seq[-1][2] = fid
    nonnull = [(l, a, b) for l, a, b in seq if l is not None]
    reacq = []
    for (idA, _aA, bA), (idB, aB, _bB) in zip(nonnull, nonnull[1:]):
        if idA != idB:
            reacq.append((bA, aB, idA, idB, aB - bA,
                          is_true.get(idA, False), is_true.get(idB, False)))
    n_false = sum(1 for *_x, At, Bt in reacq if At and not Bt)
    n_frag = sum(1 for *_x, At, Bt in reacq if At and Bt)
    n_recov = sum(1 for *_x, At, Bt in reacq if not At and Bt)
    n_n2n = sum(1 for *_x, At, Bt in reacq if not At and not Bt)
    dur_min = nfr / 30 / 60

    # Re-lock latency: frames the lock spends OFF the true ball (on noise OR lost)
    # in a gap BETWEEN two on-true periods -- i.e. how long the tracker took to get
    # back onto the true ball after leaving it. The pre-first-lock acquisition run
    # and a trailing off-true run at the very end (never re-locked) are excluded from
    # the gap list; the tail is reported separately.
    ontrue_flag = [(lpf[fid] is not None and is_true.get(lpf[fid], False))
                   for fid, _ in tframes]
    relock = []                 # interior off-true gap lengths (frames)
    seen_true = False; gap = 0
    for flag in ontrue_flag:
        if flag:
            if seen_true and gap > 0:
                relock.append(gap)
            seen_true = True; gap = 0
        elif seen_true:
            gap += 1
    tail_gap = gap if seen_true else 0     # off-true at end that never re-locked
    max_relock = max(relock) if relock else 0
    med_relock = sorted(relock)[len(relock) // 2] if relock else 0
    n_slow_relock = sum(1 for g in relock if g > 15)   # gaps over the 15-frame target

    if verbose:
        print(f"===== {label} =====")
        print(f"frames={nfr}  ids={len(life)}  dets: true={n_true_det} "
              f"noise={n_noise_det} unlabelled(no truth)={n_unlabelled}")
        print(f"lock ON TRUE ball : {on_true}/{nfr} ({on_true/nfr*100:.1f}%)")
        print(f"lock ON NOISE     : {on_noise}/{nfr} ({on_noise/nfr*100:.1f}%)")
        print(f"lock lost/none    : {nfr-on_true-on_noise}/{nfr} "
              f"({(nfr-on_true-on_noise)/nfr*100:.1f}%)")
        print(f"switches total={len(reacq)}  FALSE true->noise={n_false} "
              f"({n_false/dur_min:.1f}/min)  frag true->true={n_frag}  "
              f"noise->true={n_recov}  noise->noise={n_n2n}")
        print(f"re-lock latency (frames off-true between on-true periods): "
              f"n={len(relock)} max={max_relock} median={med_relock} "
              f">15fr={n_slow_relock}"
              + (f"  [tail never re-locked: {tail_gap} fr]" if tail_gap else ""))
        for lostF, reF, a, b, gap, At, Bt in reacq:
            kind = ("FALSE true->NOISE" if (At and not Bt) else
                    "frag same-true" if (At and Bt) else
                    "recover noise->true" if (not At and Bt) else "noise->noise")
            print(f"    lost@{lostF} reacq@{reF} gap={gap:>3}  {a}->{b}   {kind}")
    return dict(nfr=nfr, on_true=on_true, on_noise=on_noise,
                true_pct=on_true / nfr * 100 if nfr else 0,
                noise_pct=on_noise / nfr * 100 if nfr else 0,
                n_switch=len(reacq), n_false=n_false, n_frag=n_frag,
                n_recov=n_recov, n_n2n=n_n2n, nids=len(life),
                n_true_det=n_true_det, n_noise_det=n_noise_det,
                max_relock=max_relock, med_relock=med_relock,
                n_slow_relock=n_slow_relock, tail_gap=tail_gap)


if __name__ == '__main__':
    clean_p, noisy_p = sys.argv[1], sys.argv[2]
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
    clean_frames, _ = parse_raw(clean_p)
    noisy_frames, _ = parse_raw(noisy_p)
    truth, multi = truth_from_clean(clean_frames, nb)
    first = [f for f, _ in clean_frames if f < nb]
    print(f"clean: {len(first)} frames <{nb}, truth for {len(truth)} "
          f"({len(truth)/nb*100:.1f}% of clip), {multi} multi-det frames")
    shift, med = find_shift(truth, noisy_frames, nb)
    print(f"clean/noisy alignment: shift={shift:+d} (median nearest-det dist {med:.4f})")
    analyze_labeled(noisy_frames, truth, ONBOX,
                    f"noisy @ track_buffer={ONBOX['track_buffer']}",
                    nb_frames=nb, shift=shift)
