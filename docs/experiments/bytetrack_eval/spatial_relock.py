"""Prototype + sweep the spatial re-lock controller fix against exact ground truth.

The failure (see bytetrack-fast-target-2026-07-18.md): the true ball is detected in
98% of frames, but when its ByteTrack id fragments (fast motion + blackouts), the
controller clears the lock after CLEAR_AFTER frames and re-acquires by GLOBAL highest
confidence -- which is a distractor ball scoring as high (0.81) as the true ball. So
the lock sits on noise 25% of the time while the true ball is right there.

Fix, config can't reach it because it's controller logic: when the locked track is
absent, re-acquire the detection nearest the motion-PREDICTED last-known position,
within a radius, instead of the global-highest-confidence one. This both stops the
noise-grab and re-locks immediately (no CLEAR_AFTER wait).

replay_lock_spatial mirrors what _pick_target_detection + its caller will do on-box:
the function picks one detection; the caller adopts its track_id as the new lock and
feeds back the last picked position/velocity. Sweep here, then port the winner.

    python3 spatial_relock.py <clean.log.gz> <noisy.log.gz> [nb_frames]
"""
import sys

from harness import parse_raw, run_tracker, ONBOX, CONFIDENCE_MIN, CLEAR_AFTER
from groundtruth import truth_from_clean, find_shift, analyze_labeled


def replay_lock_spatial(tframes, radius=0.10, use_vel=True, fallback=True,
                        max_gap=6):
    """Controller lock replay: today's logic + a SPATIAL GATE on re-acquisition.

    Identical to harness.replay_lock (follow the locked id; hold through a
    CLEAR_AFTER grace; then clear and re-pick) EXCEPT the re-pick rule. Instead of
    the globally highest-confidence detection -- which in dense noise is a distractor
    scoring as high as the true ball -- prefer the highest-confidence detection NEAR
    the motion-predicted last-known position.

    radius:   max normalised distance from the predicted position a re-acquired
              detection may sit at.
    use_vel:  extrapolate the last position by the per-frame velocity estimated
              while the target was followed (the ball is fast, so where it WAS is a
              poor seed; where it is HEADED is much better).
    fallback: if no detection is near the prediction, fall back to the global
              highest-confidence pick (today's behaviour) so the lock still
              re-acquires rather than hunting a stale position forever. If False,
              stay lost until something appears near the prediction.
    max_gap:  cap the gap used for velocity extrapolation, so a long loss doesn't
              fling the predicted point off-frame.

    Returns (lpf, switches), same convention as harness.replay_lock.
    """
    locked = None
    last_seen = None
    have_history = False
    last_pos = None
    last_vel = None
    lpf = {}
    switches = []
    for fid, dets in tframes:
        pool = [d for d in dets if d[1] >= CONFIDENCE_MIN]
        picked = None
        if pool:
            if locked is not None:
                cand = [d for d in pool if d[0] == locked]
                picked = max(cand, key=lambda d: d[1]) if cand else None
            else:
                # (re)acquire. Prefer highest-conf det near the predicted position.
                if have_history and last_pos is not None:
                    gap = min(fid - last_seen, max_gap) if last_seen is not None else 0
                    if use_vel and last_vel is not None:
                        px = last_pos[0] + last_vel[0] * gap
                        py = last_pos[1] + last_vel[1] * gap
                    else:
                        px, py = last_pos
                    near = [d for d in pool
                            if ((d[2] - px) ** 2 + (d[3] - py) ** 2) ** 0.5 <= radius]
                    if near:
                        picked = max(near, key=lambda d: d[1])
                    elif fallback:
                        picked = max(pool, key=lambda d: d[1])
                else:
                    # first acquisition ever: nothing to seed from -> highest conf
                    picked = max(pool, key=lambda d: d[1])
        if picked is not None:
            prev = locked
            if have_history and last_pos is not None and last_seen is not None and fid > last_seen:
                g = fid - last_seen
                last_vel = ((picked[2] - last_pos[0]) / g, (picked[3] - last_pos[1]) / g)
            locked = picked[0]
            last_seen = fid
            last_pos = (picked[2], picked[3])   # always refresh -> never freezes
            have_history = True
            if prev is not None and locked != prev:
                switches.append((fid, prev, locked))
            lpf[fid] = locked
        elif last_seen is not None and (fid - last_seen) <= CLEAR_AFTER and have_history:
            lpf[fid] = locked          # grace: still holding the id (estimator drives)
        else:
            locked = None              # clear -> next frame re-acquires (spatial gate)
            lpf[fid] = None
    return lpf, switches


def replay_lock_motion(tframes, v_min=0.01, hist_len=5, static_drop=4, near_radius=None):
    """Controller lock replay exploiting the MOTION discriminator.

    On this data the true ball moves ~0.05/frame while distractor balls are static
    (~0.0005/frame, a 100x gap). So:
      - (Re)acquire: among the pool, prefer the highest-confidence detection whose
        own recent track speed is >= v_min (a mover). Static distractors are skipped.
        A brand-new track (the true ball reappearing under a fresh id) has no speed
        history yet, so if near_radius is set we also accept a detection near the
        motion-predicted position regardless of its (unknown) speed.
      - Hold: follow the locked track, but if it has been static (speed < v_min) for
        static_drop consecutive frames, it is a distractor we latched onto -> drop it
        and re-acquire. This kills the long dwell on a persistent noise ball.

    Track speed is estimated causally from the last hist_len positions of each id.

    Returns (lpf, switches), same convention as harness.replay_lock.
    """
    from collections import deque, defaultdict
    hist = defaultdict(lambda: deque(maxlen=hist_len))   # tid -> recent (fid,cx,cy)
    locked = None
    last_seen = None
    have_history = False
    last_pos = None
    last_vel = None
    static_run = 0
    lpf = {}
    switches = []

    def speed(tid):
        h = hist.get(tid)
        if not h or len(h) < 2:
            return None
        (f0, x0, y0), (f1, x1, y1) = h[0], h[-1]
        return (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (f1 - f0)) if f1 > f0 else 0.0

    for fid, dets in tframes:
        for d in dets:
            if d[0] is not None:
                hist[d[0]].append((fid, d[2], d[3]))
        pool = [d for d in dets if d[1] >= CONFIDENCE_MIN]
        picked = None
        if pool:
            if locked is not None:
                cand = [d for d in pool if d[0] == locked]
                if cand:
                    picked = max(cand, key=lambda d: d[1])
                    sp = speed(picked[0])
                    static_run = static_run + 1 if (sp is not None and sp < v_min) else 0
                    if static_run >= static_drop:      # latched onto a static distractor
                        picked = None
                        locked = None
                        static_run = 0
            if picked is None and locked is None:
                gap = min(fid - last_seen, 6) if last_seen is not None else 0
                movers = [d for d in pool if (speed(d[0]) or 0.0) >= v_min]
                if near_radius is not None and have_history and last_pos is not None:
                    if last_vel is not None:
                        px, py = last_pos[0] + last_vel[0] * gap, last_pos[1] + last_vel[1] * gap
                    else:
                        px, py = last_pos
                    # among movers, prefer any that is near the motion prediction, and
                    # pick the NEAREST of those (the reappearing ball beats far high-conf
                    # moving-noise); else fall back to the highest-conf mover.
                    near = [(d, ((d[2] - px) ** 2 + (d[3] - py) ** 2) ** 0.5) for d in movers]
                    near = [(d, dd) for d, dd in near if dd <= near_radius]
                    if near:
                        picked = min(near, key=lambda x: x[1])[0]
                    elif movers:
                        picked = max(movers, key=lambda d: d[1])
                elif movers:
                    picked = max(movers, key=lambda d: d[1])
                if picked is None and not have_history:
                    picked = max(pool, key=lambda d: d[1])   # first acquisition seed
        if picked is not None:
            prev = locked
            if have_history and last_pos is not None and last_seen is not None and fid > last_seen:
                g = fid - last_seen
                last_vel = ((picked[2] - last_pos[0]) / g, (picked[3] - last_pos[1]) / g)
            if picked[0] != locked:
                static_run = 0
            locked = picked[0]
            last_seen = fid
            last_pos = (picked[2], picked[3])
            have_history = True
            if prev is not None and locked != prev:
                switches.append((fid, prev, locked))
            lpf[fid] = locked
        elif last_seen is not None and (fid - last_seen) <= CLEAR_AFTER and have_history and locked is not None:
            lpf[fid] = locked
        else:
            locked = None
            lpf[fid] = None
    return lpf, switches


def replay_lock_motion2(tframes, v_min=0.01, hist_len=5, static_drop=3,
                        radius=0.12, ema=0.5):
    """Motion discriminator + NEAREST-to-prediction re-acquisition.

    Difference from replay_lock_motion: on re-acquire, pick the detection NEAREST the
    motion-predicted position (within `radius`) instead of the highest-confidence
    one. The true ball reappears where its trajectory was heading; a distractor,
    being static, does not sit near the (moving) prediction -- so nearest-to-
    prediction picks the ball and ignores even a high-confidence distractor. No
    global-highest-confidence fallback: the ball always re-detects within <=9 frames
    near the trajectory (measured), so staying briefly lost beats grabbing noise.
    Velocity is an EMA over successive picks, so a single noisy frame can't fling the
    prediction off. Hold still drops a lock that has gone static for static_drop frames.
    """
    from collections import deque, defaultdict
    hist = defaultdict(lambda: deque(maxlen=hist_len))
    locked = last_seen = last_pos = vel = None
    have_history = False
    static_run = 0
    lpf = {}
    switches = []

    def speed(tid):
        h = hist.get(tid)
        if not h or len(h) < 2:
            return None
        (f0, x0, y0), (f1, x1, y1) = h[0], h[-1]
        return (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (f1 - f0)) if f1 > f0 else 0.0

    for fid, dets in tframes:
        for d in dets:
            if d[0] is not None:
                hist[d[0]].append((fid, d[2], d[3]))
        pool = [d for d in dets if d[1] >= CONFIDENCE_MIN]
        picked = None
        if pool:
            if locked is not None:
                cand = [d for d in pool if d[0] == locked]
                if cand:
                    picked = max(cand, key=lambda d: d[1])
                    sp = speed(picked[0])
                    static_run = static_run + 1 if (sp is not None and sp < v_min) else 0
                    if static_run >= static_drop:
                        picked = None
                        locked = None
                        static_run = 0
            if picked is None and locked is None and have_history and last_pos is not None:
                gap = min(fid - last_seen, 6) if last_seen is not None else 0
                if vel is not None:
                    px, py = last_pos[0] + vel[0] * gap, last_pos[1] + vel[1] * gap
                else:
                    px, py = last_pos
                near = [(d, ((d[2] - px) ** 2 + (d[3] - py) ** 2) ** 0.5) for d in pool]
                near = [(d, dd) for d, dd in near if dd <= radius]
                if near:
                    picked = min(near, key=lambda x: x[1])[0]     # nearest to prediction
            elif picked is None and locked is None and not have_history:
                picked = max(pool, key=lambda d: d[1])            # first acquisition seed
        if picked is not None:
            prev = locked
            if have_history and last_pos is not None and last_seen is not None and fid > last_seen:
                g = fid - last_seen
                inst = ((picked[2] - last_pos[0]) / g, (picked[3] - last_pos[1]) / g)
                vel = inst if vel is None else (ema * inst[0] + (1 - ema) * vel[0],
                                                ema * inst[1] + (1 - ema) * vel[1])
            if picked[0] != locked:
                static_run = 0
            locked, last_seen, last_pos = picked[0], fid, (picked[2], picked[3])
            have_history = True
            if prev is not None and locked != prev:
                switches.append((fid, prev, locked))
            lpf[fid] = locked
        elif last_seen is not None and (fid - last_seen) <= CLEAR_AFTER and have_history and locked is not None:
            lpf[fid] = locked
        else:
            locked = None
            lpf[fid] = None
    return lpf, switches


def replay_lock_reject_static(tframes, v_min=0.02, hist_len=5, min_obs=2):
    """THE SHIPPED RULE (ported to drone_controller._pick_target_detection +
    _TrackMotionTracker). Follow the locked track; never drop it. At re-acquisition,
    refuse any track that is KNOWN-static (observed >= min_obs times over the last
    hist_len frames, moving < v_min per frame) -- a persistent static distractor --
    and pick the highest-confidence detection among the rest. A brand-new/unknown
    track stays eligible (initial acquisition of a slow target is unaffected). If the
    only visible detections are known-static clutter, stay lost (estimator coasts).

    Result on the 2026-07-18 noisy FAST clip: 88.3% on-true, 22 false switches
    (baseline global re-pick: 76.3% / 39). Verified byte-for-byte against the real
    controller code.
    """
    from collections import deque, defaultdict
    hist = defaultdict(lambda: deque(maxlen=hist_len))
    locked = last_seen = None
    have = False
    lpf = {}
    switches = []

    def known_static(tid):
        h = hist.get(tid)
        if not h or len(h) < min_obs:
            return False
        (f0, x0, y0), (f1, x1, y1) = h[0], h[-1]
        if f1 <= f0:
            return False
        return (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (f1 - f0)) < v_min

    for fid, dets in tframes:
        for d in dets:
            if d[0] is not None:
                hist[d[0]].append((fid, d[2], d[3]))
        pool = [d for d in dets if d[1] >= CONFIDENCE_MIN]
        picked = None
        if pool:
            if locked is not None:
                cand = [d for d in pool if d[0] == locked]
                picked = max(cand, key=lambda d: d[1]) if cand else None
            if picked is None and locked is None:
                cand = [d for d in pool if not known_static(d[0])]
                if cand:
                    picked = max(cand, key=lambda d: d[1])
        if picked is not None:
            prev = locked
            locked, last_seen, have = picked[0], fid, True
            if prev is not None and locked != prev:
                switches.append((fid, prev, locked))
            lpf[fid] = locked
        elif last_seen is not None and (fid - last_seen) <= CLEAR_AFTER and have and locked is not None:
            lpf[fid] = locked
        else:
            locked = None
            lpf[fid] = None
    return lpf, switches


def main():
    clean_p, noisy_p = sys.argv[1], sys.argv[2]
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 18000
    clean, _ = parse_raw(clean_p)
    noisy, _ = parse_raw(noisy_p)
    truth, _ = truth_from_clean(clean, nb)
    shift, med = find_shift(truth, noisy, nb)
    print(f"truth {len(truth)}/{nb}  shift={shift:+d} med={med:.4f}\n")

    # tracker params = the Stage-1 config win (nms.02 + mmd.35)
    params = dict(ONBOX)
    params.update(nms_dist_thresh=0.02, match_max_dist=0.35)
    tframes, _life = run_tracker([(f, d) for f, d in noisy if f < nb], params)

    hdr = (f"{'lock rule':<40}{'on_true%':>9}{'on_noise%':>10}{'lost%':>7}"
           f"{'FALSE':>6}{'switch':>7}{'relockMx':>9}{'>15fr':>6}")
    print(hdr)
    print("-" * len(hdr))

    def score(label, lpf):
        res = _score_lpf(tframes, truth, lpf, shift)
        print(f"{label:<40}{res['true_pct']:>9.1f}{res['noise_pct']:>10.1f}"
              f"{res['lost_pct']:>7.1f}{res['n_false']:>6}{res['n_switch']:>7}"
              f"{res['max_relock']:>9}{res['n_slow']:>6}")
        return res

    # The progression this experiment established, best config (nms.02 mmd.35):
    from harness import replay_lock
    score("baseline: global-max-conf re-pick", replay_lock(tframes)[0])
    score("spatial: near last pos (fragile)", replay_lock_spatial(tframes, radius=0.05, use_vel=True, fallback=True)[0])
    score("motion: prefer movers + drop static", replay_lock_motion(tframes, v_min=0.02, static_drop=3)[0])
    score(">> SHIPPED: reject known-static", replay_lock_reject_static(tframes, v_min=0.02)[0])


def _score_lpf(tframes, truth, lpf, shift):
    """Score a precomputed per-frame lock (lpf) with exact labels -- same logic as
    groundtruth.analyze_labeled but without re-running the tracker or re-picking."""
    from collections import defaultdict
    from groundtruth import label_det
    tr = {f - shift: t for f, t in truth.items()} if shift else truth
    votes = defaultdict(lambda: [0, 0])
    for fid, fr in tframes:
        for tid, conf, cx, cy, w, h in fr:
            lab = label_det(tr, fid, cx, cy)
            if tid is not None and lab is not None:
                votes[tid][0 if lab else 1] += 1
    is_true = {t: v[0] > v[1] for t, v in votes.items()}
    nfr = len(tframes)
    on_true = on_noise = 0
    for fid, _fr in tframes:
        lid = lpf[fid]
        if lid is None:
            continue
        if is_true.get(lid, False):
            on_true += 1
        else:
            on_noise += 1
    # switches + re-lock latency
    seq = []
    for fid, _ in tframes:
        lid = lpf[fid]
        if not seq or seq[-1][0] != lid:
            seq.append([lid, fid, fid])
        else:
            seq[-1][2] = fid
    nonnull = [(l, a, b) for l, a, b in seq if l is not None]
    n_false = 0
    for (idA, _a, _b), (idB, _c, _d) in zip(nonnull, nonnull[1:]):
        if idA != idB and is_true.get(idA, False) and not is_true.get(idB, False):
            n_false += 1
    n_switch = sum(1 for (a, *_1), (b, *_2) in zip(nonnull, nonnull[1:]) if a != b)
    ontrue_flag = [(lpf[f] is not None and is_true.get(lpf[f], False)) for f, _ in tframes]
    relock = []
    seen = False
    gap = 0
    for flag in ontrue_flag:
        if flag:
            if seen and gap > 0:
                relock.append(gap)
            seen = True
            gap = 0
        elif seen:
            gap += 1
    return dict(true_pct=on_true / nfr * 100, noise_pct=on_noise / nfr * 100,
                lost_pct=(nfr - on_true - on_noise) / nfr * 100,
                n_false=n_false, n_switch=n_switch,
                max_relock=max(relock) if relock else 0,
                n_slow=sum(1 for g in relock if g > 15))


if __name__ == '__main__':
    main()
