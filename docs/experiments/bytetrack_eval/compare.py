"""Compare detection-stream health + locking between logs (old vs new hardware).
Analyzes every log with the SAME baseline tracker params (buf30) so the only
variable is the captured RAW detection stream."""
import sys, re, gzip
from collections import defaultdict
from harness import parse_raw, run_tracker, replay_lock, ONBOX

TS = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d+).*?frame=#(\d+) !!! RAWDETS n=(\d+)")
BASE = dict(ONBOX); BASE['track_buffer'] = 30   # baseline, matches the original runs

def wallclock(path):
    op = gzip.open if path.endswith('.gz') else open
    t0 = t1 = None; nframes = 0
    with op(path, 'rt', errors='replace') as f:
        for line in f:
            m = TS.search(line)
            if not m:
                continue
            h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            t = h*3600 + mi*60 + s + int(str(ms).ljust(3,'0')[:3])/1000
            if t0 is None:
                t0 = t
            t1 = t; nframes += 1
    dur = (t1 - t0) if (t0 is not None and t1 is not None) else 0
    return dur, nframes

def blackouts(frames, minlen=5):
    st = []; run = 0; start = None
    for fid, d in frames:
        if len(d) == 0:
            if run == 0: start = fid
            run += 1
        else:
            if run >= minlen: st.append((start, run))
            run = 0
    if run >= minlen: st.append((start, run))
    return st

def stats(path, label):
    frames, _ = parse_raw(path)
    dur, nfr = wallclock(path)
    fps = nfr/dur if dur else 0
    dh = defaultdict(int)
    for _, d in frames: dh[len(d)] += 1
    bo = blackouts(frames)
    blank = sum(r for _, r in bo)
    longest = max((r for _, r in bo), default=0)
    # offline tracking at baseline buf30
    tf, life = run_tracker(frames, BASE)
    lpf, _ = replay_lock(tf)
    seq = []
    for fid, _ in tf:
        lid = lpf[fid]
        if not seq or seq[-1][0] != lid: seq.append([lid, fid, fid])
        else: seq[-1][2] = fid
    nn = [(l,a,b) for l,a,b in seq if l is not None]
    PERSIST = len(tf)*0.10
    reacq = [(a,b) for (a,_,ea),(b,sb,_) in zip(nn,nn[1:]) if a!=b]
    n_false = sum(1 for a,b in reacq if len(life.get(a,[]))>PERSIST and len(life.get(b,[]))<=PERSIST)
    n_frag  = sum(1 for a,b in reacq if len(life.get(a,[]))>PERSIST and len(life.get(b,[]))>PERSIST)
    lock_fr = sum(1 for f,_ in tf if lpf[f] is not None)
    true_id = max(life, key=lambda t: len(life[t])) if life else None
    true_cov = len(life[true_id])/len(tf) if true_id is not None else 0
    fseq = [f for f, _ in frames]
    span_fr = (max(fseq) - min(fseq) + 1) if fseq else 0
    dropped = span_fr - len(fseq)
    droppct = dropped/span_fr*100 if span_fr else 0
    return dict(span_fr=span_fr, dropped=dropped, droppct=droppct,
                label=label, dur=dur, nfr=nfr, fps=fps, dethist=dict(sorted(dh.items())),
                nblk=len(bo), blank=blank, blankpct=blank/nfr*100 if nfr else 0,
                longest=longest, longest_s=longest/fps if fps else 0,
                nids=len(life), true_cov=true_cov, lock_pct=lock_fr/len(tf)*100,
                nreacq=len(reacq), n_false=n_false, n_frag=n_frag)

if __name__ == '__main__':
    rows = [stats(p, l) for p, l in (eval(sys.argv[1]))]
    cols = [("run","label"),("wall_s","dur"),("frames","nfr"),("fps","fps"),
            ("blackouts","nblk"),("blank_fr","blank"),("blank%","blankpct"),
            ("longest_fr","longest"),("longest_s","longest_s"),
            ("#ids","nids"),("lock%","lock_pct"),("true_cov%","true_cov"),
            ("reacq","nreacq"),("FALSE","n_false"),("frag","n_frag")]
    w = 12
    print("".join(h.ljust(w) for h,_ in cols))
    for r in rows:
        cells=[]
        for h,k in cols:
            v=r[k]
            if k=='true_cov': v=f"{v*100:.0f}"
            elif isinstance(v,float): v=f"{v:.1f}"
            cells.append(str(v).ljust(w))
        print("".join(cells))
