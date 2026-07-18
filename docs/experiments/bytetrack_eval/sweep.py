import harness as H
from harness import parse_raw, analyze, ONBOX
configs = [
    ("baseline on-box (buf30, m0.3, mmd0.2)", dict()),
    ("track_buffer 60",                        dict(track_buffer=60)),
    ("track_buffer 90",                        dict(track_buffer=90)),
    ("track_buffer 120",                       dict(track_buffer=120)),
    ("buf90 + recovery_max_dist 0.2",          dict(track_buffer=90, recovery_max_dist=0.2)),
    ("buf90 + match_max_dist 0.4",             dict(track_buffer=90, match_max_dist=0.4)),
    ("buf90 + match_thresh 0.1",               dict(track_buffer=90, match_thresh=0.1)),
    ("buf90 + track_thresh0.2 det0.3",         dict(track_buffer=90, track_thresh=0.2, det_thresh=0.3)),
    ("buf90 + rec0.2 + mmd0.4 (combo)",        dict(track_buffer=90, recovery_max_dist=0.2, match_max_dist=0.4)),
]
for logf,lbl in [("logs/runA.txt.gz","RUN A"),("logs/runB.txt.gz","RUN B")]:
    frames_raw,_=parse_raw(logf)
    print(f"\n################ {lbl} — param sweep ################")
    print(f"{'config':<40} {'#ids':>5} {'true%':>6} {'switch':>7} {'FALSE':>6} {'frag':>5}")
    for name,over in configs:
        p=dict(ONBOX); p.update(over)
        r=analyze(frames_raw,p,name,verbose=False)
        print(f"{name:<40} {r['nids']:>5} {r['true_cov']*100:>5.0f}% {r['n_switch']:>7} {r['n_false']:>6} {r['n_frag']:>5}")
