# Handoff: size-driven tiling ladder — re-land + first rig test

**Date:** 2026-07-23
**Branch:** `feat/tiling-ladder` (local only, **NOT pushed**), based on `fix/tiling-bbox-double-remap`.
**Status:** re-land complete, host suite green, first on-rig run done. **One blocking bug found**
(branch-switch blinds the callback on file input). The ladder's mechanics work; the eval is
not yet trustworthy until the bug is fixed.

---

## 1. TL;DR — where to pick up

The size-driven tiling ladder (3×2 → 2×1 → 1×1, switches on target size) is fully wired and
replaces the old binary whole↔tile switch. It ran on the rig against the clean pink-ball clip:
**17 handovers, 0 aborts, 0 stalls, 0 tracebacks, ball detected at conf 0.75–0.81, `primary_side`
tracking the ball.** So valves/selector/coordinator/watchdog/size-feed all work.

**BUT** a pre-existing latent bug dominates on file input: **every branch switch blinds
`app_callback` for as long as the *other* branch was last active — 53% of the run's frames
(7099/13333) were dropped by `FrameOrderGuard`.** The policy was deciding on a strobe-lit view,
so threshold-correctness is **unverified**. This is the #1 thing to fix. See §4–5.

**Next action:** implement the source-side frame-id probe (§5), redeploy, re-run the clean clip
(§7), confirm drop bursts are gone, then re-check threshold correctness.

Related memories (read these): `[[branch-switch-blinds-file-input]]`, `[[tiling-size-ladder-branch]]`,
`[[tiling-bbox-double-remap-bug]]`, `[[bdd-box-launch-recipe]]`, `[[nv12-hailo8-model]]`,
`[[file-input-e2e-latency-tiling-matrix]]`, `[[camera-resolution-tiling-format-matrix]]`.

---

## 2. Branch topology

```
origin/main
  └─ fix/tiling-bbox-double-remap        (this is where the working-tree commits landed)
       e176cce  fix(tiling): tiled branch double-remapped bboxes (filter_letterbox)  [pre-existing]
       0403bc9  fix(latency): reconstruct StageB start ts on file-input path
       6f14c10  fix(camera): plumb config.camera resolution into source capsfilter
       b0dc1a0  fix(overlay): move #track_id label out of confidence text
       84140ea  chore(config): rig tuning snapshot (red-plate HEF, PD/thrust/takeoff)
       6090489  chore: check in CLAUDE.md; ignore graphify-out/ + requirements.txt_INSTALLED
         └─ feat/tiling-ladder            (the re-land — 11 commits)
              e4b764d  docs: spec
              50341aa  docs: size metric max(bw,bh)
              1cf8b16  docs: implementation plan
              36a4500  docs: ladder-only cleanup
              74b7eef  feat: config schema (ladder tiers, drop legacy fields)
              1d15f6e  test(config): follow shipped ulog_trace rate to 0
              5d17ced  feat: ladder policy; port Coordinator + StallWatchdog to tiers
              66588a3  feat: N-branch pipeline builder; drop two-branch section
              e6d262b  feat: wire the ladder end to end; binary switch gone
              7b54345  feat: ship the 3-rung ladder ON by default
              60fd4a0  fix: main() still referenced deleted `tiles` var  [found on rig]
```

Nothing is pushed. Working tree is clean as of this handoff. Host suite: **348 passed, 8 skipped**
(`cd BDD && /usr/bin/python3 -m pytest -q` — note: uses SYSTEM python `/usr/bin/python3`, which has
cv2+numpy<2; the flight_review venv has numpy 2.x and no cv2, so it can't run the suite).

---

## 3. What the ladder is (architecture, for orientation)

The ladder is the **sole** tiling mechanism. The binary whole↔tile switch, its config fields
(`tiles_x/tiles_y/switchable/start_on_tiling/switch_conf/lost_frames_to_tile/locked_frames_to_whole/
locked_streak_miss_penalty`), the `--tiles`/`--switch-tiles` CLI flags, and the static single-grid
path are all **deleted**.

- **Config** (`BDD/config.py` `Config.Tiling`): `ladder: list[Tier]` where `Tier(tiles_x, tiles_y,
  up_side, down_side)`, index 0 = most tiles, last MUST be 1×1. Plus `overlap`, `tile_iou_threshold`,
  `auto_switch`, `lost_to_2x1_s`, `lost_to_3x2_s`, `stall_timeout_s`, `stall_cooldown_s`.
  Empty/absent ladder ⇒ plain whole-frame. "switchable" is derived (`len(ladder) >= 2`).
- **Policy** (`BDD/tiling_policy.py`): `build_ladder(tiling)` validates + returns `[LadderTier]`.
  `TilingLadderPolicy.note(side, now_s)` returns the tier to switch to (one step) or None.
  `side = max(bbox_w, bbox_h)` of the largest MATCHED track (needs bytetrack). Climb toward whole
  when `side >= up_side`; descend when `side < down_side` (asymmetric = hysteresis). No matched
  track ⇒ target-lost time escalation: rung n-2 at `lost_to_2x1_s`, tier 0 at `lost_to_3x2_s`.
  `TilingSwitchCoordinator` (serializes every switch, ported bool→int from main) and
  `BranchStallWatchdog` (reverts to whole-frame if an accepted rung later dies) both carried over.
- **Pipeline** (`BDD/tiling_pipeline.py` `build_switchable_detection_section`): one valve-gated
  branch per rung merging at one `input-selector`; only the last rung's valve (whole-frame) boots
  open. This module has NO hailo import so its tests actually RUN on host (unlike `pipelines.py`,
  which `importorskip`s and pins nothing off-device — that's WHY the builder lives here).
- **app_base.py**: `switch_to_tier(int)` (make-before-break handover, graceful revert),
  `_init_selector_startup_pad()` (points selector at sink_{last}=whole after parse_launch;
  the selector defaults to sink_0=most-tiled, which would read a shut valve → instant stall),
  `active_tier`, `kill_tile_branch_for_test()` (fault injection on the active tiled rung).
- **app.py**: callback computes `primary_side` from matched tracks and calls `note_tiling(side)`;
  `main()` builds the ladder, derives switchable, wires policy through the coordinator, spawns the
  stall watchdog thread; `--test-switch-s` steps through all rungs; `--test-kill-tile-after-s`
  drives the watchdog test.

Shipped `config.yaml` default: `3x2 → 2x1 → 1x1`, `auto_switch: true` (user chose this).

---

## 4. THE BLOCKING BUG — branch switch blinds the callback (file input)

**Symptom:** 7099 of 13333 source frames (53%) rejected by `FrameOrderGuard`; drop bursts start
within ~0.1s of every `BRANCH SWITCH` and last as long as the previous branch was active.

**Mechanism (file input only):**
1. `normalized_frame_id()` (`app.py`) priority: picamera2 frame-id meta → `buffer.offset` →
   `buffer.pts` → wallclock. File input has NO picamera meta, so it uses **`buffer.offset`**.
2. `buffer.offset` is stamped **per branch** by each rung's hailo aggregator, and a rung's counter
   only advances while its valve is open (a shut valve drops buffers → that aggregator counts
   nothing). Evidence: after a switch the ids restart LOW (`frame 1, 2, 3…`).
3. `FrameOrderGuard` is a SINGLE high-water mark shared across all rungs (keyed on camera_id only).
4. So switching into rung B feeds the guard B's local counter, which is behind the shared mark by
   exactly the frames the other rung produced while B was idle. Every B frame is `<= mark` → rejected
   at `app.py:324` (early `return Gst.PadProbeReturn.OK`), for the whole catch-up window.

During a rejected frame the callback returns BEFORE: RAWDETS log, `ByteTracker.update`,
`note_tiling` (no size feed → policy frozen), and the queue put (controller starved). So "blind"
is total for that frame.

**Why it self-oscillates and is asymmetric:** the run ping-pongs tier2↔tier1. Descends to tier1
land it blind for ~a full size cycle; it becomes visible only at the top of the cycle (ball big),
whose first size sample immediately says "climb", so tier1 does ~no useful work and the drop bursts
always sit on the tier1 (post-descend) side; tier2 resumes clean. Net: the ladder is effectively
whole-frame-only AND blind for half the run — and blind **exactly during the small-ball windows
where tiling was supposed to help.**

**Not a ladder regression** — `normalized_frame_id`, the guard, and per-branch offsets all
pre-date this branch; main's binary switch has the same latent mechanism. The ladder just switches
often enough to make it dominate. NOT yet reproduced on main. **Camera path is believed unaffected**
(priority-1 picamera meta is stamped at source, survives the branches) but NOT verified this run.

---

## 5. RECOMMENDED FIX (the pending todo)

Stamp a **branch-independent frame id on the source, before `branch_src_tee`**, into the same
`frame-id/x-picamera2` reference meta the callback already reads at priority 1. Because it is set
before the fan-out, every rung inherits the identical id and the callback never touches
`buffer.offset`. Reference metas are proven to survive the branches (the unix-ts and picamera metas
already do).

Mirror the existing `_install_detection_start_probe` (`app_base.py:422`) — same
`add_reference_timestamp_meta` idiom — but on a **pre-tee** element (NOT `inference_wrapper_input_q`,
which is inside a branch). Good candidates: the pace `identity` on the file path, or the queue just
ahead of `branch_src_tee`. Only install it when the meta is absent (so the camera path is untouched).

**Value — two options:**
- **A (recommended here): real video frame index from PTS.** On CFR, `idx = round(buffer.pts /
  (Gst.SECOND / fps))`. Genuine file position; branch-independent; **matches the burned-in frame
  counter overlay in clean_720.mp4** (great for ground-truth correlation in the eval). Resets to 0
  per loop — already covered by `on_stream_rewound → FrameOrderGuard.reset()` (wired to the file
  loop; see `frame_order.py:42` docstring). Needs known fps (plumb from `config.camera.fps`).
  Risk: trusts PTS present+CFR — true on the file path (the pace `identity sync=true` depends on PTS).
- **B (most conservative): monotonic counter.** `self._src_frame_no += 1` per probe call. Strictly
  monotonic across loops (no reset needed), 1:1 with decoded frames if placed before any leaky queue.
  Not the literal file index.

Either fixes the blinding (the win is stamping pre-tee). Recommend A for the eval-correlation bonus,
keeping the existing loop guard-reset. Est. ~15 lines. After landing: `pytest` (host), then §7.

**Also pending (small):** log `primary_side` AT the switch decision in the coordinator/policy so
threshold-correctness is directly verifiable from the log (currently only a 1-Hz `TILING-SIZE` line
exists, too coarse to confirm which side of a threshold a switch fired on).

---

## 6. Rig test results (2026-07-22, before the fix)

Run: `clean_720.mp4`, ball HEF, `--DEBUG --no-record --vision-only`, 442s, ladder 3×2→2×1→1×1.

| Signal | Result |
|---|---|
| Tracebacks / ConfigError / NameError | 0 (after fixing the `tiles` NameError, commit 60fd4a0) |
| Ball detection | conf 0.75–0.81, `tracks=1`, `primary_side` tracks the ball |
| BRANCH SWITCH | 17 (14 climb + 14 descend across the analysis window) |
| Switch aborts / "no buffer" / STALL | 0 / 0 |
| DETS frames with n=0 | 0 |
| **Frames dropped by guard** | **7099 / 13333 = 53%** ← the bug |
| Tier 0 (3×2) reached | **never** (ball never held matched side <0.02 while visible, nor lost 10s) |
| Threshold correctness | **UNVERIFIED** — contaminated by the strobing + raw-vs-matched size mismatch |

Ball size in the clip (measured, `measure_ball.py`): side sweeps **0.008 → 0.186** on a ~36s
period. side≥0.10 (climb): 49% of frames; <0.05 (descend to 2×1): ~25%; <0.02 (descend to 3×2): 10%.
So the clip exercises the 2×1↔whole thresholds heavily but only grazes the 3×2 rung.

---

## 7. Rig runbook (what actually works — DNS is broken, use the IP)

**Host:** `bdd@100.91.134.66` (Tailscale; hostname `bdd-rpi-testrig-1`, reports `bdd-sd9`).
Tailscale DNS is currently down → **hostnames don't resolve, use the IP.** SSH is relay-routed and
can drop mid-command; the tmux run survives, so wrap remote calls and retry. The box is tmpfs:
`BDD/`, `scripts/`, `models/` are **wiped on reboot** — always redeploy. `ls BDD/*.py | wc -l` and
`ls /home/bdd/models/` to check.

**Assets (already on the box as of this handoff, but tmpfs — re-push after any reboot):**
- HEF: `/home/bdd/models/2026-04-09_11n_ball_v3_nv12_h8.hef`
  (laptop: `/media/enmk/BDD/_MODELS/experimental/_2/2026-04-09_11n_ball_v3_nv12_h8.hef`).
  Verified `HAILO8` + `NV12(320x640x3)` via `hailortcli parse-hef`. This is the true NV12+H8 ball
  detector; most `*nv12*` filenames LIE (see `[[nv12-hailo8-model]]`).
- Video: `/home/bdd/testdata/clean_720.mp4`
  (laptop: `/media/enmk/BDD/_BACKUPS/_TMP/2026-07-18/clean_720.mp4`). 1280×720 30fps 18000 frames,
  single pink ball bouncing + growing/shrinking, **burned-in frame-counter overlay** (usable as
  ground truth once PTS-index is stamped). MP4 (SOURCE_PIPELINE uses qtdemux; MKV crashes). Loops.
- Config: `/home/bdd/hailo-rpi5-examples/BDD/config.ladder-test.yaml` — a copy of `config.yaml` with
  ONLY `hef_model_path` sed-replaced to the ball HEF. **You must recreate it after each deploy**
  (the archive overwrites the tree). The shipped `config.yaml` points `hef_model_path` at a
  red-plate path that does NOT exist on this box, so a plain run fails config validation.

**Deploy (ships exactly the git ref, leaves venv/_DEBUG untouched):**
```bash
git archive --format=tar feat/tiling-ladder | gzip -6 > /tmp/ladder.tgz     # from repo root
scp -q /tmp/ladder.tgz bdd@100.91.134.66:/tmp/
ssh -n bdd@100.91.134.66 '
  cd /home/bdd/hailo-rpi5-examples && tar -xzf /tmp/ladder.tgz
  cd BDD && sed -e "s|^  hef_model_path:.*|  hef_model_path: /home/bdd/models/2026-04-09_11n_ball_v3_nv12_h8.hef|" config.yaml > config.ladder-test.yaml'
# verify tree landed: hash tracked tree both sides and compare (see [[bdd-box-launch-recipe]])
```

**Launch (vision-only, file input):**
```bash
ssh -n bdd@100.91.134.66 '
  tmux kill-server 2>/dev/null ||:
  pkill -9 -x mavsdk_server 2>/dev/null ||:
  APP=$(fuser /dev/hailo0 2>/dev/null | tr -d " "); [ -n "$APP" ] && kill -9 "$APP"; sleep 2
  tmux new-session -d -s bdd-app "cd /home/bdd && exec bash hailo-rpi5-examples/scripts/bdd.sh \
    --DEBUG --no-record --vision-only \
    --config /home/bdd/hailo-rpi5-examples/BDD/config.ladder-test.yaml \
    --input /home/bdd/testdata/clean_720.mp4"'
```
Liveness = `fuser /dev/hailo0` (NEVER `pgrep app.py` — setproctitle renames it "Hailo Detection").
Stop = `kill -TERM` then `-9` (SIGINT does not stop it). Let it run ≥5 min for several size cycles.

**Analyze the log:**
```bash
LOG=$(ssh -n bdd@100.91.134.66 'ls -t /home/bdd/hailo-rpi5-examples/_DEBUG/BDD_*.log | head -1')
# counts:
ssh -n bdd@100.91.134.66 "grep -ac 'BRANCH SWITCH' '$LOG'; grep -ac 'out-of-order' '$LOG'; \
  grep -ac STALL '$LOG'; grep -acE 'Traceback|NameError' '$LOG'"
# pull events for correlation (strip ANSI): TILING-SIZE, BRANCH SWITCH, DETS frame, RAWDETS, out-of-order
```
Key check post-fix: **`out-of-order` drop count should collapse to ~0** (only the known handful of
duplicate-at-handover drops, which are the guard doing its job — see `switch_to_tier` docstring).
The drop-burst-vs-switch correlation and per-frame threshold check were done with throwaway python
in the session scratchpad (now gone); re-derive from the RAWDETS + BRANCH SWITCH streams. The
burst law: each burst's length ≈ (time the other branch was active) × 30fps, ids restart low.

---

## 8. Remaining work (priority order)

1. **[BLOCKER] Source-side frame-id probe** (§5). Redeploy, re-run §7, confirm drops ~0.
2. **Log `primary_side` at the switch decision** so threshold correctness is verifiable.
3. **Re-verify threshold correctness** on the clean clip after 1–2 (climb at side≥up, descend at
   side<down, one step per decision, hysteresis holds — the dead-band walk that the branch never
   validated on a moving target).
4. **Exercise tier 0 (3×2)** — needs a clip where the ball goes very small/far, or a target-lost
   ≥10s window. Consider `sky_red_ball_10-200px_10m.mp4` (has 10px balls) transcoded to 720p CFR;
   note it's 4K mpeg4 as-is. Or a hold-still + occlusion clip.
5. **Watchdog on rig:** `--test-kill-tile-after-s N` should shut the active tiled rung's valve and
   the watchdog should revert to whole-frame within ~`stall_timeout_s`.
6. **Manual cycle gate:** `--test-switch-s N` steps through all 3 rungs — confirm glitch-free.
7. **Reconsider the shipped default ladder** vs the latency matrix (60/90/217/297 ms for
   1×1/2×1/2×2/3×2; only ≤2×1 holds the ~200ms control budget). 3×2 is defensible ONLY as a
   reacquire rung. User chose to ship 3×2→2×1→1×1; revisit if the eval shows the 3×2 rung is
   never usefully reached or costs too much when it is.
8. **Camera-path (live) run** to confirm the blinding is truly file-input-only, and a real
   hysteresis walk with a physically moving target.
9. **Finish the branch:** decide merge/PR for `fix/tiling-bbox-double-remap` (6 commits incl. the
   uncommitted-work snapshot) AND `feat/tiling-ladder`. Nothing is pushed. The `84140ea` config
   snapshot is a rig-tuning snapshot, NOT validated — flag before merging to main.

---

## 9. Gotchas / traps hit this session

- **`py_compile` does NOT catch the class of bug that killed launch #1** (a `NameError` on a deleted
  var in a function body). `app.py`/`app_base.py` aren't importable on a dev host (hailo/GStreamer),
  so the host suite can't cover `main()`. **Use `pylint -E0602` as the host gate** for app.py/
  app_base.py (it's clean now): `flight_review/.venv/bin/pylint --disable=all --enable=E0602,E1120,E1123
  --ignored-modules=gi,hailo,hailo_apps,cv2,numpy app.py app_base.py tiling_policy.py tiling_pipeline.py`.
- **Two pythons:** run pytest with `/usr/bin/python3` (has cv2 + numpy<2). The auto-activated
  `flight_review/.venv` has numpy 2.x and no cv2 — pytest collection fails there.
- **Shipped `config.yaml` HEF path** is a red-plate model absent on the rig → use `--config` with a
  sed-patched copy, don't edit the tracked file.
- **`--config` flag exists** (app.py) — use it; don't mutate the deployed `config.yaml`.
- **MP4 only** for file input (qtdemux); MKV crashes. File loops; kill by pid.
- **`config.ladder-test.yaml` must be recreated after every deploy** (archive overwrites the tree).
- Rig hostname resolution is down (Tailscale DNS) — **use `100.91.134.66`**, not `bdd-sd9-*`.
- `--vision-only` = no control thread, so no `LATENCY sensor→command` line (fine — we're testing
  tiling behaviour, not e2e latency here).

---

## 10. Files changed on `feat/tiling-ladder` (vs main)

```
BDD/config.py                     ladder schema (Tier/ladder/thresholds), legacy fields deleted
BDD/config.yaml                   ladder ON by default (3x2->2x1->1x1, auto_switch: true)
BDD/config.test-autoswitch.yaml   migrated to ladder schema
BDD/config.test-ladder-single.yaml  new: single-cam 3-rung ladder fixture
BDD/tiling_policy.py              TilingLadderPolicy + build_ladder; Coordinator/Watchdog ported to tiers
BDD/test_tiling_policy.py         ladder policy tests + ported coordinator/watchdog tests (tier ints)
BDD/tiling_pipeline.py            new: host-testable N-branch build_switchable_detection_section
BDD/test_tiling_pipeline.py       new: builder tests incl. name-contract pins
BDD/pipelines.py                  deleted SWITCHABLE_DETECTION_SECTION (two-branch)
BDD/test_pipelines.py             dropped two-branch pins; tile_->tier0_ rename
BDD/app_base.py                   switch_to_tier/ladder_grids/_init_selector_startup_pad/active_tier
BDD/app.py                        note_tiling(side); main() builds ladder; --tiles/--switch-tiles gone
BDD/test_config.py                ladder schema tests; ulog_trace rate follow
BDD/test_debug_drone_controller.py  pin ulog_trace rate=5 for the aggregation test
BDD/TODO.md                       flag cleanup
docs/superpowers/{specs,plans}/2026-07-07-dynamic-tiling-ladder*  spec + plan (cherry-picked)
```

**The fix (§5) goes in `BDD/app_base.py`** (new `_install_source_frame_id_probe`, called from
`create_pipeline` after parse_launch, targeting a pre-`branch_src_tee` element) — plus plumbing
`config.camera.fps` if you take the PTS-index route.
