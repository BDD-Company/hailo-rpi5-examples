# Experiment: runtime tiling / branch switching on a shared Hailo-8

**Status:** design only (no prototype yet). **Date:** 2026-06-17.
**Goal:** change tiling (tile size + count) at runtime, and/or switch between two GStreamer detection
branches that share one Hailo-8, with minimal capture→command latency and minimal frames lost — during
both steady-state and the switch itself.

**Decision locked in:** every branch uses the **same HEF** (same detector network; branches differ only
in crop geometry / tile layout). Single constant-FPS camera, shared across all branches.

---

## TL;DR / handoff (read this first)

- **What:** switch tiling geometry (and/or detection branches) at runtime on one shared Hailo-8L, minimal
  latency + frames lost. Tiling geometry is NOT live-tunable → implement as **pre-built valve-gated branches
  behind an `input-selector`**, switched via a 4-step **hot handover**. Same HEF on all branches ⇒ **no PCIe
  weight reload**. See *Proposed architecture*.
- **Switch cost:** essentially free on delivered commands (handover hides it); the visible change is the
  steady **L_A→L_B step** = `(N_B−N_A)·t_inf` (~+30–50 ms whole→3×2). Overlap transient ~sub-frame for 2–3 frames.
- **Tile budget on current rig (Pi5 + 8L + yolo11n, PCIe x1):** device ceiling ~**45–50 inf/s** (~20 ms/tile).
  Realistic max ≈ **4 tiles (2×2)**; 2 comfortable; **6 (3×2) is the hard edge**; 9+ not realistic. See *Tile budget*.
- **Hardware:** upgrading 8L→Hailo-8 is a **weak buy for yolo11n** (only 1.18×); generic 8L HEF on an H8 gives
  nothing (must recompile `--hw-arch hailo8`). **The real lever is model, not chip: yolo11n→yolov8n** (faster on
  BOTH chips, free, −1.4 mAP). H8 only pays off paired with compute-bound yolov8n (~5×). See *Hardware option*.
- **PREREQUISITE — camera Stage-A (separate task/doc).** Tiling's +45–60 ms is only affordable once Stage-A
  latency is cut. **DONE on branch `camera-stage-a-latency`:** buffer_count 3→2 + pin/auto-pin exposure
  (Fixes 2 & 3) — tail p99/max ~halved. **STILL TO DO, and it belongs to THIS task: Fix 1 = capture 30 fps +
  drop surplus frames newest-first & CHEAPLY** (re-queue the picamera2 request without make_array/numpy). It
  was deferred on purpose — Fix 1 only pays off once inference runs *slower* than capture, which is exactly
  what 2×2 tiling causes. **So implement Fix 1 as part of the tiling work** (memory
  `fix1-drop-surplus-deferred-to-tiling`); cheap-drop is mandatory or the extra ISP/CPU memcpy throttles the
  no-cooling Pi 5 and erases the win. **Full handoff: `camera-stage-a-latency.md`.**
- **Measure first (no prototyping yet):** (1) does same-HEF give one network group or two (per-branch vs shared
  `hailonet`)? (2) real `t_inf`/tile and 8L throughput ceiling (A-alone vs 2×2, read `detection delay`).

---

## Feasibility findings

1. **Tiling cannot be retuned in-place.** `hailotilecropper` geometry props
   (`tiles-along-x-axis`, `tiles-along-y-axis`, `overlap-x/y-axis`, `tiling-mode`) are all
   `GST_PARAM_MUTABLE_READY` → settable only in `NULL`/`READY`, never while `PLAYING`. There is **no
   tile-size property**: tile = frame ÷ count, then every tile is resized to the HEF's fixed input.
   ⇒ "runtime tiling change" = **select between pre-built branches**, each with a fixed geometry.
   (Only if geometry must be continuous/unbounded do we fall back to a per-branch bin restart:
   IDLE-block → `READY` → set props → `PLAYING`; tens of ms + possible 1-frame gap; reserve for rare reconfig.)

2. **Device sharing works and is half-wired already.** A single Hailo-8 vdevice runs multiple network
   groups time-shared by the HailoRT Model Scheduler. Enable by giving every branch's `hailonet` the
   **same `vdevice-group-id`** (non-`"UNIQUE"`, e.g. `"SHARED"`) + same `scheduling-algorithm=round_robin`.
   `BDD/pipelines.py` already emits `vdevice-group-id`; the stack already creates VDevices with ROUND_ROBIN.
   **Never hot-swap `hef-path` on a live `hailonet`** (Hailo staff: can segfault) — pre-build branches.

3. **Same-HEF ⇒ no PCIe weight-reload penalty.** Hailo-8 has no on-chip DRAM, so activating a *different
   network group* re-streams weights over PCIe (~5 ms on Pi 5, first inference ~42 ms). Because all
   branches share one HEF, switching does **not** change which weights are resident — the device just runs
   more or fewer crops per frame. Per-branch latency differs (more tiles = more inferences/frame), but the
   *switch* is cheap.

4. **Current code uses whole-buffer cropper, not tiling.** `BDD/pipelines.py` ~289–294 uses whole-buffer
   `hailocropper`/`hailoaggregator`. Tiling lives only in the installed `hailo_apps` `TILE_CROPPER_PIPELINE`.

---

## Proposed architecture

```
camera (single, constant FPS) → videoconvert/scale (common RGB caps)
  → tee
      ├─ queue(leaky=downstream,max=1) → valve_A → BRANCH_A  (whole-frame: hailocropper → hailonet → hailoaggregator)
      ├─ queue(leaky=downstream,max=1) → valve_B → BRANCH_B  (tiled 3×2: hailotilecropper → hailonet → hailotileaggregator)
      ├─ … more discrete configs (2×2, 4×3, …)
  → input-selector (active-pad, sync-streams=false) → hailotracker → identity(user_callback) → command
```

- All `hailonet`s: same `hef-path`, `vdevice-group-id="SHARED"`, `scheduling-algorithm=round_robin`, `batch-size=1`.
- Source bin **never changes state** → zero source-side frame loss across any switch (matches single-shared-camera design).

**Steady state:** exactly ONE valve open → one branch feeds Hailo → full latency budget, no contention.
(All-branches-always-on is rejected: permanent N× device time-sharing → permanent latency hit.)

**Hot handover A→B (zero command gap, loss ≤ incoming leaky-queue depth):**
1. `valve_B.drop=false` — B starts feeding Hailo (brief overlap; same-HEF so no reload).
2. Wait for first valid buffer out of B (pad probe on B aggregator src, or K≈2–3 warmup frames).
3. `input-selector.active-pad = B_pad` — atomic, next-buffer effect; no gap because B already producing.
4. `valve_A.drop=true` — A goes idle.

Raw inference frames have no keyframe dependency → cutover is per-frame clean with `sync-streams=false`.
(Alt: `funnel` instead of `input-selector` — simpler, but duplicates detections during the overlap;
tracker dedups anyway.)

---

## Switch-latency estimate (A → B)

Grounded numbers: **1280×720 @ 30 fps** (frame period T ≈ 33.3 ms), **yolo11n on Hailo-8L**, current
detection ~16 fps (camera/pipeline-limited → Hailo has headroom for whole-frame). Same HEF on every
branch ⇒ **no PCIe weight reload (0 ms)**. Let `t_inf` = one 640×640 yolo11n inference's device
occupancy on the 8L; **estimate t_inf ≈ 6–10 ms** (unmeasured — the 16 fps is pipeline-limited, not chip).
Example: branch A = whole-frame (`N_A = 1` inference/frame), branch B = 3×2 tiles (`N_B ≈ 6`).

"Extra latency when switching" is three separate things; only the third is a true switch-only transient:

1. **Delivered-command latency from the switch mechanics ≈ 0.** The hot handover keeps A serving until
   B is warm (step 2 waits for B's first valid buffer), so no gap/stall is injected — command latency
   simply *steps* from L_A to L_B. (Same HEF ⇒ no reload to hide.)

2. **Steady-state step L_B − L_A** (the real cost, but inherent to "more tiles", not a switching artifact).
   Device-bound: `L_B − L_A ≈ (N_B − N_A) · t_inf`. Whole-frame→3×2:
   **≈ 5 × (6–10) ≈ +30 to +50 ms ≈ 1–1.5 frames @ 30 fps.**

3. **Overlap transient** (the only switch-only cost). During the K≈2–3 frame overlap both branches feed
   the device, so incoming B-frames see A's inferences interleaved ahead under round-robin:
   `extra ≈ N_A · t_inf ≈` **6–10 ms (~¼ frame)**, lasting only ~2–3 frames (~70–100 ms wall-clock).
   May also drop 1–2 frames on B's leaky queue if the device saturates during overlap.

**Throughput caveat (can dominate the above):** 3×2 tiles = 6 inf/frame × 30 fps = **180 inf/s**. The
Hailo-8L likely can't sustain that for 640-input yolo11n (~100–160 inf/s band), so **branch B may not hold
30 fps regardless of switching** — leaky queues drop ~every other frame, and a staler delivered frame *is*
added effective latency (up to +1 frame ≈ 33 ms). This caps usable tile count more than the switch transient does.

**Bottom line:** the switch itself adds ~sub-frame (6–10 ms for ~2–3 frames) + maybe 1–2 dropped frames on
overlap; the visible change is the steady L_A→L_B step (~+30–50 ms for whole→3×2), i.e. the price of more
tiles, not of switching. All numbers hinge on `t_inf` and the 8L throughput ceiling — both measurable by
running A-alone vs B-alone and reading the existing `detection delay` line.

## Tile budget — max tiles the current rig can afford

Each tile is resized to 640×640 and is a **full inference regardless of tile size** ⇒ N tiles = N serialized
inferences/frame. Device ceiling (Pi5 + 8L + yolo11n, PCIe **x1**) ≈ Model-Zoo 157 fps × ~0.31 (x1 derate) ≈
**~45–50 inf/s ⇒ ~20 ms device occupancy/tile**. Marginal added latency ≈ ~15–20 ms per added tile (device
portion of the ~30 ms Stage B). Estimates — not a direct tiled run.

| Tiles | Layout | Max sustainable fps (≈48/N) | Added inference latency vs whole-frame |
|---|---|---|---|
| 1 | whole | 48 (camera-capped) | 0 |
| 2 | 1×2 | 24 | +~15–20 ms |
| **4** | **2×2** | **12** | **+~45–60 ms** |
| 6 | 3×2 | 8 | +~75–100 ms |
| 9 | 3×3 | 5 | +~135–180 ms |
| 12 | 4×3 | 4 | +~180–240 ms |

- **2 tiles** comfortable (~24 fps). **4 (2×2) = realistic max** (≈12 fps, near today's 16; +45–60 ms, tolerable
  ONLY after Stage-A fix). **6 (3×2) = hard edge** (≈8 fps, over budget). **9–12 not realistic** (rate collapses).
- Levers that raise it: yolo11n→yolov8n (~1.3× on 8L, free); Hailo-8 + yolov8n (~5×, only path to dense 9–12);
  single-scale tiling saves one inference.

## Hardware option: Hailo-8 vs 8L (does the chip help latency?)

Scope first: per the latency campaign, inference (Stage B) is only ~30 ms of ~234 ms e2e p50; the
dominant ~135–170 ms is camera/ISP (Stage A). So a faster NPU barely moves **whole-frame** e2e — the chip
matters only for the **tiling** case, where NPU throughput (3×2 = 180 inf/s) is the bottleneck.

Two HEF cases:
- **Generic 8L HEF on a Hailo-8 (no recompile): no benefit.** HEF runs but HailoRT warns "compiled for
  Hailo8L … lower performance"; perf stays capped at ~8L level (allocation baked to the smaller fabric at
  compile time). A HAILO8 HEF *fails* to load on an 8L (`HAILO_INVALID_HEF`). To benefit you must recompile
  `--hw-arch hailo8`.
- **Dedicated Hailo-8 HEF (recompiled): yolo11n barely gains.** Model Zoo @640 B1 (PCIe Gen3 x4):
  yolo11n **185 fps (5.4 ms) on H8 vs 157 fps (6.4 ms) on 8L = only 1.18×** (post-proc/arch-bound, not
  compute-bound). vs yolov8n 1036 vs 202 fps = 5.13×. So recompiling yolo11n for H8 shaves ~1 ms/inf and
  lifts the ceiling only ~157→185 inf/s — does NOT fix the 180 inf/s tiling problem. **Pi 5 is PCIe x1
  (Model Zoo is x4), so real throughput is ~2–5× lower on both chips** — neither sustains 6 tiles @30 fps.

Key lever — model, not chip: **yolov8n on the existing 8L (202 fps) already beats yolo11n on 8L (157 fps)**,
and on H8 yolov8n is ~5×. So for tiling throughput:
1. **Switch yolo11n → yolov8n** — bigger win than the chip swap and free on the current 8L (cost: −1.4 mAP, 36.4 vs 37.8).
2. **Hailo-8 only pays off paired with a compute-bound model (yolov8n, ~5×)**; H8 + yolo11n is a weak buy.
3. PCIe x1 may bottleneck before the chip — measure on-rig.

Both chips lack on-chip DRAM; same-HEF design already avoids weight reload, so no 8-vs-8L difference there.

## Camera Stage-A latency (PREREQUISITE — now a separate task)

Tiling latency and camera latency **add into the same fixed budget** (e2e = StageA + StageB + StageC; control
loop needs ~50–120 ms). Today e2e p50 ≈ 234 ms with **Stage A ≈ 135–170 ms dominant** (camera/ISP depth) — the
budget is already blown, so there's no room to spend +45–60 ms on 2×2 until Stage A is cut. **Sequence: fix
Stage A first, then spend the recovered headroom on tiles.**

**This work is split into its own handoff:** see **`camera-stage-a-latency.md`** (capture 30 fps + drop
surplus newest-first; `buffer_count` 3→2 not 1; pin exposure short; cheap drops; validation plan). Target:
Stage A ~150 ms → ~60–70 ms. **That task gates this one.**

## Open question to resolve first (1-hour bench)

Do two `hailonet` elements with the **same** `hef-path` + same `vdevice-group-id` share **one** network
group, or create **two** (and thus reload on switch despite same weights)? Stand up a 2-branch
valve-gated pipeline, instrument capture→command latency across a switch, watch for a ~5 ms step.
- If **shared / no step** → one `hailonet` per branch is fine (simplest).
- If **two groups / step appears** → route all branches through a **single shared `hailonet`**
  (multiple croppers → one net via selector *before* the net, aggregator after).

---

## Next steps (when prototyping is authorized)
1. Run the bench measurement above; decide per-branch vs. shared `hailonet`.
2. Parameterize the cropper builder in `BDD/pipelines.py` to emit whole-buffer `hailocropper` **or**
   `hailotilecropper tiles-along-x-axis=… tiles-along-y-axis=…`; add a `valve` per branch + `input-selector` at merge.
3. Add `switch_branch(name)` to `app_base.py` implementing the 4-step handover.
4. Validate: steady-state latency unchanged vs. today; switch causes no command gap, ≤1–2 frames lost.

Related: latency-first design principle, latency budget campaign.
