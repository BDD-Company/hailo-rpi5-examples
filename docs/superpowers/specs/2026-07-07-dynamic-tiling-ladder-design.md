# Design: size-driven N-tier tiling ladder

**Status:** approved design (brainstorming complete) — ready for implementation plan.
**Date:** 2026-07-07. **Branch:** `tiling-size-ladder` (from `origin/main`, which contains the merged PR #9 tiling work).

## Goal

Replace the current **binary, confidence-driven** whole↔tile switch with a **config-driven
ordered ladder** of tiling tiers, driven by the **tracked target's on-screen size** (with
hysteresis), plus a **time-based loss escalation**. Reuse the existing valve / input-selector
hot-handover, generalized from 2 branches to N. Same HEF on every branch ⇒ no PCIe weight reload
on a switch (invariant preserved from the existing design).

## Context — what already exists on `origin/main`

- **2-branch valve/selector infra** — `BDD/app_base.py` `switch_tiling(to_tiling)`: a `tee` →
  `valve_whole` / `valve_tile` → `branch_selector` (input-selector) 4-step hot handover with
  warmup-wait and **graceful revert** (if the incoming branch never warms up within
  `warmup_timeout_s`, re-close its valve, keep the branch still delivering, return `False`).
- **Pure policy** — `BDD/tiling_policy.py` `TilingSwitchPolicy`: whole↔tile from **confidence
  streaks** (`lost_frames_to_tile`, `locked_frames_to_whole`, `switch_conf`). Unit-tested in
  `BDD/test_tiling_policy.py`.
- **Config** — `Config.Tiling` (`BDD/config.py`): `tiles_x/y`, `overlap`, `switchable`,
  `auto_switch`, `lost_frames_to_tile`, `locked_frames_to_whole`, `switch_conf`. Mirrored in
  `BDD/test_config.py` `TestConfig`.
- **Callback** — `BDD/app.py`: after `hailo` detections + BYTETracker `update()`,
  `_match_track_to_detection` maps tracks↔detections; currently feeds `note_detection(best_conf)`.
- **`--plus-one`** benchmark path (open both valves, infer whole+tiles each frame). **To be removed.**

## Decisions (locked in brainstorming)

1. **Ladder shape:** config-driven ordered N-tier ladder (not hardcoded 3 tiers).
2. **Size metric:** `min(bw, bh)` — the **smaller normalized side** of the tracked target's bbox
   (bbox coords normalized 0..1). Area is **not** used. Thresholds are **linear fractions of the
   frame side**. (Differs from camera switching's area-based `switch_to_wide_size`; that stays as-is.)
3. **Lost handling:** tracker-driven "lost" (no matched track this frame) → **time-based
   escalation** toward more tiles. On reacquire, resume size policy.
4. **`--plus-one`:** removed entirely (flag, `switch_tiling_plus_one`/benchmark method, wiring).

## 1. Ladder model

Ordered tiers, index `0` = **most tiles** (smallest / long-lost target) → last = **whole frame**
(index N-1, grid `1×1`). Defaults:

| tier | grid | `up_side` (→ fewer tiles) | `down_side` (→ more tiles) |
|------|------|--------------------------|-----------------------------|
| 0 | 3×2 | 0.05 (→ 2×1) | — (already max tiles) |
| 1 | 2×1 | 0.10 (→ whole) | 0.02 (→ 3×2) |
| 2 | 1×1 whole | — (top) | 0.05 (→ 2×1) |

Let `s = min(bw, bh)` for the primary tracked target. Per decision, the policy moves **at most one
tier**:
- **climb** (toward whole / fewer tiles) if `s >= tiers[current].up_side` → `current-1`
- **descend** (toward more tiles) if `s < tiers[current].down_side` → `current+1`
- else stay.

**Hysteresis** is the asymmetry between adjacent tiers' thresholds: e.g. whole→2×1 at `s < 0.05`,
but 2×1→whole at `s >= 0.10`; 2×1→3×2 at `s < 0.02`, but 3×2→2×1 at `s >= 0.05`. The gaps
(0.05..0.10 and 0.02..0.05) are dead-bands that prevent thrash on jitter.

## 2. Loss escalation (time-based, tracker-driven)

"Lost" = **no matched tracked target** this frame (via `_match_track_to_detection`). Size is then
undefined, so drive by **elapsed lost time**, escalating toward more tiles (reacquire mode):

- lost ≥ `lost_to_2x1_s` (default 1.0 s) → ensure tier ≤ 2×1 (descend if currently whole)
- lost ≥ `lost_to_3x2_s` (default 10.0 s) → tier 0 (3×2)

Loss only ever **descends** (more tiles); it never forces whole-frame. On reacquire the lost-timer
resets and the size policy resumes from the current tier. The lost timeouts are expressed against a
monotonic clock passed **into** the policy as `now_s` (see §4) so they are unit-testable without
sleeping.

**Primary target selection** when multiple tracks match: the matched track with the **largest
`min(bw, bh)`** (the most solidly-large target), consistent with the size metric. `None` when no
track matches.

## 3. Pipeline & handover — generalize 2 → N branches

**Builder** (`BDD/app_base.py`, the switchable-branch section): iterate the tier ladder, emit one
valve-gated branch per tier:

```
tee → [per tier i]  queue(leaky=downstream, max=1) → valve_tier{i}
        → cropper_i → hailonet(shared vdevice, same hef) → aggregator_i → branch_selector.sink_{i}
branch_selector (input-selector) → <common tail: tracker/callback/command>
```

- grid `1×1` → whole-buffer `hailocropper` (existing path); grid `N×M` → `hailotilecropper`
  `tiles-along-x-axis=N tiles-along-y-axis=M overlap-x-axis=<overlap> overlap-y-axis=<overlap>`.
- All `hailonet`s: **same `hef-path`, `vdevice-group-id="SHARED"`, `scheduling-algorithm=round_robin`,
  `batch-size=1`** (unchanged invariant → no reload on switch).
- Valves named `valve_tier{i}`. **Startup:** open the whole-frame tier (index N-1), all others
  `drop=true`; selector active-pad defaults to `sink_{N-1}`.

**Handover** `switch_to_tier(target_i: int, warmup_timeout_s=1.0) -> bool` (replaces
`switch_tiling(bool)`): the same proven 4 steps, indexed —
1. `valve_tier{target_i}.drop = false` (open incoming)
2. wait for first buffer out of that branch's aggregator (warmup pad-probe, `warmup_timeout_s`)
3. `branch_selector.active-pad = sink_{target_i}` (atomic)
4. close the previously-active valve (`drop = true`)

**Graceful revert preserved:** if the incoming branch never warms up in time, re-close its valve,
leave selector + outgoing valve on the branch still delivering, return `False`. Because the policy
moves one tier per decision, each switch is a single adjacent handover (brief warmup overlap only,
exactly as today).

`--plus-one` and its benchmark method are deleted.

## 4. Policy object, config schema, testing

### `BDD/tiling_policy.py` — rewrite as a ladder state machine (pure, host-importable)

```python
class TilingLadderPolicy:
    def __init__(self, tiers, lost_to_2x1_s, lost_to_3x2_s, current_i):
        # tiers: ordered list of Tier(grid, up_side, down_side); index 0 = most tiles.
        ...
    def note(self, min_side: Optional[float], now_s: float) -> Optional[int]:
        # min_side is None  => target lost => time-based escalation using (now_s - lost_since_s)
        # min_side is float => size hysteresis vs tiers[current].up_side / down_side
        # returns the tier index to switch TO (one step), or None to stay.
    def committed(self, tier_i: int):   # adopt tier_i as active + reset streak/lost state
```

- `now_s` and `min_side` are **parameters**, not read inside the policy — keeps loss timers and
  hysteresis deterministically unit-testable.
- The `_fire_switch` worker-thread wrapper in `app.py` (single in-flight switch, `_switch_pending`
  lock, revert-on-abort backoff) stays, retargeted from `switch_tiling(bool)` to
  `switch_to_tier(i)` / `note(min_side, now_s)`.

### Callback wiring (`BDD/app.py`)

After tracking, compute the primary target's `min_side = min(bw, bh)` (largest-`min_side` matched
track) or `None` if no matched track, and the current `now_s` (monotonic). Replace the
`note_detection(best_conf)` call with `user_data.note_tiling(min_side, now_s)`.

### Config (`BDD/config.py`, mirror into `test_config.py`)

```yaml
tiling:
  switchable: true
  auto_switch: true          # size + loss ladder policy on
  overlap: 0.10              # applied to every tiled branch
  lost_to_2x1_s: 1.0
  lost_to_3x2_s: 10.0
  ladder:                    # ordered: most tiles -> whole; last MUST be 1x1
    - {grid: [3, 2], up_side: 0.05}
    - {grid: [2, 1], up_side: 0.10, down_side: 0.02}
    - {grid: [1, 1], down_side: 0.05}
```

**Validation:** ladder non-empty; last tier grid `== [1,1]`; tile counts strictly decreasing down
the list (index 0 most tiles); every threshold in `[0,1]`. Threshold presence follows position:
tier 0 (most tiles) has **no `down_side`** (nothing below it); the last tier (whole) has **no
`up_side`** (nothing above it); every middle tier has **both**.

**Back-compat:** if `ladder:` is absent, synthesize a 2-tier ladder `[{grid:[tiles_x,tiles_y]},
{grid:[1,1]}]` from the legacy fields so existing configs still run. Legacy
`lost_frames_to_tile`/`locked_frames_to_whole`/`switch_conf` are retained-but-unused by the new
policy when `ladder:` is present (kept parseable so old configs don't error).

### Testing

- `BDD/test_tiling_policy.py` (host-only, no GStreamer/hailo):
  - hysteresis at each boundary — a jitter band around a threshold produces no switch;
  - one-tier-per-decision (a huge size jump still steps a single tier per `note`);
  - loss escalation fires at `lost_to_2x1_s` and `lost_to_3x2_s` using injected `now_s`;
  - reacquire resets the lost timer and resumes size policy;
  - `committed()` adopts the tier and clears transient state.
- `BDD/test_config.py`: ladder parsing, validation errors (bad last tier, non-monotonic grids,
  out-of-range thresholds), and legacy-field back-compat synthesis.

### On-device validation (the rig) — commit after each passes

1. Config schema + validation + host tests. *(host-only; no rig)*
2. `TilingLadderPolicy` + host unit tests. *(host-only; no rig)*
3. N-branch pipeline + `switch_to_tier` handover; `--plus-one` removed. **Rig:** pipeline builds &
   runs whole-frame active; `--switch-test-s N` cycles all tiers glitch-free (no command gap, graceful
   revert on a dead branch).
4. Callback → policy wiring. **Rig:** auto size-switch with a target moving toward/away from the
   camera walks the ladder with hysteresis; removing the target escalates 2×1 @1 s then 3×2 @10 s;
   reacquire returns to size policy.

## Incremental commit plan

1. Config schema + ladder validation + tests (host-only).
2. `TilingLadderPolicy` + unit tests (host-only).
3. Pipeline generalization to N branches + `switch_to_tier` handover; remove `--plus-one`; validate
   build + manual `--switch-test-s` cycle on rig.
4. Wire callback (tracked-target `min_side` + loss timing) → policy → `_fire_switch`; validate auto
   size-switch + loss escalation on rig.

## Out of scope / non-goals

- Throughput: 3×2 = 6 inf/frame won't hold 30 fps on the rig (per the runtime-tiling design doc).
  Acceptable — the ladder only sits at 3×2 for tiny/long-lost targets and climbs back to whole for
  locked control. No model/HEF change here.
- No change to camera switching (`switch_to_wide_size`/`switch_to_zoom_size`, area-based) — separate
  subsystem, deliberately left as-is.
- No continuous/unbounded tile geometry — discrete pre-built branches only.
