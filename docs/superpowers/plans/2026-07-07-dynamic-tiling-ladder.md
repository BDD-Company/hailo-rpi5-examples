# Dynamic Tiling Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary confidence-driven whole↔tile switch with a config-driven N-tier tiling ladder driven by the tracked target's on-screen size (with hysteresis) plus a time-based loss escalation.

**Architecture:** A pure, host-testable `TilingLadderPolicy` decides *which tier* to switch to from `side = max(bw,bh)` of the primary tracked target (or elapsed lost-time when no target). `app.py`'s callback feeds it and fires the switch on a worker thread. `app_base.py` builds one valve-gated inference branch per ladder tier behind a shared-hef `input-selector`; `switch_to_tier(i)` performs the existing warmup-wait + graceful-revert handover, generalized from 2 branches to N. `--plus-one` is removed.

**Tech Stack:** Python 3, GStreamer (`hailocropper`/`hailotilecropper`/`input-selector`/`valve`), HailoRT, frozen-slotted-dataclass config with declarative `parse_config` validation, pytest (host-only for policy + config).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-07-dynamic-tiling-ladder-design.md` — authoritative.
- **Branch:** `tiling-size-ladder` (already created from `origin/main`).
- **Size metric:** `side = max(bw, bh)` — larger normalized side (0..1) of the primary tracked target. Area is NOT used.
- **Ladder order:** index `0` = most tiles (smallest/long-lost target); last index = whole-frame (`1×1`).
- **Tier motion:** at most ONE tier per policy decision. Climb (toward whole) when `side >= tiers[cur].up_side`; descend (more tiles) when `side < tiers[cur].down_side`.
- **Loss:** tracker-driven ("no matched track this frame"). At `lost_to_2x1_s` (default 1.0 s) drop to the rung just above whole (index `len-2`); at `lost_to_3x2_s` (default 10.0 s) drop to tier 0 (most tiles). Loss only ever ADDS tiles (moves toward index 0), never climbs toward whole; the lost timer is absolute — measured from when the target was lost, NOT reset by an intermediate switch.
- **Index direction:** index 0 = most tiles, last = whole. `note()` returns `current_i + 1` to add *fewer* tiles (climb toward whole) and `current_i - 1` to add *more* tiles (descend). Do not invert this.
- **Invariant:** every branch uses the SAME hef + `vdevice-group-id` (shared) + `scheduling-algorithm=1` (round-robin) → no PCIe weight reload on a switch. Do not change this.
- **Policy purity:** `TilingLadderPolicy` and `build_ladder` take `side` and `now_s` as PARAMETERS (no `time.monotonic()` inside) so loss timers are unit-testable without sleeping. No hailo/GStreamer imports in `tiling_policy.py`.
- **Repo rule (memory `config-testconfig-mirror`):** any new `parse_config` capability must be mirrored into `test_config.py`'s `TestConfig` with a test.
- **Tests run from `BDD/`:** `cd BDD && python -m pytest ...` (imports are bare, e.g. `from tiling_policy import ...`).
- **Commit after each task.** Tasks 3 & 4 also require on-rig validation before their commit (call it out; if the rig is unavailable, stop and report — do not claim success).

---

## File Structure

- `BDD/tiling_policy.py` — **rewrite**: `LadderTier` namedtuple, `TilingLadderPolicy` state machine, `build_ladder(tiling_cfg)` (validation; empty ladder → `[]`). Pure/host-importable.
- `BDD/test_tiling_policy.py` — **rewrite**: hysteresis, one-step, loss-escalation (injected `now_s`), reacquire, `committed`, `build_ladder` validation + empty-ladder.
- `BDD/config.py` — **modify** `Config.Tiling`: add nested `Tier` + `ladder: list[Tier]`, `lost_to_2x1_s`, `lost_to_3x2_s`; keep `overlap`, `auto_switch`; **DELETE** legacy fields `tiles_x/tiles_y/switchable/switch_conf/lost_frames_to_tile/locked_frames_to_whole` (ladder-only, no back-compat).
- `BDD/test_config.py` — **modify**: mirror the "list of present-by-default nested dataclasses with Optional fields" parser feature into `TestConfig` + a test.
- `BDD/app_base.py` — **modify**: extract pure `build_switchable_detection_section(...)`; generalize `switch_tiling(bool)` → `switch_to_tier(int)`; delete `enable_plus_one`; App ctor drops `tiles`/`tiles_x/tiles_y` (static path removed → non-switchable = whole-frame) and accepts `ladder_grids`.
- `BDD/test_app_base_pipeline.py` — **create**: host test for the pure branch-string builder.
- `BDD/app.py` — **modify** (Task 3): rewrite `user_app_callback_class` policy wrapper (`configure_tiling_policy`, `note_tiling`, `_fire_switch(int)`); callback computes `side=max(w,h)` of the primary tracked target; `main()` builds the ladder, derives `switchable=len(grids)>=2`, wires the policy, cycles tiers via `--switch-test-s`; removes `--plus-one`/`--tiles`/`--switch-tiles` + the static path.
- `BDD/config.yaml` — **modify**: rewrite the `tiling:` section to the surviving fields (Task 1) + enable/tune the ladder (Task 4).

---

### Task 1: Config schema — ladder tiers + loss timeouts

**Files:**
- Modify: `BDD/config.py` (`Config.Tiling`, ~line 157-181)
- Modify: `BDD/test_config.py` (`TestConfig`, add a nested-list mirror + test)
- Modify: `BDD/config.yaml` (commented example)
- Test: `BDD/test_config.py`

**Interfaces:**
- Produces (surviving `Config.Tiling` after cleanup): `Config.Tiling.Tier(tiles_x:int, tiles_y:int, up_side:Optional[float], down_side:Optional[float])`; `ladder: list[Tier]` (default `[]`); `overlap: float = 0.0`; `auto_switch: bool = False`; `lost_to_2x1_s: float = 1.0`; `lost_to_3x2_s: float = 10.0`.
- DELETED from `Config.Tiling` (Decision 5, ladder-only): `tiles_x`, `tiles_y`, `switchable`, `switch_conf`, `lost_frames_to_tile`, `locked_frames_to_whole`. "Switchable" is derived downstream as `len(ladder) >= 2`.

- [ ] **Step 1: Write the failing test** — append to `BDD/test_config.py` (after the existing tiling/defaults tests). Construct `Config.Tiling` DIRECTLY (no full-config YAML needed — the real `Config` requires many sections + a present HEF, absent on host; the parser machinery for a list-of-nested-tiers is proven separately in Step 5 via `TestConfig`). This pins the new fields + defaults:

```python
def test_tiling_tier_and_ladder_fields():
    from config import Config
    Tier = Config.Tiling.Tier
    t = Config.Tiling(
        auto_switch=True, overlap=0.10,
        lost_to_2x1_s=1.0, lost_to_3x2_s=10.0,
        ladder=[
            Tier(tiles_x=3, tiles_y=2, up_side=0.05, down_side=None),
            Tier(tiles_x=2, tiles_y=1, up_side=0.10, down_side=0.02),
            Tier(tiles_x=1, tiles_y=1, up_side=None, down_side=0.05),
        ],
    )
    assert len(t.ladder) == 3
    assert (t.ladder[0].tiles_x, t.ladder[0].tiles_y) == (3, 2)
    assert t.ladder[0].up_side == 0.05 and t.ladder[0].down_side is None
    assert t.ladder[2].down_side == 0.05 and t.ladder[2].up_side is None
    assert t.lost_to_2x1_s == 1.0 and t.lost_to_3x2_s == 10.0


def test_tiling_defaults_empty_ladder():
    from config import Config
    d = Config.Tiling()                     # all-defaults (built by the field factory)
    assert d.ladder == []
    assert d.lost_to_2x1_s == 1.0 and d.lost_to_3x2_s == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd BDD && python -m pytest test_config.py::test_tiling_tier_and_ladder_fields -v`
Expected: FAIL (`Config.Tiling` has no attribute `Tier` / `ladder`).

- [ ] **Step 3: Rewrite `Config.Tiling`** in `BDD/config.py` (~line 157-181): DELETE the legacy fields (`tiles_x`, `tiles_y`, `switchable`, `switch_conf`, `lost_frames_to_tile`, `locked_frames_to_whole`) and their comments; keep `overlap` + `auto_switch`; add the nested `Tier`, `ladder`, and loss timeouts. Result is the WHOLE new dataclass:

```python
    @dataclass(slots=True, kw_only=True, frozen=True)
    class Tiling:
        # Ladder-only tiling control (Decision 5): the size-driven ladder is the sole
        # tiling mechanism. Legacy fields (tiles_x, tiles_y, switchable, switch_conf,
        # lost_frames_to_tile, locked_frames_to_whole) and the static --tiles path are
        # REMOVED. "Switchable" is derived at runtime as len(ladder) >= 2.

        @dataclass(slots=True, kw_only=True, frozen=True)
        class Tier:
            # One rung of the ladder. tiles_x×tiles_y = the grid this rung runs
            # (1×1 = whole-frame). up_side/down_side are the max(bw,bh) linear-fraction
            # thresholds for leaving this rung: climb toward whole when side >= up_side,
            # descend toward more tiles when side < down_side. End rungs omit the
            # threshold that runs off the ladder (tier 0 = most tiles has no down_side;
            # the whole-frame rung has no up_side).
            tiles_x: Annotated[int, Range(min=1)] = 1
            tiles_y: Annotated[int, Range(min=1)] = 1
            # Optional fields must NOT declare a default (check_schema rule — the parser
            # auto-Nones on omission). So no `= None`; direct construction passes them.
            up_side:   Optional[Annotated[float, Range(0.0, 1.0)]]
            down_side: Optional[Annotated[float, Range(0.0, 1.0)]]

        # Ordered ladder (index 0 = MOST tiles -> last = whole-frame). Empty/absent =
        # plain whole-frame (no tiling, not switchable). Structural validation lives in
        # tiling_policy.build_ladder (last rung 1×1, strictly-decreasing tile counts).
        ladder: list[Tier] = field(default_factory=list)
        # Fractional tile overlap on both axes (0..1); applied to every tiled branch.
        overlap: Annotated[float, Range(0.0, 1.0)] = 0.0
        # Run the size+loss policy automatically. When false the ladder pipeline is
        # still built (if len(ladder)>=2) but only --switch-test-s drives switching.
        auto_switch: bool = False
        # Target-lost escalation: drop to the rung above whole after lost_to_2x1_s of
        # no matched track, then to tier 0 (most tiles) after lost_to_3x2_s. Seconds.
        lost_to_2x1_s: Annotated[float, Range(min=0.0)] = 1.0
        lost_to_3x2_s: Annotated[float, Range(min=0.0)] = 10.0
    tiling: Tiling = field(default_factory=Tiling)
```

Ensure `Optional` is imported in `config.py` (it already imports from `typing` — verify; add if not). `Tier` must be defined BEFORE the `ladder` field references it.

> **Grep for stale readers before moving on:** `grep -rn "\.tiles_x\|\.tiles_y\|\.switchable\|switch_conf\|lost_frames_to_tile\|locked_frames_to_whole" BDD/*.py` — every hit outside a comment must be updated or removed in Tasks 3/4 (App ctor, get_pipeline_string, main() wiring, the old policy block). The config change lands first; the code that read those fields is fixed in the later tasks, so `app.py`/`app_base.py` may not run until Task 3/4 — that's expected (host config tests still pass).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd BDD && python -m pytest test_config.py::test_tiling_tier_and_ladder_fields test_config.py::test_tiling_defaults_empty_ladder -v`
Expected: PASS.

- [ ] **Step 5: Mirror the parser feature into `TestConfig`** (repo rule `config-testconfig-mirror`). Add a list-of-defaulted-nested-dataclass section to `TestConfig` in `test_config.py` and a test that it parses through the REAL loader (`covert_to_config` → `loads_config`), pinning the generic parser capability independently of `Config`. Add the nested class + field inside the `TestConfig` body (near `DefaultsSection`):

```python
        @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
        class Rung:
            # Mirrors Config.Tiling.Tier: a nested dataclass used as a LIST element
            # where every field is defaulted (incl. an Optional), so the list may be
            # omitted entirely (default_factory=list) — the exact shape `ladder` uses.
            a: Annotated[int, Range(min=1)] = 1
            lo: Optional[Annotated[float, Range(0.0, 1.0)]]   # no default (check_schema rule)
        rungs: list[Rung] = dataclasses.field(default_factory=list)
```

Add near the other aliases (~line 121): `Rung = TestConfig.Rung`. Then a test using the file's existing helpers (`covert_to_config`, `tvalid_with`, both already imported/defined in the file):

```python
def test_optional_list_of_defaulted_nested_dataclasses_parses():
    cfg = covert_to_config(tvalid_with(rungs=[{'a': 2, 'lo': 0.3}, {'a': 1}]))
    assert len(cfg.rungs) == 2
    assert (cfg.rungs[0].a, cfg.rungs[0].lo) == (2, 0.3)
    assert cfg.rungs[1].lo is None            # Optional defaults to None

def test_omitted_list_of_nested_dataclasses_defaults_empty():
    assert covert_to_config(tvalid()).rungs == []

def test_nested_list_element_range_is_validated():
    import pytest
    from parse_config import ConfigError
    with pytest.raises(ConfigError):
        covert_to_config(tvalid_with(rungs=[{'a': 0}]))   # a has Range(min=1)
```

- [ ] **Step 6: Run the full config test file**

Run: `cd BDD && python -m pytest test_config.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 7: Rewrite the `tiling:` section of `BDD/config.yaml`** (currently ~lines 37-58). The live file still sets the DELETED fields (`tiles_x`, `tiles_y`, `switchable`, `lost_frames_to_tile`, `locked_frames_to_whole`, `switch_conf`) — the strict parser rejects unknown keys, so those lines MUST go or the app won't start. Replace the whole section with the surviving fields (defaults = whole-frame, ladder commented so the shipped default stays low-latency whole-frame):

```yaml
# Inference tiling — size-driven LADDER only (index 0 = most tiles -> last = whole).
# Empty/absent ladder = plain whole-frame (lowest latency, the default). "Switchable"
# is derived: a ladder with >= 2 rungs builds the N-branch hot-switch pipeline.
tiling:
  # Fractional tile overlap on both axes (0..1); applied to every tiled branch.
  overlap: 0.10
  # Run the size+loss policy automatically. When false the pipeline is still built
  # (if the ladder has >= 2 rungs) but only --switch-test-s drives switching.
  auto_switch: false
  # Target-lost escalation (seconds): drop to the rung above whole after lost_to_2x1_s
  # of no matched track, then to tier 0 (most tiles) after lost_to_3x2_s.
  lost_to_2x1_s: 1.0
  lost_to_3x2_s: 10.0
  # Size-driven ladder. side = max(bbox_w, bbox_h) of the tracked target (0..1).
  # Climb toward whole when side >= up_side; descend toward more tiles when
  # side < down_side. Last rung MUST be 1x1. Uncomment + set auto_switch: true to use.
  # ladder:
  #   - {tiles_x: 3, tiles_y: 2, up_side: 0.05}
  #   - {tiles_x: 2, tiles_y: 1, up_side: 0.10, down_side: 0.02}
  #   - {tiles_x: 1, tiles_y: 1, down_side: 0.05}
```

Verify nothing else in `config.yaml` references the removed keys (`grep -n "tiles_x\|tiles_y\|switchable\|switch_conf\|lost_frames\|locked_frames" BDD/config.yaml` → no hits).

- [ ] **Step 8: Commit**

```bash
git add BDD/config.py BDD/test_config.py BDD/config.yaml
git commit -m "feat(tiling): config schema for size-driven ladder + loss timeouts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `TilingLadderPolicy` + `build_ladder` (pure, host-tested)

**Files:**
- Rewrite: `BDD/tiling_policy.py`
- Rewrite: `BDD/test_tiling_policy.py`

**Interfaces:**
- Consumes: `Config.Tiling` (duck-typed: reads `.ladder`, `.lost_to_2x1_s`, `.lost_to_3x2_s`). No legacy fields.
- Produces:
  - `LadderTier = namedtuple("LadderTier", "tiles_x tiles_y up_side down_side")`
  - `build_ladder(tiling) -> list[LadderTier]` — returns `[]` when `tiling.ladder` is empty (= whole-frame, not switchable); otherwise maps + validates it (raises `ValueError` on a malformed ladder). No legacy synthesis.
  - `TilingLadderPolicy(tiers: list[LadderTier], lost_to_2x1_s: float, lost_to_3x2_s: float, current_i: int)`
    - `note(side: Optional[float], now_s: float) -> Optional[int]` — target tier index to switch TO (one step), or `None`.
    - `committed(tier_i: int)` — adopt tier + reset lost/transient state.
    - attribute `current_i: int` (active tier index).

- [ ] **Step 1: Write the failing tests** — replace `BDD/test_tiling_policy.py` entirely:

```python
"""Unit tests for the size-driven tiling ladder policy (host-runnable; pure — no
hailo/GStreamer). Pins the tier decision that user_app_callback_class.note_tiling
delegates to, plus build_ladder's validation and empty-ladder handling.

side = max(bbox_w, bbox_h) of the primary tracked target (0..1); None = lost.
now_s is an injected monotonic timestamp so loss timers test without sleeping.
"""
import pytest
from tiling_policy import TilingLadderPolicy, LadderTier, build_ladder

# Default 3-rung ladder: index 0 = 3x2 (most tiles) -> 2 = whole.
def ladder3():
    return [
        LadderTier(3, 2, up_side=0.05, down_side=None),
        LadderTier(2, 1, up_side=0.10, down_side=0.02),
        LadderTier(1, 1, up_side=None, down_side=0.05),
    ]

def policy(current_i, l2=1.0, l10=10.0):
    return TilingLadderPolicy(ladder3(), lost_to_2x1_s=l2, lost_to_3x2_s=l10, current_i=current_i)

# --- size hysteresis: one tier per decision, dead-bands don't thrash ----------
def test_whole_descends_to_2x1_below_down_side():
    p = policy(current_i=2)                 # whole
    assert p.note(0.04, now_s=0.0) == 1     # side < 0.05 -> descend to 2x1
    assert p.note(0.03, now_s=0.0) == 1     # not yet committed -> keeps asking

def test_whole_holds_in_deadband():
    p = policy(current_i=2)                 # whole; leaves only at side<0.05
    assert p.note(0.07, now_s=0.0) is None  # 0.05..0.10 dead-band -> stay whole

def test_2x1_climbs_to_whole_at_up_side():
    p = policy(current_i=1)
    assert p.note(0.10, now_s=0.0) == 2     # side >= 0.10 -> climb to whole
    assert p.note(0.20, now_s=0.0) == 2

def test_2x1_descends_to_3x2_below_down_side():
    p = policy(current_i=1)
    assert p.note(0.01, now_s=0.0) == 0     # side < 0.02 -> descend to 3x2

def test_2x1_holds_between_thresholds():
    p = policy(current_i=1)                 # leaves at >=0.10 up or <0.02 down
    assert p.note(0.05, now_s=0.0) is None

def test_one_tier_per_decision_even_on_huge_jump():
    p = policy(current_i=0)                 # 3x2, tiny target suddenly huge
    assert p.note(0.9, now_s=0.0) == 1      # only ONE step (to 2x1), not to whole
    p.committed(1)
    assert p.note(0.9, now_s=0.0) == 2      # next decision climbs one more

# --- loss escalation (time-based, injected now_s) -----------------------------
def test_lost_descends_after_1s_then_after_10s():
    p = policy(current_i=2)                 # whole
    assert p.note(None, now_s=0.0) is None  # just became lost
    assert p.note(None, now_s=0.5) is None  # < 1s
    assert p.note(None, now_s=1.0) == 1     # >= lost_to_2x1_s -> descend one rung
    p.committed(1)
    assert p.note(None, now_s=9.9) is None  # < 10s from lost start, already 2x1
    assert p.note(None, now_s=10.0) == 0    # >= lost_to_3x2_s -> tier 0 (3x2)

def test_lost_never_climbs_toward_whole():
    p = policy(current_i=0)                 # already max tiles
    assert p.note(None, now_s=100.0) is None  # loss only descends; nothing to do

def test_reacquire_resets_lost_timer_and_resumes_size():
    p = policy(current_i=2)                 # whole
    p.note(None, now_s=0.0)                 # lost starts at t=0
    assert p.note(0.5, now_s=0.5) is None   # target back, large -> stay whole
    # a fresh loss must re-time from now, not from the old t=0.
    assert p.note(None, now_s=0.6) is None
    assert p.note(None, now_s=1.6) == 1     # 0.6 -> 1.6 is >= 1s

# --- committed() --------------------------------------------------------------
def test_committed_adopts_tier():
    p = policy(current_i=2)
    p.note(0.04, now_s=0.0)                 # asks for 2x1
    p.committed(1)
    assert p.current_i == 1
    assert p.note(0.10, now_s=0.0) == 2     # reverse threshold from the new rung

# --- build_ladder: validation + empty-ladder ----------------------------------
class _Cfg:  # duck-typed stand-in for Config.Tiling
    def __init__(self, ladder=None, l2=1.0, l10=10.0):
        self.ladder = ladder or []
        self.lost_to_2x1_s, self.lost_to_3x2_s = l2, l10

class _T:  # duck-typed Config.Tiling.Tier
    def __init__(self, tx, ty, up=None, down=None):
        self.tiles_x, self.tiles_y, self.up_side, self.down_side = tx, ty, up, down

def test_build_ladder_uses_explicit_ladder():
    tiers = build_ladder(_Cfg(ladder=[_T(3,2,up=0.05), _T(2,1,up=0.10,down=0.02), _T(1,1,down=0.05)]))
    assert [(t.tiles_x, t.tiles_y) for t in tiers] == [(3,2),(2,1),(1,1)]

def test_build_ladder_empty_returns_empty():
    assert build_ladder(_Cfg(ladder=[])) == []      # whole-frame only, not switchable

def test_build_ladder_rejects_last_not_whole():
    with pytest.raises(ValueError):
        build_ladder(_Cfg(ladder=[_T(3,2,up=0.05), _T(2,1,down=0.02)]))  # last is 2x1

def test_build_ladder_rejects_non_decreasing_tile_counts():
    with pytest.raises(ValueError):
        build_ladder(_Cfg(ladder=[_T(2,1,up=0.05), _T(3,2,up=0.10,down=0.02), _T(1,1,down=0.05)]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd BDD && python -m pytest test_tiling_policy.py -v`
Expected: FAIL / import error (`TilingLadderPolicy`, `LadderTier`, `build_ladder` undefined).

- [ ] **Step 3: Rewrite `BDD/tiling_policy.py`**:

```python
"""Size-driven tiling ladder policy — pure decision logic, no GStreamer/hailo.

Host-importable and unit-testable (app.py pulls in hailo/GStreamer). Decides WHICH
inference tier to hot-switch to, given the tracked target's on-screen size
(side = max(bbox_w, bbox_h), normalized 0..1) or elapsed lost-time when no target.
The actual switch (threads + GStreamer valve/input-selector handover) stays in
app.py / app_base.py.

Ladder order: index 0 = MOST tiles (smallest / long-lost target) -> last = whole.
"""
from collections import namedtuple
from typing import Optional

LadderTier = namedtuple("LadderTier", "tiles_x tiles_y up_side down_side")


def build_ladder(tiling) -> list:
    """Return the ordered ladder (list[LadderTier]) from a Config.Tiling.

    The ladder is the SOLE source (no legacy synthesis). An empty tiling.ladder means
    plain whole-frame (not switchable) -> returns []. A non-empty ladder is mapped and
    validated; raises ValueError on a malformed ladder.
    """
    if not getattr(tiling, "ladder", None):
        return []                                  # whole-frame only, not switchable
    tiers = [LadderTier(t.tiles_x, t.tiles_y, t.up_side, t.down_side)
             for t in tiling.ladder]

    if (tiers[-1].tiles_x, tiers[-1].tiles_y) != (1, 1):
        raise ValueError("tiling ladder: last rung must be whole-frame (1x1), "
                         f"got {tiers[-1].tiles_x}x{tiers[-1].tiles_y}")
    counts = [t.tiles_x * t.tiles_y for t in tiers]
    if any(b >= a for a, b in zip(counts, counts[1:])):
        raise ValueError(f"tiling ladder: tile counts must strictly decrease "
                         f"(index 0 = most tiles); got {counts}")
    return tiers


class TilingLadderPolicy:
    """Per-frame tier decision from target size (hysteresis) + lost-time escalation.

    Feed :meth:`note` the primary target's ``side`` (max(bw,bh), 0..1) or ``None``
    when no target is matched, plus a monotonic ``now_s``. It returns the tier index
    to switch TO (always ONE step from the current tier) or ``None`` to stay. The
    switch is asynchronous, so call :meth:`committed` once it lands.

    Size (target present): climb toward whole (fewer tiles) when
    ``side >= tiers[current].up_side``; descend (more tiles) when
    ``side < tiers[current].down_side``. The asymmetric adjacent thresholds are the
    hysteresis dead-band that prevents thrash.

    Loss (no target): after ``lost_to_2x1_s`` descend one rung; after
    ``lost_to_3x2_s`` go to tier 0 (most tiles). Loss never climbs toward whole.
    """

    def __init__(self, tiers, lost_to_2x1_s, lost_to_3x2_s, current_i):
        self.tiers = list(tiers)
        self.lost_to_2x1_s = lost_to_2x1_s
        self.lost_to_3x2_s = lost_to_3x2_s
        self.current_i = int(current_i)
        self._lost_since_s = None     # None => target currently present

    def note(self, side: Optional[float], now_s: float) -> Optional[int]:
        if side is None:
            return self._note_lost(now_s)
        self._lost_since_s = None         # target present -> clear the lost timer
        cur = self.tiers[self.current_i]
        # index 0 = MOST tiles, last = whole. Bigger target (side up) => FEWER tiles
        # => climb toward a HIGHER index. Smaller target (side down) => MORE tiles
        # => descend toward a LOWER index.
        if self.current_i < len(self.tiers) - 1 and cur.up_side is not None \
                and side >= cur.up_side:
            return self.current_i + 1        # fewer tiles (toward whole)
        if self.current_i > 0 and cur.down_side is not None and side < cur.down_side:
            return self.current_i - 1        # more tiles
        return None

    def _note_lost(self, now_s: float) -> Optional[int]:
        if self._lost_since_s is None:
            self._lost_since_s = now_s
        lost_for = now_s - self._lost_since_s
        n = len(self.tiers)
        # Elapsed-time -> target rung (ABSOLUTE, keyed to time not the current tier):
        #   >= lost_to_3x2_s -> tier 0 (most tiles);
        #   >= lost_to_2x1_s -> the rung just above whole (index n-2).
        # Only act if the target adds tiles (strictly LOWER index than current).
        target = None
        if lost_for >= self.lost_to_3x2_s:
            target = 0
        elif lost_for >= self.lost_to_2x1_s:
            target = max(n - 2, 0)
        if target is not None and target < self.current_i:
            return target
        return None

    def committed(self, tier_i: int):
        # Adopt the tier. Do NOT clear the lost timer here: if the target is still
        # lost the escalation must keep timing from the ORIGINAL loss, so the 10 s
        # rung fires 10 s after loss (not 10 s after the 1 s switch). note() clears
        # the timer the instant a target reappears.
        self.current_i = int(tier_i)
```

> **Direction/timer invariants (do not "simplify" away):** index 0 = most tiles, last = whole, so `note()` returns `current_i + 1` to add *fewer* tiles (climb) and `current_i - 1` to add *more* tiles (descend). Loss maps its two timeouts to FIXED rungs (index `n-2` at `lost_to_2x1_s`, index `0` at `lost_to_3x2_s`) and only ever moves toward more tiles; `committed()` deliberately does NOT reset `_lost_since_s`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd BDD && python -m pytest test_tiling_policy.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add BDD/tiling_policy.py BDD/test_tiling_policy.py
git commit -m "feat(tiling): size-driven ladder policy + build_ladder (pure, host-tested)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: N-branch pipeline + `switch_to_tier`; remove `--plus-one`

**Files:**
- Modify: `BDD/app_base.py` (`get_pipeline_string` switchable section ~1430-1446; `switch_tiling` ~367-419; delete `enable_plus_one` ~421-444; App ctor ~422/1284-1293)
- Create: `BDD/test_app_base_pipeline.py`
- Modify: `BDD/app.py` (App ctor forwarding of ladder grids; remove `--plus-one` flag + `_enable_plus_one` wiring ~603-605, ~805-811)

**Interfaces:**
- Consumes: `LadderTier` list from Task 2 (via `main()` in Task 4; here App accepts `ladder_grids: list[tuple[int,int]]`).
- Produces:
  - Pure `build_switchable_detection_section(grids, branch_factory) -> str` at module scope in `app_base.py`. `grids` is `[(tx,ty), ...]` ladder order (index 0 = most tiles, last = (1,1) whole). `branch_factory(name, tx, ty) -> str` returns one inference-wrapper branch string. Emits `valve_tier{i}` + `branch_selector.sink_{i}`; opens the LAST valve (whole, index N-1) at startup (`drop=false`), all others `drop=true`.
  - `App.switch_to_tier(target_i: int, warmup_timeout_s: float = 1.0) -> bool` replacing `switch_tiling`.

- [ ] **Step 1: Write the failing host test** for the pure string builder — create `BDD/test_app_base_pipeline.py`:

```python
"""Host test for the switchable N-branch detection-section string builder.
Pure string assembly — no GStreamer instantiation, so it runs off-device."""
from app_base import build_switchable_detection_section

def _fake_branch(name, tx, ty):
    return f"[{name}:{tx}x{ty}]"

def test_builds_one_valve_and_selector_pad_per_tier():
    grids = [(3, 2), (2, 1), (1, 1)]          # index 0 = most tiles, last = whole
    s = build_switchable_detection_section(grids, _fake_branch)
    for i in range(3):
        assert f"valve_tier{i}" in s
        assert f"branch_selector.sink_{i}" in s
    assert "input-selector name=branch_selector" in s

def test_only_whole_valve_open_at_startup():
    grids = [(3, 2), (2, 1), (1, 1)]
    s = build_switchable_detection_section(grids, _fake_branch)
    # whole is the LAST tier (index 2): its valve starts open, the tile valves closed.
    assert "valve name=valve_tier2 drop=false" in s
    assert "valve name=valve_tier0 drop=true" in s
    assert "valve name=valve_tier1 drop=true" in s

def test_two_tier_ladder_still_builds():
    s = build_switchable_detection_section([(2, 1), (1, 1)], _fake_branch)
    assert "valve_tier0" in s and "valve_tier1" in s
    assert "valve name=valve_tier1 drop=false" in s   # whole open
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd BDD && python -m pytest test_app_base_pipeline.py -v`
Expected: FAIL (`build_switchable_detection_section` undefined).

- [ ] **Step 3: Add the pure builder at module scope in `BDD/app_base.py`** (near the top, after imports / `QUEUE` is importable there):

```python
def build_switchable_detection_section(grids, branch_factory):
    """Assemble the switchable N-branch detection section string.

    grids: ordered [(tiles_x, tiles_y), ...]; index 0 = most tiles, last = (1,1) whole.
    branch_factory(name, tx, ty) -> one inference-wrapper branch string.

    One valve-gated branch per tier merges at a single input-selector, tier i on
    sink_i. At startup only the LAST tier's valve (whole-frame) is open (drop=false),
    every tile valve is closed (drop=true). The selector's request-pad default active
    pad is sink_0, which does NOT match the open (whole) valve, so App must set
    active-pad = sink_{last} right after the pipeline is built (see
    _init_selector_startup_pad). switch_to_tier(i) does the runtime handover.
    """
    parts = ['tee name=branch_src_tee ']
    last = len(grids) - 1
    for i, (tx, ty) in enumerate(grids):
        drop = 'false' if i == last else 'true'
        branch = branch_factory(f'tier{i}', tx, ty)
        parts.append(
            f'branch_src_tee. ! {QUEUE(name=f"tier{i}_gate_q")} ! '
            f'valve name=valve_tier{i} drop={drop} ! '
            f'{branch} ! branch_selector.sink_{i} '
        )
    parts.append('input-selector name=branch_selector sync-streams=false ! ')
    return ''.join(parts)
```

> **Selector startup pad:** `input-selector`'s default active-pad is the first-requested pad (`sink_0` = tier 0 = most tiles), but only the whole valve is open, so only whole buffers flow. To make the ACTIVE pad match the open valve at startup, App must set `active-pad = sink_{last}` right after `create_pipeline()`. Add that in Step 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd BDD && python -m pytest test_app_base_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the builder into `get_pipeline_string` and generalize the App** in `BDD/app_base.py`:

  1. **App ctor** — DELETE the `tiles` param and the `self.tiles_x/self.tiles_y` assignment (lines ~1284-1286); the static grid is gone. Add a `ladder_grids=None` param (keep `tiling_overlap` + `switchable_tiling`). Store `self.ladder_grids = [tuple(map(int, g)) for g in ladder_grids] if ladder_grids else None`. (`switchable_tiling` is a plain bool computed by `main()` as `len(ladder_grids) >= 2` — App just receives it.)

  2. **`get_pipeline_string` switchable block** — replace the two-branch literal (lines ~1430-1446) with (no static fallback — when switchable, `self.ladder_grids` is guaranteed len ≥ 2):

```python
        if self.switchable_tiling:
            # N valve-gated branches (one per ladder tier) merge at one input-selector.
            # Whole-frame (last tier) active at startup; switch_to_tier() does the
            # warmup-wait + graceful-revert handover. Same hef on every hailonet
            # (shared vdevice, scheduling-algorithm=1) => no PCIe weight reload.
            detection_section = build_switchable_detection_section(
                self.ladder_grids,
                lambda name, tx, ty: _inference_branch(name, tx, ty, share_device=True),
            )
```

  2b. **Non-switchable path is now WHOLE-FRAME only** (static tiling removed). The single-branch wrapper (line ~1378) used `self.tiles_x, self.tiles_y`; hard-code whole-frame:

```python
        # Non-switchable path: plain whole-frame (static single-grid tiling removed).
        detection_pipeline_wrapper = _inference_branch('inference', 1, 1, share_device=False)
```

The `else: detection_section = f'{detection_pipeline_wrapper} ! '` branch (line ~1447-1448) stays as-is — it now always means whole-frame.

  3. **Set the startup active-pad** — find where `create_pipeline()` completes / the pipeline object exists (search `self.pipeline` assignment). After the pipeline is built, set the selector to the whole (last) pad. Add a small method and call it where `self.pipeline` first becomes valid:

```python
    def _init_selector_startup_pad(self):
        """Point the input-selector at the whole-frame tier (last sink pad) at
        startup so the ACTIVE pad matches the only open valve."""
        if not self.switchable_tiling or not self.ladder_grids:
            return
        sel = self.pipeline.get_by_name("branch_selector")
        if sel is None:
            return
        n = len(self.ladder_grids)
        pad = sel.get_static_pad(f"sink_{n - 1}")
        if pad is not None:
            sel.set_property("active-pad", pad)
```

Call `self._init_selector_startup_pad()` immediately after the pipeline is instantiated. If `create_pipeline()` is inherited from the base and not overridden, call it right after `self.create_pipeline()` in `__init__` (line ~1345): `self.create_pipeline(); self._init_selector_startup_pad()`.

  4. **Replace `switch_tiling` with `switch_to_tier`** (lines ~367-419):

```python
    def switch_to_tier(self, target_i: int, warmup_timeout_s: float = 1.0) -> bool:
        """Hot-switch the active inference branch to ladder tier `target_i`
        (switchable pipeline only). 4-step make-before-break handover with zero
        command gap: open the incoming valve, wait its first output buffer, flip the
        input-selector active-pad, close the outgoing valve. Same hef on every
        hailonet => no PCIe weight reload. Safe from any thread.

        GRACEFUL DEGRADATION: if the incoming branch produces NO buffer within
        warmup_timeout_s, ABORT — re-close the incoming valve, leave the selector +
        the currently-producing valve untouched, return False. Returns True if the
        active tier is target_i after the call (incl. no-op when already there)."""
        sel = self.pipeline.get_by_name("branch_selector")
        if sel is None:
            logger.warning("switch_to_tier: switchable-tiling pipeline not present; ignoring")
            return False
        target_pad_name = f"sink_{target_i}"
        cur = sel.get_property("active-pad")
        cur_name = cur.get_name() if cur is not None else None
        if cur_name == target_pad_name:
            return True  # already there
        incoming_valve = self.pipeline.get_by_name(f"valve_tier{target_i}")
        if incoming_valve is None:
            logger.warning("switch_to_tier: no valve_tier%d; ignoring", target_i)
            return False
        # the currently-active tier index (from the active pad name "sink_K")
        outgoing_valve = self.pipeline.get_by_name(cur_name.replace("sink_", "valve_tier")) \
            if cur_name else None
        target_pad = sel.get_static_pad(target_pad_name)

        incoming_valve.set_property("drop", False)        # 1. open incoming branch
        ready = threading.Event()                         # 2. wait its first buffer
        def _first_buf(pad, info, _u):
            ready.set()
            return Gst.PadProbeReturn.REMOVE
        probe_id = target_pad.add_probe(Gst.PadProbeType.BUFFER, _first_buf, None)
        if not ready.wait(warmup_timeout_s) and not ready.is_set():
            logger.warning("switch_to_tier: incoming tier %d gave no buffer in %.1fs; "
                           "ABORTING, staying on %s", target_i, warmup_timeout_s, cur_name)
            incoming_valve.set_property("drop", True)      # undo step 1
            try: target_pad.remove_probe(probe_id)
            except Exception: pass
            return False
        sel.set_property("active-pad", target_pad)         # 3. flip selector (atomic)
        if outgoing_valve is not None:
            outgoing_valve.set_property("drop", True)       # 4. idle the outgoing branch
        logger.warning("!!! BRANCH SWITCH -> tier %d (%s)", target_i, target_pad_name)
        return True
```

  5. **Delete `enable_plus_one`** entirely (lines ~421-444). Remove any `self._tiling_active` reference left dangling (it was set only in `switch_tiling`; grep and remove reads of it, or leave it unset — grep `_tiling_active`).

- [ ] **Step 6: Rewrite the policy wrapper in `user_app_callback_class`** (`BDD/app.py` ~63-118) — the old `TilingSwitchPolicy` no longer exists (deleted in Task 2), so `app.py` won't import until this lands. Replace the `__init__` policy fields:

```python
        # Size-driven tiling ladder policy. The tier decision lives in the pure,
        # unit-tested TilingLadderPolicy; this class only fires the (blocking) switch
        # on a worker thread. Enabled by main() via configure_tiling_policy().
        self.request_switch = None            # callable(tier_i:int) -> app.switch_to_tier
        self.auto_switch = False
        self._policy = None                   # set by configure_tiling_policy
        self._switch_pending = False
        self._switch_lock = threading.Lock()
```

and replace `configure_switch_policy` / `note_detection` / `_fire_switch` with:

```python
    def configure_tiling_policy(self, request_switch, tiers, lost_to_2x1_s, lost_to_3x2_s):
        """Enable the size+loss ladder policy. request_switch(tier_i) performs the
        actual (blocking) handover. tiers is the ordered LadderTier list (index 0 =
        most tiles, last = whole); the pipeline starts on the whole tier."""
        self.request_switch = request_switch
        self.auto_switch = True
        self._policy = TilingLadderPolicy(
            tiers, lost_to_2x1_s=lost_to_2x1_s, lost_to_3x2_s=lost_to_3x2_s,
            current_i=len(tiers) - 1,          # whole-frame active at startup
        )

    def note_tiling(self, side, now_s: float):
        """Per-frame tier policy. side = max(bbox_w, bbox_h) of the primary tracked
        target (0..1), or None if no target. Fires the (blocking) switch on a worker
        thread so the GStreamer streaming thread is never stalled. No-op unless
        auto_switch is configured."""
        if not self.auto_switch or self._policy is None or self.request_switch is None:
            return
        target = self._policy.note(side, now_s)
        if target is not None:
            self._fire_switch(target)

    def _fire_switch(self, target_i: int):
        with self._switch_lock:
            if self._switch_pending or target_i == self._policy.current_i:
                return
            self._switch_pending = True
        def _run():
            switched = False
            try:
                switched = bool(self.request_switch(target_i))
                if switched:
                    self._policy.committed(target_i)   # adopt tier
            finally:
                self._switch_pending = False
        threading.Thread(target=_run, name="tiling-policy-switch", daemon=True).start()
```

Update the import at the top of `app.py`: replace `from tiling_policy import TilingSwitchPolicy` with `from tiling_policy import TilingLadderPolicy, build_ladder`.

- [ ] **Step 7: Compute `side` in the callback and feed the policy** (`BDD/app.py`, the `_best_conf`/`note_detection` block ~345-347). The tracked-target size comes from the matched tracks (built ~360-390), so place the feed AFTER `track_id_map` exists. Replace the `note_detection` call with:

```python
    # Size-driven tiling ladder: side = max(w,h) of the primary tracked target
    # (the matched track with the largest max(w,h)), or None if no track matched.
    # Fed after tracking so it uses stable track identity, not raw detections.
    primary_side = None
    if USE_TRACKER and track_id_map:
        matched_sides = [max(raw_dets[i][0].width, raw_dets[i][0].height)
                         for i in track_id_map]      # track_id_map keys are det indices
        primary_side = max(matched_sides) if matched_sides else None
    user_data.note_tiling(primary_side, time.monotonic())
```

> **Implementer:** verify `Rect` exposes `.width`/`.height` (grep `class Rect` in `helpers.py`; edges are `.left_edge/.right_edge/.top_edge/.bottom_edge` per app.py:398 — if there's no `.width`, use `r.right_edge - r.left_edge` / `r.bottom_edge - r.top_edge`). bbox coords are normalized 0..1, so `side` is already a frame fraction. Keep the `_best_conf` line + the throttled `DETS` log (it uses `_best_conf`); optionally add `primary_side` to that log for on-rig visibility. The old `note_detection(_best_conf)` call is removed.

- [ ] **Step 8: Remove the tiling CLI flags from `main()`** — delete the parse lines for `--tiles` (`tiles_spec`/`tiles_override`, ~590-591), `--switch-tiles` (`switch_tiles_spec`/`switch_tiles_override`, ~595-596), and `--plus-one` (`plus_one`, ~605), and the `_enable_plus_one` thread block (~805-811). Grep `plus_one|plus-one|enable_plus_one|switch_tiles_override|tiles_override|--tiles` across `BDD/` → zero remaining references (except docs).

- [ ] **Step 9: Replace the tiling-decision block in `main()`** (`BDD/app.py` ~659-698) with the ladder build + switchable derivation. Replace the whole `switchable = ... / if switchable: ... else: ...` block (lines ~659-681) with:

```python
    # Tiling is LADDER-ONLY: build the ordered ladder from config.tiling.ladder.
    # Empty ladder => whole-frame (not switchable). >= 2 rungs => build the N-branch
    # hot-switch pipeline (whole-frame = last rung, active at startup).
    ladder_tiers = build_ladder(config.tiling)
    ladder_grids = [(t.tiles_x, t.tiles_y) for t in ladder_tiers]
    switchable = len(ladder_grids) >= 2
    if switchable:
        logger.info("!!! SWITCHABLE tiling ladder: %s (whole active at startup)", ladder_grids)
    else:
        logger.info("!!! Inference: whole-frame (no tiling ladder configured)")
    if config.tiling.auto_switch and (not switchable or not USE_TRACKER):
        logger.warning("!!! tiling.auto_switch set but %s — policy will not run",
                       "ladder has < 2 rungs" if not switchable else "bytetrack is disabled")
```

Then fix the `tiles`-referencing NV12 warning (lines ~696-698) — `tiles` no longer exists:

```python
    if any(g != (1, 1) for g in ladder_grids) and video_format == 'NV12':
        logger.warning("!!! NV12 capture + tiling together: extra format handling on the "
                       "tile path may further raise latency/CPU — measure before relying on it.")
```

Then at the `App(...)` construction (~745-753): replace `tiles=tiles` with `ladder_grids=ladder_grids`; keep `tiling_overlap=config.tiling.overlap`, `switchable_tiling=switchable`. Finally replace the OLD `configure_switch_policy` block (~759-770) with:

```python
    # Size-driven tiling ladder policy: give the callback a switch_to_tier handle and
    # the ladder. Runs only when the pipeline is switchable AND auto_switch is on.
    if switchable and config.tiling.auto_switch:
        user_data.configure_tiling_policy(
            request_switch=app.switch_to_tier,
            tiers=ladder_tiers,
            lost_to_2x1_s=config.tiling.lost_to_2x1_s,
            lost_to_3x2_s=config.tiling.lost_to_3x2_s,
        )
        logger.info("!!! auto-switch tiling ladder: %s, lost->2x1 @%.1fs, lost->3x2 @%.1fs",
                    ladder_grids, config.tiling.lost_to_2x1_s, config.tiling.lost_to_3x2_s)
```

- [ ] **Step 10: Update `--switch-test-s` to cycle tiers** (~795-803). Replace the boolean toggle:

```python
    if switch_test_s and switchable:
        n_tiers = len(ladder_grids)
        def _tiling_switch_test():
            i = 0
            while True:
                time.sleep(switch_test_s)
                app.switch_to_tier(i)
                i = (i + 1) % n_tiers
        threading.Thread(target=_tiling_switch_test, name="tiling-switch-test", daemon=True).start()
        logger.info("!!! tiling-switch-test: cycling tiers 0..%d every %.1fs", n_tiers - 1, switch_test_s)
```

- [ ] **Step 11: Host smoke**

Run: `cd BDD && python -m pytest test_app_base_pipeline.py test_tiling_policy.py test_config.py -v && python -m py_compile app.py app_base.py`
Expected: PASS + clean compile. Then grep-audit: `grep -rn "TilingSwitchPolicy\|note_detection\|switch_tiling\b\|configure_switch_policy\|enable_plus_one\|\.tiles_x\|\.tiles_y" BDD/app.py BDD/app_base.py` → zero hits (all renamed/removed).

- [ ] **Step 12: ON-RIG validation — manual tier cycle** (Pi + Hailo; deploy per `bdd-sd9-deploy-model`). Config: 3-rung `ladder`, `auto_switch: false`. Run:

Run (on rig): `python app.py --switch-test-s 8 --vision-only`
Expected: startup `SWITCHABLE tiling ladder: [(3, 2), (2, 1), (1, 1)]`; recurring `!!! BRANCH SWITCH -> tier N (sink_N)` cycling 0→1→2; continuous `DETS frame=...` across each switch (no gap); no `ABORTING` unless a branch really fails to warm; detection FPS recovers to whole-frame rate on tier 2. No crash.

> If the rig is unavailable, STOP and report — do not mark done or commit Task 3 as validated.

- [ ] **Step 13: Commit**

```bash
git add BDD/app_base.py BDD/app.py BDD/test_app_base_pipeline.py
git commit -m "feat(tiling): N-branch ladder pipeline + switch_to_tier + policy wiring

Generalizes the switchable pipeline to one valve-gated branch per ladder
tier behind a shared input-selector; switch_to_tier(i) keeps the warmup-wait
+ graceful-revert handover. Callback feeds side=max(w,h) of the primary
tracked target to TilingLadderPolicy. Removes --plus-one/--tiles/--switch-tiles
and the static single-grid path (ladder-only). Manual tier cycle validated on-rig.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Enable and validate the auto size/loss policy on-rig

All code lands in Task 3; this task turns the policy ON and confirms the size-hysteresis
and loss-escalation behavior against a live target, tuning the thresholds if needed. Its
deliverable — "verified auto policy on the rig" — is independently reviewable: it can be
rejected (behavior wrong, needs tuning) while Task 3's plumbing stands.

**Files:**
- Modify: `BDD/config.yaml` (uncomment/enable the 3-rung `ladder`, set `auto_switch: true`, tune thresholds).

- [ ] **Step 1: Enable the policy in `config.yaml`** — uncomment the `ladder:` block (Task 1 Step 7) and set `auto_switch: true`, `overlap: 0.10`, `lost_to_2x1_s: 1.0`, `lost_to_3x2_s: 10.0`. Confirm `bytetrack` is enabled (the size feed needs the tracker — `USE_TRACKER`).

- [ ] **Step 2: ON-RIG validation — auto size + loss** (Pi + Hailo). Run `python app.py --vision-only`. Verify in the log:
  - startup `auto-switch tiling ladder: [(3, 2), (2, 1), (1, 1)], lost->2x1 @1.0s, lost->3x2 @10.0s`;
  - a LARGE target (max side > 0.10) sits on / climbs to tier 2 (whole): `BRANCH SWITCH -> tier 2`;
  - move it away so `side` crosses 0.05 then 0.02 → `-> tier 1` then `-> tier 0`, ONE step per crossing; jitter around a threshold must NOT thrash (hysteresis);
  - REMOVE the target → ~1 s later `-> tier 1`, ~10 s later `-> tier 0`;
  - reintroduce → resumes size-based tiers, whole-frame restored when large/close;
  - continuous `DETS`, no `ABORTING` (unless a branch genuinely fails to warm).

> If the rig is unavailable, STOP and report — do not mark done or commit as validated.

- [ ] **Step 3: Tune if needed, then commit** — if the target's real on-screen size makes the default thresholds switch too eagerly / too late, adjust `up_side`/`down_side`/`lost_*` in `config.yaml` and re-run Step 2 until the ladder tracks the target sensibly.

```bash
git add BDD/config.yaml
git commit -m "feat(tiling): enable + tune the auto size/loss ladder policy (on-rig verified)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the executor

- **No legacy, no static:** the old `tiles_x/tiles_y/switchable/switch_conf/lost_frames_to_tile/locked_frames_to_whole` fields, the `--tiles`/`--switch-tiles`/`--plus-one` CLI flags, and the static single-grid tiling path are all REMOVED (Decision 5). The ladder is the sole tiling control; an empty ladder = whole-frame.
- **Import break between tasks is expected:** Task 2 deletes `TilingSwitchPolicy`, so `app.py` won't import until Task 3 Step 6 lands. This is fine — `app.py` only runs on the rig, and Task 2's deliverable is the host-tested pure policy. Task 3 makes `app.py` whole again before its rig gate.
- **Off-device limits:** `app.py`/`app_base.py` import hailo/GStreamer and cannot be imported on the host — the pure seams (`tiling_policy`, `build_switchable_detection_section`) carry the unit tests, and `py_compile` is the host guard for the rest. Never claim the pipeline works from host tests alone; the on-rig steps are mandatory gates for Tasks 3 & 4.
- **Graphify:** after code changes land, `graphify update .` (AST-only, per CLAUDE.md).
