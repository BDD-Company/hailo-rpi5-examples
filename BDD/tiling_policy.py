"""Size-driven tiling ladder policy — pure decision logic, no GStreamer/hailo.

Host-importable and unit-testable (app.py pulls in hailo/GStreamer). Decides WHICH
inference tier to hot-switch to, given the tracked target's on-screen size
(side = max(bbox_w, bbox_h), normalized 0..1) or elapsed lost-time when no target.
The actual switch (threads + GStreamer valve/input-selector handover) stays in
app.py / app_base.py.

The ladder is the SOLE tiling mechanism — there is no binary whole<->tile switch and
no static grid. Ladder order: index 0 = MOST tiles (smallest / long-lost target) ->
last = whole-frame (lowest latency, active at startup).

Three classes:
  - TilingLadderPolicy: the decision (size hysteresis + loss escalation -> "go to tier i").
  - TilingSwitchCoordinator: the serialization. Every switch — the automatic policy's and
    the --test-switch-s harness's — goes through it, so the policy's belief about the
    active tier can never diverge from the pipeline.
  - BranchStallWatchdog: the recovery. Reverts to whole-frame when a tier that warmed up
    successfully later stops delivering callbacks.
"""
import logging
import threading
import time
from collections import namedtuple
from typing import Optional

logger = logging.getLogger(__name__)

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

    Size (target present): climb toward whole (fewer tiles, higher index) when
    ``side >= tiers[current].up_side``; descend (more tiles, lower index) when
    ``side < tiers[current].down_side``. The asymmetric adjacent thresholds are the
    hysteresis dead-band that prevents thrash.

    Loss (no target): at ``lost_to_2x1_s`` drop to the rung just above whole; at
    ``lost_to_3x2_s`` drop to tier 0 (most tiles). Loss only ever adds tiles.
    """

    def __init__(self, tiers, lost_to_2x1_s, lost_to_3x2_s, current_i):
        self.tiers = list(tiers)
        self.lost_to_2x1_s = lost_to_2x1_s
        self.lost_to_3x2_s = lost_to_3x2_s
        self.current_i = int(current_i)
        self._lost_since_s = None     # None => target currently present

    @property
    def whole_i(self) -> int:
        """Index of the whole-frame rung — the last one, and the known-good branch."""
        return len(self.tiers) - 1

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


class TilingSwitchCoordinator:
    """The ONE entry point for changing the active inference tier.

    Before this existed, two threads could drive the handover independently: the
    ladder policy's worker, and the ``--test-switch-s`` harness, which called the
    switch directly and never updated the policy. Two failures followed.

    1. DESYNC. The harness flipped the pipeline without telling the policy, so the
       policy's ``current_i`` went stale. Because a switch request is skipped when it
       already matches the believed tier, the policy could then never command the
       reverse switch — the rig latched onto a >200 ms tiled rung while the policy
       believed it was on whole-frame. Permanent, and exactly the latency regression
       the whole campaign exists to prevent.
    2. RACE. The switch held no lock, so two callers could interleave their
       valve/selector set_property calls and leave the input-selector pointing at a
       branch whose valve is shut (a stall).

    Both are fixed by funnelling every switch through :meth:`request`, which is
    serialized and updates the policy on the way out.

    WHY THE LOCK IS NOT HELD ACROSS THE HANDOVER: ``switch_fn`` blocks for up to
    ``warmup_timeout_s`` (1 s) waiting for the incoming branch's first buffer. The
    GStreamer streaming thread calls :meth:`note` on every frame. If one lock covered
    both, that thread would stall for ~30 frames on every switch — trading a rare
    desync for a guaranteed control-loop stall. So the lock guards policy state only,
    is never held while ``switch_fn`` runs, and ``switch_fn`` never reaches back for
    it. The GStreamer-side lock (GStreamerApp._switch_lock) is strictly below this one
    in the ordering, so the two can never deadlock.
    """

    # After a switch to a tiled rung FAILS, refuse further tiled switches for this long.
    # Without it the policy — whose condition (small target / still lost) has not changed
    # — returns the same target on the very next frame and app.py spawns a fresh handover
    # worker per frame against a branch that just proved dead.
    FAILED_SWITCH_BACKOFF_S = 1.0

    def __init__(self, policy: TilingLadderPolicy, switch_fn, now=time.monotonic):
        self._policy = policy
        self._switch_fn = switch_fn      # callable(tier_i: int) -> bool
        self._now = now                  # injectable clock (tests)
        self._lock = threading.Lock()
        self._in_flight = False
        self._tiles_blocked_until = 0.0

    @property
    def whole_i(self) -> int:
        return self._policy.whole_i

    @property
    def current_tier(self) -> int:
        """The tier the policy believes is active (kept true by :meth:`request`)."""
        with self._lock:
            return self._policy.current_i

    @property
    def on_whole(self) -> bool:
        """True when the known-good whole-frame rung is the active one."""
        with self._lock:
            return self._policy.current_i == self._policy.whole_i

    @property
    def switch_in_flight(self) -> bool:
        with self._lock:
            return self._in_flight

    @property
    def tiles_blocked(self) -> bool:
        with self._lock:
            return self._now() < self._tiles_blocked_until

    def block_tiles(self, seconds: float):
        """Refuse switches AWAY from whole-frame for `seconds`. Used by
        BranchStallWatchdog after it reverts a tiled rung that died: without a cooldown
        the policy — still seeing a lost or small target — would re-escalate within ~1 s
        and dive straight back into the rung we just proved dead, a switch-revert thrash
        loop."""
        with self._lock:
            self._tiles_blocked_until = self._now() + seconds

    def note(self, side: Optional[float], now_s: float) -> Optional[int]:
        """Streaming-thread hot path. Feed this frame's primary-target ``side``
        (``None`` when no target is matched); returns the tier to switch TO if one
        should be started now, else None. Never blocks on a handover: while one is in
        flight it reports None, so the caller does not spawn a worker per frame (the
        policy keeps returning a target for as long as the condition holds). Returns
        None for tiled rungs while they are in post-stall cooldown, so no worker is
        spawned only to be refused."""
        with self._lock:
            if self._in_flight:
                return None
            target = self._policy.note(side, now_s)
            if target is None or target == self._policy.current_i:
                return None
            if target != self._policy.whole_i and self._now() < self._tiles_blocked_until:
                return None
            return target

    def request(self, tier_i: int) -> bool:
        """Switch to ``tier_i``, blocking until the handover settles. Safe from any
        thread; concurrent callers do not interleave. Returns True if that tier is
        active afterwards (including a no-op when already there).

        On failure — a dead incoming branch, or ``switch_fn`` raising — the rung that
        was already delivering detections keeps running, and tiled switches back off for
        FAILED_SWITCH_BACKOFF_S instead of being retried on every frame.

        A request for a TILED rung during the post-stall cooldown is refused outright.
        Requests for WHOLE-FRAME are never blocked: that is the known-good rung and the
        watchdog must always be able to get back to it.
        """
        whole_i = self._policy.whole_i
        with self._lock:
            if self._in_flight:
                # Another switch is settling; don't queue behind it or interleave.
                return tier_i == self._policy.current_i
            if tier_i == self._policy.current_i:
                return True
            if tier_i != whole_i and self._now() < self._tiles_blocked_until:
                logger.warning("tier %d switch refused: tiling is in post-stall cooldown "
                               "for another %.1fs",
                               tier_i, self._tiles_blocked_until - self._now())
                return False
            self._in_flight = True

        switched = False
        try:
            switched = bool(self._switch_fn(tier_i))
        except Exception:
            # A daemon worker must never die with a bare stderr traceback: the log is
            # the only forensic record of a run. Treat it as a failed switch.
            logger.exception("tiling switch to tier %d failed", tier_i)
        finally:
            with self._lock:
                if switched:
                    self._policy.committed(tier_i)      # adopt the tier
                elif tier_i != whole_i:
                    self._tiles_blocked_until = max(
                        self._tiles_blocked_until,
                        self._now() + self.FAILED_SWITCH_BACKOFF_S)
                self._in_flight = False
        return switched


class BranchStallWatchdog:
    """Recover from an inference branch that warmed up and then DIED.

    `switch_to_tier`'s make-before-break handover already refuses to move onto a branch
    that never produces a first buffer. It cannot help once a branch has been accepted
    and later stops delivering: `app_callback` simply stops firing, DET FPS goes to 0,
    and the control loop starves — which on this rig can wreck hardware. Nothing else
    notices, because every component downstream is *waiting* rather than failing.

    So: poll the callback's frame counter. If it stops advancing while a TILED rung is
    active, revert to whole-frame — the rung that was known good at startup and that the
    pipeline is built around. Degrade to last-good rather than abort; a stalled control
    loop is worse than a slow one.

    Deliberately does NOT act when whole-frame is the active rung. A stall there means
    the source or the device is gone, and switching to a tiled rung (fed by the same
    source tee, inferring on the same hailonet) cannot help — it would just thrash while
    the operator needs a clean error in the log.

    Stays disarmed until the pipeline has delivered its first callback, so it does not
    mistake startup (camera + device coming up) for a stall.

    Pure logic: `frame_count_fn` and `now` are injected, so this is unit-testable without
    GStreamer. The polling thread lives in app.py.
    """

    def __init__(self, coordinator: TilingSwitchCoordinator, frame_count_fn,
                 stall_timeout_s: float = 2.0, cooldown_s: float = 30.0,
                 now=time.monotonic):
        self._coord = coordinator
        self._frame_count = frame_count_fn
        self._stall_timeout_s = stall_timeout_s
        self._cooldown_s = cooldown_s
        self._now = now
        self._last_count = None
        self._last_progress = None
        self._whole_stall_reported = False

    def poll(self) -> bool:
        """Call periodically. Returns True iff it just reverted a stalled tiled rung."""
        now = self._now()
        count = self._frame_count()

        if self._last_count is None:            # first call: start the clock
            self._last_count, self._last_progress = count, now
            return False

        if count == 0:
            # Nothing has EVER been delivered — the camera and the device are still coming
            # up. A watchdog for "warmed up and then died" must first observe "warmed up",
            # otherwise it screams STALL a couple of seconds into every launch (it did).
            self._last_count, self._last_progress = count, now
            return False

        if count != self._last_count:           # frames are flowing
            self._last_count, self._last_progress = count, now
            self._whole_stall_reported = False
            return False

        # A handover legitimately pauses delivery; don't count it as a stall.
        if self._coord.switch_in_flight:
            self._last_progress = now
            return False

        if now - self._last_progress < self._stall_timeout_s:
            return False

        if self._coord.on_whole:
            # Whole-frame is active and stalled: nothing safer to fall back to.
            if not self._whole_stall_reported:
                logger.error("!!! STALL: no callbacks for %.1fs on the WHOLE-FRAME branch. "
                             "The source or the device is gone; a tier switch cannot help.",
                             now - self._last_progress)
                self._whole_stall_reported = True
            return False

        logger.error("!!! STALL: no callbacks for %.1fs on tier %d — the tiled branch died "
                     "after warming up. Reverting to whole-frame.",
                     now - self._last_progress, self._coord.current_tier)
        # Block BEFORE reverting, or the policy can re-request a tiled rung the moment the
        # revert lands and dive straight back into the dead branch.
        self._coord.block_tiles(self._cooldown_s)
        reverted = self._coord.request(self._coord.whole_i)
        # Restart the stall clock either way: on success the frame counter is about to
        # move; on failure we want a fresh timeout before retrying, not an instant retry.
        self._last_progress = self._now()
        if not reverted:
            logger.error("!!! STALL: revert to whole-frame FAILED; will retry in %.1fs",
                         self._stall_timeout_s)
        return reverted
