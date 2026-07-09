"""Detection-state tiling switch policy — pure decision logic, no GStreamer/hailo.

Split out of app.py so it is host-importable and unit-testable (app.py pulls in
hailo/GStreamer and cannot be imported off-device). This decides WHEN to hot-switch
the inference branch between whole-frame (low latency) and tiling (small-object
recall); the actual GStreamer valve/input-selector handover stays in app_base.py's
GStreamerApp.switch_tiling, and the worker thread that calls it stays in app.py.

Three classes:
  - TilingSwitchPolicy: the decision (streaks -> "switch to X").
  - TilingSwitchCoordinator: the serialization. Every switch — the automatic policy's
    and the --test-switch-s harness's — goes through it, so the policy's belief about
    the active branch can never diverge from the pipeline.
  - BranchStallWatchdog: the recovery. Reverts to whole-frame when a branch that warmed
    up successfully later stops delivering callbacks.
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TilingSwitchPolicy:
    """Per-frame whole<->tile switch decision driven by detection-confidence streaks.

    Feed each frame's best target confidence to :meth:`note`; it returns the branch
    to switch TO (``True`` = tiling, ``False`` = whole-frame) when a threshold is
    crossed, else ``None`` (stay put). Because the switch itself is asynchronous,
    call :meth:`committed` once it has actually taken effect so the policy adopts the
    new branch and its streak counters reset.

    Thresholds:
      - ``lost_frames_to_tile`` consecutive frames with no confident target
        (best_conf < ``switch_conf``) while on whole-frame -> switch to tiling to
        reacquire small/distant objects.
      - ``locked_frames_to_whole`` consecutive confident frames while tiling ->
        switch back to whole-frame to restore low-latency control.
    """

    def __init__(self, switch_conf: float = 0.4, lost_frames_to_tile: int = 10,
                 locked_frames_to_whole: int = 5, tiling_on: bool = False):
        self.switch_conf = switch_conf
        self.lost_frames_to_tile = lost_frames_to_tile
        self.locked_frames_to_whole = locked_frames_to_whole
        self.tiling_on = tiling_on            # currently-active branch (False = whole)
        self._lost_streak = 0
        self._lock_streak = 0

    def note(self, best_conf: float) -> Optional[bool]:
        """Update the streaks with this frame's best confidence and return the branch
        to switch to if a threshold is crossed, else ``None``."""
        if best_conf >= self.switch_conf:
            self._lock_streak += 1
            self._lost_streak = 0
        else:
            self._lost_streak += 1
            self._lock_streak = 0
        if not self.tiling_on and self._lost_streak >= self.lost_frames_to_tile:
            return True          # target lost -> tile to reacquire
        if self.tiling_on and self._lock_streak >= self.locked_frames_to_whole:
            return False         # confidently locked -> whole-frame (low latency)
        return None

    def committed(self, tiling_on: bool):
        """Adopt ``tiling_on`` as the active branch and reset the streak counters
        (call once the switch has actually taken effect)."""
        self.tiling_on = tiling_on
        self.reset_streaks()

    def reset_streaks(self):
        self._lost_streak = 0
        self._lock_streak = 0


class TilingSwitchCoordinator:
    """The ONE entry point for changing the active inference branch.

    Before this existed, two threads could drive the handover independently: the
    detection policy's worker, and the ``--test-switch-s`` harness, which called
    app.switch_tiling directly and never updated the policy. Two failures followed.

    1. DESYNC. The harness flipped the pipeline without telling the policy, so
       ``policy.tiling_on`` went stale. Because a switch request is skipped when it
       already matches ``tiling_on``, the policy could then never command the reverse
       switch — the rig latched onto the >200 ms tiling branch while the policy
       believed it was on whole-frame. Permanent, and exactly the latency regression
       the whole campaign exists to prevent.
    2. RACE. ``switch_tiling`` held no lock, so two callers could interleave their
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

    def __init__(self, policy: TilingSwitchPolicy, switch_fn, now=time.monotonic):
        self._policy = policy
        self._switch_fn = switch_fn      # callable(to_tiling: bool) -> bool
        self._now = now                  # injectable clock (tests)
        self._lock = threading.Lock()
        self._in_flight = False
        self._tiling_blocked_until = 0.0

    @property
    def tiling_on(self) -> bool:
        """The branch the policy believes is active (kept true by :meth:`request`)."""
        with self._lock:
            return self._policy.tiling_on

    @property
    def switch_in_flight(self) -> bool:
        with self._lock:
            return self._in_flight

    @property
    def tiling_blocked(self) -> bool:
        with self._lock:
            return self._now() < self._tiling_blocked_until

    def block_tiling(self, seconds: float):
        """Refuse switches TO tiling for `seconds`. Used by BranchStallWatchdog after it
        reverts a tile branch that died: without a cooldown the policy would immediately
        rebuild its lost-streak (~0.7 s at 28 fps) and dive straight back into the branch
        we just proved dead — a switch-revert thrash loop."""
        with self._lock:
            self._tiling_blocked_until = self._now() + seconds
            self._policy.reset_streaks()

    def note(self, best_conf: float) -> Optional[bool]:
        """Streaming-thread hot path. Feed this frame's best confidence; returns the
        branch to switch TO if one should be started now, else None. Never blocks on a
        handover: while one is in flight it reports None, so the caller does not spawn
        a worker per frame (``TilingSwitchPolicy.note`` keeps returning a target once
        its streak threshold is crossed). Returns None for tiling while the branch is in
        its post-stall cooldown, so no worker is spawned only to be refused."""
        with self._lock:
            if self._in_flight:
                return None
            target = self._policy.note(best_conf)
            if target is None or target == self._policy.tiling_on:
                return None
            if target is True and self._now() < self._tiling_blocked_until:
                return None
            return target

    def request(self, to_tiling: bool) -> bool:
        """Switch to ``to_tiling``, blocking until the handover settles. Safe from any
        thread; concurrent callers do not interleave. Returns True if the active branch
        is ``to_tiling`` afterwards (including a no-op when already there).

        On failure — a dead incoming branch, or ``switch_fn`` raising — the policy's
        streaks are reset so it backs off and retries after a fresh streak instead of
        thrashing, and the branch that was already delivering detections keeps running.

        A request TO tiling during the post-stall cooldown is refused outright. Requests
        to WHOLE-FRAME are never blocked: that is the known-good branch and the watchdog
        must always be able to get back to it.
        """
        with self._lock:
            if self._in_flight:
                # Another switch is settling; don't queue behind it or interleave.
                return to_tiling == self._policy.tiling_on
            if to_tiling == self._policy.tiling_on:
                return True
            if to_tiling and self._now() < self._tiling_blocked_until:
                logger.warning("tiling switch refused: branch is in post-stall cooldown "
                               "for another %.1fs",
                               self._tiling_blocked_until - self._now())
                self._policy.reset_streaks()
                return False
            self._in_flight = True

        switched = False
        try:
            switched = bool(self._switch_fn(to_tiling))
        except Exception:
            # A daemon worker must never die with a bare stderr traceback: the log is
            # the only forensic record of a run. Treat it as a failed switch.
            logger.exception("tiling switch to %s failed",
                             "TILING" if to_tiling else "WHOLE-FRAME")
        finally:
            with self._lock:
                if switched:
                    self._policy.committed(to_tiling)   # adopt branch + reset streaks
                else:
                    self._policy.reset_streaks()
                self._in_flight = False
        return switched


class BranchStallWatchdog:
    """Recover from an inference branch that warmed up and then DIED.

    `switch_tiling`'s make-before-break handover already refuses to move onto a branch
    that never produces a first buffer. It cannot help once a branch has been accepted
    and later stops delivering: `app_callback` simply stops firing, DET FPS goes to 0,
    and the control loop starves — which on this rig can wreck hardware. Nothing else
    notices, because every component downstream is *waiting* rather than failing.

    So: poll the callback's frame counter. If it stops advancing while the TILING branch
    is active, revert to whole-frame — the branch that was known good at startup and that
    the pipeline is built around. Degrade to last-good rather than abort; a stalled
    control loop is worse than a slow one.

    Deliberately does NOT act when whole-frame is the active branch. A stall there means
    the source or the device is gone, and switching to tiling (fed by the same source tee,
    inferring on the same hailonet) cannot help — it would just thrash while the operator
    needs a clean error in the log.

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
        """Call periodically. Returns True iff it just reverted a stalled tile branch."""
        now = self._now()
        count = self._frame_count()

        if self._last_count is None:            # first call: start the clock
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

        if not self._coord.tiling_on:
            # Whole-frame is active and stalled: nothing safer to fall back to.
            if not self._whole_stall_reported:
                logger.error("!!! STALL: no callbacks for %.1fs on the WHOLE-FRAME branch. "
                             "The source or the device is gone; a branch switch cannot help.",
                             now - self._last_progress)
                self._whole_stall_reported = True
            return False

        logger.error("!!! STALL: no callbacks for %.1fs while TILING is active — the tile "
                     "branch died after warming up. Reverting to whole-frame.",
                     now - self._last_progress)
        # Block BEFORE reverting, or the policy can re-request tiling the moment the
        # revert lands and dive straight back into the dead branch.
        self._coord.block_tiling(self._cooldown_s)
        reverted = self._coord.request(False)
        # Restart the stall clock either way: on success the frame counter is about to
        # move; on failure we want a fresh timeout before retrying, not an instant retry.
        self._last_progress = self._now()
        if not reverted:
            logger.error("!!! STALL: revert to whole-frame FAILED; will retry in %.1fs",
                         self._stall_timeout_s)
        return reverted
