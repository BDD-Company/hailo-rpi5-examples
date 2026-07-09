"""Detection-state tiling switch policy — pure decision logic, no GStreamer/hailo.

Split out of app.py so it is host-importable and unit-testable (app.py pulls in
hailo/GStreamer and cannot be imported off-device). This decides WHEN to hot-switch
the inference branch between whole-frame (low latency) and tiling (small-object
recall); the actual GStreamer valve/input-selector handover stays in app_base.py's
GStreamerApp.switch_tiling, and the worker thread that calls it stays in app.py.

Two classes:
  - TilingSwitchPolicy: the decision (streaks -> "switch to X").
  - TilingSwitchCoordinator: the serialization. Every switch — the automatic policy's
    and the --test-switch-s harness's — goes through it, so the policy's belief about
    the active branch can never diverge from the pipeline.
"""
import logging
import threading
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

    def __init__(self, policy: TilingSwitchPolicy, switch_fn):
        self._policy = policy
        self._switch_fn = switch_fn      # callable(to_tiling: bool) -> bool
        self._lock = threading.Lock()
        self._in_flight = False

    @property
    def tiling_on(self) -> bool:
        """The branch the policy believes is active (kept true by :meth:`request`)."""
        with self._lock:
            return self._policy.tiling_on

    @property
    def switch_in_flight(self) -> bool:
        with self._lock:
            return self._in_flight

    def note(self, best_conf: float) -> Optional[bool]:
        """Streaming-thread hot path. Feed this frame's best confidence; returns the
        branch to switch TO if one should be started now, else None. Never blocks on a
        handover: while one is in flight it reports None, so the caller does not spawn
        a worker per frame (``TilingSwitchPolicy.note`` keeps returning a target once
        its streak threshold is crossed)."""
        with self._lock:
            if self._in_flight:
                return None
            target = self._policy.note(best_conf)
            if target is None or target == self._policy.tiling_on:
                return None
            return target

    def request(self, to_tiling: bool) -> bool:
        """Switch to ``to_tiling``, blocking until the handover settles. Safe from any
        thread; concurrent callers do not interleave. Returns True if the active branch
        is ``to_tiling`` afterwards (including a no-op when already there).

        On failure — a dead incoming branch, or ``switch_fn`` raising — the policy's
        streaks are reset so it backs off and retries after a fresh streak instead of
        thrashing, and the branch that was already delivering detections keeps running.
        """
        with self._lock:
            if self._in_flight:
                # Another switch is settling; don't queue behind it or interleave.
                return to_tiling == self._policy.tiling_on
            if to_tiling == self._policy.tiling_on:
                return True
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
