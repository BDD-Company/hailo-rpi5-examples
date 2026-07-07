"""Detection-state tiling switch policy — pure decision logic, no GStreamer/hailo.

Split out of app.py so it is host-importable and unit-testable (app.py pulls in
hailo/GStreamer and cannot be imported off-device). This decides WHEN to hot-switch
the inference branch between whole-frame (low latency) and tiling (small-object
recall); the actual switch — threads plus the GStreamer valve/input-selector
handover — stays in app.py's user_app_callback_class.
"""
from typing import Optional


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
