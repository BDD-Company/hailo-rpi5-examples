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
