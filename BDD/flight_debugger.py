#!/usr/bin/env python3
"""Multi-view synchronized flight debugger.

Three synchronized, dockable views:
  - Debug video playback
  - Textual log with per-frame highlighting
  - 3D telemetry visualization

Usage:
    cd BDD
    python flight_debugger.py path/to/log_file.log [--video path/to/video.mp4]
    python flight_debugger.py --dir path/to/_DEBUG_dir/
"""

import bisect
import math
import os
import sys
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Suppress harmless Wayland warning about mouse grabbing
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.input=false")

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QSlider, QLabel, QPlainTextEdit, QTextEdit,
    QSizePolicy, QCheckBox, QSpinBox, QDialog, QListWidget,
)
from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, QPoint
from PyQt6.QtGui import (
    QImage, QPixmap, QTextCursor, QTextCharFormat, QColor, QFont,
    QAction, QShortcut, QKeySequence,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.lines import Line2D

from debug_telemetry_position import FramePose, parse_telemetry_log
from telemetry_position import Quaternion, rotate_frd_to_ned


# ===========================================================================
# Highlight style — edit these to customize log line highlighting
# ===========================================================================

class HighlightStyle:
    """Controls how current-frame log lines are highlighted.

    BACKGROUND : QColor — line background colour
    FOREGROUND : QColor | None — text colour (None keeps the default)
    BOLD       : bool — whether highlighted text is bold
    """
    BACKGROUND = QColor(50, 50, 100)
    FOREGROUND = None                     # keep default text colour
    BOLD = False


# ===========================================================================
# Constants
# ===========================================================================

FRAME_RE = re.compile(r"frame=#(\d+)")
TARGET_RE = re.compile(r"frame=#(\d+).*?!!! target : XY\(([^,]+),\s*([^)]+)\)")


# Drone body geometry (FRD frame) — arms at 45° for X-shaped quad
ARM_LEN = 5
_R45 = math.sqrt(2) / 2
_D = ARM_LEN * _R45          # arm component along each axis
_ND = ARM_LEN * 1.15 * _R45  # nose tip along arm direction
_NP = 0.08 * _R45            # nose tip perpendicular offset
BODY_VERTS_FRD = np.array([
    [ _D,  _D, 0.0],                   # 0: front-right arm tip
    [-_D, -_D, 0.0],                   # 1: back-left arm tip
    [-_D,  _D, 0.0],                   # 2: back-right arm tip
    [ _D, -_D, 0.0],                   # 3: front-left arm tip
    [0.0, 0.0, 0.0],                   # 4: centre
    [_ND - _NP, _ND + _NP, 0.0],      # 5: nose marker
    [_ND + _NP, _ND - _NP, 0.0],      # 6: nose marker
])

# Prop circles: unit circle in FRD body plane (Z=0), 24 segments
PROP_RADIUS = ARM_LEN / 2
_PROP_N = 24
_prop_theta = np.linspace(0, 2 * np.pi, _PROP_N + 1)
_PROP_UNIT_X = np.cos(_prop_theta) * PROP_RADIUS
_PROP_UNIT_Y = np.sin(_prop_theta) * PROP_RADIUS
# Arm tip centres in FRD: forward, back, right, left (indices 0-3 of BODY_VERTS_FRD)
PROP_CENTRES_FRD = BODY_VERTS_FRD[:4]

VECTOR_SCALE = {"vel": 1, "acc": 1, "mag": 8.0}
TARGET_COLOR = (0.0, 0.75, 0.75)  # cyan/teal
PLAYBACK_INTERVAL_MS = 50

# Camera FOV pyramid — camera points along body -Z (up from drone back).
# Tip sits at the normal endpoint; base extends outward.
CAMERA_HFOV_DEG = 107.0   # horizontal full field-of-view
CAMERA_VFOV_DEG = 85.0   # vertical full field-of-view
CAMERA_PYRAMID_LEN = 10  # visual length of the pyramid (metres)
CAMERA_COLOR =       (1,    0.5,  0.5,  0.1 )
GROUND_COLOR =       (0.2,  1,    0.2,  0.1 )
GROUND_EDGE_COLOR =  (0.2,  0.5,  0.2,  0.3 )
GROUND_GRIND_COLOR = (0.2,  0.5,  0.2,  0.4 )
VELOCITY_ARROW_COLOR = "green"
ACCELERATION_ARROW_COLOR = "red"


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _quat_rotate_array(q: Quaternion, pts: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts)
    for i in range(pts.shape[0]):
        n, e, d = rotate_frd_to_ned(q, pts[i, 0], pts[i, 1], pts[i, 2])
        out[i] = [n, e, d]
    return out


def _ned_to_plot(n: float, e: float, d: float):
    """NED → plot coords (East, North, Up)."""
    return e, n, -d


def _ned_array_to_plot(arr: np.ndarray) -> np.ndarray:
    return arr[:, [1, 0, 2]] * np.array([1, 1, -1])


# ===========================================================================
# Data loading
# ===========================================================================

def parse_log_lines(path: Path) -> tuple[list[str], dict[int, list[int]]]:
    """Read every line from the log and build a frame_id → line-indices map."""
    lines = path.read_text(errors="replace").splitlines()
    frame_to_lines: dict[int, list[int]] = {}
    for i, line in enumerate(lines):
        m = FRAME_RE.search(line)
        if m:
            frame_to_lines.setdefault(int(m.group(1)), []).append(i)
    return lines, frame_to_lines


def parse_target_xy(lines: list[str]) -> dict[int, tuple[float, float]]:
    """Extract target XY per frame from log lines containing '!!! target'."""
    targets: dict[int, tuple[float, float]] = {}
    for line in lines:
        m = TARGET_RE.search(line)
        if m:
            fid = int(m.group(1))
            x, y = float(m.group(2)), float(m.group(3))
            targets[fid] = (x, y)
    return targets


class VideoReader:
    """Read frames from one or more video files (MP4, MKV, etc.).

    Uses sequential reading with on-demand caching because MKV
    frame-based seeking is unreliable with many codecs.
    Supports timestamp-based frame lookup for log synchronization.
    """

    def __init__(self, path: Path | list[Path] | None):
        self._files: list[Path] = []
        self._file_counts: list[int] = []   # frames per file
        self._total = 0
        self._cache: dict[int, np.ndarray] = {}
        self._frame_times_ms: list[float] = []   # ms from video start
        self._video_start: datetime | None = None
        # Sequential reader state
        self._cur_cap: cv2.VideoCapture | None = None
        self._cur_file_idx = -1
        self._cur_global_pos = 0   # next frame to be read globally

        if path is None:
            return
        if isinstance(path, list):
            files = sorted(path)
        elif path.is_dir():
            files = sorted(path.glob("*.mp4")) + sorted(path.glob("*.mkv"))
        else:
            files = [path] if path.exists() else []
        cum_ms = 0.0
        for f in files:
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                continue
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            times: list[float] = []
            if video_frame_count <= 0:
                # MKV / unreliable count — scan and collect timestamps
                while cap.grab():
                    times.append(cum_ms + cap.get(cv2.CAP_PROP_POS_MSEC))
                video_frame_count = len(times)
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                times = [cum_ms + i * 1000.0 / fps for i in range(video_frame_count)]
            cap.release()
            self._files.append(f)
            self._file_counts.append(video_frame_count)
            self._frame_times_ms.extend(times)
            if times:
                cum_ms = times[-1] + (times[-1] - times[-2] if len(times) > 1 else 33.0)
        self._total = sum(self._file_counts)

        # Auto-detect video start time from first filename
        if self._files:
            self._video_start = self._parse_start_time(self._files[0])

    @staticmethod
    def _parse_start_time(path: Path) -> datetime | None:
        """Try to extract a start timestamp from the filename.
        Supports pattern: *YYYYMMDD-HHMMSS*
        """
        m = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", path.stem)
        if m:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            )
        return None

    @property
    def total(self) -> int:
        return self._total

    @property
    def available(self) -> bool:
        return self._total > 0

    @property
    def has_time_sync(self) -> bool:
        return self._video_start is not None and len(self._frame_times_ms) > 0

    def find_frame_for_time(self, log_ts: datetime) -> int | None:
        """Find the closest video frame for a given log timestamp."""
        if not self.has_time_sync:
            return None
        delta_ms = (log_ts - self._video_start).total_seconds() * 1000.0
        if delta_ms < 0 or delta_ms > self._frame_times_ms[-1] + 500:
            return None
        idx = bisect.bisect_left(self._frame_times_ms, delta_ms)
        if idx >= self._total:
            idx = self._total - 1
        if idx > 0:
            # pick whichever frame is closer
            if abs(self._frame_times_ms[idx] - delta_ms) > abs(self._frame_times_ms[idx - 1] - delta_ms):
                idx = idx - 1
        return idx

    def read(self, idx: int) -> np.ndarray | None:
        if idx < 0 or idx >= self._total:
            return None
        if idx in self._cache:
            return self._cache[idx]
        # Need to read forward, or restart if target is behind cursor
        if idx < self._cur_global_pos:
            self._restart()
        while self._cur_global_pos <= idx:
            frame = self._read_next()
            if frame is None:
                break
        return self._cache.get(idx)

    def _restart(self):
        """Rewind to the beginning of the first file."""
        if self._cur_cap is not None:
            self._cur_cap.release()
            self._cur_cap = None
        self._cur_file_idx = -1
        self._cur_global_pos = 0

    def _open_next_file(self) -> bool:
        if self._cur_cap is not None:
            self._cur_cap.release()
        self._cur_file_idx += 1
        if self._cur_file_idx >= len(self._files):
            self._cur_cap = None
            return False
        self._cur_cap = cv2.VideoCapture(str(self._files[self._cur_file_idx]))
        return self._cur_cap.isOpened()

    def _read_next(self) -> np.ndarray | None:
        """Read the next sequential frame, advancing files as needed."""
        if self._cur_cap is None:
            if not self._open_next_file():
                return None
        ok, frame = self._cur_cap.read()
        if not ok:
            # Try next file
            if not self._open_next_file():
                return None
            ok, frame = self._cur_cap.read()
            if not ok:
                return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._cache[self._cur_global_pos] = rgb
        self._cur_global_pos += 1
        return rgb

    def close(self):
        if self._cur_cap is not None:
            self._cur_cap.release()
            self._cur_cap = None


def precompute_telemetry(frames: list[FramePose]):
    """Pre-compute position / vector arrays for fast 3D rendering."""
    n = len(frames)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    mag = np.zeros((n, 3))
    for i, fp in enumerate(frames):
        p = fp.pose
        pos[i] = _ned_to_plot(p.position.north_m, p.position.east_m, p.position.down_m)
        vel[i] = _ned_to_plot(p.velocity.north_m_s, p.velocity.east_m_s, p.velocity.down_m_s)
        if p.acceleration:
            an, ae, ad = rotate_frd_to_ned(
                p.quaternion, p.acceleration.forward_m_s2,
                p.acceleration.right_m_s2, p.acceleration.down_m_s2)
            acc[i] = _ned_to_plot(-an, -ae, -ad)
        if p.magnetic_field:
            mn, me, md = rotate_frd_to_ned(
                p.quaternion, p.magnetic_field.forward_gauss,
                p.magnetic_field.right_gauss, p.magnetic_field.down_gauss)
            mag[i] = _ned_to_plot(mn, me, md)
    origin = pos[0].copy()
    return pos - origin, vel, acc, mag


# ===========================================================================
# Frame controller
# ===========================================================================

class FrameController(QObject):
    """Holds the current frame index and notifies views on change."""
    frame_changed = pyqtSignal(int)

    def __init__(self, total: int):
        super().__init__()
        self._idx = 0
        self._total = total

    @property
    def index(self):
        return self._idx

    @property
    def total(self):
        return self._total

    def set_frame(self, idx: int):
        idx = max(0, min(self._total - 1, idx))
        if idx != self._idx:
            self._idx = idx
            self.frame_changed.emit(idx)

    def step(self, delta: int):
        self.set_frame(self._idx + delta)

    def at_end(self):
        return self._idx >= self._total - 1


# ===========================================================================
# Log view
# ===========================================================================

class LogView(QWidget):
    """Scrollable log with current-frame lines highlighted."""

    # Customize the colour used to highlight text-search matches
    SEARCH_HIGHLIGHT = QColor(200, 100, 255, 200)   # soft purple

    def __init__(self, lines: list[str], frame_to_lines: dict[int, list[int]],
                 frames: list[FramePose]):
        super().__init__()
        self._lines = lines
        self._ftl = frame_to_lines
        self._frames = frames
        self._last_idx = 0
        self._search_text = ""

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        self._edit = QPlainTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setFont(QFont("Monospace", 9))
        self._edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._edit.setPlainText("\n".join(lines))
        self._edit.selectionChanged.connect(self._on_selection_changed)
        lo.addWidget(self._edit)

    def _on_selection_changed(self):
        """Update search term from user selection and re-highlight."""
        selected = self._edit.textCursor().selectedText().strip()
        if selected:
            self._search_text = selected
        elif not self._edit.textCursor().hasSelection():
            # Only clear when the user explicitly deselects (clicks away),
            # not when setTextCursor clears it during scrolling
            self._search_text = ""
        self._apply_highlights(self._last_idx)

    def update_frame(self, idx: int):
        self._last_idx = idx
        self._apply_highlights(idx)

        # Scroll to first highlighted line — block signals to avoid
        # clearing the search text
        fid = self._frames[idx].frame_id
        line_idxs = self._ftl.get(fid, [])
        if line_idxs:
            doc = self._edit.document()
            block = doc.findBlockByNumber(line_idxs[0])
            if block.isValid():
                self._edit.blockSignals(True)
                cursor = self._edit.textCursor()
                cursor.setPosition(block.position())
                self._edit.setTextCursor(cursor)
                self._edit.centerCursor()
                self._edit.blockSignals(False)

    def _apply_highlights(self, idx: int):
        fid = self._frames[idx].frame_id
        line_idxs = self._ftl.get(fid, [])

        # Frame-line highlight format
        fmt = QTextCharFormat()
        fmt.setBackground(HighlightStyle.BACKGROUND)
        if HighlightStyle.FOREGROUND:
            fmt.setForeground(HighlightStyle.FOREGROUND)
        if HighlightStyle.BOLD:
            fmt.setFontWeight(QFont.Weight.Bold)

        doc = self._edit.document()
        sels: list[QTextEdit.ExtraSelection] = []

        # 1) Current-frame line highlights
        for li in line_idxs:
            block = doc.findBlockByNumber(li)
            if not block.isValid():
                continue
            sel = QTextEdit.ExtraSelection()
            sel.format = fmt
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock,
                                QTextCursor.MoveMode.KeepAnchor)
            sel.cursor = cursor
            sels.append(sel)

        # 2) Highlight all occurrences of the persisted search text
        selected = self._search_text
        if len(selected) >= 2:
            search_fmt = QTextCharFormat()
            search_fmt.setBackground(self.SEARCH_HIGHLIGHT)
            cursor = QTextCursor(doc)
            while True:
                cursor = doc.find(selected, cursor)
                if cursor.isNull():
                    break
                sel = QTextEdit.ExtraSelection()
                sel.format = search_fmt
                sel.cursor = cursor
                sels.append(sel)

        self._edit.setExtraSelections(sels)


# ===========================================================================
# Video view
# ===========================================================================

class VideoView(QWidget):
    """Displays one video frame, scaled to fit the widget."""

    def __init__(self, reader: VideoReader):
        super().__init__()
        self._reader = reader
        self._current_rgb: np.ndarray | None = None

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("No video")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Ignored)
        self._label.setMinimumSize(1, 1)
        self._label.setStyleSheet("background: black; color: white;")
        lo.addWidget(self._label)

    def update_frame(self, idx: int):
        vid_idx = idx
        self._current_rgb = (
            self._reader.read(vid_idx) if self._reader.available else None
        )
        self._display()

    def _display(self):
        if self._current_rgb is None:
            self._label.setPixmap(QPixmap())
            self._label.setText("No frame")
            return
        h, w, _ = self._current_rgb.shape
        qimg = QImage(self._current_rgb.data, w, h,
                      self._current_rgb.strides[0],
                      QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self._label.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.setText("")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._display()


# ===========================================================================
# Draggable info overlay (for the 3-D view)
# ===========================================================================

class DraggableInfoBox(QLabel):
    """Semi-transparent text box that can be repositioned by dragging."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(255,255,255,210); color: #000000;"
            "border: 1px solid #888;"
            "padding: 6px; font-family: monospace; font-size: 10px;"
        )
        self.setWordWrap(False)
        self.move(10, 10)
        self._drag_offset = QPoint()
        self.show()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = ev.position().toPoint()
            ev.accept()

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            new_pos = self.mapToParent(ev.position().toPoint()) - self._drag_offset
            self.move(new_pos)
            ev.accept()


# ===========================================================================
# Telemetry 3-D view
# ===========================================================================

class TelemetryView(QWidget):
    """Matplotlib-based 3-D visualisation of the drone state."""

    def __init__(self, frames: list[FramePose], pos, vel, acc, mag,
                 target_xy: dict[int, tuple[float, float]]):
        super().__init__()
        self._frames = frames
        self._pos = pos
        self._vel = vel
        self._acc = acc
        self._mag = mag
        self._target_xy = target_xy

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)

        self._fig = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        lo.addWidget(self._canvas, 1)

        # ---- vector visibility controls ----
        self._show_velocity = True
        self._show_acceleration = True
        self._show_normal = True
        self._show_target = True
        self._drone_view = False
        self._saved_view = None
        self._last_idx = 0

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 4, 2)
        self._cb_vel = QCheckBox("Velocity")
        self._cb_vel.setChecked(True)
        self._cb_acc = QCheckBox("Acceleration")
        self._cb_acc.setChecked(True)
        self._cb_normal = QCheckBox("Normal")
        self._cb_normal.setChecked(True)
        self._cb_target = QCheckBox("Target")
        self._cb_target.setChecked(True)
        self._cb_drone_view = QCheckBox("Drone View")
        self._cb_drone_view.setChecked(False)

        for cb in (self._cb_vel, self._cb_acc, self._cb_normal, self._cb_target, self._cb_drone_view):
            cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            controls.addWidget(cb)

        controls.addStretch()

        self._camera_pyramid_len = CAMERA_PYRAMID_LEN
        lbl_pyr = QLabel("Camera pyramid:")
        self._spin_pyramid = QSpinBox()
        self._spin_pyramid.setRange(1, 100)
        self._spin_pyramid.setValue(CAMERA_PYRAMID_LEN)
        self._spin_pyramid.setSingleStep(1)
        self._spin_pyramid.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._spin_pyramid.valueChanged.connect(self._on_pyramid_len_changed)
        controls.addWidget(lbl_pyr)
        controls.addWidget(self._spin_pyramid)

        lo.addLayout(controls)

        self._cb_vel.toggled.connect(lambda c: self._toggle("_show_velocity", c))
        self._cb_acc.toggled.connect(lambda c: self._toggle("_show_acceleration", c))
        self._cb_normal.toggled.connect(lambda c: self._toggle("_show_normal", c))
        self._cb_target.toggled.connect(lambda c: self._toggle("_show_target", c))
        self._cb_drone_view.toggled.connect(self._on_toggle_drone_view)

        self._ax = self._fig.add_subplot(111, projection="3d")

        # Draggable telemetry text
        self._info = DraggableInfoBox(self._canvas)

        # ---- static elements ----
        self._ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                      "-", color="0.70", lw=0.8, zorder=1)
        self._trail, = self._ax.plot([], [], [], "-",
                                     color="steelblue", lw=1.5, zorder=2)
        self._marker, = self._ax.plot([], [], [], "ko", ms=5, zorder=5)
        self._arm1, = self._ax.plot([], [], [], "-k", lw=2.5, zorder=6)
        self._arm2, = self._ax.plot([], [], [], "-k", lw=2.5, zorder=6)
        self._prop_lines = [
            self._ax.plot([], [], [], "-", color="0.35", lw=1.0, zorder=6)[0]
            for _ in range(4)
        ]
        self._quiv_vel = None
        self._quiv_acc = None
        self._quiv_mag = None
        self._quiv_normal = None
        self._quiv_front = None
        self._quiv_target = None
        self._cam_polys = None

        # Axis limits
        pad = 3.0
        mn = pos.min(0) - pad
        mx = pos.max(0) + pad
        rng = (mx - mn).max()
        mid = (mn + mx) / 2
        self._ax.set_xlim(mid[0] - rng / 2, mid[0] + rng / 2)
        self._ax.set_ylim(mid[1] - rng / 2, mid[1] + rng / 2)
        self._ax.set_zlim(mid[2] - rng / 2, mid[2] + rng / 2)
        self._ax.set_xlabel("East (m)")
        self._ax.set_ylabel("North (m)")
        self._ax.set_zlabel("Up (m)")
        # self._ax.set_title("Drone Telemetry")
        self._ax.view_init(elev=30, azim=45)

        # Ground plane at z=0 with cross-hatch grid
        ground_scale = 0.6
        x0, x1 = mid[0] - rng * ground_scale, mid[0] + rng * ground_scale
        y0, y1 = mid[1] - rng * ground_scale, mid[1] + rng * ground_scale
        ground = Poly3DCollection(
            [[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]],
            color=GROUND_COLOR, edgecolor=(0.2, 0.5, 0.2, 0.3),
            linewidth=0.5, zorder=0)
        self.ground_poly : Poly3DCollection = self._ax.add_collection3d(ground)

        # Diagonal cross-hatch lines
        grid_step = max(1.0, round(rng * ground_scale * 2 / 10))
        grid_color = (0.2, 0.5, 0.2, 0.25)
        span = max(x1 - x0, y1 - y0)
        for d in np.arange(-span, span + grid_step, grid_step):
            # Lines along x+y = const (bottom-left to top-right)
            lx0, ly0 = x0, x0 + d - x0  # y = d - x => at x=x0: y=d-x0
            lx1, ly1 = x1, d - x1
            lx0, ly0, lx1, ly1 = np.clip([lx0, ly0, lx1, ly1],
                                          [x0, y0, x0, y0], [x1, y1, x1, y1])
            # Clip: solve for intersections with box edges
            pts = []
            for bx, by in [(x0, d - x0), (x1, d - x1),
                           (d - y0, y0), (d - y1, y1)]:
                if x0 <= bx <= x1 and y0 <= by <= y1:
                    pts.append((bx, by))
            if len(pts) >= 2:
                self._ax.plot([pts[0][0], pts[1][0]],
                              [pts[0][1], pts[1][1]], [0, 0],
                              "-", color=grid_color, lw=0.4, zorder=0)
            # Lines along x-y = const (top-left to bottom-right)
            pts2 = []
            for bx, by in [(x0, x0 - d), (x1, x1 - d),
                           (d + y0, y0), (d + y1, y1)]:
                if x0 <= bx <= x1 and y0 <= by <= y1:
                    pts2.append((bx, by))
            if len(pts2) >= 2:
                self._ax.plot([pts2[0][0], pts2[1][0]],
                              [pts2[0][1], pts2[1][1]], [0, 0],
                              "-", color=grid_color, lw=0.4, zorder=0)

        self._ax.legend(handles=[
            Line2D([0], [0], color=VELOCITY_ARROW_COLOR, lw=3, label="Velocity"),
            Line2D([0], [0], color=ACCELERATION_ARROW_COLOR, lw=3, label="Acceleration"),
            Line2D([0], [0], color=TARGET_COLOR, lw=3, label="Target"),
            # Line2D([0], [0], color="dodgerblue", lw=2, label="Mag. bearing"),
            Line2D([0], [0], color="steelblue", lw=3, label="Past trail"),
        ], loc="upper right", fontsize=7)

        try:
            self._fig.tight_layout()
        except ValueError:
            pass  # widget has no size yet; layout adjusts on first resize

    # -----------------------------------------------------------------

    _TOGGLE_TO_QUIV = {
        "_show_velocity": "_quiv_vel",
        "_show_acceleration": "_quiv_acc",
        "_show_normal": "_quiv_normal",
        "_show_target": "_quiv_target",
    }

    def _toggle(self, attr: str, checked: bool):
        setattr(self, attr, checked)
        q = getattr(self, self._TOGGLE_TO_QUIV[attr])
        if q is not None:
            q.set_visible(checked)
        self._canvas.draw_idle()

    def _on_pyramid_len_changed(self, value: int):
        self._camera_pyramid_len = value
        self.update_frame(self._last_idx)

    def _on_toggle_drone_view(self, checked: bool):
        if checked:
            self._saved_view = (
                self._ax.elev, self._ax.azim,
                getattr(self._ax, 'roll', 0),
                self._ax.get_xlim(), self._ax.get_ylim(), self._ax.get_zlim(),
            )
            self._drone_view = True
            self.update_frame(self._last_idx)
        else:
            self._drone_view = False
            if self._saved_view:
                elev, azim, roll, xlim, ylim, zlim = self._saved_view
                self._ax.view_init(elev=elev, azim=azim, roll=roll)
                self._ax.set_xlim(xlim)
                self._ax.set_ylim(ylim)
                self._ax.set_zlim(zlim)
            self._canvas.draw_idle()

    def update_frame(self, idx: int):
        self._last_idx = idx
        fp = self._frames[idx]
        p = fp.pose
        cx, cy, cz = self._pos[idx]

        # Past trail
        self._trail.set_data_3d(
            self._pos[:idx + 1, 0],
            self._pos[:idx + 1, 1],
            self._pos[:idx + 1, 2])
        self._marker.set_data_3d([cx], [cy], [cz])

        # Drone body cross
        rotated = _quat_rotate_array(p.quaternion, BODY_VERTS_FRD)
        pv = _ned_array_to_plot(rotated) + self._pos[idx]
        self._arm1.set_data_3d(
            [pv[0, 0], pv[1, 0]], [pv[0, 1], pv[1, 1]], [pv[0, 2], pv[1, 2]])
        self._arm2.set_data_3d(
            [pv[2, 0], pv[3, 0]], [pv[2, 1], pv[3, 1]], [pv[2, 2], pv[3, 2]])

        # Prop circles at each arm tip
        for pi in range(4):
            centre = PROP_CENTRES_FRD[pi]
            # Circle points in FRD body plane (Z=0), centred at arm tip
            circle_frd = np.column_stack([
                centre[0] + _PROP_UNIT_X,
                centre[1] + _PROP_UNIT_Y,
                np.zeros(_PROP_N + 1),
            ])
            circle_ned = _quat_rotate_array(p.quaternion, circle_frd)
            cp = _ned_array_to_plot(circle_ned) + self._pos[idx]
            self._prop_lines[pi].set_data_3d(cp[:, 0], cp[:, 1], cp[:, 2])


        # Remove old quiver arrows and camera polys
        for attr in ("_quiv_vel", "_quiv_acc", "_quiv_front", "_quiv_normal", "_quiv_target"):
            q = getattr(self, attr)
            if q is not None:
                q.set_visible(False)
                setattr(self, attr, None)
        if self._cam_polys is not None:
            self._cam_polys.remove()
            self._cam_polys = None

        # Front direction arrow (FRD +X projected onto body plane)
        fwd_frd = np.array([[ARM_LEN * 0.5, 0.0, 0.0]])
        fwd_ned = _quat_rotate_array(p.quaternion, fwd_frd)
        f = _ned_array_to_plot(fwd_ned)[0]
        self._quiv_front = self._ax.quiver(
            cx, cy, cz, f[0], f[1], f[2],
            color="black", arrow_length_ratio=0.25, lw=3, zorder=7)

        # Body normal (FRD -Z = "up" from the drone)
        normal_frd = np.array([[0.0, 0.0, -ARM_LEN * 0.5]])
        normal_ned = _quat_rotate_array(p.quaternion, normal_frd)
        normal_plot = _ned_array_to_plot(normal_ned)[0]
        self._quiv_normal = self._ax.quiver(
            cx, cy, cz, normal_plot[0], normal_plot[1], normal_plot[2],
            color="black", arrow_length_ratio=0.25, lw=2.5, zorder=7)
        self._quiv_normal.set_visible(self._show_normal)

        # Velocity
        v = self._vel[idx] * VECTOR_SCALE["vel"]
        self._quiv_vel = self._ax.quiver(
            cx, cy, cz, v[0], v[1], v[2],
            color=VELOCITY_ARROW_COLOR, arrow_length_ratio=0.15, lw=1.8)
        self._quiv_vel.set_visible(self._show_velocity)

        # Acceleration
        a = self._acc[idx] * VECTOR_SCALE["acc"]
        self._quiv_acc = self._ax.quiver(
            cx, cy, cz, a[0], a[1], a[2],
            color=ACCELERATION_ARROW_COLOR, arrow_length_ratio=0.15, lw=1.8)
        self._quiv_acc.set_visible(self._show_acceleration)

        # Camera FOV pyramid — tip at drone centre, extends along body -Z (up)
        tip = np.array([cx, cy, cz])
        L = self._camera_pyramid_len
        hh = L * math.tan(math.radians(CAMERA_HFOV_DEG / 2))
        hv = L * math.tan(math.radians(CAMERA_VFOV_DEG / 2))
        base_frd = np.array([
            [-hv, -hh, -L],
            [-hv,  hh, -L],
            [ hv,  hh, -L],
            [ hv, -hh, -L],
        ])
        base_ned = _quat_rotate_array(p.quaternion, base_frd)
        base_plot = _ned_array_to_plot(base_ned) + np.array([cx, cy, cz])
        b = base_plot
        faces = [
            [tip, b[0], b[1]],
            [tip, b[1], b[2]],
            [tip, b[2], b[3]],
            [tip, b[3], b[0]],
            [b[0], b[1], b[2], b[3]],
        ]
        self._cam_polys = Poly3DCollection(
            faces, color=CAMERA_COLOR, edgecolor=(0.4, 0.4, 0.4, 0.1),
            linewidth=0.5, zorder=4)
        self._ax.add_collection3d(self._cam_polys)

        # Target direction vector
        target = self._target_xy.get(fp.frame_id)
        if target is not None:
            tx, ty = target
            target_frd = np.array([[-ty * 2 * hv, -tx * 2 * hh, -L]])
            target_ned = _quat_rotate_array(p.quaternion, target_frd)
            t = _ned_array_to_plot(target_ned)[0]
            self._quiv_target = self._ax.quiver(
                cx, cy, cz, t[0], t[1], t[2],
                color=TARGET_COLOR, arrow_length_ratio=0.05, lw=4.0)
            self._quiv_target.set_visible(self._show_target)

        # Body normal orientation (normal_plot is in plot coords: East, North, Up)
        n_horiz = math.hypot(normal_plot[0], normal_plot[1])
        normal_elev = math.degrees(math.atan2(normal_plot[2], n_horiz))
        normal_azim = math.degrees(math.atan2(normal_plot[0], normal_plot[1])) % 360

        # Info overlay
        ref = self._frames[0].pose
        vel_m = np.linalg.norm(self._vel[idx])
        acc_m = np.linalg.norm(self._acc[idx])
        self._info.setText(
            f"Frame #{fp.frame_id:04d}  "
            f"{fp.timestamp.strftime('%H:%M:%S.%f')[:-3]}\n"
            f"N={p.position.north_m - ref.position.north_m:+.2f}m  "
            f"E={p.position.east_m - ref.position.east_m:+.2f}m  "
            f"Alt={p.position.altitude_m:.2f}m\n"
            f"Yaw={p.orientation.yaw_deg:.1f}\u00b0  "
            f"Pitch={p.orientation.pitch_deg:.1f}\u00b0  "
            f"Roll={p.orientation.roll_deg:.1f}\u00b0\n"
            # f"Normal: El={normal_elev:.1f}\u00b0  "
            # f"Az={normal_azim:.1f}\u00b0\n"
            f"|V|={vel_m:.2f} m/s  |A|={acc_m:.2f} m/s\u00b2"
        )
        self._info.adjustSize()

        # too much clutter in drone view
        self.ground_poly.set_visible(not self._drone_view)

        # Drone-view: bypass elev/azim/roll — build view matrix directly from
        # the camera pyramid body axes so the view is pixel-locked to them.
        if self._drone_view:
            # Body axes in plot coords (East, North, Up)
            # toward-eye  = body -Z  (normal, up from drone)
            n_vec = normal_plot / np.linalg.norm(normal_plot)
            # screen right = body -Y  (right-handed: u×v = n)
            rt_n, rt_e, rt_d = rotate_frd_to_ned(p.quaternion, 0, -1, 0)
            u_vec = np.array(_ned_to_plot(rt_n, rt_e, rt_d), dtype=float)
            u_vec /= np.linalg.norm(u_vec)
            # screen up    = body -X  (forward points down on screen)
            up_n, up_e, up_d = rotate_frd_to_ned(p.quaternion, -1, 0, 0)
            v_vec = np.array(_ned_to_plot(up_n, up_e, up_d), dtype=float)
            v_vec /= np.linalg.norm(v_vec)

            # Centre view along camera look direction (body +Z = down)
            center = np.array([cx, cy, cz]) - n_vec * CAMERA_PYRAMID_LEN * 0.5
            view_range = CAMERA_PYRAMID_LEN * 0.7
            self._ax.set_xlim(center[0] - view_range, center[0] + view_range)
            self._ax.set_ylim(center[1] - view_range, center[1] + view_range)
            self._ax.set_zlim(center[2] - view_range, center[2] + view_range)

            # Build the full projection matrix directly from body axes,
            # bypassing matplotlib's elev/azim/roll decomposition entirely.
            ax = self._ax
            box_aspect = ax._roll_to_vertical(ax._box_aspect)
            worldM = proj3d.world_transformation(
                *ax.get_xlim3d(), *ax.get_ylim3d(), *ax.get_zlim3d(),
                pb_aspect=box_aspect)
            R = 0.5 * box_aspect
            eye = R + ax._dist * n_vec

            Mr = np.eye(4)
            Mt = np.eye(4)
            Mr[:3, :3] = [u_vec, v_vec, n_vec]
            Mt[:3, -1] = -eye
            viewM = Mr @ Mt

            projM = proj3d.ortho_transformation(-ax._dist, ax._dist)
            ax.M = projM @ viewM @ worldM

        self._canvas.draw_idle()


# ===========================================================================
# Navigation bar
# ===========================================================================

class NavigationBar(QWidget):
    """Transport controls: step back, play / pause, step forward, slider."""

    switch_requested = pyqtSignal()

    def __init__(self, controller: FrameController, frames: list[FramePose],
                 show_switch: bool = False):
        super().__init__()
        self._ctrl = controller
        self._frames = frames
        self._playing = False

        self._timer = QTimer()
        self._timer.setInterval(PLAYBACK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        main = QVBoxLayout(self)
        main.setContentsMargins(8, 2, 8, 4)

        # Row 1: buttons + frame label
        row1 = QHBoxLayout()
        self._btn_prev = QPushButton("\u25c0 Step")
        self._btn_play = QPushButton("\u25b6 Play")
        self._btn_next = QPushButton("Step \u25b6")
        self._lbl = QLabel()
        self._lbl.setFont(QFont("Monospace", 10))

        for b in (self._btn_prev, self._btn_play, self._btn_next):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        row1.addWidget(self._btn_prev)
        row1.addWidget(self._btn_play)
        row1.addWidget(self._btn_next)
        row1.addSpacing(20)
        row1.addWidget(self._lbl)
        row1.addStretch()

        if show_switch:
            self._btn_switch = QPushButton("Switch\u2026")
            self._btn_switch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._btn_switch.clicked.connect(self.switch_requested.emit)
            row1.addWidget(self._btn_switch)

        main.addLayout(row1)

        # Row 2: progress slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, controller.total - 1)
        main.addWidget(self._slider)

        # Connections
        self._btn_prev.clicked.connect(lambda: controller.step(-1))
        self._btn_next.clicked.connect(lambda: controller.step(1))
        self._btn_play.clicked.connect(self.toggle_play)
        self._slider.valueChanged.connect(controller.set_frame)
        controller.frame_changed.connect(self._on_frame_changed)

        self._update_label(0)

    def _update_label(self, idx: int):
        fp = self._frames[idx]
        self._lbl.setText(
            f"#{fp.frame_id:04d}  "
            f"{fp.timestamp.strftime('%H:%M:%S.%f')[:-3]}  "
            f"[{idx}/{self._ctrl.total - 1}]"
        )

    def _on_frame_changed(self, idx: int):
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._update_label(idx)

    def toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self._btn_play.setText("\u275A\u275A Pause")
            self._timer.start()
        else:
            self._btn_play.setText("\u25b6 Play")
            self._timer.stop()

    def _tick(self):
        if self._ctrl.at_end():
            self.toggle_play()
            return
        self._ctrl.step(1)

    def reload(self, controller: FrameController, frames: list[FramePose]):
        """Replace the underlying controller and frame data for a new session."""
        if self._playing:
            self.toggle_play()
        self._ctrl.frame_changed.disconnect(self._on_frame_changed)
        self._slider.valueChanged.disconnect(self._ctrl.set_frame)
        self._btn_prev.clicked.disconnect()
        self._btn_next.clicked.disconnect()

        self._ctrl = controller
        self._frames = frames
        self._slider.setRange(0, controller.total - 1)
        self._slider.setValue(0)

        self._btn_prev.clicked.connect(lambda: controller.step(-1))
        self._btn_next.clicked.connect(lambda: controller.step(1))
        self._slider.valueChanged.connect(controller.set_frame)
        controller.frame_changed.connect(self._on_frame_changed)
        self._update_label(0)


# ===========================================================================
# Main window
# ===========================================================================

def _titled(title: str, widget: QWidget) -> QGroupBox:
    """Wrap a widget in a titled QGroupBox."""
    box = QGroupBox(title)
    lo = QVBoxLayout(box)
    lo.setContentsMargins(2, 2, 2, 2)
    lo.addWidget(widget)
    return box


class FlightDebugger(QWidget):
    def __init__(self, frames: list[FramePose], log_lines: list[str],
                 frame_to_lines: dict[int, list[int]],
                 video: VideoReader, log_path: Path,
                 video_path: Path | None = None,
                 pairs: list[tuple[Path, list[Path]]] | None = None):
        super().__init__()
        self.setWindowTitle("Flight Debugger")
        self.showMaximized()
        self._pairs = pairs

        self._ctrl = FrameController(len(frames))

        # ---- views ----
        self._log_view = LogView(log_lines, frame_to_lines, frames)
        self._video_view = VideoView(video)

        pos, vel, acc, mag = precompute_telemetry(frames)
        target_xy = parse_target_xy(log_lines)
        self._telem_view = TelemetryView(frames, pos, vel, acc, mag, target_xy)

        self._nav = NavigationBar(self._ctrl, frames,
                                  show_switch=bool(pairs))

        # ---- layout: splitters ----
        vid_title = f"Video — {video_path.name}" if video_path else "Video"

        self._settings = QSettings("FlightDebugger", "FlightDebugger")

        # Top row: video | telemetry (horizontal splitter)
        self._top_split = QSplitter(Qt.Orientation.Horizontal)
        self._top_split.addWidget(_titled(vid_title, self._video_view))
        self._top_split.addWidget(_titled("Telemetry 3D", self._telem_view))

        # Main: top_split / log (vertical splitter)
        self._main_split = QSplitter(Qt.Orientation.Vertical)
        self._main_split.addWidget(self._top_split)
        self._main_split.addWidget(_titled(f"Log — {log_path.name}", self._log_view))

        # Restore saved sizes or use defaults (50/50 top, 50/50 vertical)
        saved_top = self._settings.value("top_split_sizes")
        saved_main = self._settings.value("main_split_sizes")
        if saved_top:
            self._top_split.setSizes([int(s) for s in saved_top])
        else:
            self._top_split.setSizes([1, 1])
        if saved_main:
            self._main_split.setSizes([int(s) for s in saved_main])
        else:
            self._main_split.setSizes([1, 1])

        # Save on every resize
        self._top_split.splitterMoved.connect(self._save_splitter_sizes)
        self._main_split.splitterMoved.connect(self._save_splitter_sizes)

        # Outer layout: main_split + nav bar
        outer = QVBoxLayout(self)
        outer.addWidget(self._main_split, 1)
        outer.addWidget(self._nav, 0)

        if self._pairs:
            self._nav.switch_requested.connect(self._on_switch_session)

        # ---- connect views to controller ----
        self._ctrl.frame_changed.connect(self._log_view.update_frame)
        self._ctrl.frame_changed.connect(self._video_view.update_frame)
        self._ctrl.frame_changed.connect(self._telem_view.update_frame)

        # ---- keyboard shortcuts (use methods so they track self._ctrl) ----
        self._sc_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._sc_left.activated.connect(self._step_back)
        self._sc_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._sc_right.activated.connect(self._step_forward)
        self._sc_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._sc_space.activated.connect(self._toggle_play)

        # ---- initial render ----
        self._log_view.update_frame(0)
        self._video_view.update_frame(0)
        self._telem_view.update_frame(0)

    def _save_splitter_sizes(self):
        self._settings.setValue("top_split_sizes", self._top_split.sizes())
        self._settings.setValue("main_split_sizes", self._main_split.sizes())

    def _step_back(self):
        self._ctrl.step(-1)

    def _step_forward(self):
        self._ctrl.step(1)

    def _toggle_play(self):
        self._nav.toggle_play()

    def _on_switch_session(self):
        if self._nav._playing:
            self._nav.toggle_play()
        dlg = SessionPicker(self._pairs, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        result = dlg.selected_pair()
        if result is None:
            return
        self._load_session(*result)

    def _load_session(self, log_path: Path, vid_files: list[Path]):
        """Load a new log + video pair into all views."""
        frames = parse_telemetry_log(log_path)
        if not frames:
            return
        log_lines, frame_to_lines = parse_log_lines(log_path)
        new_video = VideoReader(vid_files or None)

        # Tear down old state
        self._video_view._reader.close()
        # Disconnect only the three view slots; leave nav's connection
        # intact so that nav.reload() can disconnect it properly.
        self._ctrl.frame_changed.disconnect(self._log_view.update_frame)
        self._ctrl.frame_changed.disconnect(self._video_view.update_frame)
        self._ctrl.frame_changed.disconnect(self._telem_view.update_frame)

        # New controller and views
        self._ctrl = FrameController(len(frames))
        self._log_view = LogView(log_lines, frame_to_lines, frames)
        self._video_view = VideoView(new_video)
        pos, vel, acc, mag = precompute_telemetry(frames)
        target_xy = parse_target_xy(log_lines)
        self._telem_view = TelemetryView(frames, pos, vel, acc, mag, target_xy)

        # Replace group boxes in splitters (preserve sizes)
        top_sizes = self._top_split.sizes()
        main_sizes = self._main_split.sizes()

        vid_title = f"Video \u2014 {vid_files[0].name}" if vid_files else "Video"
        for idx, new_box in [
            (0, _titled(vid_title, self._video_view)),
            (1, _titled("Telemetry 3D", self._telem_view)),
        ]:
            old = self._top_split.widget(idx)
            self._top_split.replaceWidget(idx, new_box)
            old.deleteLater()

        old_log_box = self._main_split.widget(1)
        self._main_split.replaceWidget(
            1, _titled(f"Log \u2014 {log_path.name}", self._log_view))
        old_log_box.deleteLater()

        self._top_split.setSizes(top_sizes)
        self._main_split.setSizes(main_sizes)

        # Reconnect controller -> views
        self._ctrl.frame_changed.connect(self._log_view.update_frame)
        self._ctrl.frame_changed.connect(self._video_view.update_frame)
        self._ctrl.frame_changed.connect(self._telem_view.update_frame)

        # Reload navigation bar
        self._nav.reload(self._ctrl, frames)

        # Initial render
        self._log_view.update_frame(0)
        self._video_view.update_frame(0)
        self._telem_view.update_frame(0)


# ===========================================================================
# Directory scanning & session matching
# ===========================================================================

_LOG_TS_RE = re.compile(r"BDD_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
_LOG_TS_RE2 = re.compile(r"BDD_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")
_VID_TS_RE = re.compile(r"(\d{8}-\d{6})")
_VID_PREFIX_RE = re.compile(r"^(.+?)_?\d{8}-\d{6}")
_MAX_PAIR_GAP_S = 5.0


def _extract_log_timestamp(path: Path) -> datetime | None:
    """Extract timestamp from log filename like BDD_2026_04_11_13_11_33_02_00_.log."""
    m = _LOG_TS_RE.search(path.stem)
    if not m:
        m = _LOG_TS_RE2.search(path.stem)

    if m:
        return datetime(*(int(m.group(i)) for i in range(1, 7)))

    return None


def _extract_video_timestamp(path: Path) -> datetime | None:
    """Extract timestamp from video filename like debug_20260411-131133_00000.mkv."""
    m = _VID_TS_RE.search(path.stem)
    if m:
        s = m.group(1)
        return datetime.strptime(s, "%Y%m%d-%H%M%S")
    return None


def discover_pairs(directory: Path) -> list[tuple[Path, list[Path]]]:
    """Scan *directory* for .log + video pairs, matched by filename timestamp.

    Returns a list of ``(log_path, [video_files])`` sorted by log timestamp.
    Prefers ``debug_`` videos when both ``debug_`` and ``RAW_`` exist.
    """
    logs = sorted(
        (p for p in directory.iterdir()
         if p.suffix == ".log" and _extract_log_timestamp(p) is not None),
        key=lambda p: _extract_log_timestamp(p),
    )
    videos = sorted(
        p for p in directory.iterdir()
        if p.suffix in (".mkv", ".mp4")
    )

    # Group video files by (prefix, timestamp_key).
    # prefix = "debug", "RAW", "clip", etc.  ts_key = "YYYYMMDD-HHMMSS".
    vid_groups: dict[str, dict[str, list[Path]]] = {}   # ts_key → prefix → files
    for v in videos:
        ts_m = _VID_TS_RE.search(v.stem)
        pref_m = _VID_PREFIX_RE.match(v.stem)
        if not ts_m:
            continue
        ts_key = ts_m.group(1)
        prefix = pref_m.group(1) if pref_m else ""
        vid_groups.setdefault(ts_key, {}).setdefault(prefix, []).append(v)

    # For each timestamp pick the best prefix (prefer "debug").
    best_videos: dict[str, list[Path]] = {}
    for ts_key, by_prefix in vid_groups.items():
        for pref in ("debug", "RAW"):
            if pref in by_prefix:
                best_videos[ts_key] = sorted(by_prefix[pref])
                break
        else:
            best_videos[ts_key] = sorted(next(iter(by_prefix.values())))

    vid_entries = sorted(
        ((datetime.strptime(k, "%Y%m%d-%H%M%S"), k) for k in best_videos),
        key=lambda x: x[0],
    )

    # Match each log to the closest video group where video_time >= log_time.
    used: set[str] = set()
    pairs: list[tuple[Path, list[Path]]] = []
    for log_path in logs:
        log_ts = _extract_log_timestamp(log_path)
        best_key: str | None = None
        best_diff = float("inf")
        for vid_ts, ts_key in vid_entries:
            diff = (vid_ts - log_ts).total_seconds()
            if 0 <= diff <= _MAX_PAIR_GAP_S and diff < best_diff and ts_key not in used:
                best_diff = diff
                best_key = ts_key
        if best_key is not None:
            used.add(best_key)
            pairs.append((log_path, best_videos[best_key]))
        else:
            pairs.append((log_path, []))

    return pairs


class SessionPicker(QDialog):
    """Dialog for choosing a log + video pair from a scanned directory."""

    def __init__(self, pairs: list[tuple[Path, list[Path]]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select recording session")
        self.setMinimumSize(700, 400)
        self._pairs = pairs

        layout = QVBoxLayout(self)
        self._list = QListWidget()
        self._list.setFont(QFont("Monospace", 10))

        for log_path, vid_files in pairs:
            log_ts = _extract_log_timestamp(log_path)
            ts_str = log_ts.strftime("%Y-%m-%d %H:%M:%S") if log_ts else "?"
            log_mb = log_path.stat().st_size / (1024 * 1024)
            if vid_files:
                base = vid_files[0].stem.rsplit("_", 1)[0]
                n = len(vid_files)
                vid_mb = sum(f.stat().st_size for f in vid_files) / (1024 * 1024)
                vid_info = f"{base}  ({n} clip{'s' if n != 1 else ''}, {vid_mb:.1f} MB)"
            else:
                vid_info = "(no video)"
            self._list.addItem(
                f"{ts_str}   {log_path.name} ({log_mb:.1f} MB)"
                f"  \u2192  {vid_info}"
            )

        self._list.setCurrentRow(len(pairs) - 1)   # pre-select last (most recent)
        self._list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self._list)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("Open")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def selected_pair(self) -> tuple[Path, list[Path]] | None:
        row = self._list.currentRow()
        if 0 <= row < len(self._pairs):
            return self._pairs[row]
        return None


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-view synchronized flight debugger")
    parser.add_argument("log_file", nargs="?", type=Path, default=None,
                        help="Path to BDD .log file")
    parser.add_argument("--video", type=Path, default=None,
                        help="Video file or directory of MP4s")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory with log + video files; "
                             "auto-discovers pairs and shows a picker")
    parser.add_argument("--video-offset", type=int, default=0,
                        help="Video frame offset: video_idx = frame_idx + offset")
    parser.add_argument("--allow-no-video", type=bool, default=False,
                        help="Do not error if video can't be loaded")
    args = parser.parse_args()

    # --dir mode: scan directory, show picker, then fall through to common path
    video_source: Path | list[Path] | None = args.video
    video_display_path: Path | None = args.video
    pairs = None

    if args.dir:
        if not args.dir.is_dir():
            print(f"ERROR: {args.dir} is not a directory")
            sys.exit(1)
        pairs = discover_pairs(args.dir)
        if not pairs:
            print(f"ERROR: no log files found in {args.dir}")
            sys.exit(1)
        # Let user pick a session
        app = QApplication(sys.argv)
        dlg = SessionPicker(pairs)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        result = dlg.selected_pair()
        if result is None:
            sys.exit(0)
        args.log_file, vid_files = result
        video_source = vid_files or None
        video_display_path = vid_files[0] if vid_files else None
    else:
        if args.log_file is None:
            parser.error("log_file is required (or use --dir)")
        app = QApplication(sys.argv)

    print(f"Loading log: {args.log_file}")
    frames = parse_telemetry_log(args.log_file)
    log_lines, frame_to_lines = parse_log_lines(args.log_file)
    print(f"  {len(frames)} telemetry frames, {len(log_lines)} log lines")

    if not frames:
        print("ERROR: no telemetry frames found in log file.")
        sys.exit(1)

    video = VideoReader(video_source)
    if video.available:
        print(f"  {video.total} video frames")
        if video.has_time_sync:
            print(f"  video start: {video._video_start} (auto-detected from filename)")
        else:
            print("  WARNING: could not detect video start time; using frame offset")
    else:
        if not args.dir and not args.allow_no_video:
            print(f"Can't load video from {args.video}")
            sys.exit(-1)
        print("  (no video)")

    win = FlightDebugger(frames, log_lines, frame_to_lines,
                         video, args.log_file, video_display_path,
                         pairs=pairs)
    ret = app.exec()
    video.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
