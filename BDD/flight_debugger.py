#!/usr/bin/env python3
"""Multi-view synchronized flight debugger.

Three synchronized, dockable views:
  - Debug video playback
  - Textual log with per-frame highlighting
  - 3D telemetry visualization

Usage:
    cd BDD
    python flight_debugger.py path/to/log_file.log [--video path/to/video.mp4]
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
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
from PyQt6.QtGui import (
    QImage, QPixmap, QTextCursor, QTextCharFormat, QColor, QFont,
    QAction, QShortcut, QKeySequence,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

# Qt property id for full-width extra-selection highlight
_FULL_WIDTH_SELECTION = 0x06010

# Drone body geometry (FRD frame)
ARM_LEN = 5
BODY_VERTS_FRD = np.array([
    [ ARM_LEN, 0.0, 0.0], [-ARM_LEN, 0.0, 0.0],
    [0.0,  ARM_LEN, 0.0], [0.0, -ARM_LEN, 0.0],
    [0.0, 0.0, 0.0],
    [ARM_LEN * 1.15,  0.08, 0.0],
    [ARM_LEN * 1.15, -0.08, 0.0],
])

VECTOR_SCALE = {"vel": 3.0, "acc": 0.3, "mag": 8.0}
PLAYBACK_INTERVAL_MS = 50

# Camera FOV pyramid — camera points along body -Z (up from drone back).
# Tip sits at the normal endpoint; base extends outward.
CAMERA_HFOV_DEG = 120.0   # horizontal full field-of-view
CAMERA_VFOV_DEG = 90.0   # vertical full field-of-view
CAMERA_PYRAMID_LEN = 50  # visual length of the pyramid (metres)
CAMERA_COLOR = (1,   0.5, 0.5, 0.1)
GROUND_COLOR = (0.2, 1, 0.2, 0.3) #


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


class VideoReader:
    """Read frames from one or more video files (MP4, MKV, etc.).

    Uses sequential reading with on-demand caching because MKV
    frame-based seeking is unreliable with many codecs.
    Supports timestamp-based frame lookup for log synchronization.
    """

    def __init__(self, path: Path | None):
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
        files = (
            sorted(path.glob("*.mp4")) + sorted(path.glob("*.mkv"))
            if path.is_dir()
            else [path] if path.exists() else []
        )
        cum_ms = 0.0
        for f in files:
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                continue
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            times: list[float] = []
            if n <= 0:
                # MKV / unreliable count — scan and collect timestamps
                while cap.grab():
                    times.append(cum_ms + cap.get(cv2.CAP_PROP_POS_MSEC))
                n = len(times)
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                times = [cum_ms + i * 1000.0 / fps for i in range(n)]
            cap.release()
            self._files.append(f)
            self._file_counts.append(n)
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
            acc[i] = _ned_to_plot(an, ae, ad)
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
    SEARCH_HIGHLIGHT = QColor(200, 100, 255, 90)   # soft purple

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
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
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

    def __init__(self, frames: list[FramePose], pos, vel, acc, mag):
        super().__init__()
        self._frames = frames
        self._pos = pos
        self._vel = vel
        self._acc = acc
        self._mag = mag

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)

        self._fig = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        lo.addWidget(self._canvas)

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
        self._nose_poly = None
        self._quiv_vel = None
        self._quiv_acc = None
        self._quiv_mag = None
        self._quiv_normal = None
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
        self._ax.set_title("Drone Telemetry")
        self._ax.view_init(elev=30, azim=45)

        self._ax.legend(handles=[
            Line2D([0], [0], color="green", lw=2, label="Velocity"),
            Line2D([0], [0], color="red", lw=2, label="Acceleration"),
            Line2D([0], [0], color="dodgerblue", lw=2, label="Mag. bearing"),
            Line2D([0], [0], color="orange", lw=2, label="Body normal (up)"),
            Line2D([0], [0], color="steelblue", lw=1.5, label="Past trail"),
        ], loc="upper right", fontsize=7)

        try:
            self._fig.tight_layout()
        except ValueError:
            pass  # widget has no size yet; layout adjusts on first resize

    # -----------------------------------------------------------------

    def update_frame(self, idx: int):
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

        # Nose triangle
        if self._nose_poly is not None:
            self._nose_poly.remove()
        self._nose_poly = Poly3DCollection(
            [pv[[0, 5, 6]]], color="red", alpha=0.8, zorder=7)
        self._ax.add_collection3d(self._nose_poly)

        # Remove old quiver arrows
        for q in (self._quiv_vel, self._quiv_acc, self._quiv_mag, self._quiv_normal):
            if q is not None:
                q.remove()

        v = self._vel[idx] * VECTOR_SCALE["vel"]
        self._quiv_vel = self._ax.quiver(
            cx, cy, cz, v[0], v[1], v[2],
            color="green", arrow_length_ratio=0.15, lw=1.8)

        a = self._acc[idx] * VECTOR_SCALE["acc"]
        self._quiv_acc = self._ax.quiver(
            cx, cy, cz, a[0], a[1], a[2],
            color="red", arrow_length_ratio=0.15, lw=1.8)

        m = self._mag[idx] * VECTOR_SCALE["mag"]
        self._quiv_mag = self._ax.quiver(
            cx, cy, cz, m[0], m[1], m[2],
            color="dodgerblue", arrow_length_ratio=0.15, lw=1.8)

        # Body-normal (FRD "up" = 0, 0, -1) rotated to NED then plot coords
        nn, ne, nd = rotate_frd_to_ned(p.quaternion, 0.0, 0.0, -1.0)
        up = np.array(_ned_to_plot(nn, ne, nd)) * ARM_LEN * 1.5
        self._quiv_normal = self._ax.quiver(
            cx, cy, cz, up[0], up[1], up[2],
            color="orange", arrow_length_ratio=0.15, lw=2.0)

        # Camera FOV pyramid — tip at drone centre, extends along body -Z (up)
        if self._cam_polys is not None:
            self._cam_polys.remove()
            self._cam_polys = None
        tip = np.array([cx, cy, cz])  # drone centre = base of normal
        L = CAMERA_PYRAMID_LEN
        hh = L * math.tan(math.radians(CAMERA_HFOV_DEG / 2))
        hv = L * math.tan(math.radians(CAMERA_VFOV_DEG / 2))
        # Base corners in FRD: camera looks along -Z, X=forward, Y=right
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
            f"|V|={vel_m:.2f} m/s  |A|={acc_m:.2f} m/s\u00b2"
        )
        self._info.adjustSize()

        self._canvas.draw_idle()


# ===========================================================================
# Navigation bar
# ===========================================================================

class NavigationBar(QWidget):
    """Transport controls: step back, play / pause, step forward, slider."""

    def __init__(self, controller: FrameController, frames: list[FramePose]):
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
            self._btn_play.setText("\u23f8 Pause")
            self._timer.start()
        else:
            self._btn_play.setText("\u25b6 Play")
            self._timer.stop()

    def _tick(self):
        if self._ctrl.at_end():
            self.toggle_play()
            return
        self._ctrl.step(1)


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
                 video_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("Flight Debugger")
        self.showMaximized()

        self._ctrl = FrameController(len(frames))

        # ---- views ----
        self._log_view = LogView(log_lines, frame_to_lines, frames)
        self._video_view = VideoView(video)

        pos, vel, acc, mag = precompute_telemetry(frames)
        self._telem_view = TelemetryView(frames, pos, vel, acc, mag)

        self._nav = NavigationBar(self._ctrl, frames)

        # ---- layout: splitters ----
        vid_title = f"Video — {video_path.name}" if video_path else "Video"

        # Top row: video | telemetry (horizontal splitter)
        top_split = QSplitter(Qt.Orientation.Horizontal)
        top_split.addWidget(_titled(vid_title, self._video_view))
        top_split.addWidget(_titled("Telemetry 3D", self._telem_view))
        top_split.setSizes([1, 1])   # equal 50/50

        # Main: top_split / log (vertical splitter)
        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.addWidget(top_split)
        main_split.addWidget(_titled(f"Log — {log_path.name}", self._log_view))
        main_split.setStretchFactor(0, 4)   # top gets 4/5
        main_split.setStretchFactor(1, 1)   # log gets 1/5

        # Outer layout: main_split + nav bar
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(main_split, 1)
        outer.addWidget(self._nav, 0)

        # ---- connect views to controller ----
        self._ctrl.frame_changed.connect(self._log_view.update_frame)
        self._ctrl.frame_changed.connect(self._video_view.update_frame)
        self._ctrl.frame_changed.connect(self._telem_view.update_frame)

        # ---- keyboard shortcuts ----
        self._sc_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._sc_left.activated.connect(lambda: self._ctrl.step(-1))
        self._sc_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._sc_right.activated.connect(lambda: self._ctrl.step(1))
        self._sc_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._sc_space.activated.connect(self._nav.toggle_play)

        # ---- initial render ----
        self._log_view.update_frame(0)
        self._video_view.update_frame(0)
        self._telem_view.update_frame(0)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-view synchronized flight debugger")
    parser.add_argument("log_file", type=Path,
                        help="Path to BDD .log file")
    parser.add_argument("--video", type=Path, default=None,
                        help="Video file or directory of MP4s")
    parser.add_argument("--video-offset", type=int, default=0,
                        help="Video frame offset: video_idx = frame_idx + offset")
    args = parser.parse_args()

    print(f"Loading log: {args.log_file}")
    frames = parse_telemetry_log(args.log_file)
    log_lines, frame_to_lines = parse_log_lines(args.log_file)
    print(f"  {len(frames)} telemetry frames, {len(log_lines)} log lines")

    if not frames:
        print("ERROR: no telemetry frames found in log file.")
        sys.exit(1)

    video = VideoReader(args.video)
    if video.available:
        print(f"  {video.total} video frames")
        if video.has_time_sync:
            print(f"  video start: {video._video_start} (auto-detected from filename)")
        else:
            print("  WARNING: could not detect video start time; using frame offset")
    else:
        print("  (no video)")

    app = QApplication(sys.argv)
    win = FlightDebugger(frames, log_lines, frame_to_lines,
                         video, args.log_file, args.video)
    ret = app.exec()
    video.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
