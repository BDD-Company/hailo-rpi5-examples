#/usr/bin/env python
"""Uniform handling of captured frames across pixel formats (RGB / NV12).

Under NV12 capture, get_numpy_from_buffer returns a planar (Y, UV) tuple instead
of an RGB ndarray. Frame wraps either representation so consumers can ask for what
they need without caring about the storage:

  - to_gray() for the latency-critical control path — for NV12 the Y plane IS the
    full-res luma image, so grayscale is free (no colour conversion).
  - to_rgb() where colour is actually rendered (annotation, recording) — the NV12
    planes are reassembled and converted lazily, then cached.
"""
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class PixelFormat(Enum):
    RGB = "RGB"
    NV12 = "NV12"


@dataclass(eq=False, repr=False)
class Frame:
    """A captured video frame, format-agnostic to its consumers.

    Stored in Detections.frame in place of a raw ndarray. Holds either an RGB
    ndarray (H, W, 3) or an NV12 (Y, UV) plane tuple and exposes a single
    interface: .shape/.width/.height plus .to_gray()/.to_rgb().
    """
    _data: "np.ndarray | tuple[np.ndarray, np.ndarray]"
    format: PixelFormat
    _rgb_cache: "np.ndarray | None" = None

    @classmethod
    def coerce(cls, x) -> "Frame | None":
        """Normalize Frame | ndarray | NV12 (Y, UV) tuple | None into Frame | None.

        Lets legacy call sites/tests keep passing raw arrays; NV12 tuples are the
        only new input. Format is inferred from the data (tuple => NV12).
        """
        if x is None or isinstance(x, cls):
            return x
        fmt = PixelFormat.NV12 if isinstance(x, tuple) else PixelFormat.RGB
        return cls(x, fmt)

    @property
    def height(self) -> int:
        plane = self._data[0] if self.format is PixelFormat.NV12 else self._data
        return plane.shape[0]

    @property
    def width(self) -> int:
        plane = self._data[0] if self.format is PixelFormat.NV12 else self._data
        return plane.shape[1]

    @property
    def shape(self) -> tuple:
        """Logical colour shape (H, W, 3), independent of planar storage — so
        consumers that size sinks/canvases by .shape behave the same for both."""
        return (self.height, self.width, 3)

    def to_gray(self) -> np.ndarray:
        """2-D grayscale image. NV12 -> Y plane (zero cost); RGB -> RGB2GRAY."""
        if self.format is PixelFormat.NV12:
            return self._data[0]  # Y plane is the full-res luma image
        d = self._data
        return cv2.cvtColor(d, cv2.COLOR_RGB2GRAY) if d.ndim == 3 else d

    def to_rgb(self) -> np.ndarray:
        """(H, W, 3) RGB image. NV12 reconstructed + cached; RGB passthrough."""
        if self._rgb_cache is None:
            self._rgb_cache = self._compute_rgb()
        return self._rgb_cache

    def _compute_rgb(self) -> np.ndarray:
        if self.format is not PixelFormat.NV12:
            return self._data  # already RGB
        y, uv = self._data
        h, w = y.shape
        # Rebuild packed NV12 — Y rows, then interleaved UV rows — for cvtColor.
        packed = np.vstack([y, np.ascontiguousarray(uv).reshape(h // 2, w)])
        return cv2.cvtColor(packed, cv2.COLOR_YUV2RGB_NV12)

    def __repr__(self) -> str:
        # Compact — never dump pixel arrays into logs (Detections repr includes this).
        return f"Frame({self.format.value}, {self.width}x{self.height})"


def to_gray(frame) -> "np.ndarray | None":
    """Grayscale for any frame form (Frame | ndarray | NV12 tuple | None)."""
    f = Frame.coerce(frame)
    return None if f is None else f.to_gray()


def to_rgb(frame) -> "np.ndarray | None":
    """RGB for any frame form (Frame | ndarray | NV12 tuple | None)."""
    f = Frame.coerce(frame)
    return None if f is None else f.to_rgb()
