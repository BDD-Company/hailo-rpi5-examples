#!/usr/bin/env python3
import numpy as np
import pytest
from helpers import Rect, XY
from estimate_distance import measure_object_size


def _make_frame(h=100, w=100):
    """White frame (bright sky)."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _draw_dark_rect(frame, x1, y1, x2, y2):
    """Draw a dark rectangle (drone) on the frame."""
    frame[y1:y2, x1:x2] = 20
    return frame


def test_returns_none_when_frame_is_none():
    bbox = Rect.from_xyxy(0.1, 0.1, 0.5, 0.5)
    assert measure_object_size(None, bbox) is None


def test_returns_none_when_roi_too_small():
    frame = _make_frame(100, 100)
    # bbox maps to 3x3 px — below 4x4 guard
    bbox = Rect.from_xyxy(0.0, 0.0, 0.03, 0.03)
    assert measure_object_size(frame, bbox) is None


def test_returns_none_when_no_dark_object():
    # Uniform white frame — Otsu will find nothing meaningful
    frame = _make_frame(100, 100)
    bbox = Rect.from_xyxy(0.0, 0.0, 1.0, 1.0)
    result = measure_object_size(frame, bbox)
    # Either None (noise guard) or a tiny value — must not crash
    assert result is None


def test_detects_dark_object_smaller_than_bbox():
    frame = _make_frame(100, 100)
    # Dark rect occupies 20x20 px in the center of a 60x60 bbox
    _draw_dark_rect(frame, 20, 20, 40, 40)
    bbox = Rect.from_xyxy(0.1, 0.1, 0.7, 0.7)  # 60x60 px bbox
    result = measure_object_size(frame, bbox)
    assert result is not None
    # Object is 20px wide / 100px frame → 0.20; allow ±5%
    assert abs(result.x - 0.20) < 0.05
    assert abs(result.y - 0.20) < 0.05


def test_result_smaller_than_bbox():
    frame = _make_frame(100, 100)
    _draw_dark_rect(frame, 30, 30, 50, 50)
    bbox = Rect.from_xyxy(0.2, 0.2, 0.8, 0.8)
    result = measure_object_size(frame, bbox)
    assert result is not None
    assert result.x < bbox.size.x
    assert result.y < bbox.size.y


def test_clips_bbox_to_frame_bounds():
    frame = _make_frame(100, 100)
    _draw_dark_rect(frame, 90, 90, 99, 99)
    # bbox extends beyond frame edges
    bbox = Rect.from_xyxy(0.8, 0.8, 1.2, 1.2)
    # Must not raise, may return None or a result
    result = measure_object_size(frame, bbox)
    assert result is None or isinstance(result, XY)
