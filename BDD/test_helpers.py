#!/usr/bin/env python3

from helpers import *


def test_xy_format():
    """XY format specifiers are applied to x and y individually."""
    v = XY(1.23456, 0.5)

    assert f'{v}'       == 'XY(1.235, 0.500)'
    assert f'{v:.2f}'   == 'XY(1.23, 0.50)'
    assert f'{v:.3f}'   == 'XY(1.235, 0.500)'
    assert f'{v: .3}'   == 'XY( 1.23,  0.5)'
    assert f'{v:>8.2f}' == 'XY(    1.23,     0.50)'
    assert str(v)       == 'XY(1.235, 0.500)'
    assert repr(v)      == 'XY(x=1.23456, y=0.5)'


class _FakeTrack:
    """Duck-typed stand-in for STrack: only .bbox is read."""
    def __init__(self, x1, y1, x2, y2):
        self.bbox = [x1, y1, x2, y2]


def test_kalman_or_raw_bbox_uses_raw_when_disabled():
    raw = Rect.from_xyxy(0.10, 0.10, 0.20, 0.20)
    track = _FakeTrack(0.30, 0.30, 0.40, 0.40)
    # Feature off => the raw detector rect is returned untouched, even with a track.
    assert kalman_or_raw_bbox(raw, track, use_kalman_bbox=False) is raw


def test_kalman_or_raw_bbox_uses_raw_when_no_track():
    raw = Rect.from_xyxy(0.10, 0.10, 0.20, 0.20)
    # Enabled but this detection matched no track => fall back to the raw rect.
    assert kalman_or_raw_bbox(raw, None, use_kalman_bbox=True) is raw


def test_kalman_or_raw_bbox_uses_track_bbox_when_enabled_and_matched():
    raw = Rect.from_xyxy(0.10, 0.10, 0.20, 0.20)
    track = _FakeTrack(0.30, 0.31, 0.40, 0.41)
    out = kalman_or_raw_bbox(raw, track, use_kalman_bbox=True)
    assert out == Rect.from_xyxy(0.30, 0.31, 0.40, 0.41)
    assert out is not raw


def main():
    test_xy_format()

    x=0.170
    y=0.428
    w=0.041
    h=0.056
    r = Rect.from_xyxy(x, y, x + w, y + h)
    # assert r.width == w
    # assert r.height == h
    print(r.area())


if __name__ == "__main__":
    main()