#!/usr/bin/env python3
"""Tests for frame_utils.Frame — uniform NV12/RGB frame handling."""
import numpy as np
import pytest

from frame_utils import Frame, PixelFormat, to_gray, to_rgb


def _rgb_frame(h=100, w=120):
    """Bright RGB frame with a dark square (drone-on-sky)."""
    f = np.full((h, w, 3), 255, dtype=np.uint8)
    f[20:40, 20:40] = 20
    return f


def _nv12_planes(h=100, w=120):
    """NV12 (Y, UV) tuple with a dark square in the luma plane, neutral chroma."""
    y = np.full((h, w), 255, dtype=np.uint8)
    y[20:40, 20:40] = 20
    uv = np.full((h // 2, w // 2, 2), 128, dtype=np.uint8)  # neutral (grayscale)
    return (y, uv)


# ---- construction / coerce -------------------------------------------------

def test_coerce_none_returns_none():
    assert Frame.coerce(None) is None


def test_coerce_ndarray_is_rgb():
    f = Frame.coerce(_rgb_frame())
    assert isinstance(f, Frame)
    assert f.format is PixelFormat.RGB


def test_coerce_tuple_is_nv12():
    f = Frame.coerce(_nv12_planes())
    assert isinstance(f, Frame)
    assert f.format is PixelFormat.NV12


def test_coerce_frame_is_identity():
    f = Frame.coerce(_rgb_frame())
    assert Frame.coerce(f) is f


# ---- shape / dimensions ----------------------------------------------------

def test_shape_rgb():
    f = Frame.coerce(_rgb_frame(100, 120))
    assert f.shape == (100, 120, 3)
    assert (f.height, f.width) == (100, 120)


def test_shape_nv12_reports_logical_colour_shape():
    f = Frame.coerce(_nv12_planes(100, 120))
    # Full-res luma dimensions, logical (H, W, 3) regardless of planar storage.
    assert f.shape == (100, 120, 3)
    assert (f.height, f.width) == (100, 120)


# ---- to_gray ---------------------------------------------------------------

def test_to_gray_rgb_is_2d():
    f = Frame.coerce(_rgb_frame())
    g = f.to_gray()
    assert g.ndim == 2
    assert g.shape == (100, 120)


def test_to_gray_nv12_is_the_y_plane_zero_copy():
    y, uv = _nv12_planes()
    f = Frame.coerce((y, uv))
    g = f.to_gray()
    assert g.ndim == 2
    assert g.shape == y.shape
    # Y plane IS the grayscale image — returned directly, no conversion/copy.
    assert np.shares_memory(g, y)


# ---- to_rgb ----------------------------------------------------------------

def test_to_rgb_rgb_passthrough():
    src = _rgb_frame()
    f = Frame.coerce(src)
    assert np.shares_memory(f.to_rgb(), src)


def test_to_rgb_nv12_reconstructs_colour_shape():
    f = Frame.coerce(_nv12_planes(100, 120))
    rgb = f.to_rgb()
    assert rgb.shape == (100, 120, 3)
    assert rgb.dtype == np.uint8


def test_to_rgb_nv12_preserves_luma_contrast():
    # Dark square in Y must stay darker than the bright field after NV12->RGB.
    f = Frame.coerce(_nv12_planes(100, 120))
    rgb = f.to_rgb()
    dark = rgb[25:35, 25:35].mean()
    bright = rgb[60:90, 60:90].mean()
    assert dark < bright
    assert bright > 200          # bright field stays near-white
    assert dark < 80             # dark square stays dark


def test_to_rgb_nv12_is_cached():
    f = Frame.coerce(_nv12_planes())
    assert f.to_rgb() is f.to_rgb()  # second call returns the cached array


# ---- free-function helpers (accept Frame | ndarray | tuple | None) ---------

def test_free_to_gray_handles_all_inputs():
    assert to_gray(None) is None
    assert to_gray(_rgb_frame()).ndim == 2
    y, uv = _nv12_planes()
    assert np.shares_memory(to_gray((y, uv)), y)
    assert to_gray(Frame.coerce(_nv12_planes())).ndim == 2


def test_free_to_rgb_handles_all_inputs():
    assert to_rgb(None) is None
    assert to_rgb(_rgb_frame()).shape == (100, 120, 3)
    assert to_rgb(_nv12_planes()).shape == (100, 120, 3)
    assert to_rgb(Frame.coerce(_rgb_frame())).shape == (100, 120, 3)


# ---- repr ------------------------------------------------------------------

def test_repr_is_compact():
    r = repr(Frame.coerce(_nv12_planes(100, 120)))
    assert "NV12" in r and "120x100" in r
    assert "array" not in r  # never dump pixel data into logs
