# NV12 `Frame` dataclass — design

**Date:** 2026-07-07
**Branch:** `feat/nv12-frame-dataclass` (off `main`)
**Status:** approved, implementing

## Problem

Under NV12 capture (NV12-input HEF), `get_numpy_from_buffer(buffer, format, w, h)`
returns a **planar `(Y, UV)` tuple**, not an RGB `ndarray`:

- `Y`  — full-resolution luma plane, shape `(H, W)`, `uint8` (this *is* the grayscale image)
- `UV` — half-resolution interleaved chroma, shape `(H/2, W/2, 2)`, `uint8`

That tuple is stored verbatim into `Detections.frame` at `app.py`, then consumed
~400 lines away by code written for the old RGB `ndarray`. The first casualty was
`estimate_distance._extract_largest_object_contour` →
`AttributeError: 'tuple' object has no attribute 'shape'`. Because the drone control
loop wraps its body in a bare `except:` (intentional graceful degradation), the crash
was swallowed on every confident detection — silently disabling optical size/centre
refinement and distance estimation for the whole NV12 run (observed `nan` distance,
zero `MoveCommand`).

Frame pixels are consumed in multiple places, each assuming a raw `ndarray`:

| Consumer | Thread | Needs |
|---|---|---|
| `estimate_distance` (via `OpticalObjectInfo`), from drone + platform control | latency-critical | grayscale only |
| `debug_output` (annotate → `RecorderSink` / `OpenCVShowImageSink` / `MultiSink`) | debug/recording | colour |
| `video_sink_gstreamer` (validates `ndim==3, shape[2]==3`) | recording | colour |

A minimal `_frame_to_gray` guard already fixed the crash in `estimate_distance`
(committed as the branch starting point). This spec covers the proper, uniform fix.

## Key fact

**NV12 does not lose colour.** It is YUV 4:2:0 — full colour, stored as luma + subsampled
chroma. Taking only `Y` yields grayscale (correct and free for the segmentation path);
colour is recovered where needed by reconstructing RGB from both planes. No consumer that
renders colour should ever receive only `Y`.

## Design

New module `BDD/frame_utils.py` with a `Frame` dataclass that becomes the type stored in
`Detections.frame` (replacing the raw `ndarray`/tuple). It wraps either representation and
exposes one interface.

```python
class PixelFormat(Enum):
    RGB = "RGB"
    NV12 = "NV12"

@dataclass
class Frame:
    _data: "np.ndarray | tuple[np.ndarray, np.ndarray]"
    format: PixelFormat
    _rgb_cache: "np.ndarray | None" = field(default=None, repr=False, compare=False)

    @property
    def width(self)  -> int: ...          # full-res W (Y-plane / ndarray)
    @property
    def height(self) -> int: ...          # full-res H
    @property
    def shape(self)  -> tuple:            # logical colour shape (H, W, 3), format-agnostic
        return (self.height, self.width, 3)

    def to_gray(self) -> np.ndarray:      # NV12 -> Y plane (zero cost); RGB -> RGB2GRAY
    def to_rgb(self)  -> np.ndarray:      # NV12 -> reassemble packed + YUV2RGB (cached); RGB -> passthrough

    @classmethod
    def coerce(cls, x) -> "Frame | None": # Frame | ndarray | tuple | None -> Frame | None
```

### Behaviour

- **`to_gray()`** — for NV12 returns `_data[0]` (the `Y` plane) directly: zero conversion,
  full resolution. For RGB `ndarray`, `cvtColor(..., RGB2GRAY)`. The latency-critical control
  path uses only this, so NV12 adds no cost there.
- **`to_rgb()`** — for NV12: reassemble packed NV12 `vstack([Y, UV.reshape(H//2, W)])` (with
  `ascontiguousarray` on `UV`) then `cvtColor(..., COLOR_YUV2RGB_NV12)`; result cached in
  `_rgb_cache` so repeated / multi-thread reads pay once. For RGB: passthrough. Returns RGB
  to match today's convention (`get_numpy_from_buffer` returned RGB for RGB HEFs).
- **`shape`** — always `(H, W, 3)` so consumers that size sinks/canvases by `.shape` behave
  identically for RGB and NV12 (both planes share the same `H, W`).
- **`coerce()`** — normalises any legacy input (raw ndarray / tuple / None) into a `Frame`,
  so existing tests/mocks that pass plain arrays keep working; NV12 is the only new behaviour.
  Format is inferred from the data (`tuple` ⇒ NV12, else RGB).
- Compact `__repr__` (`Frame(NV12, 640x480)`) — also stops the exception logs from dumping
  whole pixel arrays.

## Consumer wiring (minimal, non-invasive)

| File | Change |
|---|---|
| `app.py` (frame construction) | `frame = Frame.coerce(get_numpy_from_buffer(...))` — the single construction site |
| `estimate_distance.py` | `_extract_largest_object_contour` uses `Frame.coerce(frame).to_gray()` (evolves `_frame_to_gray`) |
| `debug_output.py` | `.shape` works via `Frame.shape`; annotation draws on `Frame.coerce(frame).to_rgb().copy()` |
| `helpers.Detections` | `frame` annotation → `Frame \| None` |
| `platform_controller.py`, `video_sink_gstreamer.py` | **no change** — `Frame` passes through; the sink still receives a real ndarray (the annotated `to_rgb` output) |

## Testing (`BDD/test_frame_utils.py`)

- `to_gray`, `to_rgb`, `shape`, `width`, `height` for both RGB ndarray and NV12 `(Y, UV)` inputs.
- NV12 → RGB reassembly on a synthetic frame: a dark square on a bright field survives the
  round-trip (bright stays bright, dark stays dark) — guards the plane reassembly maths.
- `coerce` on `Frame` / ndarray / tuple / None.
- Existing `estimate_distance` NV12 test stays green (now exercised through `Frame`).

## Risks / notes

- The NV12 → RGB reassembly (`UV.reshape` + `vstack` + `COLOR_YUV2RGB_NV12`) can only be
  fully validated on-device against a real Hailo NV12 buffer (no Hailo on the dev host).
  Synthetic tests guard the maths; **on-device colour sanity check required** before trusting
  recorded/preview colours.
- `format` inferred from `isinstance(_data, tuple)`. If `get_numpy_from_buffer` ever returns a
  packed single-ndarray NV12, that assumption breaks — documented, YAGNI for now.
- `_rgb_cache` is written lazily from possibly two reader threads (control never calls
  `to_rgb`; debug/recording does). Double-compute is harmless and idempotent; no lock needed.
