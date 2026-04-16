# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hailo-rpi5-examples** demonstrates AI inference applications on Raspberry Pi 5 using Hailo AI accelerators (Hailo-8 at 26 TOPS, Hailo-8L at 13 TOPS). Examples cover object detection, pose estimation, instance segmentation, and depth estimation, plus 13+ community projects (robotics, games, pet monitoring, autonomous navigation).

## Setup and Environment

```bash
# Initial installation (handles venv, dependencies, and hailo-apps-infra)
./install.sh

# Required at the start of every new terminal session
source setup_env.sh
```

The `install.sh` script reads `config.yaml` for versions, auto-detects hardware (RPi vs x86, Hailo-8 vs Hailo-8L), creates a Python venv, and runs `hailo-post-install` which generates the `.env` file with resource paths.

## Running Examples

```bash
# Basic pipelines (run from repo root after sourcing setup_env.sh)
python basic_pipelines/detection.py --input rpi        # RPi camera
python basic_pipelines/detection.py --input usb        # USB webcam
python basic_pipelines/detection.py --input /path/to/video.mp4

python basic_pipelines/pose_estimation.py
python basic_pipelines/instance_segmentation.py
python basic_pipelines/depth.py

# List available USB cameras
get-usb-camera
```

## Tests

```bash
./run_tests.sh
```

Tests live in `tests/test_hailo_rpi5_examples.py` (pytest). Each pipeline test runs for `TEST_RUN_TIME = 10` seconds. Logs are written to `logs/`. Tests validate camera connectivity, device architecture detection, model compatibility, and pipeline execution.

## Architecture

### Core Pattern: GStreamer Pipeline + User Callback

All basic pipelines follow this structure:

```python
# 1. User callback class holds per-frame state
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # custom state here

# 2. Callback function — called as a GStreamer pad probe, must return quickly
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    # ... process detections ...
    return Gst.PadProbeReturn.OK

# 3. Entry point
user_data = user_app_callback_class()
app = GStreamerDetectionApp(app_callback, user_data)
app.run()
```

`GStreamerDetectionApp`, `GStreamerPoseEstimationApp`, `GStreamerInstanceSegmentationApp`, and `GStreamerDepthApp` come from the `hailo-apps-infra` package (installed as a dependency).

### Hailo Metadata Types

Extracted via `roi.get_objects_typed(...)`:

| Task | Hailo type |
|------|-----------|
| Detection | `hailo.HAILO_DETECTION` (has `.get_bbox()`, `.get_label()`, `.get_confidence()`) |
| Pose keypoints | `hailo.HAILO_LANDMARKS` |
| Segmentation masks | `hailo.HAILO_CONF_CLASS_MASK` |
| Depth map | `hailo.HAILO_DEPTH_MASK` |
| Tracking IDs | `hailo.HAILO_UNIQUE_ID` |

### Frame Access

```python
format, width, height = get_caps_from_pad(pad)
frame = get_numpy_from_buffer(buffer, format, width, height)  # RGB numpy array
# draw with OpenCV, then:
user_data.set_frame(frame)  # async display by app
```

### Environment File

`HAILO_ENV_FILE` env var points to the generated `.env` (created by `hailo-post-install`). Each community project sets this at startup:

```python
project_root = Path(__file__).resolve().parent.parent
os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
```

### Community Projects

Each project under `community_projects/` is self-contained with its own `requirements.txt` and `download_resources.sh`. Advanced patterns used:
- **Multi-process** (Fruit Ninja, WLED Display): GStreamer callback enqueues data; worker process handles heavy computation
- **State machine** (TAILO): enum-based states with cooldown tracking and GPIO/servo control
- **Parallel pipelines** (BDD drone at `BDD/`): multiple GStreamer pipelines with latency profiling

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Central config: HailoRT version, TAPPAS variant, app-infra branch |
| `setup_env.sh` | Sets PYTHONPATH and activates venv — source before running anything |
| `install.sh` | Main installer |
| `.env` | Generated at install time; contains resource/model paths |
| `basic_pipelines/detection_simple.py` | Minimal detection example — best starting point for new users |
| `doc/basic-pipelines.md` | Developer guide for building new pipelines |
