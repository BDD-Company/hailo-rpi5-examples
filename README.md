
![Banner](doc/images/hailo_rpi_examples_banner.png)

# Hailo Raspberry Pi 5 Examples
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hailo-ai/hailo-rpi5-examples)

Welcome to the Hailo Raspberry Pi 5 Examples repository. This project showcases various community projects and examples demonstrating the capabilities of the Hailo AI processor. These examples will help you get started with AI on embedded devices.
The examples in this repository are designed to work with the Raspberry Pi AI Kit and AI HAT, and x86_64 Ubuntu machine supporting both the Hailo8 (26 TOPS) and Hailo8L (13 TOPS) AI processors.
Visit the [Hailo Official Website](https://hailo.ai/) and [Hailo Community Forum](https://community.hailo.ai/) for more information.

# BDD Interception Controller (this fork)

This fork runs an autonomous **air-intercept controller** on the Raspberry Pi 5 + Hailo
NPU. It receives video (RTP) and telemetry (MAVSDK/MAVLink) from a PX4 + Gazebo SITL host
(or a real drone), detects the target with a Hailo YOLO model, and sends offboard
velocity setpoints back to the flight controller.

## Running

```bash
cd ~/Developer/bdd-python
source venv_hailo_rpi_examples/bin/activate
python BDD/app.py --action drone \
    --mavsdk-connection udp://0.0.0.0:14540 \   # telemetry in / setpoints out
    --rtp-port 5600 \                            # incoming video stream
    --hef-path <model>.hef \                     # Hailo detector model
    --control-config <params>.json               # per-run config overrides (optional)
```

The host SITL must stream to this Pi (`PX4_VIDEO_HOST_IP` / `PX4_MAVLINK_TARGET_IP` set
to the Pi's IP). For the orchestrated SITL harness see the PX4-Autopilot fork's README.

## Configuration

- **`BDD/intercept_config.yaml`** — single source of truth for all defaults (grouped +
  commented). Loaded and validated by `BDD/intercept_config.py` (`InterceptConfig`).
- **`--control-config <json>`** — per-run overrides merged over the YAML defaults; only
  keys that already exist are accepted (a typo fails loudly). The tuned flight configs
  (e.g. `phased_adaptive.json`, `best_*.json`) are such override files. They enable the
  operational mode (e.g. `guidance_phased: 1`) and the tuned gains on top of the defaults.

## Key parameters

**Target & optics (geometry of the size→range estimate)**
- `target_size_m: [x, y]` — real target size (m); the red sphere is 3 m (Shahed-sized).
- `frame_angular_size_deg: [x, y]` — camera FOV (horizontal, vertical) in degrees.
- `distance_scale` — empirical multiplier on the monocular range estimate.
- `size_measure_contour` — measure apparent size via Otsu contour (true, deployed) vs the
  raw detector bbox; the bbox is vertically elongated and undershoots range.

**Estimation**
- `estimation_3d` / `estimation_3d_method` — enable 3-D world-frame target estimation.
- `estimation_3d_max_distance_m` — beyond this range fall back to 2-D (range unreliable).
- `pronav_use_kalman`, `pronav_kalman_q/r` — Kalman process/measurement noise for the
  target pos/vel estimate fed to guidance.

**Guidance mode (pick one; flight configs set the mode)**
- `guidance_phased` — 3-stage range-banded FAR→MID→CLOSE (the deployed mode).
- `guidance_lead`, `guidance_pronav`, `guidance_visual` — alternative modes.

**Phased mode**
- `phased_mid_dist` — FAR→MID handoff range (m).
- `phased_far_vmax` — FAR velocity-command magnitude (m/s); the closing-speed lever.
- `phased_far_speed` — drone speed used in the FAR lead-point time-to-go solve.
- `phased_far_vz_max` — FAR vertical (climb) command cap (m/s).
- `phased_adaptive` — scale mid_dist/pronav_n/lead_t_max/closing by estimated target speed.
- `pronav_n`, `pronav_closing_speed`, `pronav_v_max`, `pronav_vz_max` — MID pro-nav gains/caps.

**Terminal (visual servo, last metres)**
- `visual_near_thresh` / `visual_mid_thresh` — apparent-size thresholds selecting the
  NEAR/MID terminal phase.
- `visual_v_close`, `visual_term_gain`, `visual_n_gain` — closing speed + LOS-rate gains.
- `visual_term_perp_cap` — cap the lateral LOS-rate term to N·v_close so a fast crosser
  can't make the servo swing sideways instead of driving in (commit-to-close; 99 = off).
- `visual_climb_min`, `visual_v_max` — min climb while the target is above / overall cap.

**Lead / handoff**
- `lead_visual_dist` — range at which the phased mode hands off to the visual terminal.
- `lead_max_alt_m` — altitude cap (per cell the harness sets it to target_alt + margin).
- `lead_t_max` — lead time-to-go horizon.

**Takeoff & detection**
- `delay_takeof_until_n_detection_frames` — wait for N detections before takeoff.
- `safe_takeoff_period_ns` — gentle-thrust window after takeoff.
- `confidence_min` — minimum detection confidence to act on.
- `bytetrack_*` — ByteTrack tracker thresholds (track/det/match/NMS, target lock).

(See the inline comments in `BDD/intercept_config.yaml` for the full list.)

## Hailo Apps Infra Repository
Hailo's official examples and pipelines are available in the [Hailo Apps Infra repo](https://github.com/hailo-ai/hailo-apps-infra) repository.
See the Hailo Apps Infra repo for more information and documentation on how to use the pipelines and development guide.

## Install Hailo Hardware and Software Setup on Raspberry Pi

For instructions on how to set up Hailo's hardware and software on the Raspberry Pi 5, see the [Hailo Raspberry Pi 5 installation guide](doc/install-raspberry-pi5.md#how-to-set-up-raspberry-pi-5-and-hailo).

# Hailo RPi5 Basic Pipelines
The basic pipelines examples demonstrate object detection, human pose estimation, and instance segmentation, providing a solid foundation for your own projects.
This repo is using our new [Hailo Apps Infra](https://github.com/hailo-ai/hailo-apps-infra) repo as a dependency.
See our Developement Guide for more information on how to use the pipelines to create your own custom pipelines.

## Installation

### Clone the Repository
```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
```
Navigate to the repository directory:
```bash
cd hailo-rpi5-examples
```

### Installation
Run the following script to automate the installation process:
```bash
./install.sh
```

### Documentation
For additional information and documentation on how to use the pipelines to create your own custom pipelines, see the [Basic Pipelines Documentation](doc/basic-pipelines.md).

### Running The Examples
When opening a new terminal session, ensure you have sourced the environment setup script:
```bash
source setup_env.sh
```
### Detection Example
For more information see [Detection Example Documentation.](doc/basic-pipelines.md#detection-example)

![Detection Example](doc/images/detection.gif)

#### Run the simple detection example:
```bash
python basic_pipelines/detection_simple.py
```
To close the application, press `Ctrl+C`.

This is lightweight version of the detection example, mainly focusing on demonstrating Hailo performance while minimizing CPU load. The internal GStreamer video processing pipeline is simplified by minimizing video processing tasks, and the YOLOv6 Nano model is used.

#### Run the full detection example:
This is the full detection example, including object tracker and multiple video resolution support - see more information [Detection Example Documentation](doc/basic-pipelines.md#detection-example):

```bash
python basic_pipelines/detection.py
```
To close the application, press `Ctrl+C`.

#### Running with Raspberry Pi Camera input:
```bash
python basic_pipelines/detection.py --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
python basic_pipelines/detection.py --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
python basic_pipelines/detection.py --input /dev/video<X>
```

For additional options, execute:
```bash
python basic_pipelines/detection.py --help
```

#### Retrained Networks Support
The retrain guide is available in the [Hailo Apps Infra repo: Retraining Example](https://github.com/hailo-ai/hailo-apps-infra/blob/main/doc/developer_guide/retraining_example.md).

### Pose Estimation Example
For more information see [Pose Estimation Example Documentation.](doc/basic-pipelines.md#pose-estimation-example)

![Pose Estimation Example](doc/images/pose_estimation.gif)

#### Run the pose estimation example:
```bash
python basic_pipelines/pose_estimation.py
```
To close the application, press `Ctrl+C`.
See Detection Example above for additional input options examples.

### Instance Segmentation Example
For more information see [Instance Segmentation Example Documentation.](doc/basic-pipelines.md#instance-segmentation-example)

![Instance Segmentation Example](doc/images/instance_segmentation.gif)

#### Run the instance segmentation example:
```bash
python basic_pipelines/instance_segmentation.py
```
To close the application, press `Ctrl+C`.
See Detection Example above for additional input options examples.

### Depth Estimation Example
For more information see [Depth Estimation Example Documentation.](doc/basic-pipelines.md#depth-estimation-example)

![Depth Estimation Example](doc/images/depth.gif)

#### Run the depth estimation example:
```bash
python basic_pipelines/depth.py
```
To close the application, press `Ctrl+C`.
See Detection Example above for additional input options examples.

### Community Projects

Get involved and make your mark! Explore our Community Projects and start contributing today, because together, we build better things! 🚀
Check out our [Community Projects](community_projects/community_projects.md) for more information.

# Additional Examples and Resources

## Hailo Apps Infra
Hailo RPi5 Examples are using the [Hailo Apps Infra Repository](https://github.com/hailo-ai/hailo-apps-infra) as a dependency. The Hailo Apps Infra repository contains the infrastructure of Hailo applications and pipelines.
It is aimed for to provide tools for developers who want to create their own custom pipelines and applications. It features a simple and easy-to-use API for creating custom pipelines and applications.
It it installed as a pip package and can be used as a dependency in your own projects. See more information in its documentation and Development Guide.

### CLIP Application

CLIP (Contrastive Language-Image Pre-training) predicts the most relevant text prompt on real-time video frames using Hailo8/8l AI processor.
See the [hailo-CLIP Repository](https://github.com/hailo-ai/hailo-CLIP) for more information.
Click the image below to watch the demo on YouTube.

[![Watch the demo on YouTube](https://img.youtube.com/vi/XXizBHtCLew/0.jpg)](https://youtu.be/XXizBHtCLew)


#### Frigate Integration

Frigate is an open-source video surveillance software that runs on a Raspberry Pi.
Hailo is officially integrated into Frigate framework starting from version 0.16.0.
See [Hailo Official Integration with Frigate](https://community.hailo.ai/t/hailo-official-integration-with-frigate/13679) for more information.


### Raspberry Pi Official Examples

#### rpicam-apps

Raspberry Pi [rpicam-apps](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-apps) Hailo post-processing examples.
This is Raspberry Pi's official example for AI post-processing using the Hailo AI processor integrated into their CPP camera framework.
The documentation on how to use rpicam-apps can be found [here](https://www.raspberrypi.com/documentation/computers/ai.html).

#### picamera2

Raspberry Pi [picamera2](https://github.com/raspberrypi/picamera2) is the libcamera-based replacement for Picamera, which was a Python interface to the Raspberry Pi's legacy camera stack. Picamera2 also presents an easy-to-use Python API.

## Additional Resources

### Hailo Python API
The Hailo Python API is now available on the Raspberry Pi 5. This API allows you to run inference on the Hailo-8L AI processor using Python.
For examples, see our [Python code examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python).
Additional examples can be found in RPi [picamera2](#picamera2) code.
Visit our [HailoRT Python API documentation](https://hailo.ai/developer-zone/documentation/hailort-v4-18-0/?page=api%2Fpython_api.html#module-hailo_platform.drivers) for more information.

### Hailo Dataflow Compiler (DFC)

The Hailo Dataflow Compiler (DFC) is a software tool that enables developers to compile their neural networks to run on the Hailo-8/8L AI processors.
The DFC is available for download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/) (Registration required).
For examples, tutorials, and retrain instructions, see the [Hailo Model Zoo Repo](https://github.com/hailo-ai/hailo_model_zoo).
Additional documentation and [tutorials](https://hailo.ai/developer-zone/documentation/dataflow-compiler/latest/?sp_referrer=tutorials/tutorials.html) can be found in the [Hailo Developer Zone Documentation](https://hailo.ai/developer-zone/documentation/).
For a full end-to-end training and deployment example, see the [Hailo Apps Infra repo: Retraining Example](https://github.com/hailo-ai/hailo-apps-infra/blob/main/doc/developer_guide/retraining_example.md).

## Contributing

We welcome contributions from the community. You can contribute by:
1. Contribute to our [Community Projects](community_projects/community_projects.md).
2. Reporting issues and bugs.
3. Suggesting new features or improvements.
4. Joining the discussion on the [Hailo Community Forum](https://community.hailo.ai/).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This code example is provided by Hailo solely on an “AS IS” basis and “with all faults.” No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness, or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.
