#!/usr/bin/env bash

set -ex

gst-launch-1.0 -v rtspsrc location=rtsp://192.168.18.5:8554/raw_0 protocols=tcp latency=0   ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink &
gst-launch-1.0 -v rtspsrc location=rtsp://192.168.18.5:8564/processed_0 protocols=tcp latency=0   ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink &

wait
