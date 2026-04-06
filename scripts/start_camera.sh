#!/usr/bin/env bash

CAMERA=${1:-0}
DESTINATION=${2-'tcp://0.0.0.0:7001'}
shift 2
EXTRA_ARGS="$@"

CAMERA_NAME=$(rpicam-hello --list-camera | grep -E ^${CAMERA} | awk '{print $3}')

while true;
do
  echo "`date` will start streaming from camera ${CAMERA} to ${DESTINATION}"
  # ignore errors to make sure server is always started

  set -x
  rpicam-vid -t 0 \
    --camera ${CAMERA} \
    --width 1280 --height 720 \
    --framerate 30 --codec mjpeg \
    --inline --listen \
    -o ${DESTINATION} \
    --tuning-file /usr/share/libcamera/ipa/rpi/pisp/${CAMERA_NAME}_noir.json \
    ${EXTRA_ARGS}

done
