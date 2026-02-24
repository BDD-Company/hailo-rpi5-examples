#!/usr/bin/env bash

set -e
NUMBER_OF_CAMERAS=$(rpicam-hello --list-cameras | grep -iPc '^\d : ')

TASKS=()

# kill all background child processes
function cleanup()
{
  # kill child tasks
  pkill -9 -g $$
  #kill -9 ${TASKS[*]}
}

trap cleanup EXIT


function start_recording()
{
  local CAMERA="${1}"
  local CAMERA_NAME=$(rpicam-hello --list-camera | grep -E ^${CAMERA} | awk '{print $3}')

  while true;
  do
    DESTINATION=camera_${CAMERA}_$(date | tr [:space:][:punct:] _).avi
    echo "`date` recording from camera ${CAMERA} to ${DESTINATION}"

    rpicam-vid \
      -v 0\
      --camera ${CAMERA} \
      --width 1280 --height 720 \
      --framerate 30 \
      --codec libav --libav-format avi --libav-audio \
      -t 30s \
      --output ${DESTINATION} \
      --tuning-file /usr/share/libcamera/ipa/rpi/pisp/${CAMERA_NAME}_noir.json \
      &> auto_recording_camera_${CAMERA}.log ||:

  done
}


for i in $(seq 0 $(( ${NUMBER_OF_CAMERAS} - 1 )) ); do
  start_recording $i &
  task_pid=$!
  echo started recording task "$task_pid" for camera $i

  TASKS+=(${task_pid})
done

wait -n ${TASKS[*]} || exit_code=$?
echo "Exit code: " $exit_code

