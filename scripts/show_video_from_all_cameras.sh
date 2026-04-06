#!/usr/bin/env bash


EXTRA_ARGS="$@"

# kill all background child processes
function cleanup()
{
  # kill child tasks
  pkill -9 -g $$
  #kill -9 ${TASKS[*]}
}

trap cleanup EXIT

readarray -t ALL_CAMERAS < <(cam --list 2>/dev/null | tail -n +2 | cut -c 4- | tr -d \'\(\))
echo "Cameras: ${ALL_CAMERAS[*]}"

function start_camera()
{
  CAMERA=${1:-0}
  shift 1

  EXTRA_ARGS="$@"

  read -r CAMERA_TYPE CAMERA_NAME <<< "${ALL_CAMERAS[${CAMERA}]}"
  export LIBCAMERA_RPI_TUNING_FILE=/usr/share/libcamera/ipa/rpi/pisp/${CAMERA_TYPE}_noir.json
  rpicam-hello \
    --camera=${CAMERA} \
    -t 0 \
    --info-text "${CAMERA}:: =#%frame (%fps fps) exp %exp ag %ag dg %dg"
}

NUMBER_OF_CAMERAS=$(rpicam-hello --list-cameras | grep -iPc '^\d : ')
TASKS=()

PORT=7000

for i in $(seq 0 $(( ${NUMBER_OF_CAMERAS} - 1 )) ); do
  start_camera \
    $i \
    "tcp://0.0.0.0:"$(( ${PORT} + ${i} )) \
    ${EXTRA_ARGS} &

  task_pid=$!
  #echo started streaming task "$task_pid" for camera $i

  TASKS+=(${task_pid})
done

wait -n ${TASKS[*]} || exit_code=$?
echo "Exit code: " $exit_code

