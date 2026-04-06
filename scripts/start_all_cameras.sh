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
  STREAM_DESTINATION=${2-'tcp://0.0.0.0:7000'}
  OUPUT_PATH=${3-'.'}
  shift 3

  EXTRA_ARGS="$@"


  read -r CAMERA_TYPE CAMERA_NAME <<< "${ALL_CAMERAS[${CAMERA}]}"
  IFS=':' read -r STREAM_SCHEMA STREAM_ADDRESS STREAN_PORT <<< "${STREAM_DESTINATION}"
  STREAM_ADDRESS="$(echo ${STREAM_ADDRESS} | cut -c3-)"

  attempt=0
  while true;
  do
    echo "`date` will start streaming from camera ${CAMERA_NAME} to ${STREAM_ADDRESS}:${STREAN_PORT}, attempt: ${attempt}"
    OUTPUT_FILE="${OUPUT_PATH}/$(date --iso-8601=seconds | cut -c -19 | tr  '[:space:][:punct:][:alpha:]' _)camera_${CAMERA}.mjpeg"

    STARTED_AT=${SECONDS}
    set -x
    LIBCAMERA_RPI_TUNING_FILE=/usr/share/libcamera/ipa/rpi/pisp/${CAMERA_TYPE}_noir.json \
    gst-launch-1.0 libcamerasrc camera-name="${CAMERA_NAME}" \
      ! video/x-raw,colorimetry=bt709,format=NV12,width=1280,height=720,framerate=30/1 \
      ! jpegenc ! multipartmux \
      ! tcpserversink host="${STREAM_ADDRESS}" port="${STREAN_PORT}" sync-method=2 \
      &> auto_streaming_camera_${CAMERA}.log

    exit_code=$?
    set +x

    STOPPED_AT=${SECONDS}

    if (( ${exit_code} != 0 )) ; then
      tail -n20 auto_streaming_camera_${CAMERA}.log

      if (( $STOPPED_AT - $STARTED_AT <= 1 )) ; then
        echo streaming failed almost instantly, not restarting, there must be something wrong...
        return
      fi
    fi

  done
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

