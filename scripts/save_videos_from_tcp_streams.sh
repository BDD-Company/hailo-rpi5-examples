#!/usr/bin/env bash


EXTRA_ARGS="$@"
NUMBER_OF_CAMERAS="${1}"
STREAM_SOURCE_HOST="${2}"
OUTPUT_PATH="${3}"

# kill all background child processes
function cleanup()
{
  # kill child tasks
  pkill -9 -g $$
  #kill -9 ${TASKS[*]}
}

trap cleanup EXIT

function record_stream()
{
  CAMERA=${1:-0}
  STREAM_SOURCE_HOST=${2-'tcp://0.0.0.0:7000'}
  STREAM_SOURCE_PORT=${3}
  OUPUT_PATH=${4-'.'}
  shift 4

  EXTRA_ARGS="$@"

  attempt=0
  while true;
  do
    OUTPUT_FILE="${OUPUT_PATH}/$(date --iso-8601=seconds | cut -c -19 | tr  '[:space:][:punct:][:alpha:]' _)camera_${CAMERA}.mjpeg"
    echo "`date` will record from ${STREAM_SOURCE} to ${STREAM_SOURCE_HOST}:${STREAM_SOURCE_PORT}, attempt: ${attempt}"

    STARTED_AT=${SECONDS}
    set -x

    gst-launch-1.0 \
      -v tcpclientsrc host="${STREAM_SOURCE_HOST}" port="${STREAM_SOURCE_PORT}" \
      ! multipartdemux ! jpegdec \
      ! tee name=t \
      ! queue \
      ! filesink location="${OUTPUT_FILE}" sync=false \
      t. ! queue \
      ! autovideosink \
      &> auto_recording_${CAMERA}.log

    exit_code=$?
    set +x

    STOPPED_AT=${SECONDS}

    if (( ${exit_code} != 0 )) ; then
      tail -n20 auto_recording_${CAMERA}.log

      if (( $STOPPED_AT - $STARTED_AT <= 1 )) ; then
        echo streaming failed almost instantly, not restarting, there must be something wrong...
        return
      fi
    fi

  done
}


TASKS=()

PORT=7000

for i in $(seq 0 $(( ${NUMBER_OF_CAMERAS} - 1 )) ); do
  record_stream \
    $i \
    "${STREAM_SOURCE_HOST}" \
    "$(( ${PORT} + ${i} ))" \
    "${OUTPUT_PATH}" \
    ${EXTRA_ARGS} &

  task_pid=$!
  #echo started streaming task "$task_pid" for camera $i

  TASKS+=(${task_pid})
done

wait -n ${TASKS[*]} || exit_code=$?
echo "Exit code: " $exit_code

