#!/usr/bin/env bash

set -euo pipefail

# required by hail example apps, otherwise pipeline tries to create windows on no display and crashes
export DISPLAY=:0

readonly TEST_PREFIX=${1}
readonly HEF_DIR=${2}
readonly TEST_DIR=${3}
readonly OUT_DIR=${4:-"."}
readonly VENV_DIR="venv_hailo_rpi_examples"

if [[ ! -d "${TEST_DIR}" ]]; then
    echo "TEST_DIR is not a directory: ${TEST_DIR}" >&2
    exit 1
fi

if [[ ! -d "${HEF_DIR}" ]]; then
    echo "HEF_DIR is not a directory: ${HEF}" >&2
    exit 1
fi

if [[ ! -d "${OUT_DIR}" ]]; then
    echo "OUT_DIR is not a directory: ${HEF}" >&2
    exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtualenv activation script not found: ${VENV_DIR}/bin/activate" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

function just_filename()
{
    local in_name="${1}"
    out_name="$(basename "${in_name}")"
    out_name="${out_name%.*}"

    echo "${out_name}"
}

function bench()
{
    local hef_file="${1}"

    hef_basename="$(just_filename "${hef_file}")"

    for f in "${TEST_DIR}"/*; do
        [[ -f "${f}" ]] || continue

        video_basename="$(just_filename "${f}")"
        report_name="${TEST_PREFIX}_${hef_basename}_${video_basename}.log"

        echo "${hef_file} vs ${f} => ${report_name}"
        python basic_pipelines/benchmark.py --i "${f}" --hef-path "${hef_file}" 2>/dev/null \
            | sed '/^[[:space:]]*$/d' > "${OUT_DIR}/${report_name}"

    done
}


for hef_file in "${HEF_DIR}"/*.hef; do
    bench "${hef_file}"
done