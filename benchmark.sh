#!/usr/bin/env bash

set -euo pipefail

# required by hail example apps, otherwise pipeline tries to create windows on no display and crashes
export DISPLAY=:0

readonly OUT_DIR=${1}
readonly HEF_DIR=${2}
readonly TEST_DIR=${3}
readonly VENV_DIR="venv_hailo_rpi_examples"

if [[ ! -d "${TEST_DIR}" ]]; then
    echo "TEST_DIR is not a directory: ${TEST_DIR}" >&2
    exit 1
fi

if [[ ! -d "${HEF_DIR}" ]]; then
    echo "HEF_DIR is not a directory: ${HEF_DIR}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
if [[ ! -d "${OUT_DIR}" ]]; then
    echo "OUT_DIR is not a directory: ${OUT_DIR}" >&2
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
        report_name="${hef_basename}+${video_basename}.log"

        echo "${hef_file} vs ${f} => ${report_name}"
        python basic_pipelines/benchmark.py --i "${f}" --hef-path "${hef_file}" 2>/dev/null \
            | sed '/^[[:space:]]*$/d' > "${OUT_DIR}/${report_name}"

    done
}

hailortcli fw-control identify > "${OUT_DIR}/hailo_info.log"

for hef_file in "${HEF_DIR}"/*.hef; do
    hef_basename="$(just_filename "${hef_file}")"
    hailortcli benchmark ${hef_file} &> "${OUT_DIR}/bench_${hef_basename}.log"
    bench "${hef_file}"
done
