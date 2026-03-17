#!/usr/bin/env bash

set -euo pipefail

# Path to the directory with the videos
readonly TEST_VIDEOS_DIR=${1}
# Path to the directory with the .hef files or path to a single hef file
readonly HEF_DIR=${2}
# Path to the output directory with results of the benching
readonly OUT_DIR=${3}
# venv to use
readonly VENV_DIR="venv_hailo_rpi_examples"

# required by hailo example apps, otherwise pipeline tries to create windows on no display and crashes
export DISPLAY=:0

if [[ ! -d "${TEST_VIDEOS_DIR}" ]]; then
    echo "TEST_VIDEOS_DIR is not a directory: ${TEST_VIDEOS_DIR}" >&2
    exit 1
fi

if [[ ! -d "${HEF_DIR}" && ! -f ${HEF_DIR} ]]; then
    echo "HEF_DIR must be path to directory with .hef files or a path to a single .hef file: ${HEF_DIR}" >&2
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

# NOTE: debug stuff, uncomment if needed
# set -x
# export PS4='+ ${BASH_SOURCE##*/}:${LINENO}:${FUNCNAME[0]:-main}: '

# print out error code and file/line no on error, intentionally after the arguments code.
trap 'rc=$?; echo "Error: exit code $rc at ${BASH_SOURCE[0]}:${BASH_LINENO[0]}: $BASH_COMMAND" >&2; exit $rc' ERR

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
    local report_dir="${2}"

    for f in "${TEST_VIDEOS_DIR}"/*; do
        [[ -f "${f}" ]] || continue

        video_basename="$(just_filename "${f}")"
        report_name="${video_basename}.log"

        echo "${hef_file} vs ${f} => ${report_name}"
        python basic_pipelines/benchmark.py --i "${f}" --hef-path "${hef_file}" 2>/dev/null \
            | sed '/^[[:space:]]*$/d' > "${report_dir}/${report_name}"

    done
}

echo "Collecting general info about HAILO hardware..."
hailortcli fw-control identify > "${OUT_DIR}/hailo_info.log"

function bench_single_hef()
{
    local hef_file="${1}"

    local hef_basename="$(just_filename "${hef_file}")"
    local report_dir="${OUT_DIR}/${hef_basename}"
    mkdir -p "${report_dir}" ||:

    # echo "Collecting general benchmark info about ${hef_file} ..."
    # hailortcli benchmark ${hef_file} &> "${report_dir}/bench.log"
    bench "${hef_file}" "${report_dir}"
}

# Single HEF file
if [[ -f ${HEF_DIR} ]]; then
    bench_single_hef "${HEF_DIR}"
else
    for hef_file in "${HEF_DIR}"/*.hef; do
        bench_single_hef "${hef_file}"
    done
fi
