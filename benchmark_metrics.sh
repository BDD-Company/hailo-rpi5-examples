#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <TEST_VIDEOS_DIR> <REPORTS_DIR> [OUTPUT_JSON] [ANNOTATIONS_DIR]" >&2
    exit 1
fi

readonly TEST_VIDEOS_DIR=${1}
readonly REPORTS_DIR=${2}
readonly OUTPUT_JSON=${3:-}
readonly ANNOTATIONS_DIR=${4:-${TEST_VIDEOS_DIR}}
readonly VENV_DIR="venv_hailo_rpi_examples"

if [[ ! -d "${TEST_VIDEOS_DIR}" ]]; then
    echo "TEST_VIDEOS_DIR is not a directory: ${TEST_VIDEOS_DIR}" >&2
    exit 1
fi

if [[ ! -d "${REPORTS_DIR}" ]]; then
    echo "REPORTS_DIR is not a directory: ${REPORTS_DIR}" >&2
    exit 1
fi

if [[ ! -d "${ANNOTATIONS_DIR}" ]]; then
    echo "ANNOTATIONS_DIR is not a directory: ${ANNOTATIONS_DIR}" >&2
    exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtualenv activation script not found: ${VENV_DIR}/bin/activate" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cmd=(
    python3 basic_pipelines/benchmark_metrics.py
    --videos-dir "${TEST_VIDEOS_DIR}"
    --annotations-dir "${ANNOTATIONS_DIR}"
    --reports-dir "${REPORTS_DIR}"
)

if [[ -n "${OUTPUT_JSON}" ]]; then
    cmd+=(--output-json "${OUTPUT_JSON}")
fi

"${cmd[@]}"
