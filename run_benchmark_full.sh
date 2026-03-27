#!/usr/bin/env bash

# Usage: ./run_benchmark_full.sh <videos_dir> <models_dir> <output_dir> <labels_dir>
#
# Example:
#   ./run_benchmark_full.sh benchmark/video/ benchmark/models/ benchmark/output/test1 benchmark/labels/

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <VIDEOS_DIR> <MODELS_DIR> <OUTPUT_DIR> <LABELS_DIR>" >&2
    exit 1
fi

readonly VIDEOS_DIR="${1}"
readonly MODELS_DIR="${2}"
readonly OUTPUT_DIR="${3}"
readonly LABELS_DIR="${4}"
readonly METRICS_JSON="${OUTPUT_DIR}/metrics.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1: Running inference benchmark ==="
"${SCRIPT_DIR}/benchmark.sh" "${VIDEOS_DIR}" "${MODELS_DIR}" "${OUTPUT_DIR}"

echo ""
echo "=== Step 2: Calculating metrics for all models ==="
"${SCRIPT_DIR}/benchmark_metrics.sh" "${VIDEOS_DIR}" "${OUTPUT_DIR}" "${METRICS_JSON}" "${LABELS_DIR}"

echo ""
echo "=== Full JSON report: ${METRICS_JSON} ==="
