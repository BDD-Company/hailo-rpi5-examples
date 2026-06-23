#!/usr/bin/env bash
# Live-camera (imx477 NV12) matrix. Writes /tmp/nv12_camera.txt
set -u
PY=$HOME/hailo-rpi5-examples/venv_hailo_rpi_examples/bin/python
S=$HOME/hailo-rpi5-examples/BDD/experiments/nv12_tiling_bench/nv12_tiling_bench.py
OUT=/tmp/nv12_camera.txt
DUR=${DUR:-5}; REPEATS=${REPEATS:-2}; FPS=${FPS:-30}
: > "$OUT"
run(){ local tag=$1 nx=$2 ny=$3; shift 3
  for r in $(seq 1 "$REPEATS"); do
    local line; line=$("$PY" -u "$S" --nx "$nx" --ny "$ny" --duration "$DUR" --camera --fps "$FPS" --maxq 2 "$@" 2>/dev/null | grep -E '^(RESULT|ERROR)')
    echo "$tag r$r ${line:-NO_RESULT_${nx}x${ny}}" | tee -a "$OUT"; sleep 2
  done
}
echo "## cols: tag rN RESULT mode|inf/f|sysFPS|totInf/s|tileFPS|tP50|tP90|tP99|fullFPS|fP50|fP90|e2eP50|e2eP90|capP50|capP90|tC0|tC1|out/in" | tee -a "$OUT"
echo "# === DUAL +1 (round-robin) live camera ===" | tee -a "$OUT"
for m in "2 1" "2 2" "3 2" "3 3"; do run dual $m; done
echo "# === SINGLE tiling-only (merge-load reference) live camera ===" | tee -a "$OUT"
for m in "2 1" "2 2" "3 2" "3 3"; do run single $m --single; done
echo "DONE" | tee -a "$OUT"
