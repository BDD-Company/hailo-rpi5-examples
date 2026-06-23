#!/usr/bin/env bash
# Full benchmark matrix, run on the Pi. Writes results to /tmp/nv12_matrix.txt
# No pkill (its -f pattern self-matches this script's shell); the python self-terminates
# after duration+warmup and force-frees /dev/hailo0 via its hardened teardown.
set -u
PY=$HOME/hailo-rpi5-examples/venv_hailo_rpi_examples/bin/python
S=$HOME/hailo-rpi5-examples/BDD/experiments/nv12_tiling_bench/nv12_tiling_bench.py
OUT=/tmp/nv12_matrix.txt
DUR=${DUR:-5}
REPEATS=${REPEATS:-2}
: > "$OUT"

run(){ # tag nx ny extra...
  local tag=$1 nx=$2 ny=$3; shift 3
  for r in $(seq 1 "$REPEATS"); do
    local line
    line=$("$PY" -u "$S" --nx "$nx" --ny "$ny" --duration "$DUR" "$@" 2>/dev/null | grep -E '^(RESULT|ERROR)')
    echo "$tag r$r ${line:-NO_RESULT_${nx}x${ny}}" | tee -a "$OUT"
    sleep 2
  done
}

echo "## cols: tag rN RESULT mode|inf/f|sysFPS|totInf/s|tileFPS|tP50|tP90|tP99|fullFPS|fP50|fP90|e2eP50|e2eP90|tC0|tC1|out/in" | tee -a "$OUT"
echo "# === SINGLE (tiling only, uncapped ceiling) ===" | tee -a "$OUT"
for m in "1 1" "2 1" "2 2" "3 2" "3 3"; do run single $m --uncapped --single; done
echo "# === DUAL (NxM tiles + whole-frame +1, round-robin, uncapped) ===" | tee -a "$OUT"
for m in "2 1" "2 2" "3 2" "3 3"; do run dual $m --uncapped; done
echo "# === DUAL at 30fps operating point ===" | tee -a "$OUT"
for m in "2 1" "2 2" "3 2" "3 3"; do run op30 $m; done
echo "DONE" | tee -a "$OUT"
