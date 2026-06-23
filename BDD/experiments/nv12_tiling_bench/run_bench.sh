#!/usr/bin/env bash
# Drives nv12_tiling_bench.py across the requested tiling modes on the Pi.
# Each mode runs in its own process for clean VDevice teardown.
set -u
HEF="${HEF:-/home/bdd/models/yolov11n_nv12.hef}"
PY="${PY:-$HOME/hailo-rpi5-examples/venv_hailo_rpi_examples/bin/python}"
SCRIPT="${SCRIPT:-$HOME/hailo-rpi5-examples/BDD/experiments/nv12_tiling_bench/nv12_tiling_bench.py}"
DUR="${DUR:-5}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
EXTRA="${EXTRA:-}"     # set EXTRA=--uncapped for ceiling test

MODES=("2 1" "2 2" "3 2" "3 3")

echo "# hef=$HEF dur=${DUR}s src=${WIDTH}x${HEIGHT}@${FPS} extra='${EXTRA}'"
echo "# cols: mode|inf/frame|sysFPS|totalInf/s|tileFPS|tileP50|tileP90|tileP99|fullFPS|fullP50|fullP90|e2ePipeP50|e2ePipeP90|tempC0|tempC1|out/in"
for m in "${MODES[@]}"; do
  read -r NX NY <<<"$m"
  "$PY" "$SCRIPT" --hef "$HEF" --nx "$NX" --ny "$NY" \
      --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" --duration "$DUR" $EXTRA \
      2>/dev/null | grep -E "^(RESULT|ERROR)" || echo "FAIL ${NX}x${NY}"
  sleep 1
done
