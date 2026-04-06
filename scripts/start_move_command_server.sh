#!/usr/bin/env bash

WHAT_TO_START="${1:-move_http.py}"

# Do everythin in subshell so venv doesn't affect caller's shell
(
  set -e
  cd ~/BDD/WaveShare/pt_rpi/
  source ./ugv-env/bin/activate
  while true;
  do
    echo `date` will start ${WHAT_TO_START}
    # ignore errors to make sure server is always started
    python ${WHAT_TO_START} ||:
  done
)
