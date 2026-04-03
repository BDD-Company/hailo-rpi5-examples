#! /usr/bin/env bash

cd ./hailo-rpi5-examples/
. ./setup_env.sh

readonly start_time=$(date +%Y%m%d-%H%M%S)
set -ex

(
    if [[ -d .git ]];
    then
        git -P log -n1 --oneline
        git -P describe --long --always --dirty ||:
        git -P diff ||:
    fi

#    export G_MESSAGES_DEBUG=all
#    export GST_TRACERS="latency;stats"
#    export GST_DEBUG="7"
#    export GST_DEBUG_FILE=_DEBUG/gstreamer_${start_time}.log
#    export GST_DEBUG_NO_COLOR=1
    export DISPLAY=:0
#    export HAILO_MODEL="/home/bdd/models/2026-03-13_shahed_v7_1280.hef"
    export HAILO_MODEL="/home/bdd/models/2026-03-13_shahed_v7_1280.hef"

    python \
        ./BDD/app.py \
        -i rpi \
        --hef-path="${HAILO_MODEL}" \
        -r 30 \
        "$@"
) 2>&1 | tee "./_DEBUG/BDD_${start_time}.log"
