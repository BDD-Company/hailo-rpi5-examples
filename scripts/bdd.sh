#! /usr/bin/env bash

cd ./hailo-rpi5-examples/
. ./setup_env.sh

readonly start_time=$(date +%Y%m%d-%H%M%S)
# make sure that log dir exists
mkdir -p ./_DEBUG/ ||:

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
#    export GST_DEBUG="GST_TRACER:7"
    export DISPLAY=:0
    #export HAILO_MODEL="/home/bdd/models/2026-04-07_11n_ball_v2.hef"
    export HAILO_MODEL="/home/bdd/models/ball_11n_640_v1.hef" # detects SUN
    #export HAILO_MODEL="/home/bdd/models/2026-04-13_11n_sh_v3.hef" # detects SUN


    python \
        ./BDD/app.py \
        --hef-path "${HAILO_MODEL}" \
        -i rpi \
        "$@"

) 2>&1 | tee "./_DEBUG/BDD_${start_time}.log"
