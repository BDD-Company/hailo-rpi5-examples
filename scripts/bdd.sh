#! /usr/bin/env bash

cd ./hailo-rpi5-examples/
. ./setup_env.sh

readonly start_time=$(date +%Y%m%d-%H%M%S)
# make sure that log dir exists
mkdir -p ./_DEBUG/ ||:

set -ex

(
    if git status ;
    then
        git -P log -n1 --oneline
        git -P describe --long --always --dirty ||:
        git -P diff ||:
    fi

    export GST_DEBUG=3
#    export G_MESSAGES_DEBUG=all
#    export GST_TRACERS="latency;stats"

    export DISPLAY=:0
    export HAILO_MODEL="/home/bdd/models/2026-04-22_11n_sh_v5.hef"

    python \
        ./BDD/app.py \
        --hef-path "${HAILO_MODEL}" \
        -i rpi \
        "$@"

) &> "./_DEBUG/BDD_${start_time}.log" &

tail -f "./_DEBUG/BDD_${start_time}.log"

