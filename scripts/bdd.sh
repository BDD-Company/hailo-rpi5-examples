#! /usr/bin/env bash

cd ./hailo-rpi5-examples/
. ./setup_env.sh

readonly start_time=$(date +%Y%m%d-%H%M%S)
# make sure that log dir exists
mkdir -p ./_DEBUG/ ||:

function kill_resource_holder()
{
    local resource="$1"
    for p in $(fuser -v ${resource} 2>&1 | tail -n-1 | awk '{print $3}');
    do
        echo ${p} holds ${resource} killing it...
        kill -9 ${p} ||:
    done
}

set -ex

(
    if git status ;
    then
        git -P log -n1 --oneline
        git -P describe --long --always --dirty ||:
        git -P diff ||:
    fi

    kill_resource_holder "/dev/hailo0"
    kill_resource_holder "/dev/video*"

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

