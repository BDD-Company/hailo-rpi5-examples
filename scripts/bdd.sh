#! /usr/bin/env bash

set -ex
echo `pwd`

cd ./hailo-rpi5-examples/

readonly start_time=$(date +%Y%m%d-%H%M%S)
# make sure that log dir exists
mkdir -p ./_DEBUG/ ||:


(
    #. ./setup_env.sh
    . ./venv_hailo_rpi_examples/bin/activate

    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -P log -n1 --oneline ||:
        git -P describe --long --always --dirty ||:
        git -P diff ||:
    fi

#    export G_MESSAGES_DEBUG=all
#    export GST_TRACERS="latency;stats"
#    export GST_DEBUG="GST_TRACER:7"
    export DISPLAY=:0
    #export HAILO_MODEL="/home/bdd/models/2026-04-07_11n_ball_v2.hef"
    # export HAILO_MODEL="/home/bdd/models/ball_11n_640_v1.hef" # detects SUN
    export HAILO_MODEL="/home/bdd/models/2026-04-18_11n_sh_v4.hef" # excludes Agro drone

    [[ -e "${HAILO_MODEL}" ]] || { echo "HAILO_MODEL does not exist: ${HAILO_MODEL}" >&2; exit 1; }

    # bind to 3rd core, 2 is bound to px4, effective only IF `isolcpus=2,3` is set in cmdline.txt
    taskset -c 3 \
        python \
            BDD/app.py \
            --hef-path "${HAILO_MODEL}" \
            -i rpi \
            "$@"

) 2>&1 | tee "./_DEBUG/BDD_${start_time}.log"
