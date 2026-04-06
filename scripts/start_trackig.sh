#!/usr/bin/env bash


# force forward all to pre-existing display, even when connected via ssh
#export DISPLAY=:0
#export LIBCAMERA_RPI_TUNING_FILE=/usr/share/libcamera/ipa/rpi/pisp/imx477_noir.json

readonly PATH_TO_SCRIPT="${0}"
function start()
{
    echo $BASH_SUBSHELL
    set -x
    export LIBCAMERA_RPI_TUNING_FILE="/usr/share/libcamera/ipa/rpi/pisp/arducam_64mp.json" #"/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json"
    PROJECT_ROOT="$(realpath `dirname ${PATH_TO_SCRIPT}`/..)"
    cd "${PROJECT_ROOT}"
#    source ${PROJECT_ROOT}/.venv/bin/activate
    export DISPLAY=:0
    ${PROJECT_ROOT}/src/track.py \
        --config=configs/config.json \
        "$@"
#         --platform=mock
}

(
    echo "Resetting mavlink"
    set -ex

    source /home/dima/BDD/Drone/px4/autopilot-main/venv/bin/activate

    # Reset mavlink first
    printf "mavlink stop-all\nmavlink start -x -u 14550 -r 4000000\n" | \
        /home/dima/BDD/Drone/px4/PX4-Autopilot/Tools/mavlink_shell.py  --udp :14550
)

#since it enters venv, run in subshell
(start $@)

# fix hailo: CHECK failed - max_desc_page_size given 16384 is bigger than hw max desc page size 4096
# sudo modprobe -r hailo_pci && sudo modprobe hailo_pci force_desc_page_size=256
