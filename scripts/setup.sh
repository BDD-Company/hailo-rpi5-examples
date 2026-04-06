#!/bin/env bash

set -e

if [ "$(uname -s)" != "Linux" ] || [ "$(grep '^ID=' /etc/os-release | cut -d= -f2)" != "ubuntu" ]; then
    echo "This script is only supported on Ubuntu-based Linux systems."
    exit 1
fi


sudo apt install libcap-dev
# gstreamer and dependencies for RTSP streaming, must be done before creating venv,
# so that all packages would be available in venv
sudo apt-get install -y \
    python3-gi \
    python3-gst-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-rtsp \
    gstreamer1.0-plugins-ugly

# gstreamer1.0-plugins-ugly - for x264enc
# gstreamer1.0-rtsp for RTSP

python -m vevnv ./.venv
source ./.venv/bin/activate
# pip install poetry

pip install -r ./requirements.txt

git submodule update --init

(
    cd ./external/autopilot
    pip install -e .
    pip install -r ./requirements.txt
)

# HOW to cleanup requirements
# rm ./requirements.txt
# pipx install pipreqs
# pipreqs . --force --ignore .venv,.venv_clean,src/experiments,external
# # Manually review ./requirements.txt to figure out any unnecessary packages
# git commit ./requirements.txt 'Updated requirements'
#
