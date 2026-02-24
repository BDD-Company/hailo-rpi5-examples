#! /usr/bin/env bash

cd ./hailo-rpi5-examples/
. ./setup_env.sh 

git -P status ||:
git -P diff ||:

set -ex

DISPLAY=:0 python ./BDD/app.py --hef-path /home/bdd/models/2026-02-18_n640-last.hef -i rpi -f -r 60 "$@" 

