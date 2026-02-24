#!/usr/bin/env bash

set -ex

(
	cd /home/bdd/TMP/PX4-Autopilot/; 
	sudo taskset -c 2 ./build/scumaker_pilotpi_arm64/bin/px4 -s posix-configs/rpi/pilotpi_mc.config
)

