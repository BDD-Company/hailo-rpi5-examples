#!/usr/bin/env bash

while true; do 
    rpicam-vid \
        --codec libav \
        --libav-format avi \
        --libav-audio \
        --output camera_0_$(date | tr [:space:][:punct:] _).avi \
        --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_noir.json \
        -t 1000000 \
        --width 1280 --height 720 \
        --framerate 30;
done
