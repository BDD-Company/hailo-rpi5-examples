#!/usr/bin/env bash
# Build the custom grid+whole-frame hailocropper .so on the Pi.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -fPIC -shared \
    -I/usr/include/hailo/tappas -I/usr/include/opencv4 \
    -o libtiles_and_whole.so tiles_and_whole_cropper.cpp
echo "built: $(pwd)/libtiles_and_whole.so"
