ffmpeg -f v4l2 -input_format yuyv422 -video_size 640x512 -framerate 30 -i /dev/video8 \
    -c:v libx264 -preset ultrafast -tune zerolatency -crf 28 -pix_fmt yuv420p \
    -f segment -segment_time 30 -segment_atclocktime 1 -reset_timestamps 1 -strftime 1 \
    "thermal_${CAMERA}_%Y-%m-%d_%H-%M-%S.mp4"
