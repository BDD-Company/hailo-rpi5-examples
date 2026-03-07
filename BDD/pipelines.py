#!/usr/bin/env python3

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import get_camera_resulotion

# Absolute import for your local helper
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    get_source_type,
)

from hailo_apps.hailo_app_python.core.common.defines import (
    TAPPAS_POSTPROC_PATH_KEY,
    GST_VIDEO_SINK,
    TAPPAS_POSTPROC_PATH_DEFAULT,
)

import os

"""
Set of pipelines based on hailo default examples.

This set is tweaked to minimize latency from capturing frame to consuming detection results.
"""


def QUEUE(name, max_size_buffers=3, max_size_bytes=0, max_size_time=0, leaky='no'):
    """
    Creates a GStreamer queue element string with the specified parameters.

    Args:
        name (str): The name of the queue element.
        max_size_buffers (int, optional): The maximum number of buffers that the queue can hold. Defaults to 3.
        max_size_bytes (int, optional): The maximum size in bytes that the queue can hold. Defaults to 0 (unlimited).
        max_size_time (int, optional): The maximum size in time that the queue can hold. Defaults to 0 (unlimited).
        leaky (str, optional): The leaky type of the queue. Can be 'no', 'upstream', or 'downstream'. Defaults to 'no'.

    Returns:
        str: A string representing the GStreamer queue element with the specified parameters.
    """
    q_string = f'queue name={name} leaky={leaky} max-size-buffers={max_size_buffers} max-size-bytes={max_size_bytes} max-size-time={max_size_time} '
    return q_string


def SOURCE_PIPELINE(video_source, video_width=640, video_height=640,
                    name='source', no_webcam_compression=False,
                    frame_rate=30, sync=True,
                    video_format='RGB',
                    do_timestamp=False,
                    force_framerate=None):
    """
    Creates a GStreamer pipeline string for the video source with a separate fps caps
    for frame rate control.

    Args:
        video_source (str): The path or device name of the video source.
        video_width (int, optional): The width of the video. Defaults to 640.
        video_height (int, optional): The height of the video. Defaults to 640.
        video_format (str, optional): The video format. Defaults to 'RGB'.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'source'.

    Returns:
        str: A string representing the GStreamer pipeline for the video source.
    """
    source_type = get_source_type(video_source)

    if source_type == 'usb':
        if no_webcam_compression:
            # When using uncompressed format, only low resolution is supported
            source_element = (
                f'v4l2src device={video_source} name={name} ! '
                f'video/x-raw, width=640, height=480 ! '
                'videoflip name=videoflip video-direction=horiz ! '
            )
        else:
            # Use compressed format for webcam
            width, height = get_camera_resulotion(video_width, video_height)
            source_element = (
                f'v4l2src device={video_source} name={name} ! image/jpeg, framerate=30/1, width={width}, height={height} ! '
                f'{QUEUE(name=f"{name}_queue_decode")} ! '
                f'decodebin name={name}_decodebin ! '
                f'videoflip name=videoflip video-direction=horiz ! '
            )
    elif source_type == 'rpi':
        do_timestamp_clause = ''
        if do_timestamp:
            do_timestamp_clause=" do-timestamp=true format=time "

        framere_clause=''
        if force_framerate is not None:
            framerate= int(force_framerate)
            framere_clause=f',framerate={framerate}/1'

        source_element = (
            # max-size-time is 100ms in nanoseconds
            f'appsrc name=app_source is-live=true leaky-type=downstream {do_timestamp_clause} ! '
            # 'videoflip name=videoflip video-direction=horiz ! '
            f'video/x-raw, format={video_format}, width={video_width}, height={video_height} {framere_clause} ! '
        )
    elif source_type == 'libcamera':
        source_element = (
            f'libcamerasrc name={name} ! '
            f'video/x-raw, format={video_format}, width=1536, height=864 ! '
        )
    elif source_type == 'ximage':
        source_element = (
            f'ximagesrc xid={video_source} ! '
            f'{QUEUE(name=f"{name}queue_scale_")} ! '
            f'videoscale ! '
        )
    else:
        source_element = (
            f'filesrc location="{video_source}" name={name} ! '
            f' qtdemux name=demux demux.video_0 ! '
            f'{QUEUE(name=f"{name}_queue_decode")} ! '
            f'decodebin name={name}_decodebin ! '
        )

    # Set up the fps caps.
    # If sync is True, constrain the rate with the given frame_rate.
    # Otherwise, pass through (no framerate limitation).
    if sync:
        fps_caps = f"video/x-raw, framerate={frame_rate}/1"
    else:
        fps_caps = "video/x-raw"

    source_pipeline = (
        f'{source_element} '
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f'videoscale name={name}_videoscale n-threads=2 ! '
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        f'videoconvert n-threads=3 name={name}_convert qos=false ! '
        f'video/x-raw, pixel-aspect-ratio=1/1, format={video_format}, '
        f'width={video_width}, height={video_height} ! '
        f'videorate name={name}_videorate ! capsfilter name={name}_fps_caps caps="{fps_caps}" '
    )

    return source_pipeline


def INFERENCE_PIPELINE(
    hef_path,
    post_process_so=None,
    batch_size=1,
    config_json=None,
    post_function_name=None,
    additional_params='',
    name='inference',
    # Extra hailonet parameters
    scheduler_timeout_ms=None,
    scheduler_priority=None,
    vdevice_group_id=1,
    multi_process_service=None
):
    """
    Creates a GStreamer pipeline string for inference and post-processing using a user-provided shared object file.
    This pipeline includes videoscale and videoconvert elements to convert the video frame to the required format.
    The format and resolution are automatically negotiated based on the HEF file requirements.

    Args:
        hef_path (str): Path to the HEF file.
        post_process_so (str or None): Path to the post-processing .so file. If None, post-processing is skipped.
        batch_size (int): Batch size for hailonet (default=1).
        config_json (str or None): Config JSON for post-processing (e.g., label mapping).
        post_function_name (str or None): Function name in the .so postprocess.
        additional_params (str): Additional parameters appended to hailonet.
        name (str): Prefix name for pipeline elements (default='inference').

        # Extra hailonet parameters
        Run `gst-inspect-1.0 hailonet` for more information.
        vdevice_group_id (int): hailonet vdevice-group-id. Default=1.
        scheduler_timeout_ms (int or None): hailonet scheduler-timeout-ms. Default=None.
        scheduler_priority (int or None): hailonet scheduler-priority. Default=None.
        multi_process_service (bool or None): hailonet multi-process-service. Default=None.

    Returns:
        str: A string representing the GStreamer pipeline for inference.
    """
    # config & function strings
    config_str = f' config-path={config_json} ' if config_json else ''
    function_name_str = f' function-name={post_function_name} ' if post_function_name else ''
    vdevice_group_id_str = f' vdevice-group-id={vdevice_group_id} '
    multi_process_service_str = f' multi-process-service={str(multi_process_service).lower()} ' if multi_process_service is not None else ''
    scheduler_timeout_ms_str = f' scheduler-timeout-ms={scheduler_timeout_ms} ' if scheduler_timeout_ms is not None else ''
    scheduler_priority_str = f' scheduler-priority={scheduler_priority} ' if scheduler_priority is not None else ''

    hailonet_str = (
        f'hailonet name={name}_hailonet '
        f'hef-path={hef_path} '
        f'batch-size={batch_size} '
        f'{vdevice_group_id_str}'
        f'{multi_process_service_str}'
        f'{scheduler_timeout_ms_str}'
        f'{scheduler_priority_str}'
        f'{additional_params} '
        f'force-writable=true '
    )

    inference_pipeline = (
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f'videoscale name={name}_videoscale n-threads=2 qos=false ! '
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name={name}_videoconvert n-threads=2 ! '
        f'{QUEUE(name=f"{name}_hailonet_q")} ! '
        f'{hailonet_str} ! '
    )

    if post_process_so:
        inference_pipeline += (
            f'{QUEUE(name=f"{name}_hailofilter_q")} ! '
            f'hailofilter name={name}_hailofilter so-path={post_process_so} {config_str} {function_name_str} qos=false ! '
        )

    inference_pipeline += f'{QUEUE(name=f"{name}_output_q")} '

    return inference_pipeline


def INFERENCE_PIPELINE_WRAPPER(inner_pipeline, bypass_max_size_buffers=20, name='inference_wrapper'):
    """
    Creates a GStreamer pipeline string that wraps an inner pipeline with a hailocropper and hailoaggregator.
    This allows to keep the original video resolution and color-space (format) of the input frame.
    The inner pipeline should be able to do the required conversions and rescale the detection to the original frame size.

    Args:
        inner_pipeline (str): The inner pipeline string to be wrapped.
        bypass_max_size_buffers (int, optional): The maximum number of buffers for the bypass queue. Defaults to 20.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'inference_wrapper'.

    Returns:
        str: A string representing the GStreamer pipeline for the inference wrapper.
    """
    # Get the directory for post-processing shared objects
    tappas_post_process_dir = os.environ.get(TAPPAS_POSTPROC_PATH_KEY, TAPPAS_POSTPROC_PATH_DEFAULT)
    whole_buffer_crop_so = os.path.join(tappas_post_process_dir, 'cropping_algorithms/libwhole_buffer.so')

    # Construct the inference wrapper pipeline string
    inference_wrapper_pipeline = (
        f'{QUEUE(name=f"{name}_input_q")} ! '
        f'hailocropper name={name}_crop so-path={whole_buffer_crop_so} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true '
        f'hailoaggregator name={name}_agg '
        f'{name}_crop. ! {QUEUE(max_size_buffers=bypass_max_size_buffers, name=f"{name}_bypass_q")} ! {name}_agg.sink_0 '
        f'{name}_crop. ! {inner_pipeline} ! {name}_agg.sink_1 '
        f'{name}_agg. ! {QUEUE(name=f"{name}_output_q")} '
    )

    return inference_wrapper_pipeline


def USER_CALLBACK_PIPELINE(name='identity_callback'):
    """
    Creates a GStreamer pipeline string for the user callback element.

    Args:
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'identity_callback'.

    Returns:
        str: A string representing the GStreamer pipeline for the user callback element.
    """
    # Construct the user callback pipeline string
    user_callback_pipeline = (
        f'{QUEUE(name=f"{name}_q")} ! '
        f'identity name={name} '
    )

    return user_callback_pipeline

def TRACKER_PIPELINE(class_id, kalman_dist_thr=0.8, iou_thr=0.9, init_iou_thr=0.7, keep_new_frames=2, keep_tracked_frames=15, keep_lost_frames=2, keep_past_metadata=False, qos=False, name='hailo_tracker'):
    """
    Creates a GStreamer pipeline string for the HailoTracker element.
    Args:
        class_id (int): The class ID to track. Default is -1, which tracks across all classes.
        kalman_dist_thr (float, optional): Threshold used in Kalman filter to compare Mahalanobis cost matrix. Closer to 1.0 is looser. Defaults to 0.8.
        iou_thr (float, optional): Threshold used in Kalman filter to compare IOU cost matrix. Closer to 1.0 is looser. Defaults to 0.9.
        init_iou_thr (float, optional): Threshold used in Kalman filter to compare IOU cost matrix of newly found instances. Closer to 1.0 is looser. Defaults to 0.7.
        keep_new_frames (int, optional): Number of frames to keep without a successful match before a 'new' instance is removed from the tracking record. Defaults to 2.
        keep_tracked_frames (int, optional): Number of frames to keep without a successful match before a 'tracked' instance is considered 'lost'. Defaults to 15.
        keep_lost_frames (int, optional): Number of frames to keep without a successful match before a 'lost' instance is removed from the tracking record. Defaults to 2.
        keep_past_metadata (bool, optional): Whether to keep past metadata on tracked objects. Defaults to False.
        qos (bool, optional): Whether to enable QoS. Defaults to False.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_tracker'.
    Note:
        For a full list of options and their descriptions, run `gst-inspect-1.0 hailotracker`.
    Returns:
        str: A string representing the GStreamer pipeline for the HailoTracker element.
    """
    # Construct the tracker pipeline string
    tracker_pipeline = (
        f'hailotracker name={name} class-id={class_id} kalman-dist-thr={kalman_dist_thr} iou-thr={iou_thr} init-iou-thr={init_iou_thr} '
        f'keep-new-frames={keep_new_frames} keep-tracked-frames={keep_tracked_frames} keep-lost-frames={keep_lost_frames} keep-past-metadata={keep_past_metadata} qos={qos} ! '
        f'{QUEUE(name=f"{name}_q")} '
    )
    return tracker_pipeline


def OVERLAY_PIPELINE(name='hailo_overlay'):
    """
    Creates a GStreamer pipeline string for the hailooverlay element.
    This pipeline is used to draw bounding boxes and labels on the video.

    Args:
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_overlay'.

    Returns:
        str: A string representing the GStreamer pipeline for the hailooverlay element.
    """
    # Construct the overlay pipeline string
    overlay_pipeline = (
        f'{QUEUE(name=f"{name}_q")} ! '
        f'hailooverlay name={name} '
    )

    return overlay_pipeline

def DISPLAY_PIPELINE(video_sink=GST_VIDEO_SINK, sync='true', show_fps='false', name='hailo_display'):
    """
    Creates a GStreamer pipeline string for displaying the video.
    It includes the hailooverlay plugin to draw bounding boxes and labels on the video.

    Args:
        video_sink (str, optional): The video sink element to use. Defaults to 'autovideosink'.
        sync (str, optional): The sync property for the video sink. Defaults to 'true'.
        show_fps (str, optional): Whether to show the FPS on the video sink. Should be 'true' or 'false'. Defaults to 'false'.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_display'.

    Returns:
        str: A string representing the GStreamer pipeline for displaying the video.
    """
    # Construct the display pipeline string
    display_pipeline = (
        # f'{OVERLAY_PIPELINE(name=f"{name}_overlay")} ! '
        f'{QUEUE(name=f"{name}_videoconvert_q")} ! '
        f'videoconvert name={name}_videoconvert n-threads=2 qos=false ! '
        f'{QUEUE(name=f"{name}_q")} ! '
        f'fpsdisplaysink name={name} video-sink={video_sink} sync={sync} text-overlay={show_fps} signal-fps-measurements=true '
    )

    return display_pipeline
