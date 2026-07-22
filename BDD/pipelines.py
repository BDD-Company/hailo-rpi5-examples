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
import re

"""
Set of pipelines based on hailo default examples.

This set is tweaked to minimize latency from capturing frame to consuming detection results.
"""


def QUEUE(name, max_size_buffers=1, max_size_bytes=0, max_size_time=0, leaky='downstream'):
    """
    Creates a GStreamer queue element string with the specified parameters.

    Args:
        name (str): The name of the queue element.
        max_size_buffers (int, optional): The maximum number of buffers that the queue can hold. Defaults to 1.
            Set to 1 so each stage holds at most one frame; leaky=downstream drops the oldest when full,
            ensuring downstream always receives the freshest available frame.
        max_size_bytes (int, optional): The maximum size in bytes that the queue can hold. Defaults to 0 (unlimited).
        max_size_time (int, optional): The maximum time (nanoseconds) the queue may buffer. Defaults to 0 (disabled).
            Kept at 0 so buffer count is the sole cap; a non-zero time limit grows proportionally to
            downstream slowness and can mask backpressure instead of shedding it.
        leaky (str, optional): The leaky type of the queue. Can be 'no', 'upstream', or 'downstream'. Defaults to 'downstream'.

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
            # `format=time` MUST be set on appsrc at pipeline-parse time, not later from
            # picamera_thread: with the lighter single-stream picam2 config, the main
            # thread now reaches Gst.State.PAUSED before picamera_thread runs its
            # set_property("format", Gst.Format.TIME), so appsrc emits a BYTES-format
            # segment and videorate downstream rejects it with "Got segment but doesn't
            # have GST_FORMAT_TIME value" -> Internal data stream error and the pipeline
            # dies. Declaring it here eliminates the race regardless of thread timing.
            f'appsrc name=app_source is-live=true leaky-type=downstream format=time {do_timestamp_clause} ! '
            # 'videoflip name=videoflip video-direction=horiz ! '
            f'video/x-raw, format={video_format}, width={video_width}, height={video_height} {framere_clause} ! '
        )
    elif source_type == 'libcamera':
        framerate = int(force_framerate if force_framerate is not None else frame_rate)
        source_element = (
            f'libcamerasrc name={name} exposure-time-mode=manual exposure-time=8000 ! '
            f'video/x-raw, format=NV12, width={video_width}, height={video_height}, framerate={framerate}/1 ! '
        )
    elif source_type == 'ximage':
        source_element = (
            f'ximagesrc xid={video_source} ! '
            f'{QUEUE(name=f"{name}queue_scale_")} ! '
            f'videoscale ! '
        )
    else:
        # `identity sync=true` paces the file at its own PTS, making it behave like the
        # 30fps live camera the whole downstream is built for. Without it filesrc races
        # to EOF as fast as the disk allows -- a 12MB clip in ~60ms -- so EOS fires
        # almost immediately, on_eos rewinds, and it repeats: measured 13 EOS/s for a
        # whole 80s run, seek-thrashing the pipeline while the queues fed inference a
        # mix of overlapping passes. That, not the codec, is what looked like "playback
        # wedges at frame 1 then EOS".
        #
        # It goes HERE, at the source, not on the sink. Everything downstream (leaky
        # queues 1 deep, sync=false sink) is deliberately live-tuned to never wait on a
        # clock; clocking the sink instead fights that and stalls the run at 2 frames in
        # 80s. Pacing here leaves the live path's semantics untouched: frames arrive at
        # 30fps and are handled exactly as camera frames are, drops included.
        #
        # And it goes BEFORE the decode queue, not after decodebin. That queue is
        # leaky=downstream and only 1 deep, so with an unpaced filesrc in front of it it
        # sheds COMPRESSED H.264 packets -- gutting the reference frames, so decodebin
        # emits nothing at all (0 frames in 80s, measured). Pacing ahead of it means it
        # never overflows and every packet decodes. Syncing on compressed buffers is
        # sound here because make_clips.sh encodes baseline profile / no B-frames, so
        # PTS == DTS and identity's clock wait matches presentation order.
        source_element = (
            f'filesrc location="{video_source}" name={name} ! '
            f' qtdemux name=demux demux.video_0 ! '
            f'identity name={name}_pace sync=true ! '
            f'{QUEUE(name=f"{name}_queue_decode")} ! '
            f'decodebin name={name}_decodebin ! '
        )

    # Set up the fps caps.
    # If sync is True, constrain the rate with the given frame_rate.
    # Otherwise, pass through (no framerate limitation).
    # Always pin format/size so caps queries can't intersect upstream Bayer/etc.
    # back through libcamerasrc during negotiation.
    base_caps = f"video/x-raw, format={video_format}, width={video_width}, height={video_height}"
    if sync:
        fps_caps = f"{base_caps}, framerate={frame_rate}/1"
    else:
        fps_caps = base_caps

    if source_type == 'rpi':
        # The picamera2/appsrc producer already delivers exactly video_format @
        # video_width x video_height at the camera's natural (variable) rate with correct
        # per-buffer PTS, so the whole common tail is redundant on this path:
        #   - videoscale (W->W) / videoconvert (format->format) — pure no-op full-frame
        #     passes;
        #   - videorate + fps_caps — only DUPLICATED the variable camera stream up to a
        #     fixed framerate, and those dupes got inferred by Hailo then dropped by the
        #     callback's frame-id dedup, wasting inference. Downstream times off PTS, so
        #     no fixed rate is needed for the live path.
        # The inference wrapper does its own scale/convert to the model input downstream.
        # (update_fps_caps() would look up the now-removed videorate/fps_caps elements,
        # but it has no callers and already no-ops gracefully if they are absent.)
        source_pipeline = (
            f'{source_element} '
            f'{QUEUE(name=f"{name}_scale_q")} '
        )
    else:
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


def NON_LETTERBOX_POST_FUNCTION(post_function_name):
    """Postprocess function to use on a TILED inference branch: the plain (non-letterbox)
    variant of ``post_function_name``.

    The ``*_letterbox`` postprocess variants flatten every detection by the ROI's own
    bbox (``create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox())``). On a
    tile crop the ROI bbox IS the tile bbox, so ``filter_letterbox`` pre-applies the
    tile→frame remap — and hailotileaggregator then applies the same remap again.
    Result: every tiled detection lands at ``tile_offset + x/2`` (the offset-bbox bug
    of 2026-07-20). Tile crops are never letterboxed anyway (hailotilecropper emits
    plain crops; videoscale's stretch preserves normalized coordinates), so the tiled
    branch must use the plain variant and let the aggregator do the one remap.
    """
    if post_function_name and post_function_name.endswith('_letterbox'):
        return post_function_name[: -len('_letterbox')]
    return post_function_name


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
    multi_process_service=None,
    internal_leaky='no',
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

    # When this pipeline is wrapped by hailocropper / hailoaggregator (the typical case),
    # NONE of these queues may drop buffers: the aggregator pairs sink_0 (bypass) with sink_1
    # (this branch) by per-buffer offset, so dropping a buffer here orphans the matching bypass
    # buffer and the aggregator deadlocks. internal_leaky='no' propagates backpressure all the way
    # back through the cropper to inference_wrapper_input_q, which IS leaky=downstream and sheds
    # the oldest camera frame instead — keeping both branches in lockstep.
    inference_pipeline = (
        f'{QUEUE(name=f"{name}_scale_q", leaky=internal_leaky)} ! '
        f'videoscale name={name}_videoscale n-threads=2 qos=false ! '
        f'{QUEUE(name=f"{name}_convert_q", leaky=internal_leaky)} ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name={name}_videoconvert n-threads=2 ! '
        f'{QUEUE(name=f"{name}_hailonet_q", leaky=internal_leaky)} ! '
        f'{hailonet_str} ! '
    )

    if post_process_so:
        inference_pipeline += (
            f'{QUEUE(name=f"{name}_hailofilter_q", leaky=internal_leaky)} ! '
            f'hailofilter name={name}_hailofilter so-path={post_process_so} {config_str} {function_name_str} qos=false ! '
        )

    inference_pipeline += f'{QUEUE(name=f"{name}_output_q", leaky=internal_leaky)} '

    return inference_pipeline


def INFERENCE_PIPELINE_WRAPPER(inner_pipeline, bypass_max_size_buffers=1, name='inference_wrapper',
                               tiles_x=1, tiles_y=1, tiling_overlap=0.0, tile_iou_threshold=0.4):
    """
    Creates a GStreamer pipeline string that wraps an inner pipeline with a cropper and aggregator.
    This allows to keep the original video resolution and color-space (format) of the input frame.
    The inner pipeline should be able to do the required conversions and rescale the detection to the original frame size.

    Whole-frame (tiles_x == tiles_y == 1): whole-buffer ``hailocropper`` / ``hailoaggregator`` —
    one inference per frame (the proven low-latency default).
    Tiled (tiles_x * tiles_y > 1): ``hailotilecropper`` / ``hailotileaggregator`` splits each frame
    into an ``tiles_x`` × ``tiles_y`` grid, runs ONE inference per tile (so N tiles = N serialized
    inferences/frame), and the aggregator stitches the per-tile detections back into full-frame
    coordinates. Tiling raises small-object recall at distance but multiplies inference cost — at the
    Hailo-8 ~68 inf/s ceiling, per-frame latency ≈ N × ~15 ms, so it trades reaction time for range.

    Args:
        inner_pipeline (str): The inner pipeline string to be wrapped.
        bypass_max_size_buffers (int, optional): The maximum number of buffers for the bypass queue. Defaults to 1.
            This queue MUST be non-leaky: the aggregator pairs sink_0 (bypass) with sink_1 (inference)
            by buffer; if a leaky queue drops the bypass buffer the aggregator is waiting for, the two
            streams desync and the aggregator stops emitting forever. With leaky=no, hitting the cap
            backpressures the cropper, which backpressures the input queue (which is leaky=downstream
            and sheds the oldest camera frame instead). Worst-case added latency is
            (bypass_max_size_buffers - 1) * frame_interval — keep this number small for low latency.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'inference_wrapper'.
        tiles_x, tiles_y (int): tile grid. 1×1 = whole-frame (default). >1 enables hailotilecropper.
        tiling_overlap (float): fractional tile overlap (0..1) on both axes; 0 = abutting tiles.
        tile_iou_threshold (float): IoU above which the tile aggregator merges two detections
            coming from adjacent tiles (seam dedup). Tiled path only.

    Returns:
        str: A string representing the GStreamer pipeline for the inference wrapper.
    """
    tiled = tiles_x > 1 or tiles_y > 1
    if not tiled:
        # Get the directory for post-processing shared objects
        tappas_post_process_dir = os.environ.get(TAPPAS_POSTPROC_PATH_KEY, TAPPAS_POSTPROC_PATH_DEFAULT)
        whole_buffer_crop_so = os.path.join(tappas_post_process_dir, 'cropping_algorithms/libwhole_buffer.so')
        cropper = (
            f'hailocropper name={name}_crop so-path={whole_buffer_crop_so} function-name=create_crops '
            f'use-letterbox=true resize-method=inter-area internal-offset=true '
        )
        aggregator = f'hailoaggregator name={name}_agg '
        # Whole-frame: one inference/frame, so the inference branch links STRAIGHT to
        # the aggregator with no intermediate queue — the proven lowest-latency
        # topology (an extra queue is a thread hand-off this hot path doesn't need).
        tile_q = ''
    else:
        # A letterbox postprocess inside a tiled wrapper double-remaps every bbox
        # (see NON_LETTERBOX_POST_FUNCTION) — refuse to build such a pipeline. This
        # is construction time, before anything is armed, so failing fast is safe.
        if re.search(r'function-name=\S*_letterbox\b', inner_pipeline):
            raise ValueError(
                'tiled INFERENCE_PIPELINE_WRAPPER got a letterbox postprocess '
                '(function-name=*_letterbox) in its inner pipeline; detections would '
                'be remapped tile→frame twice. Use the plain variant '
                '(NON_LETTERBOX_POST_FUNCTION) for tiled branches.')
        # hailotilecropper has built-in grid tiling (no so-path); single-scale =
        # exactly tiles_x×tiles_y crops (multi-scale would emit a full pyramid).
        overlap = (f'overlap-x-axis={tiling_overlap} overlap-y-axis={tiling_overlap} '
                   if tiling_overlap else '')
        cropper = (
            f'hailotilecropper name={name}_crop tiling-mode=single-scale '
            f'tiles-along-x-axis={tiles_x} tiles-along-y-axis={tiles_y} {overlap}'
            f'internal-offset=true '
        )
        # flatten-detections=true is REQUIRED: without it the per-tile detections
        # stay nested in tile sub-ROIs and `roi.get_objects_typed(HAILO_DETECTION)`
        # in the callback finds NOTHING (silent n=0). flatten lifts them into the
        # frame ROI (in global coords). iou-threshold dedups across tile seams.
        aggregator = (f'hailotileaggregator name={name}_agg flatten-detections=true '
                      f'iou-threshold={tile_iou_threshold} ')
        # N tiles flow per frame; hold a frame's worth in a queue so the cropper isn't
        # blocked tile-by-tile waiting on the single shared hailonet. (Only the tiled
        # path gets this queue; whole-frame links straight through — see above.)
        tile_q_depth = tiles_x * tiles_y + 1
        tile_q = f'{QUEUE(max_size_buffers=tile_q_depth, leaky="no", name=f"{name}_tile_q")} ! '

    # Construct the inference wrapper pipeline string. Two-pad fork:
    #   crop.src_0 = bypass (original full frame) -> agg.sink_0
    #   crop.src_1 = crops/tiles -> inner inference -> agg.sink_1
    inference_wrapper_pipeline = (
        f'{QUEUE(name=f"{name}_input_q")} ! '
        f'{cropper}'
        f'{aggregator}'
        f'{name}_crop. ! {QUEUE(max_size_buffers=bypass_max_size_buffers, leaky="no", name=f"{name}_bypass_q")} ! {name}_agg.sink_0 '
        f'{name}_crop. ! {tile_q}{inner_pipeline} ! {name}_agg.sink_1 '
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
