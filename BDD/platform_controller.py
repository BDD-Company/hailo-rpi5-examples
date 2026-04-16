#! /usr/bin/env python3

import asyncio
import time
from queue import Empty, Queue

from helpers import XY

from platform_mover import PlatformMover
from helpers import Detection, Detections, MoveCommand, STOP


from helpers import (
    debug_collect_call_info,
    LoggerWithPrefix
)
import logging
logger = logging.getLogger(__name__)
global_logger = logger


def platform_controlling_thread(*args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(platform_controlling_thread_async(*args, **kwargs))
    except Exception as e:
        logger.error("in drone event loop", exc_info=True, stack_info=True)
    finally:
        loop.close()


async def platform_controlling_thread_async(platform_connection_string, platform_config, detections_queue, control_config = {}, output_queue = None, signal_event_when_ready = None):
    global global_logger
    logger = global_logger

    START_TIME_MS = time.monotonic_ns() / 1000_000
    MIN_CONFIDENCE       = control_config.pop('confidence_min', 0.1)
    FADE_COEFF           = control_config.pop('target_lost_fade_per_frame', 0.5)
    FRAME_ANGLUAR_SIZE_DEG = control_config.pop('frame_angular_size_deg', XY(120, 90))
    # P-gain at centre (small error) — keeps oscillation away
    MOVE_SCALE_MIN         = control_config.pop('move_scale', 0.3)
    # P-gain at large error — enables fast catch-up; equals MOVE_SCALE_MIN if not set
    MOVE_SCALE_MAX         = control_config.pop('move_scale_max', MOVE_SCALE_MIN)
    # normalised error distance at which MOVE_SCALE_MAX is fully reached (linear ramp)
    MOVE_SCALE_RAMP_DIST   = control_config.pop('move_scale_ramp_dist', 0.3)
    # D-gain: multiplier for error derivative (e[n] - e[n-1]); dampens oscillation
    # positive value reduces command when approaching target, resists overshoot
    D_COEFF                = control_config.pop('d_coeff', 0.0)
    # sign of each axis: set -1 to flip that axis direction
    # x: +1 = pan right when target is right; y: +1 = tilt down when target is below
    AXIS_SIGNS             = control_config.pop('axis_signs', XY(1.0, 1.0))
    # normalised distance from center [0..0.5] below which we stop sending move commands
    DEAD_ZONE              = control_config.pop('dead_zone_normalized', 0.02)

    for _key in list(control_config):
        control_config.pop(_key)

    center = XY(0.5, 0.5)
    seen_target = False

    SPEED_ADJUSTMENTS = platform_config.get('speed_adjustments', XY(1.0, 1.0))

    platform = PlatformMover(destination=platform_connection_string, **platform_config)
    # Disable hardware position polling: current_pos() reads are unreliable and block the loop.
    platform.adjustable_speed = False
    logger.info("homing to 0,0…")
    platform.move_to_center()

    # Wait until the platform physically arrives at 0,0 (or timeout).
    _HOME_TOLERANCE_DEG = 5.0   # degrees — "close enough"
    _HOME_TIMEOUT_S     = 15.0
    _t0 = time.monotonic()
    _home = XY(0, 0)
    while True:
        await asyncio.sleep(0.15)
        cur = platform.current_pos()
        dist = cur.distance_to(_home)
        logger.debug("homing… pos=%s  dist=%.1f°", cur, dist)
        if dist <= _HOME_TOLERANCE_DEG:
            logger.info("homed at %s (dist=%.1f°)", cur, dist)
            break
        if time.monotonic() - _t0 > _HOME_TIMEOUT_S:
            logger.warning("homing timeout %.1fs, pos=%s — continuing anyway", _HOME_TIMEOUT_S, cur)
            break

    # current_pos() returns cached/stale value DURING movement.
    # Capture it now (at rest) so move_relative compensation works:
    #   move_relative(dx, dy) → new_pos = current_pos() + dx,dy
    # Passing (desired_pos - stale_pos) gives new_pos = desired_pos.
    stale_pos = platform.current_pos()        
    logger.info("platform ready — stale_pos=%s config=%s", stale_pos, platform_config)

    if signal_event_when_ready:
        signal_event_when_ready.set()

    platform = debug_collect_call_info(platform, history_max_size=3)

    skipped_detetions = 0
    prev_target_relative_pos = None 
    prev_angle_to_target = XY(0, 0)

    while True:
        try:
            detections_obj = Detections(-1)
            angle_to_target = XY()

            logger = global_logger
            try:
                # Keep the asyncio loop responsive while waiting for a queue item.
                r : Detections = detections_queue.get(0.01)
                if r is STOP:
                    logger.info("stopping")
                    break
                detections_obj = r

            except Empty:
                # No detections, not even frame with ID and image
                skipped_detetions += 1
                logger.warning("No frames (%d times), no detections, input queue empty? LAST ACTION: %s", skipped_detetions, platform.last_command())
                continue
            except:
                logger.exception("Serious error getting next detection from a queue", exc_info=True)
                break

            logger = LoggerWithPrefix(logger, prefix=f'frame=#{detections_obj.frame_id:04}')

            logger.debug("GOT DETECTIONS: %d objects, detect delay: %.1fms, total delay: %.1fms",
                len(detections_obj.detections),
                (detections_obj.meta.detection_end_timestamp_ns - detections_obj.meta.detection_start_timestamp_ns) / 1e6,
                (detections_obj.meta.detection_end_timestamp_ns - detections_obj.meta.capture_timestamp_ns) / 1e6,
            )
            skipped_detetions = 0
            debug_info = {'start_time_ms': START_TIME_MS}

            detections, frame = detections_obj.detections, detections_obj.frame

            detections = [d for d in detections if d is not None]
            detections.sort(reverse=True, key=lambda d: d.confidence)
            target_relative_pos = None

            detection = detections[0] if detections else Detection()
            if detection.confidence >= MIN_CONFIDENCE:
                seen_target = True
                # offset from frame centre to target, normalised [−0.5, 0.5]
                # positive x → target is to the right  → pan right
                # positive y → target is below center  → tilt down
                target_relative_pos = detection.bbox.center - center
                # D-term: derivative of error (change since last frame)
                # reduces command when approaching target, resists overshoot.
                # Clamped so D-contribution never exceeds the P-contribution
                # (prevents D from dominating and causing reverse oscillation near zero).
                d_error = (target_relative_pos - prev_target_relative_pos
                           if prev_target_relative_pos is not None else XY())
                prev_target_relative_pos = target_relative_pos.clone()
                d_contrib_x = d_error.x * D_COEFF
                d_contrib_y = d_error.y * D_COEFF
                # clamp: |D| ≤ |P| so D can damp but not reverse the direction
                if target_relative_pos.x != 0:
                    d_contrib_x = max(-abs(target_relative_pos.x), min(abs(target_relative_pos.x), d_contrib_x))
                if target_relative_pos.y != 0:
                    d_contrib_y = max(-abs(target_relative_pos.y), min(abs(target_relative_pos.y), d_contrib_y))
                pd_error = XY(target_relative_pos.x + d_contrib_x,
                              target_relative_pos.y + d_contrib_y)
                # dynamic P-gain: scale linearly from MOVE_SCALE_MIN (near centre)
                # to MOVE_SCALE_MAX (when error >= MOVE_SCALE_RAMP_DIST)
                error_dist = max(abs(target_relative_pos.x), abs(target_relative_pos.y))
                t = min(error_dist / MOVE_SCALE_RAMP_DIST, 1.0) if MOVE_SCALE_RAMP_DIST > 0 else 1.0
                move_scale = MOVE_SCALE_MIN + (MOVE_SCALE_MAX - MOVE_SCALE_MIN) * t
                # convert PD-corrected offset → degrees
                angle_to_target = pd_error.multiplied_by_XY(FRAME_ANGLUAR_SIZE_DEG) * move_scale
                prev_angle_to_target = angle_to_target

                in_dead_zone = (abs(target_relative_pos.x) < DEAD_ZONE and
                                abs(target_relative_pos.y) < DEAD_ZONE)
                if not in_dead_zone:
                    # PD-controller: each frame move by PD-corrected angular error.
                    # move_relative does:  new_pos = current_pos() + delta * speed_adjustments
                    delta = angle_to_target.multiplied_by_XY(AXIS_SIGNS)
                    logger.debug("target offset: %s  d_err: %s  angle: %s  delta: %s  scale: %.3f",
                                 target_relative_pos, d_error, angle_to_target, delta, move_scale)
                    platform.move_relative(delta.x, delta.y)
                    debug_info["mode"] = f"follow  size={detection.bbox.area():.3f}"
                else:
                    logger.debug("target offset: %s  IN DEAD ZONE (< %.3f), not moving",
                                 target_relative_pos, DEAD_ZONE)
                    debug_info["mode"] = f"follow(locked)  size={detection.bbox.area():.3f}"
            else:
                prev_target_relative_pos = None  # reset D-term when target lost
                if seen_target:
                    prev_angle_to_target *= FADE_COEFF
                    delta = prev_angle_to_target.multiplied_by_XY(AXIS_SIGNS)
                    platform.move_relative(delta.x, delta.y)
                    target_relative_pos = prev_angle_to_target.divided_by_XY(FRAME_ANGLUAR_SIZE_DEG)
                    debug_info["mode"] = "hover"
                else:
                    debug_info["mode"] = "idle"

            last_command = platform.last_command() or '<<== NO ==>>'
            debug_info["action"] = last_command
            logger.info("MODE: %s  ACTION: %s", debug_info.get('mode', ''), last_command)

            if output_queue is not None:
                output_queue.put({
                    'detections': detections_obj,
                    'selected': detection,
                    'telemetry': debug_info,
                    'selected_detection_projected_pos': target_relative_pos,
                    'move_goal': target_relative_pos,
                })

        except:
            logging.exception("Got exception: %s", detections_obj, exc_info=True)
