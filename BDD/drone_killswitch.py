#!/usr/bin/env python

import time
from pymavlink import mavutil

import logging

logger = logging.getLogger(__name__)


RC_HZ = 20.0
# SWITCH_CH = 7   # example: AUX1 on CH7


def request_message_interval(master, message_id: int, hz: float) -> None:
    interval_us = int(1_000_000 / hz)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,     # param1: MAVLink message ID
        interval_us,    # param2: interval in microseconds
        0, 0, 0, 0, 0
    )

    # Wait for ACK so you know PX4 accepted/rejected it
    ack = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=2.0)
    if ack:
        logger.debug(f"COMMAND_ACK: command={ack.command} result={ack.result}")
    else:
        logger.debug("No COMMAND_ACK received")


def classify_3pos(pwm: int) -> str:
    if pwm == 0 or pwm == 65535:
        return "UNKNOWN"
    if pwm < 1300:
        return "LOW"
    if pwm > 1700:
        return "HIGH"
    return "MID"


def kill_on_rc_switch_on_channel(mavlink_udp_port : int, killswitch_channel, drone):
    try:
        __kill_on_rc_switch_on_channel(mavlink_udp_port, killswitch_channel, drone)
    except:
        logger.exception("RC killswitch", exc_info=True)


def __kill_on_rc_switch_on_channel(mavlink_udp_port : int, killswitch_channel, drone):
    master = mavutil.mavlink_connection(f"udpin:0.0.0.0:{mavlink_udp_port}")
    master.wait_heartbeat()
    logger.debug("!!! Connected to sys=%s comp=%s", master.target_system, master.target_component)

    request_message_interval(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
        RC_HZ,
    )

    last_state = None
    last_print = 0
    while True:
        msg = master.recv_match(type="RC_CHANNELS", blocking=True, timeout=1.0)
        if msg is None:
            continue

        channels = [
            msg.chan1_raw, msg.chan2_raw, msg.chan3_raw, msg.chan4_raw,
            msg.chan5_raw, msg.chan6_raw, msg.chan7_raw, msg.chan8_raw,
            msg.chan9_raw, msg.chan10_raw, msg.chan11_raw, msg.chan12_raw,
            msg.chan13_raw, msg.chan14_raw, msg.chan15_raw, msg.chan16_raw,
            msg.chan17_raw, msg.chan18_raw,
        ]

        pwm = channels[killswitch_channel - 1]
        state = classify_3pos(pwm)

        if state != last_state:
            logger.info(f"CH{killswitch_channel}: pwm={pwm} state={state}")
            last_state = state

            if state == "HIGH" or state == "MID":
                logger.warning(">>> operator requested takeover / abort")
                drone.ABORT()

            # elif state == "MID":
            #     drone.PAUSE()
            #     logger.debug(">>> operator requested pause / hold")

            elif state == "LOW":
                logger.warning(">>> operator requested script enable")

        # now = time.time()
        # if now - last_print > 1.0:
        #     logger.debug(
        #         f"ch1={channels[0]} ch2={channels[1]} ch3={channels[2]} ch4={channels[3]} "
        #         f"ch5={channels[4]} ch6={channels[5]} ch7={channels[6]} ch8={channels[7]}"
        #     )
        #     last_print = now


# if __name__ == "__main__":
#     main()