#!/usr/bin/env python

import time
from pymavlink import mavutil

# Listen on the companion computer.
# Change port if your PX4 MAVLink instance uses a different one.
MAVLINK_UDP_IN = "udpin:0.0.0.0:14550"

RC_HZ = 20.0
SWITCH_CH = 7   # example: AUX1 on CH7


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
        print(f"COMMAND_ACK: command={ack.command} result={ack.result}")
    else:
        print("No COMMAND_ACK received")


def classify_3pos(pwm: int) -> str:
    if pwm == 0 or pwm == 65535:
        return "UNKNOWN"
    if pwm < 1300:
        return "LOW"
    if pwm > 1700:
        return "HIGH"
    return "MID"


def main():
    master = mavutil.mavlink_connection(MAVLINK_UDP_IN)
    master.wait_heartbeat()
    print(f"Connected to sys={master.target_system} comp={master.target_component}")

    request_message_interval(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
        RC_HZ,
    )

    last_state = None
    last_print = 0.0

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

        pwm = channels[SWITCH_CH - 1]
        state = classify_3pos(pwm)

        if state != last_state:
            print(f"CH{SWITCH_CH}: pwm={pwm} state={state}")
            last_state = state

            if state == "HIGH":
                print(">>> operator requested takeover / abort")
                # Example:
                # - stop sending MAVSDK manual_control inputs
                # - set a shared asyncio/threading flag
                # - switch your app state machine

            elif state == "MID":
                print(">>> operator requested pause / hold")

            elif state == "LOW":
                print(">>> operator requested script enable")

        now = time.time()
        if now - last_print > 1.0:
            print(
                f"ch1={channels[0]} ch2={channels[1]} ch3={channels[2]} ch4={channels[3]} "
                f"ch5={channels[4]} ch6={channels[5]} ch7={channels[6]} ch8={channels[7]}"
            )
            last_print = now


if __name__ == "__main__":
    main()