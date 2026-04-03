#!/usr/bin/env python3

import logging
import time
import math

from helpers import XY

from platform_mover import *


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

    DESTINATION="/dev/ttyUSB0"

    def move_test_headers(title):
        line = "=" * 60
        logging.debug("\n\n%s\n    Starting test group:\n    %s\n%s\n", line, title, line)


    def sleep(seconds):
        #logging.debug('sleep %s', seconds)
        time.sleep(seconds)


    def sleep_for_iterations(iterations):
        # enough sleep for platform to perform movement
        # sleep formula devised empirically
        sleep_miliseconds = max(1, math.sqrt(1 * iterations / 3) / 2)
        sleep(sleep_miliseconds)

    # convert input to move commands 1 to 1
    speed_adjustments = XY(1, -1)
    mover = PlatformMover(
        destination=DESTINATION,
        speed_adjustments=speed_adjustments,
        minimal_move = XY(1, 1),
        speed=0,
        acceleration=1)

    mover.move_to_center()


    for i in range(10):
        mover.move_relative(-0.3, -0.3)
        time.sleep(1)


    def test_move_RLDU():
        move_test_headers("Move RIGHT, LEFT, DOWN, UP with different speeds")

        MINIMAL_MOVE = XY(1, 1)

        logging.debug("speed adjustments: %s", speed_adjustments)

        mover = PlatformMover(
            destination=DESTINATION,
            speed_adjustments=speed_adjustments,
            # minimal_move = XY(1, 1),
            speed=0,
            acceleration=32)

        # 11 to have total move > MINIMAL_MOVE on minimal delta
        for iterations in [1, 11, 20]:
            logging.debug('\n\n %s', '!!!' * 20)
            logging.debug("Iterations: %s", iterations)

            for delta in [0.1, 0.5, 1, 2]:
                if delta * iterations < MINIMAL_MOVE.x or delta * iterations < MINIMAL_MOVE.y:
                    continue

                logging.debug('\n %s', '!!!' * 20)
                logging.debug("Move delta: %s", delta)
                logging.debug("total expected move: %s", delta * iterations)

                mover.move_to_center()

                # expected_final_pos = speed_adjustments * delta * iterations

                logging.debug("RIGHT")
                for _ in range(0, iterations):
                    mover.move_relative(delta, 0)
                # assert math.isclose(mover._current_pos.x, expected_final_pos.x)
                sleep_for_iterations(iterations * delta)

                mover.move_to_center()
                logging.debug("LEFT")
                for _ in range(0, iterations):
                    mover.move_relative(-delta, 0)
                # assert math.isclose(mover._current_pos.x, -expected_final_pos.x)
                sleep_for_iterations(iterations * delta)

                mover.move_to_center()
                logging.debug("DOWN")
                for _ in range(0, iterations):
                    mover.move_relative(0, delta)
                # assert math.isclose(mover._current_pos.y, expected_final_pos.y)
                sleep_for_iterations(iterations * delta)

                mover.move_to_center()
                logging.debug("UP")
                for _ in range(0, iterations):
                    mover.move_relative(0, -delta)
                # assert math.isclose(mover._current_pos.y, -expected_final_pos.y)
                sleep_for_iterations(iterations * delta)

                mover.move_to_center()
                # enough time to allow to get back to neutral position
                sleep_for_iterations(iterations * delta * 2)

    test_move_RLDU()

    def test_move_spiral():
        move_test_headers("Move in spiral with different steps count")

        MINIMAL_MOVE = XY(1, 1)

        # convert input to move commands 1 to 1
        speed_adjustments = XY(1, -1)
        logging.debug("speed adjustments: %s", speed_adjustments)

        mover = PlatformMover(
            destination=DESTINATION,
            speed_adjustments=speed_adjustments,
            # minimal_move = XY(0.1, 0.1),
            speed=0,
            acceleration=32)

        MIN_RADIUS = 5
        MAX_RADIUS = 10
        SPIRAL_LOOPS = 2

        for steps in [60,]:
            # mover.move_to_center()
            mover.current_pos()
            sleep(1)

            prev_x = 0
            prev_y = 0
            logging.debug('%s coils of spiral in %s steps', SPIRAL_LOOPS, steps)

            angle_step = math.pi * SPIRAL_LOOPS * 2 / steps
            angle = 0
            radius_step = (MAX_RADIUS - MIN_RADIUS) / steps
            radius = MIN_RADIUS

            end = math.pi * 2 * SPIRAL_LOOPS / steps + 1
            for _ in range(steps * SPIRAL_LOOPS):
                angle += angle_step
                x = math.cos(angle) * radius
                y = math.sin(angle) * radius
                dx = x - prev_x
                dy = y - prev_y

                prev_x = x
                prev_y = y
                radius += radius_step


                mover.move_relative(dx, dy)
                mover.current_pos()
                sleep(0.05)
                mover.current_pos()
                #sleep(2 / steps * math.sqrt(steps) / 10)

            # mover.move_to_center()

    # test_move_RLDU()

    test_move_spiral()


if __name__ == '__main__':
    main()
