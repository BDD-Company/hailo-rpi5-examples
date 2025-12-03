#!/usr/bin/env python

import sys
import asyncio
# import datetime
# import threading
import dataclasses
from enum import Enum

from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, Attitude, VelocityNedYaw, AttitudeRate
from mavsdk import System

import logging

from mavsdk.telemetry import Telemetry, EulerAngle

logger = logging.Logger("BDD_drone")
# nest_asyncio.apply()

async def _one(gen):
    async for item in gen:
        return item

def mavsdk_msg_to_dict(msg):
    if dataclasses.is_dataclass(msg):
        d = dataclasses.asdict(msg)
        return {key : mavsdk_msg_to_dict(val) for key, val in d.items()}

    # do not unroll "primitive" types and enums
    if type(msg).__module__ == "builtins" or isinstance(msg, Enum):
        if isinstance(msg, Enum):
            return f"{msg.value} ({msg.name})"

        return msg

    d = {}
    for name in dir(msg):
        if name.startswith("_"):
            continue
        v = getattr(msg, name)
        if callable(v):
            continue

        if dataclasses.is_dataclass(v):
            v = mavsdk_msg_to_dict(v)
        else:
            v = mavsdk_msg_to_dict(v)

        d[name] = v

    # nested list-like objects
    if len(d) == 0 and len(msg) != 0:
        d = tuple(i for i in msg)

    return d


DEFAULT_TAKEOFF_ALTITUDE_M = 10

class DroneMover():

    def __init__(self, drone_connection_string, config : dict|None = None) -> None:
        super().__init__()
        self.drone = System()
        self.offboard = None
        self.config = {} if config is None else config
        self.initial_pos = None
        self.initial_yaw = None
        self.cruise_altitude = self.config.get('cruise_altitude', DEFAULT_TAKEOFF_ALTITUDE_M)
        self.tasks : list[asyncio.Task] = []
        self.drone_connection_string = drone_connection_string

        # self.telemetry_data = {}
        # self.telemetry_thread = None

        # asyncio.run(self.__startup_sequence(drone_connection_string))

        # Just to kick off telemetry collection
        # self.status()


    def __del__(self):
        async def __await_tasks():
            for t in self.tasks:
                await asyncio.wait_for(t, timeout=0.001)

        asyncio.run(__await_tasks())

        if not self.drone:
            return

        async def __shutdown_async():
            # TODO: maybe try to land first?
            if self.offboard:
                await self.offboard.stop()
            await self.drone.action.disarm()

        asyncio.run(__shutdown_async())


    async def startup_sequence(self):
        logging.info("Connecting to drone... %s", self.drone_connection_string)
        await self.drone.connect(system_address = self.drone_connection_string)

        drone = self.drone

        logging.info("Connecting to PX4...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                logging.debug("connected!")
                break

        async for health in drone.telemetry.health():
            logging.debug("Health report: %s", health.__str__())
            if health.is_armable:
                logging.debug("seems armable")
            break

        async def arm():
            logging.info("arming")
            arming_exception = None
            for i in range(100):
                try:
                    await drone.action.arm()
                    arming_exception = None
                    break
                except Exception as e:
                    arming_exception = e
                    logging.warning(f"retrying to arm, attempt {i}")
                    await asyncio.sleep(0.5)

            if arming_exception is not None:
                logging.warning(f"Failed to arm, latest exception", exc_info=arming_exception)
                raise Exception('failed to arm')

        await arm()

        logging.info("armed")

        await asyncio.sleep(1) # TODO(vnemkov): maybe remove?

        logger.debug("getting initial telemetry...")
        # await drone.telemetry.set_rate_heading(50.0)
        logger.debug("initial telemetry: %s", await self.get_telemetry_async())

        # Важно: перед включением Offboard нужно отправить хотя бы одну команду
        # NED: Z вниз → 0 = текущая высота
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

        logger.debug("Entering Offboard mode...")
        try:
            await drone.offboard.start()
        except:
            logger.debug("Failed to enter Offboard mode, aborting", exc_info=True)
            await drone.action.disarm()
            return

        await asyncio.sleep(1) # TODO(vnemkov): maybe remove?
        logging.debug("Taking off...")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -1 * self.cruise_altitude, 0.0))
        await asyncio.sleep(self.cruise_altitude / 2) # 2m/s climb rate approx

        self.offboard = drone.offboard
        logging.debug("took off")

        # # NOTE(vnemkov): not required, just visual indicator for the pilot
        # THRUST_VALUE = 0.1
        # logging.debug("Little silly dance with thrust value: %s", THRUST_VALUE)
        # await self.drone.offboard.set_attitude_rate(AttitudeRate(0, 0, yaw_deg_s=90, thrust_value=THRUST_VALUE))
        # await asyncio.sleep(0.5)
        # await self.drone.offboard.set_attitude_rate(AttitudeRate(0, 0, yaw_deg_s=-90, thrust_value=THRUST_VALUE))
        # await asyncio.sleep(1)
        # await self.drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, thrust_value=THRUST_VALUE))
        # await asyncio.sleep(0.5)
        # await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -1 * self.cruise_altitude, 0.0))

        # # NOTE(vnemkov): not required, just visual indicator for the pilot
        # logging.debug("Just a little dance")
        # self.move_relative(-90, 0)
        # await asyncio.sleep(1)
        # self.move_relative(90, 0)
        # await asyncio.sleep(1)

        return


    # def _get_position_and_yaw(self)-> tuple[DronePosition, float]:
    #     async def _get_position_and_yaw_async(self):
    #         position = self.drone_controller.get_position()
    #         yaw = self.drone_controller.get_yaw()
    #         return DronePosition.from_drone_position_tuple(await position), await yaw

    #     return asyncio.run(_get_position_and_yaw_async(self))

    # def current_pos(self) -> XY:
    #     return XY()

        # if self.initial_pos is None or self.initial_yaw is None:
        #     self.initial_pos, self.initial_yaw = self._get_position_and_yaw()
        #     ic(self.initial_pos, self.initial_yaw)

        # position, yaw = self._get_position_and_yaw()
        # altitude_diff = self.initial_pos.altitude_m - position.altitude_m
        # yaw_diff = self.initial_yaw - yaw

        # return ic(XY(x = yaw_diff, y = altitude_diff))


    async def get_current_attitude_async(self) -> dict[str, float]:
        """
        MUCH (!!!) faster than getting all telemetry data at once
        """
        val : EulerAngle = await _one(self.drone.telemetry.attitude_euler())
        return mavsdk_msg_to_dict(val)


    async def get_telemetry_async(self) -> dict:
        # print("!!! TELEMETRY", file = sys.stderr, flush = True)
        logger.debug("!!! telemetry")

        telemetry_items = [
            # "position", # -- hangs
            # "battery", # pretty useless
            # "heading", # -- hangs
            # "gps_info", # -- hangs
            "odometry",
            # "attitude_angular_velocity_body", # present in odometry
            # "health", # useless after start
            # "imu", # using "attitude_euler" instead
            # "raw_imu", # -- hangs
            # "scaled_imu", # -- hangs
            "scaled_pressure",
            "attitude_euler",
            # "position_velocity_ned",
            "flight_mode"
            # "rc_status",
        ]
        # tasks = {
        #     "position": asyncio.create_task(_one(self.drone.telemetry.position())),
        #     "battery": asyncio.create_task(_one(self.drone.telemetry.battery())),
        #     "gps_info": asyncio.create_task(_one(self.drone.telemetry.gps_info())),
        #     "health": asyncio.create_task(_one(self.drone.telemetry.health())),
        #     "attitude_angular_velocity_body": asyncio.create_task(
        #         _one(self.drone.telemetry.attitude_angular_velocity_body())
        #     ),
        # }
        tasks = {
            item : asyncio.create_task(_one(getattr(self.drone.telemetry, item)())) for item in telemetry_items
        }
        logger.debug("!!! tasks")

        snapshot = {}
        # task_start_time = datetime.datetime.now().ctime()
        for key, task in tasks.items():
            # print("!!! TELEMETRY task", key, file = sys.stderr, flush = True)
            msg = await task

            # with open(key + task_start_time + '.pickle', 'wb') as f:
            #     pickle.dump(msg, f)

            snapshot[key] = mavsdk_msg_to_dict(msg)

        return snapshot


    def move_to(self, new_pos) -> None:
        print("move_to")
        new_pos.x *= -1

        async def _move_to_async():
            await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, new_pos.x))
            logger.debug('!!! Executed move_to (new_pos: %s)', new_pos)
            await asyncio.sleep(0.1)

        self.__execute_move_task(_move_to_async())


    def move_to_center(self) -> None:
        # this should move drone to initial pos
        async def _move_to_center_async():
            await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, 0))
            # self.drone_controller.goto_position(self.initial_pos.latitude_deg, self.initial_pos.longitude_deg, self.initial_pos.altitude_m, self.initial_yaw)
            await asyncio.sleep(0.1)
            logger.debug('!!! Executed move_to_center')

        self.__execute_move_task(_move_to_center_async())


    def move_forward(self, speed_ms : float = 1.0) -> None:
        print("move_forward")
        async def _move_forward():
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(speed_ms, 0, 0, 0))
            await asyncio.sleep(0.2)
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
            logger.debug('!!! Executed move_forward')
        self.__execute_move_task(_move_forward())


    def move_relative(self, dx, dy) -> None:
        print("move_relative")
        async def _move_relative_async():
            logger.debug("_move_relative_async")
            await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, dx))
            # await asyncio.sleep(0.1)
            logger.debug('!!! Executed move_relative (dx: %s, dy: %s)', dx, dy)

            # await self.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, dx))
        self.__execute_move_task(_move_relative_async())


    async def move_to_async(self, new_pos) -> None:
        # on the drone
        new_pos.x *= -1

        await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, new_pos.x))
        logger.debug('!!! Executed move_to (new_pos: %s)', new_pos)
        # await asyncio.sleep(0.1)


    async def move_to_center_async(self) -> None:
        # this should move drone to initial pos

        await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, 0))
        # self.drone_controller.goto_position(self.initial_pos.latitude_deg, self.initial_pos.longitude_deg, self.initial_pos.altitude_m, self.initial_yaw)
        await asyncio.sleep(0.1)
        logger.debug('!!! Executed move_to_center')


    async def move_to_target_async(self, yaw_degree : float, pitch_degree : float, thrust : float = 0.2) -> None:
        print("move_forward_async")
        try:
            # -1 : here ducking nose DOWN to be able to move to target which is UP (due to how quadcopters move)
            # 0.5 : doing it sligthly so target doesn't disappear from frame
            pitch_degree *= -1
            await self.drone.offboard.set_attitude(Attitude(0, pitch_deg=pitch_degree, yaw_deg=yaw_degree, thrust_value=thrust))
        except:
            logger.exception("While moving to target with pitch: %s, yaw: %s, thrust: %s",
                pitch_degree,
                yaw_degree,
                thrust)

        # await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(speed_ms, 0, 0, 0))
        # await asyncio.sleep(0.5)
        # await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
        logger.debug('!!! Executed move_forward')


    async def move_relative_async(self, dx, dy) -> None:
        print("move_relative_async")
        await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, dx))
        await asyncio.sleep(1)
        logger.debug('!!! Executed move_relative (dx: %s, dy: %s)', dx, dy)


    def __execute_move_task(self, task):
        #TODO:
        """ Maintain a move task queue, with only 1 active item,
        so whenever a new item is added, queue is cleaned out.
        Items are pulled from queue and executed one by one.
        """
        asyncio.run(task)


async def main():
    from helpers import configure_logging, XY
    configure_logging(level = logging.NOTSET)

    logger.debug("starting up drone...")

    drone = DroneMover('udp://:14550')
    drone.move_to_center()
    drone.move_forward(10)
    drone.move_to(XY(10, 10))

    logger.debug("drone started")


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    nest_asyncio.apply()
    loop.run_until_complete(main())

