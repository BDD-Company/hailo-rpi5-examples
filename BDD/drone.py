#!/usr/bin/env python

import sys
import asyncio
import nest_asyncio

from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, Attitude, VelocityNedYaw
from mavsdk import System

import logging

logger = logging.Logger(__name__)
nest_asyncio.apply()

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
        # self.telemetry = OverwriteQueue(1)
        # self.telemetry_thread = None

        asyncio.run(self.__startup_sequence(drone_connection_string))

        async def __takeoff():
            self.move_to_center()
            await asyncio.sleep(1)

            # NOTE(vnemkov): not required, just visual indicator for the pilot
            logging.debug("Just a little dance")
            self.move_relative(-90, 0)
            await asyncio.sleep(0.5)
            self.move_relative(90, 0)
            await asyncio.sleep(0.5)

        # don't wait for takeoff, unblock rest of the systems
        self.tasks.append(asyncio.create_task(__takeoff()))

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
            await self.offboard.stop()
            await self.drone.action.disarm()

        asyncio.run(__shutdown_async())

    async def __startup_sequence(self, drone_connection_string):
        logging.info("Connecting to drone...")
        await self.drone.connect(system_address = drone_connection_string)

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

        logging.info("arming")
        for i in range(100):
            try:
                await drone.action.arm()
                break
            except:
                logging.debug(f"Got exception while arming, retrying to arm (attempt {i}) after some timeout", exc_info=True)
                await asyncio.sleep(0.2)
        logging.info("armed")

        await asyncio.sleep(1) # TODO(vnemkov): maybe remove?

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

        # await asyncio.sleep(0.1) # TODO(vnemkov): maybe remove?
        logging.debug("Taking off...")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -1 * self.cruise_altitude, 0.0))
        await asyncio.sleep(5)

        self.offboard = drone.offboard

        # # NOTE(vnemkov): not required, just visual indicator for the pilot
        logging.debug("Just a little dance")
        self.move_relative(-90, 0)
        await asyncio.sleep(1)
        self.move_relative(90, 0)
        await asyncio.sleep(1)

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

    # def status(self):

    #     telemetry_collection_interval_s = self.config.get('telemetry_collection_interval_s', 1)
    #     async def telemetry_thread():
    #         while True:
    #             try:
    #                 telemetry_point = {"timestamp": datetime.datetime.now()}
    #                 # basic telemetry data
    #                 async for position in self.drone.telemetry.position():
    #                     telemetry_point.update({
    #                         "latitude": position.latitude_deg,
    #                         "longitude": position.longitude_deg,
    #                         "altitude": position.relative_altitude_m,
    #                         "absolute_altitude": position.absolute_altitude_m
    #                     })

    #                 # battery
    #                 async for battery in self.drone.telemetry.battery():
    #                     telemetry_point.update({
    #                         "battery_voltage": battery.voltage_v,
    #                         "battery_remaining": battery.remaining_percent
    #                     })
    #                     break

    #                 # GPS
    #                 async for gps_info in self.drone.telemetry.gps_info():
    #                     telemetry_point.update({
    #                         "gps_satellites": gps_info.num_satellites,
    #                         "gps_fix_type": gps_info.fix_type.name
    #                     })
    #                     break

    #                 # HEALTH
    #                 async for health in self.drone.telemetry.health():
    #                     telemetry_point.update({
    #                         "health": health
    #                     })
    #                     break

    #                 self.telemetry.put(telemetry_point)
    #                 await asyncio.sleep(telemetry_collection_interval_s)

    #             except Exception as e:
    #                 logger.exception("Error on telemetry thread", exc_info=True)

    #     # if self.telemetry_thread is None:
    #     #     self.telemetry_thread = threading.Thread(
    #     #         target=asyncio.run,
    #     #         args=(telemetry_thread(),),
    #     #         name="telemetry_thread"
    #     #     )
    #     #     self.telemetry_thread.daemon = False
    #     #     self.telemetry_thread.start()

    #     try:
    #         telemetry : dict = self.telemetry.get(block = False)
    #         return telemetry
    #     except queue.Empty as e:
    #         # no telemetry yet
    #         return None

    def move_to(self, new_pos) -> None:
        # on the drone
        new_pos.x *= -1

        async def _move_to_async():
            await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, new_pos.x))
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
        async def _move_forward():
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(speed_ms, 0, 0, 0))
            await asyncio.sleep(0.2)
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
            logger.debug('!!! Executed move_forward')
        self.__execute_move_task(_move_forward())


    def move_relative(self, dx, dy) -> None:
        async def _move_relative_async():
            await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, dx))
            await asyncio.sleep(1)
            logger.debug('!!! Executed move_relative (dx: %s, dy: %s)', dx, dy)

            # await self.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, dx))
        self.__execute_move_task(_move_relative_async())

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

