#!/usr/bin/env python

import asyncio
import nest_asyncio
import dataclasses
from enum import Enum

from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, Attitude, VelocityNedYaw, AttitudeRate
from mavsdk import System
from mavsdk.telemetry import Telemetry, EulerAngle

from helpers import MoveCommand, dotdict, XY

import logging

logger = logging.Logger("BDD_drone")

DEFAULT_TAKEOFF_ALTITUDE_M = 20


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

        # Telemetry aspects to keep cached; edit this list to add/remove streams.
        self.telemetry_aspects = [
            "attitude_euler",
            # "battery",
            "health",
            "odometry",
            # "attitude_angular_velocity_body",
        ]
        self._telemetry_tasks : dict[str, asyncio.Task] = {}
        self._telemetry_ready : dict[str, asyncio.Event] = {}
        self._telemetry_latest : dict[str, object] = {}

        # self.telemetry_data = {}
        # self.telemetry_thread = None

        # asyncio.run(self.__startup_sequence(drone_connection_string))

        # Just to kick off telemetry collection
        # self.status()


    def __del__(self):
        async def __await_tasks():
            for t in self.tasks:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=0.001)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    pass

        asyncio.run(__await_tasks())

        if not self.drone:
            return

        async def __shutdown_async():
            # TODO: maybe try to land first?
            if self.offboard:
                await self.offboard.stop()
            await self.drone.action.disarm()

        asyncio.run(__shutdown_async())


    async def startup_sequence(self, arm_attempts = 100):
        logging.info("Connecting to drone... %s", self.drone_connection_string)
        await self.drone.connect(system_address = self.drone_connection_string)
        arm_attempts = max(3, arm_attempts)

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
            for i in range(arm_attempts):
                try:
                    await drone.action.arm()
                    arming_exception = None
                    break
                except Exception as e:
                    arming_exception = e
                    logging.warning(f"retrying to arm, attempt {i}")
                    await asyncio.sleep(0.1)

            if arming_exception is not None:
                logging.warning(f"Failed to arm, latest exception", exc_info=arming_exception)

        await arm()

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
        logging.debug("Taking off to %sm...", self.cruise_altitude)
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -1 * self.cruise_altitude, 0.0))
        await asyncio.sleep(5) #self.cruise_altitude / 2) # 2m/s climb rate approx

        self.offboard = drone.offboard
        logging.debug("took off")

        await self._ensure_telemetry_cache()

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


    async def _ensure_telemetry_cache(self, aspects: list[str] | None = None):
        """
        Start background consumers for each telemetry aspect so the latest sample
        is always available without awaiting a fresh gRPC call.
        """
        aspects = self.telemetry_aspects if aspects is None else aspects

        for aspect in aspects:
            if aspect in self._telemetry_tasks:
                continue

            self._telemetry_ready[aspect] = asyncio.Event()

            async def _consume(aspect_name: str):
                async for sample in getattr(self.drone.telemetry, aspect_name)():
                    self._telemetry_latest[aspect_name] = sample
                    self._telemetry_ready[aspect_name].set()

            task = asyncio.create_task(_consume(aspect))
            self._telemetry_tasks[aspect] = task
            self.tasks.append(task)


    async def get_cached_telemetry(self, aspect: str, wait_for_first: bool = True):
        """
        Return the last received telemetry sample for `aspect`; optionally wait
        for the first sample. Add/remove aspects in `self.telemetry_aspects`.
        """
        await self._ensure_telemetry_cache([aspect])

        if wait_for_first and aspect in self._telemetry_ready:
            await self._telemetry_ready[aspect].wait()

        return self._telemetry_latest.get(aspect)


    async def get_cached_attitude(self, wait_for_first: bool = True) -> EulerAngle | None:
        return await self.get_cached_telemetry("attitude_euler", wait_for_first=wait_for_first)


    async def get_telemetry_dict(self, wait = False) -> dict:
        result = {}
        for aspect in self.telemetry_aspects:
            result[aspect] = mavsdk_msg_to_dict(await self.get_cached_telemetry(aspect, wait))

        return dotdict(result)


    def get_telemetry_dict_sync(self, wait = False) -> dict:
        return asyncio.run(self.get_telemetry_dict(wait))


    # def move_to(self, new_pos) -> None:
    #     print("move_to")
    #     new_pos.x *= -1

    #     async def _move_to_async():
    #         await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, new_pos.x))
    #         logger.debug('!!! Executed move_to (new_pos: %s)', new_pos)
    #         await asyncio.sleep(0.1)

    #     self.__execute_move_task(_move_to_async())

    # def move_forward(self, speed_ms : float = 1.0) -> None:
    #     print("move_forward")
    #     async def _move_forward():
    #         await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(speed_ms, 0, 0, 0))
    #         await asyncio.sleep(0.2)
    #         await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    #         logger.debug('!!! Executed move_forward')
    #     self.__execute_move_task(_move_forward())


    # def move_relative(self, dx, dy) -> None:
    #     print("move_relative")
    #     async def _move_relative_async():
    #         logger.debug("_move_relative_async")
    #         await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, dx))
    #         # await asyncio.sleep(0.1)
    #         logger.debug('!!! Executed move_relative (dx: %s, dy: %s)', dx, dy)

    #         # await self.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, dx))
    #     self.__execute_move_task(_move_relative_async())


    # async def move_to_async(self, new_pos) -> None:
    #     # on the drone
    #     new_pos.x *= -1

    #     await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, new_pos.x))
    #     logger.debug('!!! Executed move_to (new_pos: %s)', new_pos)
    #     # await asyncio.sleep(0.1)


    # async def move_to_center_async(self) -> None:
    #     # this should move drone to initial pos

    #     await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, 0))
    #     # self.drone_controller.goto_position(self.initial_pos.latitude_deg, self.initial_pos.longitude_deg, self.initial_pos.altitude_m, self.initial_yaw)
    #     await asyncio.sleep(0.1)
    #     logger.debug('!!! Executed move_to_center')


    async def track_target(self, x : float, y : float, forward_speed_m_s : float = 0.0) -> None:
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(forward_m_s=forward_speed_m_s, right_m_s=0, down_m_s=y / 20, yawspeed_deg_s=x))
        await asyncio.sleep(0.1)


    # async def standstill(self):
    #     await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))


    async def execute_move_command(self, move_command : MoveCommand) -> None:
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(forward_m_s=move_command.move_speed_ms, right_m_s=0, down_m_s=0, yawspeed_deg_s=move_command.adjust_attitude.x))


    async def move_xy(self, xy : XY, yaw = 0) -> None:
        await self.drone.offboard.set_position_ned(PositionNedYaw(xy.x, xy.y, -1 * self.cruise_altitude, yaw))
        # await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(forward_m_s=move_command.move_speed_ms, right_m_s=0, down_m_s=0, yawspeed_deg_s=move_command.adjust_attitude.x))


    # async def move_to_target_async(self, yaw_m_s : float, pitch_degree : float, forward_speed_m_s : float = 0.0) -> None:
    #     print("move_forward_async")
    #     try:
    #         # -1 : here ducking nose DOWN to be able to move to target which is UP (due to how quadcopters move)
    #         # 0.5 : doing it sligthly so target doesn't disappear from frame
    #         # pitch_degree *= -1
    #         # await self.drone.offboard.set_attitude(Attitude(0, pitch_deg=pitch_degree, yaw_deg=yaw_degree, thrust_value=thrust))
    #         await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(forward_m_s=forward_speed, 0, 0, yawspeed_deg_s=yaw_degree))

    #     except:
    #         logger.exception("While moving to target with pitch: %s, yaw: %s, thrust: %s",
    #             pitch_degree,
    #             yaw_degree,
    #             thrust)

        # # await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(speed_ms, 0, 0, 0))
        # # await asyncio.sleep(0.5)
        # # await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
        # logger.debug('!!! Executed move_forward')


    async def move_relative_async(self, dx, dy) -> None:
        print("move_relative_async")
        await self.drone.offboard.set_position_ned(PositionNedYaw(0, 0, -1 * self.cruise_altitude, dx))
        await asyncio.sleep(0.1)
        logger.debug('!!! Executed move_relative (dx: %s, dy: %s)', dx, dy)


    async def standstill(self) -> None:
        print("move_relative_async")
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))


async def main():
    from helpers import configure_logging, XY
    configure_logging(level = logging.NOTSET)

    logger.debug("starting up drone...")

    drone = DroneMover('udp://:14550')
    drone.move_to_center()
    await drone.move_forward_async(10)
    drone.move_to(XY(10, 10))

    logger.debug("drone started")


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    nest_asyncio.apply()
    loop.run_until_complete(main())
