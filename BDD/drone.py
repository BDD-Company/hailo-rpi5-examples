#!/usr/bin/env python

import asyncio
# import nest_asyncio
import dataclasses
from enum import Enum
import time

from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, Attitude, VelocityNedYaw, AttitudeRate
from mavsdk import System
from mavsdk.telemetry import Telemetry, EulerAngle, LandedState

from helpers import MoveCommand, dotdict, XY

import logging

logger = logging.getLogger("BDD_drone")

DEFAULT_TAKEOFF_ALTITUDE_M = 10
SAFE_TILT_DEG = 90
IDLE_THRUST = 0.1

def is_in_air(state : LandedState):
    return state == LandedState.IN_AIR


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
            "landed_state"
            # "attitude_angular_velocity_body",
        ]
        self._telemetry_tasks : dict[str, asyncio.Task] = {}
        self._telemetry_ready : dict[str, asyncio.Event] = {}
        self._telemetry_latest : dict[str, object] = {}
        self.__sticks = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "r": 0.0,
            "last_set_time": time.monotonic_ns(),
        }

    @staticmethod
    def _log_background_task_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Background task '%s' failed", task.get_name())


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
            # if self.offboard:
            #     await self.offboard.stop()
            await self.drone.action.disarm()

        asyncio.run(__shutdown_async())

    async def __manual_input_loop(self) -> None:
        """
        Sends manual control input continuously.
        MAVSDK docs recommend at least 10 Hz.
        x: pitch   (-1..1)
        y: roll    (-1..1)
        z: thrust  (0..1)
        r: yaw     (-1..1)
        """
        try:
            logger.info("!!! Started manual input loop")
            while True:
                sticks = self.__sticks
                logger.debug("!!! manual input: %s", sticks)

                now_ns = time.monotonic_ns()
                since_last_command_ms = (now_ns - sticks.get('last_set_time', 0)) / 1_000_000
                target_x = sticks["x"]
                target_y = sticks["y"]
                target_z = sticks["z"]
                target_r = sticks.get("r", 0)

                # OLD commands must eventually fade to 0
                reduce_factor = since_last_command_ms / 100
                # if reduce_factor > 1:
                #     target_x /= reduce_factor
                #     target_y /= reduce_factor
                #     target_z /= reduce_factor
                #     target_r /= reduce_factor

                logger.debug("set_manual_control_input: %s, since_last_command_ms: %s, reduced by factor: %s",
                        (target_x, target_y, target_z, target_r),
                        since_last_command_ms,
                        reduce_factor)

                await self.drone.manual_control.set_manual_control_input(
                    target_x, target_y, target_z, target_r
                )
                await asyncio.sleep(0.1)  # 10 Hz
        except:
            logger.exception("manual inpt loop", exc_info=True, stack_info=True)


    async def startup_sequence(self, arm_attempts = 100, force_arm=False):
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

        await self.move_to_xy(XY(0, 0), 0)
        manual_input_task = asyncio.create_task(self.__manual_input_loop(), name="manual_input_loop")
        manual_input_task.add_done_callback(self._log_background_task_result)
        self.tasks.append(manual_input_task)
        await asyncio.sleep(0.5)
        await self.move_to_xy(XY(0, 0), 0)

        # await drone.manual_control.start_altitude_control()

        async def arm():
            logging.info("arming")

            # Enable arming without GPS
            await drone.param.set_param_int("COM_ARM_WO_GPS", 1)
            await drone.param.set_param_int("COM_RC_IN_MODE", 1) # 1 == drone flies autonomously
            await drone.param.set_param_int("COM_RC_OVERRIDE", 1)

            # keep armed even if not took off for long time (1000 seconds)
            await drone.param.set_param_float("COM_DISARM_PRFLT", 1000.0)

            arming_exception = None
            for i in range(arm_attempts):
                try:
                    if force_arm:
                        await drone.action.arm_force()
                    else:
                        await drone.action.arm()

                    arming_exception = None
                    break
                except Exception as e:
                    arming_exception = e
                    logging.warning(f"retrying to arm, attempt {i}")
                    await asyncio.sleep(0.3)

            if arming_exception is not None:
                logging.warning(f"Failed to arm, latest exception", exc_info=arming_exception)

        await arm()

        logging.info("armed")

        # await asyncio.sleep(0.5) # TODO(vnemkov): maybe remove?
        await self.move_to_xy(XY(0, 0), IDLE_THRUST)
        await asyncio.sleep(0.5)
        await self.move_to_xy(XY(0.5, 0.5), 0.2, allow_unsafe=True)
        await asyncio.sleep(1)
        # logging.debug("offboard mode: %s", await self.offboard.is_active())

        await self._ensure_telemetry_cache()

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


    async def move_to_xy(self, xy : XY, thrust : float = 0.0, allow_unsafe = True) -> None:
        # Keep commanded tilt within a safe envelope to avoid toppling.
        SAFE_MANUAL_VALUES = 0.9
        def _clamp(value: float) -> float:
            if allow_unsafe == True:
                return value
            return max(-SAFE_MANUAL_VALUES, min(SAFE_MANUAL_VALUES, value))


        # x : float
        #      value between -1. to 1. negative -> backwards, positive -> forwards

        # y : float
        #      value between -1. to 1. negative -> left, positive -> right

        # z : float
        #      value between -1. to 1. negative -> down, positive -> up (usually for now, for multicopter 0 to 1 is expected)

        self.__sticks["x"] = _clamp(xy.y)
        self.__sticks["y"] = _clamp(xy.x)
        self.__sticks["z"] = _clamp(thrust)
        self.__sticks["r"] = 0
        self.__sticks["last_set_time"] = time.monotonic_ns()


    async def standstill(self) -> None:
        await self.move_to_xy(XY(0, 0), 0)

    async def idle(self):
        await self.move_to_xy(XY(0, 0), 0.0)

    def ABORT(self):
        self.drone = None


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
