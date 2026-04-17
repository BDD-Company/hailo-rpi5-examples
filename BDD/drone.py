#!/usr/bin/env python

import asyncio
# import nest_asyncio
import dataclasses
from enum import Enum
from math import nan, cos, radians, acos, degrees, copysign
import time

from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, Attitude, VelocityNedYaw, AttitudeRate, OffboardError
from mavsdk import System
from mavsdk.telemetry import Telemetry, EulerAngle, LandedState

from helpers import MoveCommand, dotdict, XY, debug_collect_call_info

import logging

logger = logging.getLogger("BDD_drone")

DEFAULT_TAKEOFF_ALTITUDE_M = 10
SAFE_TILT_DEG = 180
IDLE_THRUST = 0.01
UPSIDE_DOWN_ANGLE_DEG = 120
UPSIDE_DOWN_HOLD_S = 1.0

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
        self.use_set_attitude = self.config.get('use_set_attitude', False)
        self.cruise_altitude = self.config.get('cruise_altitude', DEFAULT_TAKEOFF_ALTITUDE_M)
        self.upside_down_angle_deg = self.config.get('upside_down_angle_deg', UPSIDE_DOWN_ANGLE_DEG)
        self.upside_down_hold_s = self.config.get('upside_down_hold_s', UPSIDE_DOWN_HOLD_S)
        self.min_lift_fraction = self.config.get('min_lift_fraction', 0.4)
        # How much upward velocity (m/s) fully relaxes the min_lift constraint.
        # At this velocity the effective min_lift becomes 0.
        self.lift_velocity_headroom = self.config.get('lift_velocity_headroom_ms', 3.0)
        # Same for net upward acceleration (m/s²), after subtracting gravity.
        self.lift_accel_headroom = self.config.get('lift_accel_headroom_mss', 5.0)
        self.tasks : list[asyncio.Task] = []
        self.drone_connection_string = drone_connection_string
        self.aborted = False
        self.upside_down_state = False

        # Telemetry aspects to keep cached; edit this list to add/remove streams.
        self.telemetry_aspects = [
            "attitude_euler",
            # "battery",
            # "health",
            "odometry",
            "landed_state",
            "imu"
            # "attitude_angular_velocity_body",
        ]
        self._telemetry_tasks : dict[str, asyncio.Task] = {}
        self._telemetry_ready : dict[str, asyncio.Event] = {}
        self._telemetry_latest : dict[str, object] = {}
        self._telemetry_dict_cache : dict[str, object] = {}

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

        async def arm():
            logging.info("arming")

            # # Enable arming without GPS
            await drone.param.set_param_int("COM_ARM_WO_GPS", 1)
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
            else:
                logging.info("armed")


        await arm()


        await asyncio.sleep(1) # TODO(vnemkov): maybe remove?

        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.01))

        logger.debug("Entering Offboard mode...")
        try:
            await drone.offboard.start()
        except OffboardError as e:
            logger.error("Failed to enter Offboard mode: %s, aborting", e, exc_info=True)
            await drone.action.disarm()
            raise

        self.offboard = drone.offboard

        # DURATION=3
        # THRUST=0.3
        # logger.debug("A little dance")
        # await self.move_to_target_zenith_async(roll_degree=-45, pitch_degree=0, thrust=THRUST)
        # await asyncio.sleep(DURATION)
        # await self.move_to_target_zenith_async(roll_degree=  0,  pitch_degree=0, thrust=THRUST)
        # await asyncio.sleep(DURATION / 3)
        # await self.move_to_target_zenith_async(roll_degree= 45, pitch_degree=0, thrust=THRUST)
        # await asyncio.sleep(DURATION)

        # await self.move_to_target_zenith_async(roll_degree=0, pitch_degree=0, thrust=THRUST / 2)
        # await asyncio.sleep(DURATION / 3)

        # await self.move_to_target_zenith_async(roll_degree=0, pitch_degree=-45, thrust=THRUST)
        # await asyncio.sleep(DURATION)
        # await self.move_to_target_zenith_async(roll_degree=0, pitch_degree=  0, thrust=THRUST)
        # await asyncio.sleep(DURATION / 3)
        # await self.move_to_target_zenith_async(roll_degree=0, pitch_degree= 45, thrust=THRUST)
        # await asyncio.sleep(DURATION)

        # await self.move_to_target_zenith_async(roll_degree=0, pitch_degree=0, thrust=THRUST / 2)
        # await asyncio.sleep(DURATION / 3)

        # await asyncio.sleep(0.5) # TODO(vnemkov): maybe remove?
        # await self.move_to_xy(XY(0, 0), IDLE_THRUST)
        # await asyncio.sleep(0.5)
        # for i in range(0, 10):
        #     logging.info("initial dance: %s", i)
        #     await self.move_to_xy(XY(0.5, 0.5), 0.2, allow_unsafe=True)
        #     await asyncio.sleep(1)
        #     await self.move_to_xy(XY(0.5, -0.5), 0.2, allow_unsafe=True)
        #     await asyncio.sleep(1)
        #     await self.move_to_xy(XY(-0.5, 0.5), 0.2, allow_unsafe=True)
        #     await asyncio.sleep(1)
        #     await self.move_to_xy(XY(-0.5, -0.5), 0.2, allow_unsafe=True)
        #     await asyncio.sleep(1)

        # await asyncio.sleep(1)
        # logging.debug("offboard mode: %s", await self.offboard.is_active())

        await self._ensure_telemetry_cache()
        # await self.start_upside_down_monitor()

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
                    self._telemetry_dict_cache[aspect_name] = mavsdk_msg_to_dict(sample)
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

    def get_telemetry_dict_cached(self) -> dotdict:
        """Return pre-converted telemetry as dotdict with zero async overhead.
        Values are at most one telemetry update old. Only valid after startup_sequence().
        Safe to call without await from inside the asyncio loop."""
        return dotdict({aspect: self._telemetry_dict_cache.get(aspect) for aspect in self.telemetry_aspects})

    async def move_to_target_ned(self, target_position_ned, current_telemetry = None):
        if target_position_ned is None:
            logger.warning("No NED target provided, ignoring command")
            return

        current_telemetry = current_telemetry or self.get_telemetry_dict_cached()
        # it is not critical if we can't get yaw
        yaw_deg = (current_telemetry.get('attitude_euler', {}) or {}).get('yaw_deg', 0)

        drone_offboard = debug_collect_call_info(self.drone.offboard)

        await drone_offboard.set_position_ned(
            PositionNedYaw(
                north_m = target_position_ned.north_m,
                east_m = target_position_ned.east_m,
                down_m = target_position_ned.down_m,
                yaw_deg = yaw_deg
            )
        )

        logger.info("!!! executing: %s ", drone_offboard.last_command())


    def _vertical_energy_bonus(self, current_telemetry) -> float:
        """Return a [0, 1] factor representing how much the min_lift constraint
        can be relaxed based on current upward velocity and acceleration.
        Returns 0 when no telemetry is available or drone is descending."""
        if not current_telemetry or not all(current_telemetry.keys()):
            return 0.0

        bonus = 0.0

        # Vertical velocity from odometry (NED: down is positive, so upward = negative z)
        try:
            down_m_s = current_telemetry["odometry"]["velocity_body"]["z_m_s"]
            up_velocity = max(0.0, -down_m_s)
            if self.lift_velocity_headroom > 0:
                bonus += up_velocity / self.lift_velocity_headroom
        except (TypeError, KeyError):
            pass

        # Net vertical acceleration from IMU (body-frame FRD).
        # Rotate to NED using the attitude quaternion, then subtract gravity.
        try:
            imu = current_telemetry["imu"]
            q = current_telemetry["odometry"]["q"]
            fwd = imu["acceleration_frd"]["forward_m_s2"]
            rgt = imu["acceleration_frd"]["right_m_s2"]
            dwn = imu["acceleration_frd"]["down_m_s2"]

            # Quaternion rotation: body FRD -> NED
            qw, qx, qy, qz = q["w"], q["x"], q["y"], q["z"]
            # q * v  (v as pure quaternion)
            tw = -qx * fwd - qy * rgt - qz * dwn
            tx =  qw * fwd + qy * dwn - qz * rgt
            ty =  qw * rgt + qz * fwd - qx * dwn
            tz =  qw * dwn + qx * rgt - qy * fwd
            # (q * v) * q_conjugate  -> down component
            accel_down_ned = -tw * qz + tz * qw - tx * qy + ty * qx

            # Subtract gravity (g ≈ 9.81 m/s² pointing down in NED)
            net_accel_down = accel_down_ned - 9.81
            net_accel_up = max(0.0, -net_accel_down)
            if self.lift_accel_headroom > 0:
                bonus += net_accel_up / self.lift_accel_headroom
        except (TypeError, KeyError):
            pass

        return min(bonus, 1.0)

    def _clamp_tilt_for_lift(self, roll_deg: float, pitch_deg: float, thrust: float, current_telemetry = None) -> tuple[float, float]:
        """Scale down roll and pitch so that the vertical lift component
        (thrust * cos(roll) * cos(pitch)) stays >= effective min_lift.
        The effective min_lift is reduced when the drone has upward
        velocity/acceleration (vertical energy bonus from telemetry)."""
        if thrust <= 0:
            return roll_deg, pitch_deg

        bonus = self._vertical_energy_bonus(current_telemetry)
        effective_min_lift = self.min_lift_fraction * (1.0 - bonus)

        required_cos = effective_min_lift / thrust
        if required_cos > 1.0:
            # Thrust alone can't meet the lift minimum — zero out tilt
            return 0.0, 0.0

        current_cos = cos(radians(roll_deg)) * cos(radians(pitch_deg))
        if current_cos >= required_cos:
            return roll_deg, pitch_deg

        # Scale both angles down proportionally until the constraint is met.
        # max_angle = total allowed tilt if both axes shared equally
        max_half_angle = degrees(acos(required_cos ** 0.5))

        abs_roll = abs(roll_deg)
        abs_pitch = abs(pitch_deg)
        total = abs_roll + abs_pitch
        if total == 0:
            return 0.0, 0.0

        # Distribute the budget proportionally to the requested angles
        budget_roll = max_half_angle * 2 * (abs_roll / total)
        budget_pitch = max_half_angle * 2 * (abs_pitch / total)
        safe_roll = copysign(min(abs_roll, budget_roll), roll_deg)
        safe_pitch = copysign(min(abs_pitch, budget_pitch), pitch_deg)

        # Verify and tighten if the proportional split still violates
        while cos(radians(safe_roll)) * cos(radians(safe_pitch)) < required_cos:
            safe_roll *= 0.95
            safe_pitch *= 0.95

        logger.debug("Tilt clamped for lift: roll %.1f->%.1f, pitch %.1f->%.1f (thrust=%.2f, min_lift=%.2f, eff=%.2f) bonus=%.2f",
                      roll_deg, safe_roll, pitch_deg, safe_pitch, thrust, self.min_lift_fraction, effective_min_lift, bonus)
        return safe_roll, safe_pitch

    async def move_to_target_zenith_async(self, roll_degree : float, pitch_degree : float, thrust : float = 0.0, current_telemetry = None) -> None:
        if self.upside_down_state:
            logger.warning("drone is UPSIDE-DOWN, ignoring command")
            return

        drone_offboard = debug_collect_call_info(self.drone.offboard)

        if self.use_set_attitude:
            safe_roll, safe_pitch = self._clamp_tilt_for_lift(roll_degree, pitch_degree, thrust, current_telemetry)
            yaw_deg = ((current_telemetry or {}).get('attitude_euler', {}) or {}).get('yaw_deg', 0)
            await drone_offboard.set_attitude(
                Attitude(
                    roll_deg=safe_roll,
                    pitch_deg=safe_pitch,
                    yaw_deg=yaw_deg,
                    thrust_value=thrust,
                )
            )
        else:
            await drone_offboard.set_attitude_rate(
                AttitudeRate(
                    roll_deg_s=roll_degree,
                    pitch_deg_s=pitch_degree,
                    yaw_deg_s=0,
                    thrust_value=thrust,
                )
            )

        logger.info("!!! executing: %s ", drone_offboard.last_command())

        # await self.drone.offboard.set_attitude(
        #     Attitude(
        #         roll_deg=safe_roll,
        #         pitch_deg=safe_pitch,
        #         yaw_deg=0,
        #         thrust_value=thrust,
        #     )
        # )
        # await self.drone.offboard.set_velocity_body(
        #     VelocityBodyYawspeed(
        #         forward_m_s=safe_pitch,
        #         right_m_s=safe_roll,
        #         down_m_s=thrust * 10,
        #         yawspeed_deg_s=0
        #     )
        # )

    async def standstill(self) -> None:
        await self.move_to_target_zenith_async(0, 0, IDLE_THRUST * 2)


    async def idle(self):
        for i in range(0, 0):
            await self.move_to_target_zenith_async(0, 0, IDLE_THRUST / 2)
            await asyncio.sleep(0.2)
            await self.move_to_target_zenith_async(0, 0, IDLE_THRUST)
            await asyncio.sleep(0.05)
            await self.move_to_target_zenith_async(0, 0, IDLE_THRUST / 2)

        await self.move_to_target_zenith_async(0, 0, IDLE_THRUST / 2)


    def ABORT(self):
        """Thread-safe abort: stops offboard mode, giving control back to the FC/RC."""
        self.aborted = True
        logger.info("!!! Aborting, stopping offboard mode...")
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(self._stop_offboard(), loop)

    async def _stop_offboard(self):
        try:
            await self.drone.offboard.stop()
        except OffboardError as e:
            logger.warning("Failed to stop offboard: %s", e)
        logger.info("!!! Abort done, flight controller has control")

    def _is_upside_down(self, attitude: EulerAngle) -> bool:
        return abs(attitude.roll_deg) > self.upside_down_angle_deg or \
               abs(attitude.pitch_deg) > self.upside_down_angle_deg

    async def _upside_down_monitor(self) -> None:
        """Background task: when the drone stays upside-down for
        `upside_down_hold_s` seconds, command a level attitude to recover."""
        upside_down_since: float | None = None
        logger.debug("!!! Started upside-down monitor thread with angle= %s\thold time= %s s", self.upside_down_angle_deg, self.upside_down_hold_s)

        while not self.aborted:
            attitude = await self.get_cached_attitude(wait_for_first=True)
            if attitude is None:
                await asyncio.sleep(0.05)
                self.upside_down_state = False
                continue

            now = time.monotonic()

            self.upside_down_state = self._is_upside_down(attitude)
            if self.upside_down_state:
                if upside_down_since is None:
                    upside_down_since = now
                    logger.info("Upside-down detected (roll=%.1f pitch=%.1f)",
                                attitude.roll_deg, attitude.pitch_deg)

                elapsed = now - upside_down_since
                if elapsed >= self.upside_down_hold_s:
                    logger.warning(
                        "Upside-down for %.2fs - stabilizing to level attitude", elapsed)
                    await self.drone.offboard.set_attitude(
                        Attitude(0.0, 0.0, 0.0, 0.0))
                    upside_down_since = None
            else:
                if upside_down_since is not None:
                    logger.info("No longer upside-down, resetting timer")
                upside_down_since = None

            await asyncio.sleep(0.05)

    async def start_upside_down_monitor(self) -> asyncio.Task:
        """Launch the upside-down recovery monitor as a background task."""
        task = asyncio.create_task(
            self._upside_down_monitor(), name="upside_down_monitor")
        task.add_done_callback(self._log_background_task_result)
        self.tasks.append(task)
        logger.info("Upside-down monitor started (threshold=%.0f° hold=%.2fs)",
                     self.upside_down_angle_deg, self.upside_down_hold_s)
        return task


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
