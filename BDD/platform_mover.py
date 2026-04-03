import logging
import math
import abc
import json

from urllib.parse import urlparse
from threading import Lock


from helpers import XY
import interfaces

logger = logging.getLogger(__name__)

class CommandSenderInterface(abc.ABC):
    @abc.abstractmethod
    def send_command(self, command, read_response=False) -> None:
        pass

# Various constants related to command protocol
class Protocol():
    __slots__ = () # to make values unassignable

    SERIAL_BAUD_RATE = 115200

# Some of the command codes from json_cmd.h
# https://github.com/waveshareteam/ugv_base_general/blob/91e58712176c6d98a9476a74c7a1712f3d631114/General_Driver/json_cmd.h
    FEEDBACK_BASE_INFO = 1001

    # bus servos error feedback (received FROM driver)
    # {"T":1005,"id":1,"status":1}
    CMD_BUS_SERVO_ERROR = 1005

    # MODULE TYPE
    # 0: nothing
    # 1: RoArm-M2-S
    # 2: Gimbal
    # {"T":4,"cmd":0}
    CMD_MODULE_TYPE = 4

    # {"T":130}
    CMD_BASE_FEEDBACK = 130

    # GIMBAL CTRL(SIMPLE)
    # {"T":133,"X":45,"Y":45,"SPD":0,"ACC":0}
    CMD_GIMBAL_CTRL_SIMPLE = 133

    # GIMBAL CTRL MOVE
    # {"T":134,"X":45,"Y":45,"SX":300,"SY":300}
    CMD_GIMBAL_CTRL_MOVE = 134

    # CMD_GIMBAL_CTRL_STOP
    # {"T":135}
    CMD_GIMBAL_CTRL_STOP = 135

    # set the echo mode of receiving new cmd.
    # 0: [default]off
    # 1: on
    # {"T":143,"cmd":0}
    CMD_UART_ECHO_MODE = 143


def preprocess_received_data(result):
    """"
        Trim all excess garbage from beginning and end, just in case.
        result is expected to be a JSON-bytes, hence must begin with '{' and end with '}',
    """
    assert isinstance(result, bytes)

    # not expecting any nested objects, nor strings with arbitrary content,
    # so try to extract flat JSON object, possibly surrounded by garbage
    end = result.rfind(b'}')
    if end == -1:
        return b''

    begin = result.rfind(b'{', None, end)
    if begin == -1:
        return b''

    return result[begin:end + 1]


def create_command_sender(destination : str) -> CommandSenderInterface:
    result = urlparse(destination)

    if result.scheme in ['ws', 'http', 'https']:
        # On some platforms socketio module can be unavailable (like rpi5)
        # so import it only if really required.
        import socketio

        class SocketIOSender(CommandSenderInterface):
            def __init__(self, url):
                self._sio = socketio.Client()
                self._sio.connect(url, transports=['websocket'], namespaces=['/json'])

            def __del__(self):
                self._sio.disconnect()

            def send_command(self, command, read_response=False):
                self._sio.emit('json', command, namespace='/json')

        return SocketIOSender(destination)

    else:
        import json
        import serial

        class SerialPortSender(CommandSenderInterface):
            def __init__(self, destination_device):
                self.serial = serial.Serial(destination_device, Protocol.SERIAL_BAUD_RATE, timeout=1)
                self._drain()
                self.lock = Lock()

            def __del__(self):
                if hasattr(self, 'serial'):
                    self.serial.close()

            def _drain(self):
                drained = self.serial.read(self.serial.in_waiting)
                if len(drained) != 0:
                    logger.debug('drained from serial: %s', drained)


            def send_command(self, command, read_response=False):
                result = None
                with self.lock:
                    if read_response:
                        # drain whatever is in the serial right now to avoid receiving broken data down the road
                        self._drain()

                    if not isinstance(command, bytes):
                        command = (json.dumps(command) + '\n').encode("utf-8")

                    # logger.debug("Sending command: %s", command)
                    bytes_sent_total = 0

                    while bytes_sent_total != len(command):
                        bytes_sent = self.serial.write(command[bytes_sent_total:])
                        bytes_sent_total += bytes_sent

                    self.serial.flush()

                    if read_response:
                        result = self.serial.readline()

                if result:
                    logger.debug("result = %s", result)
                    result = json.loads(preprocess_received_data(result))
                    if int(result['T']) == Protocol.CMD_BUS_SERVO_ERROR:
                        raise RuntimeError(f"Got error from servos: {result}")

                # r = self.serial.read(self.serial.in_waiting)
                # if r:
                #     logger.debug("read some data from serial: %r", r)

                return result

        return SerialPortSender(destination)


class PlatformMover():
    '''
    Actually sends commands to move platform.
    Platform API:
        # - https://www.waveshare.com/wiki/UGV01
        - https://www.waveshare.com/wiki/2-Axis_Pan-Tilt_Camera_Module#Slave_JSON_Command_Set
    '''

    _MAX_SPEED = 0
    _MAX_ACCELERATION = 0

    def __init__(self, destination, acceleration=16, speed=16, speed_adjustments = XY(1.0, 1.0), minimal_move = XY()):
        self._command_sender = create_command_sender(destination)
        logger.debug("Connected to the platform on %s via %s", destination, self._command_sender)

        self._accumulated_delta = XY(0.0, 0.0)

        self.platform_acceleration = int(acceleration)
        self.platform_speed = int(speed)

        # 1.0 to make type a float
        self.speed_adjustments = XY(x=float(speed_adjustments.x), y=float(speed_adjustments.y))
        self.minimal_move = XY(x=abs(float(minimal_move.x)), y=abs(float(minimal_move.y)))
        self.adjustable_speed = True
        self.attempt_smooth_start = False
        # min distance -> factor by which multiply the speed
        # self.speed_up_factors = ((1, 1), (4, 4), (8, 8), (100, 10))

        # {'T': 41, 'SA': 0, 'SB': 0}
        # {'T': 41, ' ': 0, 'SB': 0}
        # Unknown commands for unknown reasons, as of now Im too afraid to remove those.
        self._send_JSON_command({'T': 41, 'SA': 0, 'SB': 0})
        self._send_JSON_command({'T': 41, ' ': 0, 'SB': 0})
        # echo off:
        self._send_JSON_command({"T": Protocol.CMD_UART_ECHO_MODE, "cmd": 0})

        # self._position_polling_interval_seconds = 0.05
        # self._pos_at_time = deque(maxlen=int(10 / self._position_polling_interval_seconds)) # store position values for last 10 seconds
        # self._start_periodic_position_polling()

        # logger.debug("Zeroing the platform...")
        # self.move_to_center()
        # logger.debug("platform zeroed")
        self._send_JSON_command({"T": Protocol.CMD_MODULE_TYPE, "cmd": 2}) # set MODULE_TYPE=2 (Pan/Tilt)

    def __del__(self):
        if hasattr(self, '_command_sender'):
            del self._command_sender


    def current_pos(self):
        retries = 3
        while retries > 0:
            # self._send_JSON_command({"T": Protocol.CMD_MODULE_TYPE, "cmd": 2}) # set MODULE_TYPE=2 (Pan/Tilt)
            pos = self._command_sender.send_command({"T": Protocol.CMD_BASE_FEEDBACK}, read_response=True)

            try:
                # logger.debug("got pos: %s ", pos)
                result=XY(x=float(pos["pan"]), y=float(pos["tilt"]))
                logger.debug('got position response: %s', result)
                return result

            except:
                logger.exception("Got invalid response: %s", pos, exc_info=True)
                retries -= 1
                if retries == 0:
                    raise

        # return self._current_pos.clone()


    # def pos_at_time(self, time : Milliseconds) -> XY:
    #     # assert False, "Not implemented"
    #     if len(self._pos_at_time) == 0:
    #         return XY()

    #     # making a copy just to make sure that time_array and pos_array have same dimensions,
    #     # i.e. nothing was inserted in another thread while we were construction those temporaries
    #     pos_at_time = self._pos_at_time.copy()

    #     time_array = np.array([pos_tuple[0] for pos_tuple in pos_at_time])
    #     pos_array = np.array([complex(pos.x, pos.y) for _, pos in pos_at_time])
    #     interpolate_at = np.array([time,])
    #     # logger.debug("!!! Interpolating position at %s ms time, time_array: %s, pos_array: %s", time, time_array, pos_array)

    #     interpolated_pos = np.interp(pos_at_time, time_array, pos_array)

    #     return XY(x=interpolated_pos.real[0], y=interpolated_pos.imag[0])


    # def _start_periodic_position_polling(self):
    #     def periodic_position_polling_thread_func():
    #         while True:
    #             try:
    #                 now = get_current_time_ms()
    #                 pos = self.current_pos()
    #                 self._pos_at_time.append((now, pos))
    #                 time.sleep(self._position_polling_interval_seconds)
    #             except:
    #                 logger.warning("Got exception", exc_info=True)

    #     self._position_polling_thread = Thread(
    #         target=periodic_position_polling_thread_func,
    #         name="PlatformMover/periodic position polling"
    #     )
    #     self._position_polling_thread.start()

    def move_relative(self, dx : float, dy : float):
        self._accumulated_delta += XY(dx, dy)

        scaled_delta = self._accumulated_delta.multiplied_by_XY(self.speed_adjustments)
        if (math.isclose(dx, 0.0) and math.isclose(dy, 0.0))\
            or (scaled_delta.abs() < self.minimal_move):
            logger.debug("!!! Requested move (%s, %s) is too small (< %s), letting it accumulate", dx, dy, self.minimal_move)
            return

        new_pos = self.current_pos()
        new_pos += scaled_delta

        # logging.debug('PlatformMover.move_platform: _accumulated_delta: %s, scaled_delta: %s, new_pos: %s',
        #     self._accumulated_delta, scaled_delta, new_pos)

        logger.debug("Starting relative move by %2.2f, %2.2f, accumulated: %s", dx, dy, self._accumulated_delta)
        self.move_to(new_pos)
        logger.debug("Move complete")


    def get_speedup_factor_from_distance_squared(self, distance_squared):
        # for min_distance, factor in self.speed_up_factors:
        #     if distance <= min_distance:
        #         return factor
        return distance_squared, distance_squared

    def stop_moving(self):
        self._send_JSON_command({'T': Protocol.CMD_GIMBAL_CTRL_STOP})



    def move_to(self, new_pos : XY):
        if self.adjustable_speed:

            move_distance_squared = self.current_pos().distance_squared_to(new_pos)
            acceleration_factor, speed_factor = self.get_speedup_factor_from_distance_squared(move_distance_squared)

            if acceleration_factor > 1 or speed_factor > 1:
                logger.debug('!!! speeding up ! distance: %s, acc factor: %s, speed factor: %s',
                    move_distance_squared, acceleration_factor, speed_factor)

            acceleration_factor = max(1.0, acceleration_factor)
            speed_factor = max(1.0, speed_factor)

        else:
            acceleration_factor, speed_factor = (1.0, 1.0)

        acceleration = self.platform_acceleration * acceleration_factor
        speed = self.platform_speed * speed_factor

        if acceleration != self.platform_acceleration or speed != self.platform_speed:
            logger.debug('!!! speeding up ! acceleration: %s, speed: %s',
                acceleration, speed)

        # NOTE: Just a silly attempt to do a smooth start
        # here we rely on the fact that message go over network
        # and then put into a queue on receiver's side,
        # so there is a micoseconds/miliseconds delay between commands
        # Same trick might not work on local machine with a direct connection
        # to a moving platform.
        if self.attempt_smooth_start:
            self._move_to(new_pos, speed=speed/4, acceleration=acceleration/4)
            self._move_to(new_pos, speed=speed/2, acceleration=acceleration/2)

        self._move_to(new_pos, speed=speed, acceleration=acceleration)


    def move_to_center(self):
        self._move_to(XY(0, 0), PlatformMover._MAX_SPEED, PlatformMover._MAX_ACCELERATION)


    def _move_to(self, new_pos : XY, speed, acceleration):
        # message = {
        #     'T': Protocol.CMD_GIMBAL_CTRL_SIMPLE,
        #     'X': int(new_pos.x),
        #     'Y': int(new_pos.y),
        #     "SPD": int(speed),
        #     "ACC": 0
        # }

        message = {
            # NOTE: for whatever reason movement is less jerky than with CMD_GIMBAL_CTRL_SIMPLE.
            'T': Protocol.CMD_GIMBAL_CTRL_MOVE,
            'X': new_pos.x,
            'Y': new_pos.y,
            'SX': speed,
            'SY': speed
        }
        self._send_JSON_command(message)

        self._accumulated_delta = XY()


    def _send_JSON_command(self, command):
        self._command_sender.send_command(command)
