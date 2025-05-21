from dynamixel_sdk import *

from shared_src.common import StoppableThread

from .core import MODULE_CONFIG, logger

#! IMPORTANT: This is currently only implemented for the Dynamixel XL320 servos.
#! If you want to use other servos, you need to look up the amount of bytes each register
#! takes and implement the read/write functions accordingly besides changing the type in the config file.


class _Servo:
    def __init__(self, id: int, type: str, porthandler: PortHandler, packetHandler):
        self.id = id
        self.type = type
        self.portHandler = porthandler
        self.packetHandler = packetHandler
        self._config = MODULE_CONFIG.get("dynamixel", {}).get(type, {})

        self.set_angle(0)

    def _write(self, num_bytes: int, address: int, value: int) -> bool:
        """Write a value to the servo."""
        func_map = {
            1: self.packetHandler.write1ByteTxRx,
            2: self.packetHandler.write2ByteTxRx,
            4: self.packetHandler.write4ByteTxRx,
        }

        if num_bytes in func_map:
            func = func_map[num_bytes]
            result, error = func(self.portHandler, self.id, address, value)
            if result != COMM_SUCCESS:
                logger.error(
                    f"Failed to write {num_bytes} bytes: {self.packetHandler.getTxRxResult(result)}"
                )
                return False
            elif error != 0:
                logger.error(
                    f"Error writing {num_bytes} bytes: {self.packetHandler.getRxPacketError(error)}"
                )
                return False
        else:
            logger.error(f"Invalid number of bytes: {num_bytes}")
            return False

        return True

    def _read(self, num_bytes: int, address: int) -> int:
        """Read a value from the servo."""
        func_map = {
            1: self.packetHandler.read1ByteTxRx,
            2: self.packetHandler.read2ByteTxRx,
            4: self.packetHandler.read4ByteTxRx,
        }

        if num_bytes in func_map:
            func = func_map[num_bytes]
            result, error = func(self.portHandler, self.id, address)
            if result != COMM_SUCCESS:
                logger.error(
                    f"Failed to read {num_bytes} bytes: {self.packetHandler.getTxRxResult(result)}"
                )
                return -1
            elif error != 0:
                logger.error(
                    f"Error reading {num_bytes} bytes: {self.packetHandler.getRxPacketError(error)}"
                )
                return -1
        else:
            logger.error(f"Invalid number of bytes: {num_bytes}")
            return -1

        return result

    def _toggle_torque(self, enable: bool) -> bool:
        return self._write(1, self._config.get("ADDR_TORQUE_ENABLE"), int(enable))

    @property
    def angle(self) -> int:
        """Get the current angle of the servo."""
        result = self._read(2, self._config.get("ADDR_PRESENT_POSITION"))
        if result == -1:
            logger.error("Failed to read angle.")
            return -1

        return int(
            (result - self._config.get("DXL_MINIMUM_POSITION_VALUE"))
            * 360
            / (
                self._config.get("DXL_MAXIMUM_POSITION_VALUE")
                - self._config.get("DXL_MINIMUM_POSITION_VALUE")
            )
        )

    def set_angle(self, angle: int):
        result = self._toggle_torque(True)
        if not result:
            logger.error("Failed to enable torque.")
            return

        if not 0 <= angle <= 360:
            logger.error("Angle must be between 0 and 360 degrees.")
            return

        self._write(
            2,
            self._config.get("ADDR_GOAL_POSITION"),
            int(
                angle
                * (
                    self._config.get("DXL_MAXIMUM_POSITION_VALUE")
                    - self._config.get("DXL_MINIMUM_POSITION_VALUE")
                )
                / 360
            ),
        )


class ServoManager(StoppableThread):
    def __init__(
        self,
        *args,
        broadcast_add: bool = True,
        reverse_direction: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._servo_config = MODULE_CONFIG.get("servos", {})
        self._lane_config = MODULE_CONFIG.get("lanes", {})
        self.port = self._servo_config.get("uart_port")
        self.baudrate = self._servo_config.get("baudrate")
        self._broadcast_add = broadcast_add
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(2.0)
        self.reverse_direction = reverse_direction
        self.servos = {}

        if not self.portHandler.openPort():
            logger.error("Failed to open the port")
            raise RuntimeError("Failed to open the port")
        if not self.portHandler.setBaudRate(self.baudrate):
            logger.error("Failed to change the baudrate")
            raise RuntimeError("Failed to change the baudrate")

        if self._broadcast_add:
            self.broadcast_add()

        logger.info(
            f"ServoManager initialized on port {self.port} with baudrate {self.baudrate}"
        )

    def add_servo(self, id: int, type: str):
        if type not in MODULE_CONFIG.get("dynamixel", {}):
            logger.error(f"Invalid servo type: {type}")
            return

        if id not in self.servos:
            self.servos[id] = _Servo(id, type, self.portHandler, self.packetHandler)

    def remove_servo(self, id: int):
        if id in self.servos:
            del self.servos[id]

    def dispose(self):
        self.stop()
        for servo in self.servos.values():
            servo._toggle_torque(False)
        self.portHandler.closePort()

    def broadcast_add(self):
        """Broadcast the servo IDs to the network."""
        ids, result = self.packetHandler.broadcastPing(self.portHandler)
        if result != COMM_SUCCESS:
            logger.error(
                f"Failed to broadcast ping: {self.packetHandler.getTxRxResult(result)}"
            )
            return

        if not ids:
            logger.warning("No servos found.")
            return

        for id in ids:
            if id not in self.servos:
                model_number = ids.get(id)[0]
                servos = MODULE_CONFIG.get("dynamixel", {})
                type = next(
                    (
                        k
                        for k, v in servos.items()
                        if v.get("MODEL_NUMBER") == model_number
                    ),
                    None,
                )
                if type is None:
                    logger.error(f"Unknown servo model number: {model_number}")
                    continue

                self.add_servo(id, type)

    def on_event(self, command: str, value: str):
        """Handle incoming commands from the server."""
        logger.debug(f"Received command: {command} with value: {value}")
        turning_degree = self._lane_config.get("turning_degree", 0)

        match command:
            case "exit":
                logger.info("Exit command received, disposing servos...")
                self.dispose()
            case "switch":
                #! We suppose that the servos are connected in the same order as the lanes, e.g. lane 0 => servo 0
                #! and that that the lanes are numbered from 0 to n-1 from left to right
                #! This makes it easier to manage the servos

                from_lane, to_lane = tuple(map(int, value.split("-->")))
                if from_lane < 0 or to_lane < 0:
                    logger.error(
                        f"Invalid lane mapping: Lanes {from_lane} or {to_lane} not found"
                    )
                    return

                if from_lane == to_lane:
                    logger.info(f"Already on lane {from_lane}, no action taken.")
                    return

                from_servo = self.servos.get(from_lane)
                to_servo = self.servos.get(to_lane)
                if from_servo is None or to_servo is None:
                    logger.error(
                        f"Invalid servo mapping: {from_servo} or {to_servo} not found"
                    )
                    return

                angle = (
                    180 + turning_degree
                    if ((from_lane < to_lane) ^ self.reverse_direction)
                    else 180 - turning_degree
                )
                from_servo.set_angle(angle)
                to_servo.set_angle(angle)
                logger.debug(f"Setting angle for {from_lane} to {angle}")
                logger.info(f"Switching from lane {from_lane} to lane {to_lane}")
            case _:
                logger.error(f"Unknown command: {command}")
                return
