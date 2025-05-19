from dynamixel_sdk import *

from shared_src.common import Singleton

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
        self.config = MODULE_CONFIG.get("dynamixel", {}).get(type, {})

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
        return self._write(1, self.config.get("ADDR_TORQUE_ENABLE"), int(enable))

    @property
    def angle(self) -> int:
        """Get the current angle of the servo."""
        result = self._read(2, self.config.get("ADDR_PRESENT_POSITION"))
        if result == -1:
            logger.error("Failed to read angle.")
            return -1

        return int(
            (result - self.config.get("DXL_MINIMUM_POSITION_VALUE"))
            * 360
            / (
                self.config.get("DXL_MAXIMUM_POSITION_VALUE")
                - self.config.get("DXL_MINIMUM_POSITION_VALUE")
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
            self.config.get("ADDR_GOAL_POSITION"),
            int(
                angle
                * (
                    self.config.get("DXL_MAXIMUM_POSITION_VALUE")
                    - self.config.get("DXL_MINIMUM_POSITION_VALUE")
                )
                / 360
            ),
        )


class ServoManager(metaclass=Singleton):
    def __init__(self, port: str, baudrate: int, broadcast_add: bool = True):
        self.port = port
        self.baudrate = baudrate
        self._broadcast_add = broadcast_add
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(2.0)
        self.servos = {}
        self.try_reinit()

    def try_reinit(self):
        """Try to reinitialize the port and baudrate."""
        if self.portHandler.isOpen():
            return

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
        match command:
            case "exit":
                logger.info("Exit command received, disposing servos...")
                self.dispose()
            case "switch":
                pass  # TODO: Implement switch command
            case _:
                logger.error(f"Unknown command: {command}")
                return
