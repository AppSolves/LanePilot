import subprocess as sp
import time
from enum import Enum
from typing import Optional

import cv2
import requests as req
from flask import Flask, Response, redirect, url_for

from shared_src.common import StoppableThread
from shared_src.network import NETWORK_CONFIG

from .core import MODULE_CONFIG, logger

_app = Flask(__name__)


class Direction(Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"


FPS: int = MODULE_CONFIG["display_server"].get("fps")
_current_direction: Direction = Direction.STRAIGHT
_file_name_map = MODULE_CONFIG["display_server"].get("file_name_map")


def generate_frame(direction: Direction) -> Optional[bytes]:
    """
    Generate a frame from the given file name.
    Args:
        direction (Direction): The direction to generate the frame for.
    Returns:
        bytes: The generated frame as bytes.
    """
    file_name = _file_name_map.get(direction.value)
    try:
        frame = cv2.imread(file_name)
        if frame is None:
            logger.error(f"Failed to read image from {file_name}")
            return None

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            logger.error(f"Failed to encode image from {file_name}")
            return None

        # Convert the buffer to bytes and return it
        frame_bytes = buffer.tobytes()
        return (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
    except Exception as e:
        logger.error(f"Error generating frame from {file_name}: {e}")
        return None


def generate_frames():
    """
    Generate frames for the live display.
    Yields:
        bytes: The generated frame as bytes.
    """
    while True:
        # Generate the frame based on the direction
        frame = generate_frame(_current_direction)
        if frame is not None:
            yield frame
        time.sleep(1 / FPS)  # Control the frame rate


@_app.route("/")
def index():
    return redirect(url_for("live_display"))


@_app.route("/set_direction/<direction>")
def set_direction(direction: str):
    """
    Set the current direction.
    Args:
        direction (str): The direction to set.
    Returns:
        str: A message indicating the result of the operation.
    """
    global _current_direction
    if direction in Direction.__members__:
        _current_direction = Direction[direction]
        return f"Direction set to {direction}", 200
    else:
        logger.error(f"Invalid direction: {direction}")
        return "Invalid direction", 400


@_app.route("/live_display")
def live_display():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


class DisplayServer(StoppableThread):
    def __init__(self, port: int, *args, **kwargs):
        """
        Initialize the DisplayServer.
        Args:
            port (int): The port to bind to.
        """
        super().__init__(*args, **kwargs)
        self._lane_config = MODULE_CONFIG.get("lanes", {})
        self._module = __name__.split(".")[-1]
        self._process: Optional[sp.Popen] = None
        self._ip = NETWORK_CONFIG["ips"].get("hotspot")
        self._port = port

    def run_with_exception_handling(self) -> None:
        try:
            self._process = sp.Popen(
                f"gunicorn -b {self._ip}:{self._port} {self._module}:_app",
                shell=True,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                close_fds=True,
            )
            logger.info(f"DisplayServer started on port {self._port}")
            self._process.wait()
        except Exception as e:
            logger.error(f"DisplayServer encountered an error: {e}")
            raise  # Propagate error for reconnect logic

    def on_event(self, command: str, value: str):
        """Handle incoming commands from the server."""
        logger.debug(f"Received command: {command} with value: {value}")

        match command:
            case "exit":
                logger.info("Exit command received, disposing server...")
                self.dispose()
            case "switch":
                #! We suppose that the lanes are numbered from 0 to n-1 from left to right
                #! This makes it easier to manage the directions

                from_lane, to_lane = tuple(map(int, value.split("-->")))
                if from_lane < 0 or to_lane < 0:
                    logger.error(
                        f"Invalid lane mapping: Lanes {from_lane} or {to_lane} not found"
                    )
                    return

                if from_lane == to_lane:
                    direction = Direction.STRAIGHT
                elif from_lane < to_lane:
                    direction = Direction.RIGHT
                else:
                    direction = Direction.LEFT

                req.get(
                    f"http://{self._ip}:{self._port}/set_direction/{direction.value}"
                )
                logger.info(f"Switching from lane {from_lane} to lane {to_lane}")
            case _:
                logger.error(f"Unknown command: {command}")
                return

    def dispose(self):
        """
        Dispose of the DisplayServer.
        """
        self.stop()
        if self._process:
            self._process.terminate()
            self._process.wait()
            logger.info("DisplayServer process terminated")
        else:
            logger.warning("DisplayServer process not found")
        self._process = None
