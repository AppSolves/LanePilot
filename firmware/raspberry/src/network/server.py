import subprocess as sp
import sys
from typing import Callable

import zmq
from shared_src.common import Singleton, StoppableThread

from .core import logger


def run_gstreamer(peer_ip: str, udp_port: int):
    """
    Run GStreamer pipeline to stream video from the Raspberry Pi camera to a peer using SRT.

    Args:
        peer_ip (str): The IP address of the peer to stream to.
        udp_port (int): The UDP port to use for streaming.
    """

    logger.info(f"Starting GStreamer pipeline to {peer_ip}:{udp_port}")
    try:
        return sp.run(
            [
                "gst-launch-1.0",
                "libcamerasrc",
                "!",
                "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1",
                "!",
                "videoconvert",
                "!",
                "x264enc",
                "tune=zerolatency",
                "bitrate=5000",
                "speed-preset=ultrafast",
                "!",
                "video/x-h264,profile=main" "!",
                "queue",
                "!",
                "mpegtsmux",
                "!",
                "srtsink",
                f"uri='srt://{peer_ip}:{udp_port}?mode=caller&latency=1'",
            ],
            check=True,
        )
    except (sp.TimeoutExpired, sp.CalledProcessError) as e:
        logger.error(f"GStreamer process timed out or failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


class ServerThread(StoppableThread, metaclass=Singleton):
    """A thread that runs a ZeroMQ server to receive commands from a client."""

    __listeners: list[Callable] = []

    def __init__(self, port: int, *args, **kwargs) -> None:
        """Initialize the server thread.

        Args:
            port (int): The port to bind the server to.
        """
        super().__init__(*args, **kwargs)

        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")

        logger.info(f"Server started on port {self.port}")
        logger.info("Waiting for commands...")

    def add_listener(self, listener: Callable) -> None:
        """Add a listener to the server.

        Args:
            listener: The listener to add.
        """
        self.__listeners.append(listener)

    def remove_listener(self, listener: Callable) -> None:
        """Remove a listener from the server.

        Args:
            listener: The listener to remove.
        """
        self.__listeners.remove(listener)

    def run(self) -> None:
        while self.running:
            data = self.socket.recv_json()
            command, value = data.get("command"), data.get("value")

            for listener in self.__listeners:
                listener(command, value)

            if command == "exit":
                logger.info("Exit command received, shutting down server.")
                break

            self.socket.send_json({"command": "status", "value": "ok"})

        self.dispose()

    def dispose(self) -> None:
        """Clean up the server resources."""
        self.stop()
        self.socket.send_json({"command": "exit", "value": "ok"})
        self.socket.close()
        self.context.term()
        self.__listeners.clear()
        logger.info("Server disposed.")
