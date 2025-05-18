from typing import Callable, Optional

import zmq

from ..common import StoppableThread
from .core import logger


class ServerClient(StoppableThread):
    """A thread that runs a ZeroMQ server/client to send and receive commands."""

    __listeners: list[Callable] = []

    def __init__(
        self,
        port: int,
        *args,
        is_server: bool = True,
        server_ip: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the server/client thread.

        Args:
            port (int): The port to bind to.
        """
        super().__init__(*args, **kwargs)

        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        if is_server:
            self.type = "Server"
            self.socket.bind(f"tcp://*:{self.port}")
        else:
            self.type = "Client"
            self.server_ip = server_ip
            if not self.server_ip:
                raise ValueError("Server IP is required for client mode.")
            self.socket.connect(f"tcp://{self.server_ip}:{self.port}")

        logger.info(f"{self.type} started on port {self.port}")
        logger.info("Waiting for commands...")

    def add_listener(self, listener: Callable) -> None:
        """Add a listener, which receives commands on each event.

        Args:
            listener: The listener to add.
        """
        self.__listeners.append(listener)

    def remove_listener(self, listener: Callable) -> None:
        """Remove a listener.

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
                logger.info(
                    f"Exit command received, shutting down {self.type.lower()}."
                )
                break

            self.socket.send_json({"command": "status", "value": "ok"})

        self.dispose()

    def dispose(self) -> None:
        """Clean up the resources."""
        self.stop()
        self.socket.send_json({"command": "exit", "value": "ok"})
        self.socket.close()
        self.context.term()
        self.__listeners.clear()
        logger.info(f"{self.type} disposed.")
