from typing import Callable, Optional

import zmq

from ..common import StoppableThread, get_parent_class
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

        self._disposed = False
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        if is_server:
            self.type = "ZeroMQ server"
            self.socket.bind(f"tcp://*:{self.port}")
        else:
            self.type = "ZeroMQ client"
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
        logger.info(
            f"Adding listener: {get_parent_class(listener)}.{listener.__name__}"
        )
        self.__listeners.append(listener)

    def remove_listener(self, listener: Callable) -> None:
        """Remove a listener.

        Args:
            listener: The listener to remove.
        """
        logger.info(
            f"Removing listener: {get_parent_class(listener)}.{listener.__name__}"
        )
        self.__listeners.remove(listener)

    def run_with_exception_handling(self) -> None:
        try:
            while self.running:
                try:
                    data = self.socket.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue  # No message, keep looping
                except zmq.ZMQError as e:
                    logger.error(f"{self.type} error: {e}")
                    raise ConnectionError(f"{self.type} connection lost: {e}")

                command, value = data.get("command"), data.get("value")

                for listener in self.__listeners:
                    listener(command, value)

                if command == "exit":
                    logger.info(f"Exit command received, shutting down {self.type}.")
                    break

                try:
                    self.socket.send_json({"command": "status", "value": "ok"})
                except zmq.ZMQError as e:
                    logger.error(f"{self.type} failed to send response: {e}")
                    raise ConnectionError(f"{self.type} send failed: {e}")

        except Exception as e:
            logger.error(f"{self.type} encountered an error: {e}")
            raise  # Propagate the error for reconnect logic

        finally:
            self.dispose()

    def dispose(self) -> None:
        """Clean up the resources."""
        if self._disposed:
            return
        self._disposed = True
        self.stop()
        self.__listeners.clear()
        self.socket.close()
        self.context.term()
        logger.info(f"{self.type} disposed.")
