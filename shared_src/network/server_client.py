import time
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
        check_interval_sec: int = 30,
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
        self.check_interval_sec = check_interval_sec
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
        """Run the server/client thread."""

        last_check = time.time()
        try:
            while self.running:
                # Check if the connection is alive
                now = time.time()
                if now - last_check >= self.check_interval_sec:
                    if not self._is_connection_alive():
                        logger.error(f"{self.type} connection lost (periodic check).")
                        break
                    last_check = now

                try:
                    data = self.socket.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue  # No message, keep looping
                except zmq.ZMQError as e:
                    logger.error(f"{self.type} ZMQ error: {e}")
                    raise ConnectionError(f"{self.type} connection lost: {e}")

                command, value = data.get("command"), data.get("value")

                for listener in self.__listeners:
                    listener(command, value)

                if command == "exit":
                    logger.info(
                        f"Exit command received, shutting down {self.type.lower()}."
                    )
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

    def _is_connection_alive(self) -> bool:
        """Check if the ZeroMQ socket is still alive."""
        try:
            # For server: check if socket is closed
            if self.socket.closed:
                return False
            # For client: try a non-blocking poll for events
            if hasattr(self.socket, "get_monitor_socket"):
                # If monitoring is enabled, check events
                monitor = self.socket.get_monitor_socket()
                if monitor.poll(0):
                    evt = monitor.recv_multipart()
                    if b"DISCONNECTED" in evt[0]:
                        return False
            return True
        except Exception as e:
            logger.error(f"{self.type} connection check failed: {e}")
            return False

    def dispose(self) -> None:
        """Clean up the resources."""
        self.stop()
        self.socket.send_json({"command": "exit", "value": "ok"})
        self.socket.close()
        self.context.term()
        self.__listeners.clear()
        logger.info(f"{self.type} disposed.")
