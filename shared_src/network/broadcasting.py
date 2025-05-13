import socket
from typing import Optional

from .core import PORTS, logger


def discover_peer(timeout: int = 10, port: int = PORTS["udp"]) -> Optional[str]:
    """
    Sends a broadcast message to discover peers on the network.
    Listens for a response and returns the IP address of the discovered peer.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)

        message = b"P2P_BROADCAST_REQ"
        sock.sendto(message, ("<broadcast>", port))
        logger.info("Broadcast sent, waiting for response...")

        try:
            data, addr = sock.recvfrom(1024)
            logger.info(f"Found peer at {addr[0]}: {data}")

            if data != b"P2P_BROADCAST_RES":
                logger.warning("Invalid response, expected P2P_BROADCAST_RES.")
                return None

            return addr[0]
        except socket.timeout:
            logger.error("No response, peer not found.")
            return None


def respond_to_broadcast(
    port: int = PORTS["udp"],
    stop_on_response: bool = False,
) -> Optional[str]:
    """
    Listens for broadcast messages and responds to discovery requests.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))

        logger.info(f"Listening for broadcast messages on port {port}...")

        try:
            while True:
                data, addr = sock.recvfrom(1024)
                logger.info(f"Received message from {addr[0]}: {data}")

                if data == b"P2P_BROADCAST_REQ":
                    logger.info(
                        f"Valid broadcast request received from {addr[0]}. Sending response..."
                    )
                    response = b"P2P_BROADCAST_RES"
                    sock.sendto(response, addr)
                    if stop_on_response:
                        logger.info("Response sent, stopping broadcast responder.")
                        return addr[0]
                else:
                    logger.warning(f"Invalid message received from {addr[0]}: {data}")
        except Exception as e:
            logger.error(f"Error while listening for broadcast messages: {e}")
            return None
