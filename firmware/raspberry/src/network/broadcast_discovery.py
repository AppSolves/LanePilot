import socket
from typing import Optional

from .core import PORTS, logger


def discover_peer(timeout: int = 10, port: int = PORTS["udp"]) -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
    finally:
        sock.close()
