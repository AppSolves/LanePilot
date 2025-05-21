import hashlib
import hmac
import os
import socket
from typing import Optional

from .core import NETWORK_CONFIG, logger


def discover_peer(
    port: int = NETWORK_CONFIG["ports"].get("gstreamer"),
    timeout: int = 10,
    retries: int = 3,
) -> Optional[str]:
    """
    Sends a broadcast message to discover peers on the network.
    Listens for a response and returns the IP address of the discovered peer.
    """
    broadcast_ip = NETWORK_CONFIG["ips"].get("broadcast")
    if not broadcast_ip:
        logger.error("Broadcast IP not configured in NETWORK_CONFIG.")
        return None

    for attempt in range(retries):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)
            sock.bind(("", 0))

            challenge = os.urandom(16)
            message = b"P2P_BROADCAST_REQ:" + challenge
            try:
                sock.sendto(message, (broadcast_ip, port))
                logger.info(
                    f"[{attempt+1}/{retries}] Broadcasting to {broadcast_ip}:{port}, waiting for response..."
                )

                data, addr = sock.recvfrom(1024)
                if addr[0] == NETWORK_CONFIG["ips"].get("self"):
                    logger.info(f"Received response from self ({addr[0]}), ignoring.")
                    return None

                logger.info(f"Found peer at {addr[0]}: {data}")

                if not data.startswith(b"P2P_BROADCAST_RES:"):
                    logger.warning("Invalid response, expected P2P_BROADCAST_RES.")
                    return None

                received_hmac = data.split(b":", 1)[1]
                expected_hmac = hmac.new(
                    NETWORK_CONFIG["vars"].get("handshake").encode(),
                    challenge,
                    hashlib.sha256,
                ).digest()
                if not hmac.compare_digest(received_hmac, expected_hmac):
                    logger.warning("HMAC verification failed. Peer not authorized.")
                    return None

                return addr[0]

            except socket.timeout:
                logger.warning(
                    f"Attempt {attempt+1} timed out. Retrying..."
                    if attempt < retries - 1
                    else "No response, peer not found."
                )
            except Exception as e:
                logger.error(f"Error during peer discovery: {e}")
                return None

    return None


def respond_to_broadcast(
    port: int = NETWORK_CONFIG["ports"].get("gstreamer"),
    timeout: Optional[int] = None,
    stop_on_response: bool = False,
) -> Optional[str]:
    """
    Listens for broadcast messages and responds to discovery requests.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if timeout:
            sock.settimeout(timeout)
        sock.bind(("", port))

        logger.info(f"Listening for broadcast messages on port {port}...")

        try:
            while True:
                data, addr = sock.recvfrom(1024)
                if addr[0] == NETWORK_CONFIG["ips"].get("self"):
                    logger.info(f"Received message from self ({addr[0]}), ignoring.")
                    continue

                logger.info(f"Received message from {addr[0]}: {data}")

                if data.startswith(b"P2P_BROADCAST_REQ:"):
                    challenge = data.split(b":", 1)[1]
                    response_hmac = hmac.new(
                        NETWORK_CONFIG["vars"].get("handshake").encode(),
                        challenge,
                        hashlib.sha256,
                    ).digest()
                    response = b"P2P_BROADCAST_RES:" + response_hmac
                    sock.sendto(response, addr)
                    if stop_on_response:
                        logger.info("Response sent, stopping broadcast responder.")
                        return addr[0]
                else:
                    logger.warning(f"Invalid message received from {addr[0]}: {data}")

        except Exception as e:
            logger.error(f"Error while listening for broadcast messages: {e}")
            return None
