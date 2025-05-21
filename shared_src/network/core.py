import os
import socket
from typing import Optional

from ..common import get_logger

logger = get_logger()


def _get_broadcast_addr(self_ip: Optional[str]) -> str:
    if self_ip:
        logger.info(f"Using static IP address: {self_ip}")
        ip = self_ip
    else:
        logger.info("No static IP address found, using dynamic IP address.")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(3)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]

    broadcast_ip = ".".join(ip.split(".")[:-1]) + ".255"
    logger.info(f"Broadcast address: {broadcast_ip}")

    return broadcast_ip


NETWORK_CONFIG: dict = {
    "ips": {
        "self": (__self_ip := os.environ.get("DEVICE_STATIC_IP")),
        "broadcast": _get_broadcast_addr(__self_ip),
    },
    "ports": {
        "uart": "/dev/ttyAMA0",
        "gstreamer": int(os.environ.get("GSTREAMER_PORT")),
        "zmq": int(os.environ.get("ZMQ_PORT")),
        "display_server": int(os.environ.get("DISPLAY_SERVER_PORT")),
    },
    "vars": {
        "handshake": os.environ.get("HANDSHAKE_SECRET", "default"),
        "cudacodec_enabled": os.environ.get("CUDA_CODEC_ENABLED", "false").lower()
        == "true",
    },
}
