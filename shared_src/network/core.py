import os

from shared_src.common import get_logger

logger = get_logger()

PORTS: dict = {
    "uart": "/dev/ttyAMA0",
    "udp": int(os.environ.get("UDP_PORT")),
    "tcp": int(os.environ.get("TCP_PORT")),
}
