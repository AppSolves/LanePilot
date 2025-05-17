import os

from ..common import get_logger
from .broadcasting import get_broadcast_addr

logger = get_logger()

NETWORK_CONFIG: dict = {
    "ips": {
        "self": os.environ.get("DEVICE_STATIC_IP"),
        "broadcast": get_broadcast_addr(),
    },
    "ports": {
        "uart": "/dev/ttyAMA0",
        "udp": int(os.environ.get("UDP_PORT")),
        "tcp": int(os.environ.get("TCP_PORT")),
    },
}
