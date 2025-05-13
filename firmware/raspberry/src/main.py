import signal
import sys

from shared_src.common import StoppableThread
from shared_src.network import PORTS, ServerClient, discover_peer

from .hardware_control import MODULE_CONFIG, ServoManager
from .network import run_gstreamer_caller
from .network.core import logger


def start_network(tcp_port: int, udp_port: int) -> None:
    peer_ip = discover_peer(timeout=10, port=udp_port)
    if peer_ip is None:
        logger.error("No peer found, exiting.")
        sys.exit(1)

    server_thread = ServerClient(tcp_port)
    gstreamer_thread = StoppableThread(
        target=run_gstreamer_caller, args=(peer_ip, udp_port), daemon=True
    )
    server_thread.start()
    gstreamer_thread.start()

    servo_config = MODULE_CONFIG.get("servos", {})
    servo_manager = ServoManager(
        port=servo_config.get("uart_port"), baudrate=servo_config.get("baudrate")
    )
    server_thread.add_listener(servo_manager.on_event)

    signal.signal(signal.SIGTERM, lambda _, __: server_thread.stop())
    server_thread.join()
    servo_manager.dispose()
    gstreamer_thread.stop()


if __name__ == "__main__":
    start_network(PORTS["tcp"], PORTS["udp"])
