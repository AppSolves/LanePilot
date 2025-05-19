import signal

from shared_src.common import run_with_retry
from shared_src.network import NETWORK_CONFIG, ServerClient, discover_peer

from .hardware_control import MODULE_CONFIG, ServoManager
from .network import GStreamerSender, logger


def start_network(tcp_port: int, udp_port: int) -> None:
    peer_ip = discover_peer(timeout=10, port=udp_port)
    if peer_ip is None:
        logger.error("No peer found, exiting.")
        raise RuntimeError("No peer found")

    server_thread = ServerClient(tcp_port, daemon=True)
    gstreamer_thread = GStreamerSender(peer_ip=peer_ip, udp_port=udp_port, daemon=True)
    server_thread.start()
    gstreamer_thread.start()

    servo_config = MODULE_CONFIG.get("servos", {})
    servo_manager = ServoManager(
        port=servo_config.get("uart_port"), baudrate=servo_config.get("baudrate")
    )
    server_thread.add_listener(servo_manager.on_event)

    signal.signal(signal.SIGTERM, lambda _, __: server_thread.stop())
    gstreamer_thread.join()
    server_thread.stop()
    servo_manager.dispose()

    # After joining, check for exceptions
    for t in [server_thread, gstreamer_thread]:
        if hasattr(t, "exception") and t.exception:
            raise t.exception


if __name__ == "__main__":
    run_with_retry(
        start_network,
        NETWORK_CONFIG["ports"].get("tcp"),
        NETWORK_CONFIG["ports"].get("udp"),
    )
