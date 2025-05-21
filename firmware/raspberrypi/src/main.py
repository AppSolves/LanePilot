import signal

from shared_src.common import StoppableThread, run_with_retry, stop_threads
from shared_src.network import NETWORK_CONFIG, ServerClient, discover_peer

from .hardware_control import MODULE_CONFIG, ServoManager
from .network import DisplayServer, GStreamerSender, logger


def start_network(zmq_port: int, gstreamer_port: int, display_server_port: int) -> None:
    """
    Start the network components.
    Args:
        zmq_port (int): The port for the ZeroMQ server.
        gstreamer_port (int): The port for the GStreamer server.
        display_server_port (int): The port for the display server.
    """
    logger.info("Starting network components...")
    peer_ip = discover_peer(timeout=10, port=gstreamer_port)
    if peer_ip is None:
        logger.error("No peer found, exiting.")
        raise RuntimeError("No peer found")

    server_thread = ServerClient(zmq_port, daemon=True)
    server_thread.start()

    gstreamer_thread = GStreamerSender(
        peer_ip=peer_ip, port=gstreamer_port, daemon=True
    )
    gstreamer_thread.start()

    display_server_thread = DisplayServer(
        port=display_server_port,
        daemon=True,
    )
    display_server_thread.start()
    server_thread.add_listener(display_server_thread.on_event)

    servo_manager_thread = ServoManager()
    servo_manager_thread.start()
    server_thread.add_listener(servo_manager_thread.on_event)

    threads: tuple[StoppableThread, ...] = (
        server_thread,
        gstreamer_thread,
        display_server_thread,
        servo_manager_thread,
    )
    signal.signal(signal.SIGTERM, lambda _, __: stop_threads(threads))
    gstreamer_thread.join()
    stop_threads(threads)

    # After joining, check for exceptions
    for t in threads:
        if t.exception:
            raise t.exception


if __name__ == "__main__":
    run_with_retry(
        start_network,
        NETWORK_CONFIG["ports"].get("zmq"),
        NETWORK_CONFIG["ports"].get("gstreamer"),
    )
