import signal
import sys

from shared_src.common import StoppableThread
from shared_src.network import NETWORK_CONFIG, ServerClient, respond_to_broadcast

from .network.core import logger
from .network.gstreamer import GStreamerReceiver


def stop_threads(threads: list[StoppableThread]) -> None:
    """
    Stops all threads in the provided list.
    """
    for thread in threads:
        thread.stop()
        thread.join()


def start_network(tcp_port: int, udp_port: int) -> None:
    peer_ip = respond_to_broadcast(port=udp_port, stop_on_response=True)
    if peer_ip is None:
        logger.error("No peer found, exiting.")
        sys.exit(1)

    server_thread = ServerClient(
        tcp_port, is_server=False, server_ip=peer_ip, daemon=True
    )
    gstreamer_thread = GStreamerReceiver(
        f'srtsrc uri="srt://0.0.0.0:{udp_port}?mode=listener&latency=1" ! queue ! tsdemux ! h264parse ! nvh264dec ! videoconvert ! appsink sync=false',
        daemon=True,
    )
    server_thread.start()
    gstreamer_thread.start()

    signal.signal(
        signal.SIGTERM, lambda _, __: stop_threads([server_thread, gstreamer_thread])
    )
    server_thread.join()
    gstreamer_thread.join()


if __name__ == "__main__":
    start_network(
        NETWORK_CONFIG["ports"].get("tcp"),
        NETWORK_CONFIG["ports"].get("udp"),
    )
