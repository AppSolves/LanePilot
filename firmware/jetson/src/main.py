import signal

from shared_src.common import run_with_retry, stop_threads
from shared_src.network import NETWORK_CONFIG, ServerClient, respond_to_broadcast

from .network import GStreamerReceiver, logger


def start_network(tcp_port: int, udp_port: int, nvidia_backend: bool = False) -> None:
    peer_ip = respond_to_broadcast(port=udp_port, stop_on_response=True)
    if peer_ip is None:
        logger.error("No peer found, exiting.")
        raise RuntimeError("No peer found")

    server_thread = ServerClient(
        tcp_port, is_server=False, server_ip=peer_ip, daemon=True
    )
    server_thread.start()

    decoder = "nvh264dec" if nvidia_backend else "avdec_h264"
    gstreamer_thread = GStreamerReceiver(
        f'srtsrc uri="srt://0.0.0.0:{udp_port}?mode=listener&latency=1" ! queue ! tsdemux ! h264parse ! {decoder} ! videoconvert ! appsink sync=false',
        daemon=True,
    )
    gstreamer_thread.start()

    signal.signal(
        signal.SIGTERM, lambda _, __: stop_threads([gstreamer_thread, server_thread])
    )
    gstreamer_thread.join()
    stop_threads([gstreamer_thread, server_thread])

    # After joining, check for exceptions
    for t in [server_thread, gstreamer_thread]:
        if hasattr(t, "exception") and t.exception:
            raise t.exception


if __name__ == "__main__":
    run_with_retry(
        start_network,
        NETWORK_CONFIG["ports"].get("tcp"),
        NETWORK_CONFIG["ports"].get("udp"),
        NETWORK_CONFIG["vars"].get("cudacodec_enabled"),
    )
