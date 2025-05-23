import atexit
import signal
from pathlib import Path

from firmware.jetson.src.ai_inference import ModelPipeline
from firmware.jetson.src.ai_inference.gat_inference import GATInference
from firmware.jetson.src.ai_inference.yolo_inference import YOLOInference
from shared_src.common import Config, StoppableThread, run_with_retry, stop_threads
from shared_src.network import NETWORK_CONFIG, ServerClient, respond_to_broadcast

from .network import GStreamerReceiver, logger


def start_network(
    zmq_port: int, gstreamer_port: int, nvidia_backend: bool = False
) -> None:
    """
    Start the network components.
    Args:
        zmq_port (int): The port for the ZeroMQ server.
        gstreamer_port (int): The port for the GStreamer server.
        nvidia_backend (bool): Flag to use NVIDIA backend for GStreamer.
    """
    logger.info("Starting network components...")
    peer_ip = respond_to_broadcast(port=gstreamer_port, stop_on_response=True)
    if peer_ip is None:
        logger.error("No peer found, exiting.")
        raise RuntimeError("No peer found")

    server_thread = ServerClient(
        zmq_port, is_server=False, server_ip=peer_ip, daemon=True
    )
    server_thread.start()

    decoder = "nvh264dec" if nvidia_backend else "avdec_h264"
    gstreamer_thread = GStreamerReceiver(
        f'srtsrc uri="srt://0.0.0.0:{gstreamer_port}?mode=listener&latency=1" ! queue ! tsdemux ! h264parse ! {decoder} ! videoconvert ! appsink sync=false',
        daemon=True,
    )
    gstreamer_thread.start()

    model_paths = Path(Config.get("ROOT_DIR"), "models")
    pipeline = ModelPipeline(
        [
            YOLOInference(
                Path(model_paths, "vehicle_detection", "vehicle_detection.engine"),
                return_tensors=True,
            ),
            GATInference(
                Path(model_paths, "lane_allocation", "lane_allocation.engine"),
                enable_host_code=True,
            ),
        ]
    )
    gstreamer_thread.add_listener(pipeline)

    threads: tuple[StoppableThread, ...] = (server_thread, gstreamer_thread)
    atexit.register(pipeline.dispose)
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
        NETWORK_CONFIG["vars"].get("cudacodec_enabled"),
    )
