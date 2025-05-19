import subprocess as sp

from .core import logger


# TODO: Currently only implemented for software encoding. Implement hardware encoding in the future.
def run_gstreamer_caller(peer_ip: str, udp_port: int, bitrate: int = 2_000_000):
    """
    Run GStreamer pipeline to stream video from the Raspberry Pi camera to a peer using SRT.

    Args:
        peer_ip (str): The IP address of the peer to stream to.
        udp_port (int): The UDP port to use for streaming.
    """

    logger.info(f"Starting GStreamer pipeline to {peer_ip}:{udp_port}")
    encoder = "x264enc"
    try:
        return sp.run(
            [
                "gst-launch-1.0",
                "libcamerasrc",
                "!",
                "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1",
                "!",
                "videoconvert",
                "!",
                encoder,
                "tune=zerolatency",
                f"bitrate={bitrate}",
                "speed-preset=ultrafast",
                "!",
                "video/x-h264,profile=main",
                "!",
                "queue",
                "!",
                "mpegtsmux",
                "!",
                "srtsink",
                f'uri="srt://{peer_ip}:{udp_port}?mode=caller&latency=1"',
            ],
            check=True,
        )
    except (sp.TimeoutExpired, sp.CalledProcessError) as e:
        logger.error(f"GStreamer process timed out or failed: {e}")
        raise RuntimeError(f"GStreamer process failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        logger.info("GStreamer pipeline terminated.")
