import subprocess as sp

from shared_src.common import StoppableThread

from .core import logger


class GStreamerSender(StoppableThread):
    def __init__(
        self,
        peer_ip: str,
        udp_port: int,
        bitrate: int = 2_000_000,
        encoder: str = "x264enc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._peer_ip = peer_ip
        self._udp_port = udp_port
        self._bitrate = bitrate
        self._encoder = encoder

        # TODO: Currently only implemented for software encoding. Implement hardware encoding in the future.
        assert self._encoder == "x264enc"

    def run_with_exception_handling(self):
        """
        Run GStreamer pipeline to stream video from the Raspberry Pi camera to a peer using SRT.
        """

        logger.info(f"Starting GStreamer pipeline to {self._peer_ip}:{self._udp_port}")
        try:
            sp.run(
                [
                    "gst-launch-1.0",
                    "libcamerasrc",
                    "!",
                    "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1",
                    "!",
                    "videoconvert",
                    "!",
                    self._encoder,
                    "tune=zerolatency",
                    f"bitrate={self._bitrate}",
                    "speed-preset=ultrafast",
                    "!",
                    "video/x-h264,profile=main",
                    "!",
                    "queue",
                    "!",
                    "mpegtsmux",
                    "!",
                    "srtsink",
                    f'uri="srt://{self._peer_ip}:{self._udp_port}?mode=caller&latency=1"',
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
