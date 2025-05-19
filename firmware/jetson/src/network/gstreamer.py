from typing import Callable

import cv2

from shared_src.common import Singleton, StoppableThread

from .core import logger


class GStreamerReceiver(StoppableThread, metaclass=Singleton):
    """
    A class that receives GStreamer data and provides a generator for frames.
    """

    __listeners: list[Callable] = []

    def __init__(self, pipeline: str, *args, **kwargs) -> None:
        """Initialize the GStreamer receiver.
        Args:
            pipeline (str): The GStreamer pipeline to use.
        """
        super().__init__(*args, **kwargs)
        self._pipeline = pipeline
        timeout = kwargs.get("timeout", 3000)
        self._cap = cv2.VideoCapture()
        self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout)
        self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout)
        self._cap.open(self._pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            logger.error("Failed to open video stream")
            self._cap = None

    def add_listener(self, listener: Callable) -> None:
        """Add a listener, which receives commands on each event.

        Args:
            listener: The listener to add.
        """
        self.__listeners.append(listener)

    def remove_listener(self, listener: Callable) -> None:
        """Remove a listener.

        Args:
            listener: The listener to remove.
        """
        self.__listeners.remove(listener)

    def run_with_exception_handling(self) -> None:
        try:
            for frame in self.frames:
                for listener in self.__listeners:
                    listener(frame)
        except Exception as e:
            logger.error(f"GStreamerReceiver encountered an error: {e}")
            raise  # Propagate error for reconnect logic
        finally:
            self.dispose()

    @property
    def frames(self):
        """
        A generator that yields frames from the GStreamer pipeline.
        """
        if not self._cap:
            logger.error("Video stream is not initialized")
            raise ConnectionError("Video stream is not initialized")

        while True:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("No frame received")
                raise ConnectionError("Lost connection to video stream")
            yield frame

    def dispose(self):
        """
        Release the video capture resource.
        """
        self.stop()
        self.__listeners.clear()
        if self._cap:
            self._cap.release()

        logger.info("GStreamerReceiver disposed")
