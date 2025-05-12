import threading


class StoppableThread(threading.Thread):
    """A thread that can be stopped."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__stop_event = threading.Event()

    def stop(self):
        """Stop the thread."""
        self.__stop_event.set()

    @property
    def running(self):
        """Check if the thread is running."""
        return not self.__stop_event.is_set()
