import threading
from typing import Iterable, final


class StoppableThread(threading.Thread):
    """A thread that can be stopped."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__stop_event = threading.Event()
        self.exception = None

    @final
    def stop(self):
        """Stop the thread."""
        self.__stop_event.set()

    @property
    def running(self):
        """Check if the thread is running."""
        return not self.__stop_event.is_set()

    @final
    def run(self):
        try:
            self.run_with_exception_handling()
        except Exception as e:
            self.exception = e

    def run_with_exception_handling(self):
        """Override this in subclasses instead of run()."""
        pass


def stop_threads(threads: Iterable[StoppableThread]) -> None:
    """
    Stops all threads in the provided list.
    """
    for thread in threads:
        try:
            thread.dispose()
        except:
            thread.stop()
        thread.join()
