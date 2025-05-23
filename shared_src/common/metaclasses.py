from threading import Lock


class Singleton(type):
    """A thread-safe implementation of Singleton using a metaclass."""

    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Final(type):
    """A metaclass that prevents subclassing of the class it creates."""

    def __init__(cls, name, bases, namespace):
        # Check if any base class is marked as final
        for base in bases:
            if isinstance(base, Final):
                raise TypeError(f"Cannot inherit from final class {base.__name__}")
        super().__init__(name, bases, namespace)


class FinalSingleton(Singleton, Final):
    """A thread-safe implementation of Singleton with final class enforcement."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
