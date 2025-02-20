# src/common/utils.py
import os
from threading import Lock

import yaml
from dotenv import load_dotenv


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


class Config(metaclass=Singleton):
    """Static class to store project configuration values as read-only properties."""

    __CONFIG = {}

    @staticmethod
    def _get_root_dir() -> str:
        """Get the root directory of the project."""
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    @classmethod
    def _reload_config(cls) -> None:
        """Load the configuration from a YAML file into the class variable."""
        if cls.__CONFIG:
            return

        load_dotenv()
        root_dir = cls._get_root_dir()
        config_file_path = os.path.join(root_dir, "project_config.yaml")

        with open(config_file_path, "r", encoding="utf-8") as file:
            cls.__CONFIG = yaml.safe_load(file)
            cls.__CONFIG["ROOT_DIR"] = root_dir

    @classmethod
    def get(cls, key: str, default=None):
        """Retrieve a config value like a dictionary."""
        return cls.__CONFIG.get(key, default)

    @classmethod
    def all(cls):
        """Return all configuration values."""
        return cls.__CONFIG.copy()

    def __getattr__(self, key):
        """Retrieve a config value like an attribute."""
        return self.get(key)

    def __setattr__(self, key, value):
        """Prevent modifications of config values."""
        raise AttributeError("Config properties are read-only!")

    def __delattr__(self, key):
        """Prevent modifications of config values."""
        raise AttributeError("Config properties are read-only!")


Config._reload_config()
