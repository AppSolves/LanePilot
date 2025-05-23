import hashlib
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from .metaclasses import Singleton


class Config(metaclass=Singleton):
    """Static class to store project configuration values as read-only properties."""

    __CONFIG = {}
    MAPPING = {}

    @staticmethod
    def _get_root_dir() -> Path:
        """Get the root directory of the project."""
        return Path(__file__).resolve().parent.parent.parent

    @staticmethod
    def translate(content: str, mapping: dict) -> str:
        """Translate placeholders in the content using the provided mapping."""
        for key, value in mapping.items():
            content = content.replace(key, value)
        return content

    @classmethod
    def load_config_file(cls, config_file: Path) -> dict:
        """Load a YAML configuration file."""
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_str = cls.translate(f.read(), cls.MAPPING)
            config = yaml.safe_load(config_str)

        if not config:
            raise ValueError(f"Config file is empty: {config_file}")

        return config

    @classmethod
    def _reload_global_config(cls) -> None:
        """Load the configuration from a YAML file into the class variable."""
        if cls.__CONFIG:
            return

        load_dotenv()
        root_dir = cls._get_root_dir()
        cls.MAPPING = {
            "$ROOT_DIR": root_dir.as_posix(),
        }

        config_file_path = Path(root_dir, "project_config.yaml")
        cls.__CONFIG = cls.load_config_file(config_file_path)
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


def get_file_hash(file_path: Path, hash_algo: str = "sha256"):
    hash_func = hashlib.new(hash_algo)

    with open(file_path, "rb") as file:
        # Read the file in chunks to handle large files
        chunk = file.read(8192)
        while chunk:
            hash_func.update(chunk)
            chunk = file.read(8192)

    # Return the hex representation of the hash
    return hash_func.hexdigest()


def get_parent_class(func) -> Optional[str]:
    if not hasattr(func, "__qualname__"):
        return None
    qualname = func.__qualname__
    parts = qualname.split(".")
    if len(parts) > 1:
        return parts[-2]  # This is likely the class name
    return None


Config._reload_global_config()
