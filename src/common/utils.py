# src/common/utils.py
import os

import yaml


def Singleton(cls):
    """Singleton decorator. Use this decorator to ensure that only one instance of a class is created.

    Returns:
        cls: The class instance.
    """
    __instance__ = None

    def __get_instance__(*args, **kwargs):
        nonlocal __instance__
        if __instance__ is None:
            __instance__ = cls(*args, **kwargs)
        return __instance__

    return __get_instance__


def _get_root_dir() -> str:
    """Get the root directory of the project.

    Returns:
        str: The root directory of the project.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_project_environment() -> None:
    """Setup the project environment by reading the `config.yaml` file and setting the project name as a global variable."""
    global ROOT_DIR, CONFIG_FILE, PROJECT_NAME

    if "PROJECT_NAME" in globals():
        return

    ROOT_DIR = _get_root_dir()
    CONFIG_FILE = os.path.join(ROOT_DIR, "project_config.yaml")

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    PROJECT_NAME = config["project_name"]


_setup_project_environment()
