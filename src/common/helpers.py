import os

import yaml


def Singleton(cls):
    __instance__ = None

    def __get_instance__(*args, **kwargs):
        nonlocal __instance__
        if __instance__ is None:
            __instance__ = cls(*args, **kwargs)
        return __instance__

    return __get_instance__


def setup_project_environment():
    global _THIS_DIR, CONFIG_FILE, PROJECT_NAME

    if "PROJECT_NAME" in globals():
        return

    _THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CONFIG_FILE: str = os.path.join(_THIS_DIR, "config.yaml")

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    PROJECT_NAME: str = config["project_name"]


setup_project_environment()
