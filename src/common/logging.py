# src/common/logging.py
import inspect
import logging
import logging.handlers
import os
from typing import Optional

from src.common.utils import Config


def get_logger(
    level: Optional[int | str] = None,
    include_relpath: bool = False,
    create_log_file: bool = True,
) -> logging.Logger:
    """Initialize the logger.

    Args:
        level (int, optional): The logging level. Defaults to logging.DEBUG.
        include_relpath (bool, optional): Whether to include the relative path of the file that called the logger (CAUTION: Can introduce a delay when called often). Defaults to False.
        create_log_file (bool, optional): Whether to create a log file and store it on your drive. Defaults to True.

    Returns:
        logging.Logger: The logger object.
    """

    project_name = Config.get("project_name", "Unknown")
    logging_env_name = Config.get("environment_variables", {}).get(
        "logging_level", "LANEPILOT_LOG_LEVEL"
    )
    level = level or os.environ.get(logging_env_name, logging.DEBUG)

    logger = logging.getLogger(project_name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

        if create_log_file:
            os.makedirs("logs", exist_ok=True)
            logger.addHandler(
                logging.handlers.RotatingFileHandler(
                    os.path.join("logs", f"{project_name}.log"),
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )

    sub_path = ""
    if include_relpath:
        frm = inspect.stack()[1]
        sub_path = os.path.relpath(frm.filename, Config.get("project_root"))
        sub_path = sub_path.replace("\\", "/")
        sub_path = "/" + sub_path

    formatter = logging.Formatter(
        f"{project_name}{sub_path} :: "
        + "[%(levelname)s] - [%(asctime)s] --> %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger
