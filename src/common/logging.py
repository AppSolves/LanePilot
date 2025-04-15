import inspect
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from src.common.utils import Config


def get_logger(
    level: Optional[int | str] = None,
    create_log_file: bool = True,
) -> logging.Logger:
    """Initialize the LanePilot logger.

    Args:
        level (int, optional): The logging level. Defaults to logging.DEBUG.
        create_log_file (bool, optional): Whether to create a log file and store it on your drive. Defaults to True.

    Returns:
        logging.Logger: The logger object.
    """

    project_name = Config.get("project_name", "Unknown")
    root_dir = Path(Config.get("ROOT_DIR", os.getcwd()))

    env_vars = Config.get("environment_variables", {})
    logging_env_name = env_vars.get("logging_level", None)
    logging_include_relpaths_name = env_vars.get("logging_include_relpaths", None)

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
                    Path(root_dir, "logs", f"{project_name}.log"),
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )

    sub_path = ""
    if os.environ.get(logging_include_relpaths_name, "false").lower() == "true":
        frm = inspect.stack()[1]
        try:
            sub_path = Path(frm.filename).relative_to(root_dir)
        except ValueError:
            sub_path = Path(frm.filename).relative_to(Path.cwd())
        sub_path = sub_path.as_posix()
        sub_path = "/" + sub_path

    formatter = logging.Formatter(
        f"{project_name}{sub_path} :: "
        + "[%(levelname)s] - [%(asctime)s] --> %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger
