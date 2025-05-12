import inspect
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .utils import Config


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
    root_dir = Path(Config.get("ROOT_DIR"))

    env_vars = Config.get("environment_variables", {}).get("logging", {})

    module = project_name
    if env_vars.get("include_relative_paths", False):
        frm = inspect.stack()[1]
        try:
            sub_module = Path(frm.filename).relative_to(root_dir)
        except ValueError:
            sub_module = Path(frm.filename).relative_to(Path.cwd())
        sub_module = sub_module.as_posix().split(".")[0].replace("/", ".")
        module += f".{sub_module}"

    level = level or env_vars.get("level", logging.DEBUG)
    logger = logging.getLogger(module)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

        if create_log_file:
            log_dir = Path(root_dir, "runtime", "logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.addHandler(
                logging.handlers.RotatingFileHandler(
                    Path(log_dir, f"{module}.log"),
                    maxBytes=20 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )

    formatter = logging.Formatter(
        f"{module} :: " + "[%(levelname)s] - [%(asctime)s] --> %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger
