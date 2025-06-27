import inspect
import logging
import logging.handlers
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from .utils import Config


def python_to_gst_level(py_level: int) -> int:
    if py_level >= 50:  # CRITICAL
        return 1  # ERROR
    elif py_level >= 40:
        return 1  # ERROR
    elif py_level >= 30:
        return 2  # WARNING
    elif py_level >= 20:
        return 4  # INFO
    elif py_level >= 10:
        return 5  # DEBUG
    elif py_level > 0:
        return 7  # TRACE
    else:
        return 0  # none


def python_to_trt_level(py_level: int) -> int:
    if py_level >= 50:  # CRITICAL
        return 1
    elif py_level >= 40:
        return 1
    elif py_level >= 30:
        return 2
    elif py_level >= 20:
        return 3
    elif py_level >= 10:
        return 4
    elif py_level > 0:
        return 5
    else:
        return 0


class FormattingPurpose(Enum):
    DEFAULT = "default"
    FILE = "file"

    def __str__(self):
        return self.value


def get_formatter(
    purpose: FormattingPurpose = FormattingPurpose.DEFAULT,
    module: str = str(Config.get("project_name", "Unknown")),
) -> logging.Formatter:
    """Get a logging formatter with a specific format."""
    match purpose:
        case FormattingPurpose.DEFAULT:
            return logging.Formatter(
                f"{module} :: " + "[%(levelname)s] - [%(asctime)s] --> %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        case FormattingPurpose.FILE:
            return logging.Formatter(
                "[%(levelname)s] - [%(asctime)s] --> %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        case _:
            options = ", ".join(FormattingPurpose._member_names_)
            raise ValueError(
                f"Unknown formatter purpose: {purpose}. "
                f"Available options are: {options}."
            )


def get_logger(
    level: Optional[int | str] = None,
    create_log_file: bool = True,
) -> logging.Logger:
    """Initialize the logger.

    Args:
        level (int, optional): The logging level. Defaults to logging.DEBUG.
        create_log_file (bool, optional): Whether to create a log file and store it on your drive. Defaults to True.

    Returns:
        logging.Logger: The logger object.
    """
    project_name = str(Config.get("project_name", "Unknown"))
    root_dir = Path(str(Config.get("ROOT_DIR")))

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
    try:
        logger.setLevel(level)
    except ValueError:
        # If the level is not a valid logging level, set it to INFO
        logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

        if create_log_file:
            log_dir = root_dir / "runtime" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.addHandler(
                logging.handlers.RotatingFileHandler(
                    log_dir / f"{module}.log",
                    maxBytes=20 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(
                get_formatter(purpose=FormattingPurpose.FILE, module=module)
            )
        else:
            handler.setFormatter(
                get_formatter(purpose=FormattingPurpose.DEFAULT, module=module)
            )

    return logger


# Check if _ROOT_LOGGER is already set
if "_ROOT_LOGGER" not in locals():
    _ROOT_LOGGER: logging.Logger = get_logger(create_log_file=False)
    IS_DEBUG: bool = _ROOT_LOGGER.level == logging.DEBUG

if not os.environ.get("GST_DEBUG"):
    os.environ["GST_DEBUG"] = str(python_to_gst_level(_ROOT_LOGGER.level))
