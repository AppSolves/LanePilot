# src/common/logging.py
import logging
import logging.handlers
import os

from src.common.utils import Config, Singleton


class Logger(logging.Logger, metaclass=Singleton):
    """General logger class for this project."""

    def __init__(
        self, level: int = logging.DEBUG, create_log_file: bool = True
    ) -> None:
        """Initialize the logger.

        Args:
            level (int, optional): The logging level. Defaults to logging.DEBUG.
            create_log_file (bool, optional): Whether to create a log file and store it on your drive. Defaults to True.
        """
        project_name = Config.get("project_name", "Unknown")
        super().__init__(project_name, level)
        self.addHandler(logging.StreamHandler())
        if create_log_file:
            os.makedirs("logs", exist_ok=True)
            self.addHandler(
                logging.handlers.RotatingFileHandler(
                    os.path.join("logs", f"{project_name}.log"),
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )

        formatter = logging.Formatter(
            f"{project_name} :: " + "[%(levelname)s] - [%(asctime)s] --> %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        for handler in self.handlers:
            handler.setFormatter(formatter)

    @staticmethod
    def get_logger() -> "Logger":
        """Get the logger instance.

        Returns:
            Logger: The logger instance.
        """
        return Logger()
