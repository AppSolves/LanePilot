import logging
import logging.handlers
import os

from src.common.helpers import PROJECT_NAME, Singleton


@Singleton
class Logger(logging.Logger):
    def __init__(
        self, level: int = logging.DEBUG, create_log_file: bool = True
    ) -> None:
        super().__init__(PROJECT_NAME, level)
        self.addHandler(logging.StreamHandler())
        if create_log_file:
            os.makedirs("logs", exist_ok=True)
            self.addHandler(
                logging.handlers.RotatingFileHandler(
                    os.path.join("logs", f"{PROJECT_NAME}.log"),
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )

        formatter = logging.Formatter(
            f"{PROJECT_NAME} :: " + "[%(levelname)s] - [%(asctime)s] --> %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        for handler in self.handlers:
            handler.setFormatter(formatter)
