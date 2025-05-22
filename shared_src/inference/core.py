import os
from pathlib import Path

from shared_src.common import Config, get_logger

CONFIG_FILE: Path = Path(Path(__file__).parent, "config.yaml").resolve()
logger = get_logger()

if not os.path.exists(CONFIG_FILE):
    logger.error(f"Config file not found: {CONFIG_FILE}")
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

MODULE_CONFIG = Config.load_config_file(CONFIG_FILE)
