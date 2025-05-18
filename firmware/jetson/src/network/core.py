import os

from shared_src.common import get_logger

logger = get_logger()
os.environ["GST_DEBUG"] = str(logger.level)
