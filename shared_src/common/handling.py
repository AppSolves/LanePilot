import sys
import time
from typing import Callable

from .core import logger


def run_with_retry(
    func: Callable,
    *args,
    max_retries: int = 5,
    retry_delay_sec: int = 5,
    **kwargs,
):
    """
    Run a function with automatic retries on failure.
    This function will attempt to call the provided function up to `max_retries` times.
    If the function raises an exception, it will wait for `retry_delay_sec` seconds
    before retrying. If all retries fail, the program will exit with an error message.
    """
    logger.debug(f"Initializing function {func.__name__} with {max_retries} retries.")
    retries = 0
    while retries < max_retries:
        try:
            func(*args, **kwargs)
            break  # Exit if function returns normally
        except Exception as e:
            logger.error(
                f"Exception occured: {e}. Retrying in {retry_delay_sec} seconds..."
            )
            retries += 1
            time.sleep(retry_delay_sec)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Exiting...")
            sys.exit(0)
    else:
        logger.error("Max retries reached. Exiting.")
        sys.exit(1)
