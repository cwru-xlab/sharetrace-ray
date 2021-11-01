import logging
import sys
from typing import Union


def get_stdout_logger(name: str, level: Union[int, str] = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger
