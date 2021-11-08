from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from timeit import default_timer
from typing import Any, Callable, Union

from numpy import datetime64, timedelta64

TimeDelta = Union[timedelta, timedelta64]
DateTime = Union[datetime, datetime64]

LOGGING_CONFIG = {
    'version': 1,
    'loggers': {
        'root': {
            'level': logging.DEBUG,
            'handlers': ['console']
        },
        'console': {
            'level': logging.DEBUG,
            'handlers': ['console'],
            'propagate': False
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': logging.DEBUG,
            'formatter': 'default',
            'stream': sys.stdout,
        }
    },
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(module)s | %(message)s'
        }
    }
}


class Timer:
    __slots__ = ('result', 'seconds')

    def __init__(self, result: Any, seconds: float):
        self.result = result
        self.seconds = seconds

    @classmethod
    def time(cls, func: Callable) -> Timer:
        start = default_timer()
        result = func()
        stop = default_timer()
        return Timer(result, stop - start)
