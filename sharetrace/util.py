from __future__ import annotations

import datetime
import inspect
import logging
import os
import sys
import timeit
from typing import Any, Callable, Union

import numpy as np

TimeDelta = Union[datetime.timedelta, np.timedelta64]
DateTime = Union[datetime.datetime, np.datetime64]

NOW = round(datetime.datetime.utcnow().timestamp())

LOGGING_CONFIG = {
    'version': 1,
    'loggers': {
        'root': {
            'level': logging.INFO,
            'handlers': ['console', 'file']
        },
        'console': {
            'level': logging.INFO,
            'handlers': ['console'],
            'propagate': False
        },
        'file': {
            'level': logging.INFO,
            'handlers': ['file'],
            'propagate': False,
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': logging.INFO,
            'formatter': 'default',
            'stream': sys.stdout,
        },
        'file': {
            'class': 'sharetrace.util.RandomFileHandler',
            'level': logging.INFO,
            'formatter': 'default',
            'args': ('logs', 'a'),
        }
    },
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(module)s | %(message)s'
        }
    }
}


class RandomFileHandler(logging.FileHandler):

    def __init__(self, args):
        path, mode = args
        if not os.path.exists(path):
            os.mkdir(path)
        super().__init__(f'{path}//{NOW}.log', mode)


def time(func: Callable[[], Any]) -> Timer:
    return Timer.time(func)


class Timer:
    __slots__ = ('result', 'seconds')

    def __init__(self, result: Any, seconds: float):
        self.result = result
        self.seconds = seconds

    @classmethod
    def time(cls, func: Callable[[], Any]) -> Timer:
        start = timeit.default_timer()
        result = func()
        stop = timeit.default_timer()
        return Timer(result, stop - start)


def get_mb(obj):
    return get_bytes(obj) / 1e6


def get_bytes(obj, seen=None):
    """Recursively finds size of objects in bytes

    References:
        https://github.com/bosswissam/pysize/blob/master/pysize.py
    """

    if isinstance(obj, (np.ndarray, np.void)):
        return obj.nbytes

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                gs_descriptor = inspect.isgetsetdescriptor(d)
                m_descriptor = inspect.ismemberdescriptor(d)
                if gs_descriptor or m_descriptor:
                    size += get_bytes(obj.__dict__, seen)
                break
    any_str = isinstance(obj, (str, bytes, bytearray))
    if isinstance(obj, dict):
        size += sum((get_bytes(v, seen) for v in obj.values()))
        size += sum((get_bytes(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not any_str:
        size += sum((get_bytes(i, seen) for i in obj))
    if hasattr(obj, '__slots__'):
        size += sum(
            get_bytes(getattr(obj, s), seen)
            for s in obj.__slots__
            if hasattr(obj, s))
    return size
