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

LOGS_DIR = 'logs'
LOGGERS = (
    'contact-search:100-2900',
    'contact-search:3000-5400',
    'contact-search:5500-7400',
    'contact-search:7500-8900',
    'contact-search:9000-10000',
    'risk-propagation:serial',
    'risk-propagation:lewicki',
    'risk-propagation:ray',
    'risk-propagation:100-2900',
    'risk-propagation:3000-5400',
    'risk-propagation:5500-7400',
    'risk-propagation:7500-8900',
    'risk-propagation:9000-10000')


def logging_config():
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    config = {
        'version': 1,
        'loggers': {
            'root': {
                'level': logging.INFO,
                'handlers': ['console']
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': logging.INFO,
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
    for logger in LOGGERS:
        config['loggers'][logger] = {
            'level': logging.INFO,
            'handlers': [logger]}
        config['handlers'][logger] = {
            'class': 'logging.FileHandler',
            'level': logging.INFO,
            'formatter': 'default',
            'mode': 'a',
            'filename': f'{LOGS_DIR}//{logger}.log'}
    return config


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


def approx(val):
    return val if val is None else round(float(val), 4)
