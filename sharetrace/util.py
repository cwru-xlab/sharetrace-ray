from __future__ import annotations

import inspect
import sys
import timeit
from typing import Any, Callable, Optional

import numpy as np


def time(func: Callable[[], Any]) -> Timer:
    return Timer.time(func)


class Timer:
    __slots__ = ("result", "seconds")

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
    if hasattr(obj, "__dict__"):
        for cls in obj.__class__.__mro__:
            if "__dict__" in cls.__dict__:
                d = cls.__dict__["__dict__"]
                gs_descriptor = inspect.isgetsetdescriptor(d)
                m_descriptor = inspect.ismemberdescriptor(d)
                if gs_descriptor or m_descriptor:
                    size += get_bytes(obj.__dict__, seen)
                break
    any_str = isinstance(obj, (str, bytes, bytearray))
    if isinstance(obj, dict):
        size += sum((get_bytes(v, seen) for v in obj.values()))
        size += sum((get_bytes(k, seen) for k in obj.keys()))
    elif hasattr(obj, "__iter__") and not any_str:
        size += sum((get_bytes(i, seen) for i in obj))
    if hasattr(obj, "__slots__"):
        size += sum(
            get_bytes(getattr(obj, s), seen)
            for s in obj.__slots__
            if hasattr(obj, s))
    return size


def approx(val: Optional, prec=4) -> Optional[float]:
    """Approximates the value to given precision."""
    return val if val is None else round(float(val), prec)


def sdiv(a: float, b: float) -> float:
    """Returns the quotient of a and b, or 0 if b is 0."""
    return 0 if b == 0 else a / b
