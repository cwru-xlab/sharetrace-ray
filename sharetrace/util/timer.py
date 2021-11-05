from __future__ import annotations

from timeit import default_timer
from typing import Any, Callable


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
