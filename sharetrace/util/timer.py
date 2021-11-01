from __future__ import annotations

import datetime
from typing import Any, Callable


class Timer:
    __slots__ = ('result', 'duration')

    def __init__(self, result: Any, duration: datetime.timedelta):
        self.result = result
        self.duration = duration

    @classmethod
    def time(cls, func: Callable) -> Timer:
        start = datetime.datetime.now()
        result = func()
        stop = datetime.datetime.now()
        return Timer(result, stop - start)
