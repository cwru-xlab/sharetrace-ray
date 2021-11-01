import functools
from logging import Logger
from typing import Iterable, NoReturn

import numpy as np

from sharetrace.search.base import BaseContactSearch
from sharetrace.util.timer import Timer


class TimedContactSearch(BaseContactSearch):
    __slots__ = ('wrapped', 'logger')

    def __init__(self, wrapped: BaseContactSearch, logger: Logger):
        super().__init__()
        self.wrapped = wrapped
        self.logger = logger

    def search(self, histories: np.ndarray) -> np.ndarray:
        search = functools.partial(lambda: self.wrapped.search(histories))
        timed = Timer.time(search)
        self._log_duration(timed.duration.seconds)
        return timed.result

    def _log_duration(self, duration: float) -> NoReturn:
        self.logger.info('Contact search: %.2f seconds', duration)

    def pairs(self, histories: np.ndarray) -> Iterable:
        return self.wrapped.pairs(histories)
