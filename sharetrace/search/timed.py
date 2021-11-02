import datetime
import functools
import logging
from logging import Logger
from typing import Iterable, NoReturn

import numpy as np

from sharetrace import logging_config
from sharetrace.search.base import BaseContactSearch, Contacts, Histories
from sharetrace.util.timer import Timer

from logging import config

config.dictConfig(logging_config.config)
logger = logging.getLogger(__name__)


class TimedContactSearch(BaseContactSearch):
    __slots__ = ('wrapped',)

    def __init__(self, wrapped: BaseContactSearch):
        super().__init__()
        self.wrapped = wrapped

    def search(self, histories: Histories) -> Contacts:
        search = functools.partial(lambda: self.wrapped.search(histories))
        timed = Timer.time(search)
        self._log_duration(timed.duration.microseconds / 1e6)
        return timed.result

    @staticmethod
    def _log_duration(secs: float) -> NoReturn:
        logger.info('Contact search: %.3f seconds', secs)

    def pairs(self, histories: np.ndarray) -> Iterable:
        return self.wrapped.pairs(histories)
