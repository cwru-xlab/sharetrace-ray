from abc import ABC, abstractmethod
from logging import getLogger
from logging.config import dictConfig
from typing import Iterable, Sequence, Tuple

from numpy import ndarray, timedelta64

from sharetrace import logging_config
from sharetrace.util.types import TimeDelta

Histories = Sequence[ndarray]
Contacts = ndarray
Pairs = Iterable[Tuple[ndarray, ndarray]]

ZERO = timedelta64(0, 's')
dictConfig(logging_config.config)


class BaseContactSearch(ABC):
    __slots__ = ('min_dur', 'workers', '_logger')

    def __init__(self, min_dur: TimeDelta = ZERO, workers: int = 1, **kwargs):
        super().__init__()
        self.min_dur = timedelta64(min_dur)
        self.workers = workers
        self._logger = getLogger(__name__)
        self.log_params(min_dur=self.min_dur, workers=workers, **kwargs)

    def log_params(self, **kwargs):
        self._logger.debug(
            '%s parameters: %s',
            self.__class__.__name__,
            {str(k): str(v) for k, v in kwargs.items()})

    @abstractmethod
    def search(self, histories: Histories) -> Contacts:
        pass

    @abstractmethod
    def pairs(self, histories: Histories) -> Pairs:
        pass
