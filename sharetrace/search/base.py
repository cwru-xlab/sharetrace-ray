import logging.config
from abc import ABC, abstractmethod
from collections import Iterable
from typing import Sequence, Tuple

import numpy as np

from sharetrace import logging_config
from sharetrace.util.types import TimeDelta

Histories = Sequence[np.ndarray]
Contacts = np.ndarray
Pairs = Iterable[Tuple[np.ndarray, np.ndarray]]

ZERO = np.timedelta64(0, 's')
logging.config.dictConfig(logging_config.config)


class BaseContactSearch(ABC):
    __slots__ = ('min_dur', 'n_workers', 'logger')

    def __init__(self, min_dur: TimeDelta = ZERO, n_workers: int = 1, **kwargs):
        super().__init__()
        self.min_dur = np.timedelta64(min_dur)
        self.n_workers = n_workers
        self.logger = logging.getLogger(__name__)
        self._log_params(min_dur=self.min_dur, n_workers=n_workers, **kwargs)

    def _log_params(self, **kwargs):
        self.logger.debug(
            'Parameters: %s',
            {str(k): str(v) for k, v in kwargs.items()})

    @abstractmethod
    def search(self, histories: Histories) -> Contacts:
        pass

    @abstractmethod
    def pairs(self, histories: Histories) -> Pairs:
        pass
