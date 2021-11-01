from abc import ABC, abstractmethod
from collections import Iterable
from typing import Tuple

import numpy as np

from sharetrace.util.types import TimeDelta

Pairs = Iterable[Tuple[np.ndarray, np.ndarray]]
ZERO_SECONDS = np.timedelta64(0, 's')


class BaseContactSearch(ABC):
    __slots__ = ('min_duration', 'n_workers')

    def __init__(
            self,
            min_duration: TimeDelta = ZERO_SECONDS,
            n_workers: int = 1):
        super().__init__()
        self.min_duration = np.datetime64(min_duration)
        self.n_workers = n_workers

    @abstractmethod
    def search(self, histories: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def pairs(self, histories: np.ndarray) -> Iterable:
        pass
