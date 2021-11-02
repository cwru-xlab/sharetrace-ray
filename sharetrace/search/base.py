from abc import ABC, abstractmethod
from collections import Iterable
from typing import Sequence, Tuple

import numpy as np

from sharetrace.util.types import TimeDelta

Histories = Sequence[np.ndarray]
Contacts = np.ndarray
Pairs = Iterable[Tuple[np.ndarray, np.ndarray]]

ZERO = np.timedelta64(0, 's')


class BaseContactSearch(ABC):
    __slots__ = ('min_dur', 'n_workers')

    def __init__(self, min_dur: TimeDelta = ZERO, n_workers: int = 1, **kwargs):
        super().__init__()
        self.min_dur = np.timedelta64(min_dur)
        self.n_workers = n_workers

    @abstractmethod
    def search(self, histories: Histories) -> Contacts:
        pass

    @abstractmethod
    def pairs(self, histories: Histories) -> Pairs:
        pass
