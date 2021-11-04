from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from sharetrace import model
from sharetrace.util.types import TimeDelta

TWO_DAYS = np.timedelta64(172_800, 's')
FIVE_MINUTES = np.timedelta64(5, 'm')
_CONTACT_MSG = 1
_USER_MSG = 2
_DAY = np.timedelta64(1, 'D')
_DEFAULT_MSG = model.risk_score(0, np.datetime64('0'))

NumpyMap = Mapping[Any, np.ndarray]


# TODO See partition for up-to-date constructor
class RiskPropagation(ABC):
    __slots__ = (
        'send_threshold',
        'time_buffer',
        'time_constant',
        'transmission',
        'timeout',
        'max_duration',
        'n_early_stop',
        'n_workers')

    def __init__(
            self,
            send_threshold: float = 0.75,
            time_buffer: TimeDelta = TWO_DAYS,
            time_constant: float = 1,
            transmission: float = 0.8,
            max_duration: TimeDelta = FIVE_MINUTES,
            timeout: float = np.inf,
            n_early_stop: float = np.inf,
            n_workers: int = 1):
        self.send_threshold = send_threshold
        self.time_buffer = time_buffer
        self.time_constant = time_constant
        self.transmission = transmission
        self.timeout = timeout
        self.max_duration = max_duration
        self.n_early_stop = n_early_stop
        self.n_workers = n_workers

    @abstractmethod
    def run(self, scores: np.ndarray, contacts: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def create_graph(self, contacts: np.ndarray, n_parts: int = 1):
        pass
