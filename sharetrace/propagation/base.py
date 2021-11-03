import abc
from numbers import Real
from typing import Any, Mapping

import numpy as np

import sharetrace.search.base
from sharetrace import model
from sharetrace.util.types import TimeDelta

TWO_DAYS = np.timedelta64(172_800, 's')
FIVE_MINUTES = np.timedelta64(5, 'm')
_CONTACT_MSG = 1
_USER_MSG = 2
_DAY = np.timedelta64(1, 'D')
_DEFAULT_MSG = model.score(0, np.datetime64('0'))

NumpyMap = Mapping[Any, np.ndarray]


def _contact_callback(
        msg: np.ndarray,
        ne: NumpyMap,
        nodes: NumpyMap,
        time_buffer: np.timedelta64,
        time_constant: Real,
        transmission_rate: Real) -> np.ndarray:
    sharetrace.search.base.Pairs
    def neighbor(contact: Any, user: Any):
        users = ne[contact]
        return users[user != users][0]

    user, contact, scores = msg['src'], msg['dest'], msg['value']
    dest = neighbor(contact, user)
    recent = nodes[contact]['timestamp']
    scores = scores[scores['timestamp'] <= recent + time_buffer]
    if scores.size > 0:
        diff = (scores['timestamp'] - recent) / _DAY
        np.clip(diff, -np.inf, 0, out=diff)
        weight = diff / time_constant
        max_score = scores[np.argmax(np.log(scores['value']) + weight)]
        max_score['value'] *= transmission_rate
        msg = model.message(max_score, contact, dest, _CONTACT_MSG)
    else:
        msg = model.message(_DEFAULT_MSG, contact, dest, _CONTACT_MSG)
    return np.array([msg])


class RiskPropagation:
    __slots__ = (
        'send_threshold',
        'time_buffer',
        'time_constant',
        'transmission_rate',
        'timeout',
        'max_duration',
        'n_msgs_early_stop',
        'n_workers')

    def __init__(
            self,
            send_threshold: float = 0.75,
            time_buffer: TimeDelta = TWO_DAYS,
            time_constant: float = 1,
            transmission_rate: float = 0.8,
            max_duration: TimeDelta = FIVE_MINUTES,
            timeout: float = np.inf,
            n_msgs_early_stop: float = np.inf,
            n_workers: int = 1):
        self.send_threshold = send_threshold
        self.time_buffer = time_buffer
        self.time_constant = time_constant
        self.transmission_rate = transmission_rate
        self.timeout = timeout
        self.max_duration = max_duration
        self.n_msgs_early_stop = n_msgs_early_stop
        self.n_workers = n_workers

    def run(self, scores: np.ndarray, contacts: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def partition(self):
        pass
