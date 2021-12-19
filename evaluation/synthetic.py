import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from logging import config
from typing import Optional

import numpy as np
from scipy import stats

from sharetrace import model, util
from sharetrace.util import DateTime

logging.config.dictConfig(util.logging_config())

rng = np.random.default_rng()
SEC_PER_DAY = 86400
CONTACTS_FILE_FORMAT = 'data//contacts:{}.npy'


def save_contacts(contacts, n: int):
    save_data(CONTACTS_FILE_FORMAT.format(n), contacts)


def load_contacts(n: int):
    return load(CONTACTS_FILE_FORMAT.format(n))


def save_data(file, arr: np.ndarray):
    warnings.filterwarnings("ignore")
    np.save(file, arr)


def load(filename):
    return np.load(filename, allow_pickle=True)


class DataFactory(ABC):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, users: int):
        pass


class UniformBernoulliValueFactory(DataFactory):
    __slots__ = ('per_user', 'p')

    def __init__(self, per_user: float = 15, p: float = 0.5):
        assert 0 <= p <= 1
        assert per_user > 0
        super().__init__()
        self.per_user = per_user
        self.p = p

    def __call__(self, users: int):
        per_user = self.per_user
        indicator = stats.bernoulli.rvs(self.p, size=users)
        replace = np.flatnonzero(indicator)
        values = rng.uniform(0, 0.5, size=(users, per_user))
        values[replace] = rng.uniform(0.5, 1, size=(len(replace), per_user))
        return values


class TimeFactory(DataFactory):
    __slots__ = ('days', 'per_day', 'now')

    def __init__(
            self,
            days: int = 15,
            per_day: int = 16,
            now: Optional[DateTime] = None):
        assert days > 0
        assert per_day > 0
        super().__init__()
        self.days = days
        self.per_day = per_day
        self.now = now or datetime.utcnow()

    def __call__(self, users: int):
        create_delta, concat = self.create_delta, np.concatenate
        dt_type, td_type = 'datetime64[s]', 'timedelta64[s]'
        now = np.datetime64(self.now).astype(dt_type)
        offsets = (np.arange(self.days) * SEC_PER_DAY).astype(td_type)
        starts = now - offsets
        times = np.zeros((users, self.days * self.per_day), dtype=dt_type)
        for i in range(users):
            delta = create_delta().astype(td_type)
            times[i, :] = concat([start + delta for start in starts])
        return times

    def create_delta(self):
        deltas = rng.uniform(0, SEC_PER_DAY / self.per_day, size=self.per_day)
        return deltas.cumsum()


class RandomWalkLocationFactory(DataFactory):
    __slots__ = (
        'days',
        'per_day',
        'low',
        'high',
        'step_low',
        'step_high')

    def __init__(
            self,
            days: int = 15,
            per_day: int = 16,
            low: float = -1,
            high: float = 1,
            step_low: float = -0.01,
            step_high: float = 0.01):
        assert days > 0
        assert per_day > 0
        assert low < high
        assert step_low < step_high
        super().__init__()
        self.days = days
        self.per_day = per_day
        self.low = low
        self.high = high
        self.step_low = step_low
        self.step_high = step_high

    def __call__(self, users: int):
        locs = np.zeros((users, 2, self.steps()))
        walk = self.walk
        for i in range(users):
            locs[i, ::] = walk()
        return locs

    def steps(self):
        return self.days * self.per_day

    def walk(self):
        clip = np.clip
        low, high = self.low, self.high
        (x0, y0), steps, delta = self.x0y0(), self.steps(), self.delta()
        xy = np.zeros((2, steps))
        xy[:, 0] = clip((x0, y0), low, high)
        for i in range(1, steps):
            xy[:, i] = clip(xy[:, i - 1] + delta[:, i], low, high)
        return xy

    def x0y0(self):
        loc = abs(self.high) - abs(self.low)
        scale = np.sqrt((self.high - self.low) / 2)
        return stats.norm(loc, scale).rvs(size=2)

    def delta(self):
        steps = self.steps()
        return rng.uniform(self.step_low, self.step_high, size=(2, steps))


class Dataset:
    __slots__ = ('scores', 'histories')

    def __init__(self, scores, histories):
        self.scores = scores
        self.histories = histories

    @lru_cache
    def geohashes(self, prec: int = 8):
        return model.to_geohashes(*self.histories, prec=prec)

    def save(self, path: str = '.'):
        self._mkdir(path)
        save_data(self._scores_file(path), self.scores)
        save_data(self._histories_file(path), self.histories)

    @staticmethod
    def _mkdir(path: str):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    @classmethod
    def load(cls, path: str = '.'):
        scores = load(Dataset._scores_file(path))
        histories = load(Dataset._histories_file(path))
        return Dataset(scores, histories)

    @staticmethod
    def _scores_file(path: str):
        return os.path.join(path, 'scores.npy')

    @staticmethod
    def _histories_file(path: str):
        return os.path.join(path, 'histories.npy')


class DatasetFactory(DataFactory):
    __slots__ = ('value_factory', 'time_factory', 'loc_factory')

    def __init__(
            self,
            value_factory: DataFactory,
            time_factory: DataFactory,
            loc_factory: DataFactory):
        super().__init__()
        self.value_factory = value_factory
        self.time_factory = time_factory
        self.loc_factory = loc_factory

    def __call__(self, users: int) -> Dataset:
        values = self.value_factory(users)
        times = self.time_factory(users)
        locs = self.loc_factory(users)
        scores = self.scores(values, times)
        histories = self.histories(locs, times)
        return Dataset(scores, histories)

    @staticmethod
    def scores(values, times):
        scores = []
        risk_score, append = model.risk_score, scores.append
        for uvals, utimes in zip(values, times):
            append([risk_score(v, t) for v, t in zip(uvals, utimes)])
        return np.array(scores)

    @staticmethod
    def histories(locs, times):
        histories = []
        append = histories.append
        tloc, history = model.temporal_loc, model.history
        for u, (utimes, ulocs) in enumerate(zip(times, locs)):
            tlocs = [tloc(loc, time) for time, loc in zip(utimes, ulocs.T)]
            append(history(tlocs, u))
        return np.array(histories)


class GaussianMixture:
    __slots__ = ('locs', 'scales', 'weights', 'components')

    def __init__(self, locs, scales, weights):
        self.locs = locs
        self.scales = scales
        self.weights = weights
        self.components = [
            stats.norm(loc, scale) for loc, scale in zip(locs, scales)]

    def __call__(self, n):
        component = random.choices(self.components, weights=self.weights)[0]
        return component.rvs(size=n)


def create_data(
        users: int = 10_000,
        days: int = 15,
        per_day: int = 16,
        low: float = -1,
        high: float = 1,
        step_low: float = -0.01,
        step_high: float = 0.01,
        p: float = 0.2,
        save: bool = False):
    time_factory = TimeFactory(days, per_day)
    value_factory = UniformBernoulliValueFactory(days, p)
    loc_factory = RandomWalkLocationFactory(
        days=days,
        per_day=per_day,
        low=low,
        high=high,
        step_low=step_low,
        step_high=step_high)
    dataset_factory = DatasetFactory(
        value_factory=value_factory,
        time_factory=time_factory,
        loc_factory=loc_factory)
    dataset = dataset_factory(users)
    if save:
        dataset.save('.//data')
