from __future__ import annotations

import logging.config
import os
import random
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from evaluation.synthetic.graphs import ConnectedCavemanGraphFactory
from sharetrace import model, util
from sharetrace.model import ArrayLike
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
    def __call__(self, n: int):
        pass


class UniformBernoulliValueFactory(DataFactory):
    __slots__ = ('per_user', 'p')

    def __init__(self, per_user: float = 15, p: float = 0.5):
        assert 0 <= p <= 1
        assert per_user > 0
        super().__init__()
        self.per_user = per_user
        self.p = p

    def __call__(self, n: int):
        per_user = self.per_user
        indicator = stats.bernoulli.rvs(self.p, size=n)
        replace = np.flatnonzero(indicator)
        values = rng.uniform(0, 0.5, size=(n, per_user))
        values[replace] = rng.uniform(0.5, 1, size=(len(replace), per_user))
        return values


class TimeFactory(DataFactory):
    __slots__ = ('days', 'per_day', 'now', 'random_first')

    def __init__(
            self,
            days: int = 15,
            per_day: int = 16,
            now: Optional[DateTime] = None,
            random_first: bool = False):
        assert days > 0
        assert per_day > 0
        super().__init__()
        self.days = days
        self.per_day = per_day
        self.now = now or datetime.utcnow()
        self.random_first = random_first

    def __call__(self, n: int) -> np.ndarray:
        create_delta, concat = self.create_delta, np.concatenate
        dt_type, td_type = 'datetime64[s]', 'timedelta64[s]'
        now = np.datetime64(self.now).astype(dt_type)
        offsets = (np.arange(self.days) * SEC_PER_DAY).astype(td_type)
        starts = now - offsets
        times = np.zeros((n, self.days * self.per_day), dtype=dt_type)
        for i in range(n):
            delta = create_delta().astype(td_type)
            times[i, :] = concat([start + delta for start in starts])
        if self.random_first:
            rng.shuffle(times, axis=1)
            times = times[:, 0]
        return times

    def create_delta(self) -> np.ndarray:
        deltas = rng.uniform(0, SEC_PER_DAY / self.per_day, size=self.per_day)
        return deltas.cumsum()


class LocationFactory(DataFactory):
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

    def __call__(self, n: int) -> np.ndarray:
        locs = np.zeros((n, 2, self.steps()))
        walk = self.walk
        for i in range(n):
            locs[i, ::] = walk()
        return locs

    def steps(self) -> int:
        return self.days * self.per_day

    def walk(self) -> np.ndarray:
        clip = np.clip
        low, high = self.low, self.high
        (x0, y0), steps, delta = self.x0y0(), self.steps(), self.delta()
        xy = np.zeros((2, steps))
        xy[:, 0] = clip((x0, y0), low, high)
        for i in range(1, steps):
            xy[:, i] = clip(xy[:, i - 1] + delta[:, i], low, high)
        return xy

    def x0y0(self) -> Tuple[float, float]:
        loc = abs(self.high) - abs(self.low)
        scale = np.sqrt((self.high - self.low) / 2)
        return stats.norm(loc, scale).rvs(size=2)

    def delta(self) -> np.ndarray:
        steps = self.steps()
        return rng.uniform(self.step_low, self.step_high, size=(2, steps))


class GaussianMixture:
    __slots__ = ('locs', 'scales', 'weights', 'components')

    def __init__(self, locs: ArrayLike, scales: ArrayLike, weights: ArrayLike):
        self.locs = locs
        self.scales = scales
        self.weights = weights
        self.components = [
            stats.norm(loc, scale) for loc, scale in zip(locs, scales)]

    def __call__(self, n):
        component = random.choices(self.components, weights=self.weights)[0]
        return component.rvs(size=n)


class GaussianMixtureLocationFactory(LocationFactory):
    __slots__ = ('components', 'locs', 'scales', 'weights', '_mixture')

    def __init__(
            self,
            components=None,
            locs=None,
            scales=None,
            weights=None,
            **kwargs):
        super(GaussianMixtureLocationFactory, self).__init__(**kwargs)
        if components is None:
            if locs is None:
                if scales is None:
                    if weights is None:
                        raise ValueError(
                            "Must specify 'components' or at least one of "
                            "'locs', 'scales', or 'weights'")
                    else:
                        # noinspection PyTypeChecker
                        components = len(weights)
                else:
                    components = len(scales)
            else:
                components = len(locs)
        self.components = c = components
        self.locs = locs or np.linspace(self.low, self.high, 2 + c)[1:-1]
        self.scales = scales or np.repeat(np.sqrt(np.diff(self.locs[:2])), c)
        self.weights = weights or np.repeat(1 / c, c)
        self._mixture = GaussianMixture(self.locs, self.scales, self.weights)

    def x0y0(self) -> Tuple[float, float]:
        return self._mixture(2)


class Dataset:
    __slots__ = ('scores', 'histories', 'contacts')

    def __init__(
            self,
            scores: ArrayLike,
            histories: Optional[ArrayLike] = None,
            contacts: Optional[ArrayLike] = None):
        self.scores = scores
        self.histories = histories
        self.contacts = contacts

    @lru_cache
    def geohashes(self, prec: int = 8) -> ArrayLike:
        if self.histories is None:
            raise AttributeError("'histories' is None")
        else:
            return model.to_geohashes(*self.histories, prec=prec)

    def save(self, path: str = '.') -> None:
        self._mkdir(path)
        save_data(self._scores_file(path), self.scores)
        if self.histories is not None:
            save_data(self._histories_file(path), self.histories)
        if self.contacts is not None:
            save_data(self._contacts_file(path), self.contacts)

    @staticmethod
    def _mkdir(path: str) -> None:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    @classmethod
    def load(
            cls,
            path: str = '.',
            histories: bool = False,
            contacts: bool = False
    ) -> Dataset:
        scores = load(Dataset._scores_file(path))
        histories_, contacts_ = None, None
        if histories:
            histories_ = load(Dataset._histories_file(path))
        if contacts:
            contacts_ = load(Dataset._contacts_file(path))
        return Dataset(scores, histories_, contacts_)

    @staticmethod
    def _scores_file(path: str) -> str:
        return os.path.join(path, 'scores.npy')

    @staticmethod
    def _histories_file(path: str) -> str:
        return os.path.join(path, 'histories.npy')

    @staticmethod
    def _contacts_file(path: str) -> str:
        return os.path.join(path, 'contacts.npy')


class DatasetFactory(DataFactory):
    __slots__ = ('score_factory', 'history_factory', 'contact_factory')

    def __init__(
            self,
            score_factory: ScoreFactory,
            history_factory: Optional[HistoryFactory] = None,
            contact_factory: Optional[ContactFactory] = None):
        super().__init__()
        self.score_factory = score_factory
        self.history_factory = history_factory
        self.contact_factory = contact_factory

    def __call__(self, n: int) -> Dataset:
        scores = self.score_factory(n)
        histories, contacts = None, None
        if (factory := self.history_factory) is not None:
            histories = factory(n)
        if (factory := self.contact_factory) is not None:
            contacts = factory(n)
        return Dataset(scores, histories, contacts)


class ScoreFactory(DataFactory):
    __slots__ = ('value_factory', 'time_factory')

    def __init__(self, value_factory: DataFactory, time_factory: DataFactory):
        super().__init__()
        self.value_factory = value_factory
        self.time_factory = time_factory

    def __call__(self, n: int) -> np.ndarray:
        values, times = self.value_factory(n), self.time_factory(n)
        risk_score = model.risk_score
        return np.array([
            [risk_score(v, t) for v, t in zip(vs, ts)]
            for vs, ts in zip(values, times)])


class HistoryFactory(DataFactory):
    __slots__ = ('loc_factory', 'time_factory')

    def __init__(self, loc_factory: DataFactory, time_factory: DataFactory):
        super().__init__()
        self.loc_factory = loc_factory
        self.time_factory = time_factory

    def __call__(self, n: int) -> np.ndarray:
        locations, times = self.loc_factory(n), self.time_factory(n)
        tloc, history = model.temporal_loc, model.history
        return np.array([
            history([tloc(loc, t) for t, loc in zip(ts, locs.T)], name)
            for name, (ts, locs) in enumerate(zip(times, locations))])


class ContactFactory(DataFactory):
    __slots__ = ('graph_factory', 'time_factory')

    def __init__(self, graph_factory: DataFactory, time_factory: DataFactory):
        super().__init__()
        self.graph_factory = graph_factory
        self.time_factory = time_factory

    def __call__(self, n: int) -> np.ndarray:
        graph = self.graph_factory(n)
        times = self.time_factory(graph.num_edges)
        contact = model.contact
        edges = graph.edges()
        return np.array([contact(names, t) for names, t in zip(edges, times)])


def create_data(
        users: int = 10_000,
        days: int = 15,
        per_day: int = 16,
        low: float = -1,
        high: float = 1,
        step_low: float = -0.01,
        step_high: float = 0.01,
        p: float = 0.2,
        save: bool = False) -> Dataset:
    dataset_factory = DatasetFactory(
        score_factory=ScoreFactory(
            value_factory=UniformBernoulliValueFactory(per_user=days, p=p),
            time_factory=TimeFactory(days=days, per_day=1)),
        history_factory=HistoryFactory(
            loc_factory=LocationFactory(
                days=days,
                per_day=per_day,
                low=low,
                high=high,
                step_low=step_low,
                step_high=step_high),
            time_factory=TimeFactory(days=days, per_day=per_day)),
        contact_factory=ContactFactory(
            graph_factory=ConnectedCavemanGraphFactory(2),
            time_factory=TimeFactory(days=days, per_day=1, random_first=True)))
    dataset = dataset_factory(users)
    if save:
        dataset.save('.//data')
    return dataset
