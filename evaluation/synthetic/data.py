from __future__ import annotations

import logging.config
import os
import random
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from typing import Iterable, Optional, Tuple, Union

import igraph as ig
import networkx as nx
import numpy as np
from scipy import stats

from sharetrace import model, util
from sharetrace.model import ArrayLike

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
            now: Optional[int] = None,
            random_first: bool = False):
        assert days > 0
        assert per_day > 0
        super().__init__()
        self.days = days
        self.per_day = per_day
        self.now = now or datetime.utcnow().timestamp()
        self.random_first = random_first

    def __call__(self, n: int) -> np.ndarray:
        create_delta, concat = self.create_delta, np.concatenate
        offsets = np.arange(self.days) * SEC_PER_DAY
        starts = self.now - offsets
        times = np.zeros((n, self.days * self.per_day), dtype=np.int64)
        for i in range(n):
            delta = create_delta()
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


class Graph(ABC):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        pass

    @abstractmethod
    def nodes(self, *args) -> Iterable:
        pass

    @abstractmethod
    def edges(self, *args) -> Iterable[Tuple]:
        pass


class GraphFactory(DataFactory):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, n: int) -> Graph:
        pass


class GraphReader(ABC):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @abstractmethod
    def read(self, path: str) -> Graph:
        pass


class IGraph(Graph):
    __slots__ = ('_graph',)

    def __init__(self, graph: Union[ig.Graph, nx.Graph]):
        super().__init__()
        if isinstance(graph, nx.Graph):
            graph = ig.Graph.from_networkx(graph)
        self._graph = graph

    @property
    def num_nodes(self) -> int:
        return len(self._graph.vs)

    @property
    def num_edges(self) -> int:
        return len(self._graph.es)

    def nodes(self, *args) -> Iterable:
        nodes = self._graph.vs
        if args:
            if len(args) == 1:
                k = args[0]
                nodes = ((n.index, n[k]) for n in nodes)
            else:
                nodes = ((n.index, tuple(n[k] for k in args)) for n in nodes)
        else:
            nodes = iter(nodes.indices)
        return nodes

    def edges(self, *args) -> Iterable[Tuple]:
        edges = self._graph.es
        if args:
            if len(args) == 1:
                k = args[0]
                edges = ((e.tuple, e[k]) for e in edges)
            else:
                edges = ((e.tuple, tuple(e[k] for k in args)) for e in edges)
        else:
            edges = (e.tuple for e in edges)
        return edges


class SocioPatternsGraphReader(GraphReader):
    __slots__ = ('sep',)

    def __init__(self, sep=' '):
        super().__init__()
        self.sep = sep

    def read(self, path: str) -> Graph:
        def _id(name, g, index, curr_idx):
            if name in idx:
                i = index[name]
            else:
                g.add_vertex(name)
                index[name] = i = curr_idx
                curr_idx += 1
            return i, curr_idx

        with open(path, 'r') as f:
            graph, idx, curr = ig.Graph(), {}, 0
            add_edge = graph.add_edge
            while line := f.readline():
                args = line.rstrip('\n').split(self.sep)
                t, n1, n2 = args[:3]
                i1, curr = _id(n1, graph, idx, curr)
                i2, curr = _id(n2, graph, idx, curr)
                add_edge(i1, i2, time=int(t))
            return IGraph(graph)


class SocioPatternsContactFactory(DataFactory):
    __slots__ = ('path', 'sep')

    def __init__(self, path: str, sep: str = ' '):
        super().__init__()
        self.path = path
        self.sep = sep

    def __call__(self, n: int = None) -> np.ndarray:
        graph = SocioPatternsGraphReader(self.sep).read(self.path)
        contact = model.contact
        return np.array([contact(names, t) for names, t in graph.edges('time')])
