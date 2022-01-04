from __future__ import annotations

import datetime
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Iterable, Optional, Tuple, Union

import igraph as ig
import networkx as nx
import numpy as np
from scipy import stats

from sharetrace import model
from sharetrace.model import ArrayLike

SEC_PER_DAY = 86400
CONTACTS_FILE_FORMAT = 'data//contacts:{}.npy'


def save_data(file, arr: np.ndarray):
    warnings.filterwarnings("ignore")
    np.save(file, arr)


def load(filename):
    return np.load(filename, allow_pickle=True)


class DataFactory(ABC, Callable):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, n: int, **kwargs):
        pass


class UniformBernoulliValueFactory(DataFactory):
    __slots__ = ('per_user', 'p', 'seed', '_rng')

    def __init__(self, per_user: float = 15, p: float = 0.5, seed=None):
        assert 0 <= p <= 1
        assert per_user > 0
        super().__init__()
        self.per_user = per_user
        self.p = p
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __call__(self, n: int, **kwargs) -> np.ndarray:
        rng = self._rng
        per_user = self.per_user
        indicator = stats.bernoulli.rvs(self.p, size=n)
        replace = np.flatnonzero(indicator)
        values = rng.uniform(0, 0.5, size=(n, per_user))
        values[replace] = rng.uniform(0.5, 1, size=(len(replace), per_user))
        return values


class TimeFactory(DataFactory):
    __slots__ = ('days', 'per_day', 'now', 'random_first', 'seed', '_rng')

    def __init__(
            self,
            days: int = 15,
            per_day: int = 16,
            now: Optional[int] = None,
            random_first: bool = False,
            seed=None):
        assert days > 0
        assert per_day > 0
        super().__init__()
        self.days = days
        self.per_day = per_day
        self.now = now or round(datetime.datetime.utcnow().timestamp())
        self.random_first = random_first
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, n: int, now: int = None, **kwargs) -> np.ndarray:
        now = now if now is not None else self.now
        create_delta, concat = self.create_delta, np.concatenate
        offsets = np.arange(self.days) * SEC_PER_DAY
        starts = now - offsets
        times = np.zeros((n, self.days * self.per_day), dtype=np.int64)
        for i in range(n):
            delta = create_delta()
            times[i, :] = concat([start + delta for start in starts])
        if self.random_first:
            self._rng.shuffle(times, axis=1)
            times = times[:, 0]
        return times

    def create_delta(self) -> np.ndarray:
        deltas = self._rng.uniform(
            0, SEC_PER_DAY / self.per_day, size=self.per_day)
        return deltas.cumsum()


class LocationFactory(DataFactory):
    __slots__ = (
        'days',
        'per_day',
        'low',
        'high',
        'step_low',
        'step_high',
        'seed',
        '_rng')

    def __init__(
            self,
            days: int = 15,
            per_day: int = 16,
            low: float = -1,
            high: float = 1,
            step_low: float = -0.01,
            step_high: float = 0.01,
            seed=None):
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
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __call__(self, n: int, **kwargs) -> np.ndarray:
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
        return self._rng.uniform(self.step_low, self.step_high, size=(2, steps))


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
            score_factory: DataFactory,
            history_factory: Optional[DataFactory] = None,
            contact_factory: Optional[DataFactory] = None):
        super().__init__()
        self.score_factory = score_factory
        self.history_factory = history_factory
        self.contact_factory = contact_factory

    def __call__(self, n: int, **kwargs) -> Dataset:
        scores = self.score_factory(n)
        histories, contacts = None, None
        if (factory := self.history_factory) is not None:
            histories = factory(n)
        if (factory := self.contact_factory) is not None:
            contacts = factory(n)
        return Dataset(scores=scores, histories=histories, contacts=contacts)


class ScoreFactory(DataFactory):
    __slots__ = ('value_factory', 'time_factory')

    def __init__(self, value_factory: DataFactory, time_factory: DataFactory):
        super().__init__()
        self.value_factory = value_factory
        self.time_factory = time_factory

    def __call__(self, n: int, now: int = None, **kwargs) -> np.ndarray:
        values = self.value_factory(n)
        times = self.time_factory(n, now=now)
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

    def __call__(self, n: int, **kwargs) -> np.ndarray:
        locations, times = self.loc_factory(n), self.time_factory(n)
        tloc, history = model.temporal_loc, model.history
        return np.array([
            history([tloc(loc, t) for t, loc in zip(ts, locs.T)], name)
            for name, (ts, locs) in enumerate(zip(times, locations))])


class ContactFactory(DataFactory):
    __slots__ = ('graph_factory', 'time_factory', 'graph_path')

    def __init__(
            self,
            graph_factory: DataFactory,
            time_factory: DataFactory,
            graph_path: Optional[str] = None):
        super().__init__()
        self.graph_factory = graph_factory
        self.time_factory = time_factory
        self.graph_path = graph_path

    def __call__(self, n: int, **kwargs) -> np.ndarray:
        graph = self.graph_factory(n)
        if (path := self.graph_path) is not None:
            graph.save(path)
        ts = self.time_factory(graph.num_edges)
        contact = model.contact
        es = graph.edges()
        return np.array([
            contact((n1, n2), t) for (n1, n2), t in zip(es, ts) if n1 != n2])


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

    @abstractmethod
    def save(self, path: str) -> None:
        pass


class GraphFactory(DataFactory):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, n: int, **kwargs) -> Graph:
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

    def save(self, path: str) -> None:
        self._graph.write(path)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(nodes={self.num_nodes}, edges={self.num_edges})'


class SocioPatternsGraphReader(GraphReader):
    __slots__ = ('sep',)

    def __init__(self, sep=' '):
        super().__init__()
        self.sep = sep

    def read(self, path: str) -> Graph:
        with open(path, 'r') as f:
            sep = self.sep
            triples = np.array([
                line.rstrip('\n').split(sep)[:3] for line in f.readlines()],
                dtype=np.int64)
        times, pairs = triples[:, 0], triples[:, [1, 2]]
        # Sort pairs to ensure that "reversed" pairs are not kept.
        order = np.argsort(pairs, axis=1)
        times, pairs = times[order], pairs[order]
        unique = np.unique(pairs, axis=0)
        idx = {n: i for i, n in enumerate(np.unique(pairs))}
        # Shift time forward to ensure positive timestamps.
        offset = round(datetime.datetime.utcnow().timestamp())
        # Use the latest time in a series of contacts as the contact time.
        times = np.array([
            np.max(times[(pairs == p).all(1)]) + offset for p in unique])
        graph = ig.Graph(
            edges=[(idx[n1], idx[n2]) for n1, n2 in unique],
            edge_attrs={'time': times})
        return IGraph(graph)


class SocioPatternsDatasetFactory(DataFactory):
    __slots__ = ('score_factory', 'contact_factory')

    def __init__(
            self,
            score_factory: SocioPatternsScoreFactory,
            contact_factory: SocioPatternsContactFactory):
        super().__init__()
        self.score_factory = score_factory
        self.contact_factory = contact_factory

    def __call__(self, n: int = None, **kwargs) -> Dataset:
        contacts = self.contact_factory()
        scores = self.score_factory(
            self.contact_factory.nodes, now=self.contact_factory.start)
        assert np.min(contacts['time']) - np.max(scores['time']) > 0
        return Dataset(scores=scores, contacts=contacts)


class SocioPatternsScoreFactory(ScoreFactory):
    __slots__ = ()

    def __init__(self, value_factory: DataFactory, time_factory: TimeFactory):
        super().__init__(value_factory, time_factory)

    def __call__(self, n: int, now: int = None, **kwargs) -> np.ndarray:
        return super().__call__(n, now=now)


class SocioPatternsContactFactory(DataFactory):
    __slots__ = ('path', 'sep', 'nodes', 'start', 'graph_path')

    def __init__(
            self,
            path: str,
            sep: str = ' ',
            graph_path: Optional[str] = None):
        super().__init__()
        self.path = path
        self.sep = sep
        self.graph_path = graph_path
        self.nodes = -1
        self.start = -1

    def __call__(self, n: int = None, now: int = None, **kwargs) -> np.ndarray:
        graph = SocioPatternsGraphReader(self.sep).read(self.path)
        if (path := self.graph_path) is not None:
            graph.save(path)
        self.nodes = graph.num_nodes
        # Raw contact times are nonzero and shifted forward from now. To
        # ensure all risk score timestamps are prior to the first contact,
        # we set the start time used by the time factory to be yesterday.
        # This assumes that each user only has 1 risk score with a timestamp
        # that ranges between the start time given here and 1 day later.
        start = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        self.start = round(start.timestamp())
        contact = model.contact
        return np.array([contact(names, t) for names, t in graph.edges('time')])


if __name__ == '__main__':
    SocioPatternsGraphReader().read('data//conference.txt')
