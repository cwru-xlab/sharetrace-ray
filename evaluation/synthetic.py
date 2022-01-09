from __future__ import annotations

import datetime
import itertools
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Optional, Tuple

import igraph as ig
import numpy as np
from scipy import stats

from sharetrace import model
from sharetrace.model import ArrayLike

SEC_PER_DAY = 86400
CONTACTS_FILE_FORMAT = "data//contacts:{}.npy"


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
    __slots__ = ("per_user", "p", "seed", "_rng")

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
    __slots__ = ("days", "per_day", "now", "random_first", "seed", "_rng")

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
        "days",
        "per_day",
        "low",
        "high",
        "step_low",
        "step_high",
        "seed",
        "_rng")

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
    __slots__ = ("locs", "scales", "weights", "components")

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
    __slots__ = ("components", "locs", "scales", "weights", "_mixture")

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
    __slots__ = ("scores", "graph", "histories", "contacts")

    def __init__(
            self,
            scores: ArrayLike,
            graph: Optional[ig.Graph] = None,
            histories: Optional[ArrayLike] = None,
            contacts: Optional[ArrayLike] = None):
        self.scores = scores
        self.graph = graph
        self.histories = histories
        self.contacts = contacts

    @lru_cache
    def geohashes(self, prec: int = 8) -> ArrayLike:
        if self.histories is None:
            raise AttributeError("'histories' is None")
        else:
            return model.to_geohashes(*self.histories, prec=prec)

    def save(self, path: str = ".") -> None:
        self._mkdir(path)
        save_data(self._scores_file(path), self.scores)
        if (graph := self.graph) is not None:
            ig.write(graph, self._graph_file(path))
        if (histories := self.histories) is not None:
            save_data(self._histories_file(path), histories)
        if (contacts := self.contacts) is not None:
            save_data(self._contacts_file(path), contacts)

    @staticmethod
    def _mkdir(path: str) -> None:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    @classmethod
    def load(
            cls,
            path: str = ".",
            graph: bool = False,
            histories: bool = False,
            contacts: bool = False
    ) -> Dataset:
        scores = load(Dataset._scores_file(path))
        graph_, histories_, contacts_ = None, None, None
        if graph:
            graph_ = ig.read(Dataset._graph_file(path))
        if histories:
            histories_ = load(Dataset._histories_file(path))
        if contacts:
            contacts_ = load(Dataset._contacts_file(path))
        return Dataset(scores, graph_, histories_, contacts_)

    @staticmethod
    def _scores_file(path: str) -> str:
        return os.path.join(path, "scores.npy")

    @staticmethod
    def _graph_file(path: str) -> str:
        return os.path.join(path, "graph.graphml")

    @staticmethod
    def _histories_file(path: str) -> str:
        return os.path.join(path, "histories.npy")

    @staticmethod
    def _contacts_file(path: str) -> str:
        return os.path.join(path, "contacts.npy")


class DatasetFactory(DataFactory):
    __slots__ = (
        "score_factory", "graph_factory", "history_factory", "contact_factory")

    def __init__(
            self,
            score_factory: DataFactory,
            graph_factory: Optional[DataFactory] = None,
            history_factory: Optional[DataFactory] = None,
            contact_factory: Optional[DataFactory] = None):
        super().__init__()
        self.score_factory = score_factory
        self.graph_factory = graph_factory
        self.history_factory = history_factory
        self.contact_factory = contact_factory

    def __call__(self, n: int, **kwargs) -> Dataset:
        scores = self.score_factory(n, **kwargs)
        graph, histories, contacts = None, None, None
        if (factory := self.graph_factory) is not None:
            graph = factory(n, **kwargs)
        if (factory := self.history_factory) is not None:
            histories = factory(n, **kwargs)
        if (factory := self.contact_factory) is not None:
            contacts = factory(n, **kwargs)
        return Dataset(scores, graph, histories, contacts)


class ScoreFactory(DataFactory):
    __slots__ = ("value_factory", "time_factory")

    def __init__(self, value_factory: DataFactory, time_factory: DataFactory):
        super().__init__()
        self.value_factory = value_factory
        self.time_factory = time_factory

    def __call__(self, n: int, now: int = None, **kwargs) -> np.ndarray:
        values = self.value_factory(n, **kwargs)
        times = self.time_factory(n, now=now, **kwargs)
        risk_score = model.risk_score
        return np.array([
            [risk_score(v, t) for v, t in zip(vs, ts)]
            for vs, ts in zip(values, times)])


class HistoryFactory(DataFactory):
    __slots__ = ("loc_factory", "time_factory")

    def __init__(self, loc_factory: DataFactory, time_factory: DataFactory):
        super().__init__()
        self.loc_factory = loc_factory
        self.time_factory = time_factory

    def __call__(self, n: int, **kwargs) -> np.ndarray:
        locations = self.loc_factory(n, **kwargs)
        times = self.time_factory(n, **kwargs)
        tloc, history = model.temporal_loc, model.history
        return np.array([
            history([tloc(loc, t) for t, loc in zip(ts, locs.T)], name)
            for name, (ts, locs) in enumerate(zip(times, locations))])


class ContactFactory(DataFactory):
    __slots__ = ("graph_factory", "time_factory", "graph_path")

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
        graph: ig.Graph = self.graph_factory(n, **kwargs)
        if (path := self.graph_path) is not None:
            graph.save(path)
        ts = self.time_factory(len(edges := graph.es), **kwargs)
        es = (e.tuple for e in edges)
        contact = model.contact
        return np.array([
            contact((n1, n2), t) for (n1, n2), t in zip(es, ts) if n1 != n2])


class CachedGraphFactory(DataFactory):
    __slots__ = ("graph_factory", "_graph")

    def __init__(self, graph_factory: DataFactory):
        super().__init__()
        self.graph_factory = graph_factory
        self._graph = None

    def __call__(self, n: int = None, **kwargs):
        if self._graph is None:
            self._graph = self.graph_factory(n, **kwargs)
        return self._graph


class SocioPatternsGraphFactory(DataFactory):
    __slots__ = ("path", "sep")

    def __init__(self, path: str, sep=" "):
        super().__init__()
        self.path = path
        self.sep = sep

    def __call__(self, n: int = None, **kwargs):
        def by_pair(triple):
            return triple[1:]

        def by_time(triple):
            return triple[0]

        with open(self.path, "r") as f:
            triples = sorted(map(self._parse, f.readlines()), key=by_pair)
        groups = itertools.groupby(triples, key=by_pair)
        # Use the most recent time as the time of contact.
        triples = [sorted(g, key=by_time)[-1] for _, g in groups]
        names = set(itertools.chain.from_iterable(t[1:] for t in triples))
        idx = {n: i for i, n in enumerate(names)}
        # Shift time forward to ensure positive timestamps.
        offset = round(datetime.datetime.utcnow().timestamp())
        times = [t + offset for t, _, _ in triples]
        edges = [(idx[n1], idx[n2]) for _, n1, n2 in triples]
        return ig.Graph(edges=edges, edge_attrs={"time": times})

    def _parse(self, line: str) -> Tuple[int, int, int]:
        t, n1, n2 = line.rstrip("\n").split(self.sep)[:3]
        t, n1, n2 = int(t), int(n1), int(n2)
        return t, min(n1, n2), max(n1, n2)


class SocioPatternsDatasetFactory(DataFactory):
    __slots__ = ("score_factory", "graph_factory", "contact_factory")

    def __init__(
            self,
            score_factory: DataFactory,
            graph_factory: DataFactory,
            contact_factory: SocioPatternsContactFactory):
        super().__init__()
        self.score_factory = score_factory
        self.graph_factory = graph_factory
        self.contact_factory = contact_factory

    def __call__(self, n: int = None, **kwargs) -> Dataset:
        contacts = self.contact_factory(**kwargs)
        scores = self.score_factory(
            self.contact_factory.nodes,
            now=self.contact_factory.start,
            **kwargs)
        graph = self.graph_factory(n, **kwargs)
        assert np.min(contacts["time"]) - np.max(scores["time"]) > 0
        return Dataset(scores=scores, graph=graph, contacts=contacts)


class SocioPatternsContactFactory(DataFactory):
    __slots__ = ("graph_factory", "graph_path", "nodes", "start",)

    def __init__(
            self,
            graph_factory: DataFactory,
            graph_path: Optional[str] = None):
        super().__init__()
        self.graph_factory = graph_factory
        self.graph_path = graph_path
        self.nodes = -1
        self.start = -1

    def __call__(self, n: int = None, now: int = None, **kwargs) -> np.ndarray:
        graph: ig.Graph = self.graph_factory(n, **kwargs)
        if (path := self.graph_path) is not None:
            graph.save(path)
        self.nodes = len(graph.vs)
        # Raw contact times are nonzero and shifted forward from now. To
        # ensure all risk score timestamps are prior to the first contact,
        # we set the start time used by the time factory to be yesterday.
        # This assumes that each user only has 1 risk score with a timestamp
        # that ranges between the start time given here and 1 day later.
        start = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        self.start = round(start.timestamp())
        contact = model.contact
        es = ((e.tuple, e["time"]) for e in graph.es)
        return np.array([contact(names, t) for names, t in es])
