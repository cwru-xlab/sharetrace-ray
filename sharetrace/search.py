import itertools
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import joblib
import numpy as np
import pyproj
from scipy import spatial

from sharetrace import model, util

Locations = Histories = Sequence[np.void]
Contacts = np.ndarray
Pairs = Iterable[Tuple[np.void, np.void]]

rng = np.random.default_rng()
EMPTY = ()


class Kind(Enum):
    BRUTE = 'brute'
    KD_TREE = 'kd_tree'


class BaseContactSearch(ABC):
    __slots__ = ('logger', 'min_dur', 'workers')

    def __init__(
            self,
            min_dur: float = 0,
            workers: int = 1,
            logger: Optional[logging.Logger] = None):
        super().__init__()
        self.min_dur = min_dur
        self.workers = workers
        self.logger = logger

    @abstractmethod
    def search(self, histories: Histories) -> Contacts:
        pass

    @abstractmethod
    def pairs(self, histories: Histories) -> Pairs:
        pass


class BruteContactSearch(BaseContactSearch):
    __slots__ = ('kind', '_min_dur')

    def __init__(
            self,
            min_dur: float = 0,
            workers: int = 1,
            logger: Optional[logging.Logger] = None):
        super().__init__(min_dur, workers, logger)
        self.kind = Kind.BRUTE
        self._min_dur = np.timedelta64(int(min_dur * 1e6), 'us')

    def search(self, histories: Histories) -> Contacts:
        timer = util.time(lambda: self._search(histories))
        result = timer.result
        stats = self.stats(len(histories), len(result), timer.seconds)
        if self.logger is not None:
            util.info(self.logger, json.dumps(stats))
        return result

    def stats(self, n: int, contacts: int, runtime: float) -> Dict[str, Any]:
        return {
            'Kind': self.kind.name,
            'RuntimeInSec': util.approx(runtime),
            'Workers': self.workers,
            'MinDurationInSec': self.min_dur,
            'InputSize': n,
            'Contacts': contacts}

    def _search(self, histories: Histories) -> Contacts:
        pairs = self.pairs(histories)
        par = joblib.Parallel(n_jobs=self.workers)
        contacts = par(joblib.delayed(self._find_contact)(*p) for p in pairs)
        contacts = np.array([c for c in contacts if c is not None])
        return contacts

    def pairs(self, histories: Histories) -> Pairs:
        return itertools.combinations(histories, 2)

    def _find_contact(self, h1: np.void, h2: np.void) -> Optional[np.ndarray]:
        found = None
        name1, name2 = h1['name'], h2['name']
        if name1 != name2:
            events = self._find(h1['locs'], 0, h2['locs'], 0)
            if len(events) > 0:
                found = model.contact((name1, name2), events)
        return found

    def _find(
            self,
            locs1: np.ndarray,
            i1: int,
            locs2: np.ndarray,
            i2: int
    ) -> Sequence[np.ndarray]:
        events = []
        later, create_event, find, add_events = (
            self._later, self._event, self._find, events.extend)
        len1, len2 = len(locs1) - 1, len(locs2) - 1
        loc1, loc2 = locs1[i1], locs2[i2]
        started = False
        while i1 < len1 and i2 < len2:
            if loc1['loc'] == loc2['loc']:
                if started:
                    i1, i2 = i1 + 1, i2 + 1
                    loc1, loc2 = locs1[i1], locs2[i2]
                else:
                    started = True
                    start = later(loc1, loc2)
            elif started:
                started = False
                # noinspection PyUnboundLocalVariable
                add_events(create_event(start, loc1, loc2))
            elif loc1['time'] < loc2['time']:
                i1 += 1
                loc1 = locs1[i1]
            elif loc2['time'] < loc1['time']:
                i2 += 1
                loc2 = locs2[i2]
            else:
                add_events(find(locs1, i1 + 1, locs2, i2))
                add_events(find(locs1, i1, locs2, i2 + 1))
                break
        if started:
            add_events(create_event(start, loc1, loc2))
        return events

    def _event(
            self, start, loc1: np.void, loc2: np.void) -> Iterable[np.ndarray]:
        end = self._earlier(loc1, loc2)
        dur = end - start
        return [model.event(start, dur)] if dur >= self._min_dur else EMPTY

    @staticmethod
    def _later(loc1: np.void, loc2: np.void) -> float:
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 > t2 else t2

    @staticmethod
    def _earlier(loc1: np.void, loc2: np.void) -> float:
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 < t2 else t2


class TreeContactSearch(BruteContactSearch):
    __slots__ = ('r', 'leaf_size')

    def __init__(
            self,
            min_dur: float = 0,
            workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            logger: Optional[logging.Logger] = None):
        super().__init__(min_dur, workers, logger)
        self.r = r
        self.leaf_size = leaf_size

    def pairs(self, histories: Histories) -> Pairs:
        locs = self.to_coordinates(histories)
        idx = self.index(locs)
        pairs = self.query_pairs(np.concatenate(locs))
        return self.filter(pairs, idx, histories)

    def to_coordinates(self, hists: Histories) -> Histories:
        par = joblib.Parallel(self.workers)
        return par(joblib.delayed(self._to_coordinates)(h) for h in hists)

    def stats(self, n: int, contacts: int, runtime: float) -> Dict[str, Any]:
        stats = super().stats(n, contacts, runtime)
        stats.update({'Radius': self.r, 'LeafSize': self.leaf_size})
        return stats

    @staticmethod
    def index(locations: Locations) -> np.ndarray:
        idx = np.arange(len(locations))
        repeats = [len(locs) for locs in locations]
        return np.repeat(idx, repeats)

    def query_pairs(self, locs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def filter(pairs: np.ndarray, idx: np.ndarray, hists: Histories) -> Pairs:
        selected = np.unique(idx[pairs], axis=0)
        return ((hists[h1], hists[h2]) for (h1, h2) in selected if h1 != h2)

    @staticmethod
    def _to_coordinates(hist: np.ndarray) -> np.ndarray:
        return np.vstack([model.to_coord(loc)['loc'] for loc in hist['locs']])


class KdTreeContactSearch(TreeContactSearch):
    __slots__ = ('p', 'eps')

    def __init__(
            self,
            min_dur: float = 0,
            workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            eps: int = 1e-5,
            p: float = 2,
            logger: Optional[logging.Logger] = None):
        super().__init__(min_dur, workers, r, leaf_size, logger)
        self.kind = Kind.KD_TREE
        self.eps = eps
        self.p = p

    def query_pairs(self, locs: np.ndarray) -> np.ndarray:
        locs = self._project(locs)
        kd_tree = spatial.KDTree(locs, self.leaf_size)
        return kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')

    def stats(self, n: int, contacts: int, runtime: float) -> Dict[str, Any]:
        stats = super().stats(n, contacts, runtime)
        stats.update({'Epsilon': self.eps, 'MinkowskiNorm': self.p})
        return stats

    @staticmethod
    def _project(coordinates: np.ndarray) -> np.ndarray:
        """Project from latitude-longitude to x-y Cartesian coordinates.

            References:
                https://stackoverflow.com/a/54039559

            Args:
                coordinates: A (N, 2) numpy array of latitude-longitude pairs.

            Returns:
                A (N, 2) numpy array of x-y Cartesian pairs.
        """
        lats, longs = coordinates.T
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        transformer = pyproj.Transformer.from_proj(lla, ecef)
        projected = transformer.transform(longs, lats, radians=False)
        return np.column_stack(projected)
