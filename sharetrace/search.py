from abc import ABC, abstractmethod
from itertools import combinations
from logging import getLogger
from logging.config import dictConfig
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple

from joblib import Parallel, delayed
from numpy import (
    arange, array, column_stack, concatenate, ndarray, repeat, sort,
    timedelta64, unique, vstack
)
from numpy.random import default_rng
from pyproj import Proj, Transformer
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree

from sharetrace.model import contact, event, to_coord
from sharetrace.util import LOGGING_CONFIG, TimeDelta, Timer

Locations = Histories = Sequence[ndarray]
Contacts = ndarray
Pairs = Iterable[Tuple[ndarray, ndarray]]

_rng = default_rng()
_EMPTY = ()
ZERO = timedelta64(0, 's')

dictConfig(LOGGING_CONFIG)
logger = getLogger(__name__)


class BaseContactSearch(ABC):
    __slots__ = ('min_dur', 'workers')

    def __init__(self, min_dur: TimeDelta = ZERO, workers: int = 1, **kwargs):
        super().__init__()
        self.min_dur = timedelta64(min_dur)
        self.workers = workers
        self._log_params(min_dur=self.min_dur, workers=workers, **kwargs)

    def _log_params(self, **kwargs):
        logger.debug(
            '%s parameters: %s',
            self.__class__.__name__,
            {str(k): str(v) for k, v in kwargs.items()})

    @abstractmethod
    def search(self, histories: Histories) -> Contacts:
        pass

    @abstractmethod
    def pairs(self, histories: Histories) -> Pairs:
        pass


class BruteContactSearch(BaseContactSearch):
    __slots__ = ()

    def __init__(self, min_dur: TimeDelta = ZERO, workers: int = 1, **kwargs):
        super().__init__(min_dur, workers, **kwargs)

    def search(self, histories: Histories) -> Contacts:
        timer = Timer.time(lambda: self._search(histories))
        logger.info('%s: %.2f seconds', self.__class__.__name__, timer.seconds)
        return timer.result

    def _search(self, histories: Histories) -> Contacts:
        logger.debug('Finding pairs to search...')
        pairs = self.pairs(histories)
        logger.debug('Initiating contact search...')
        par = Parallel(n_jobs=self.workers)
        contacts = par(delayed(self._find_contact)(*p) for p in pairs)
        contacts = array([c for c in contacts if c is not None])
        logger.debug('Contact search completed')
        return contacts

    def pairs(self, histories: Histories) -> Pairs:
        return combinations(histories, 2)

    def _find_contact(self, h1: ndarray, h2: ndarray) -> Optional[ndarray]:
        found = None
        name1, name2 = h1['name'], h2['name']
        if name1 != name2:
            events = self._find(h1['locs'], 0, h2['locs'], 0)
            if len(events) > 0:
                found = contact((name1, name2), events)
        return found

    def _find(
            self,
            locs1: ndarray,
            i1: int,
            locs2: ndarray,
            i2: int
    ) -> List[ndarray]:
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

    def _event(self, start, loc1: ndarray, loc2: ndarray) -> Iterable[ndarray]:
        end = self._earlier(loc1, loc2)
        dur = end - start
        return [event(start, dur)] if dur >= self.min_dur else _EMPTY

    @staticmethod
    def _later(loc1: ndarray, loc2: ndarray) -> float:
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 > t2 else t2

    @staticmethod
    def _earlier(loc1: ndarray, loc2: ndarray) -> float:
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 < t2 else t2


class TreeContactSearch(BruteContactSearch):
    __slots__ = ('r', 'leaf_size')

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            **kwargs):
        super().__init__(
            min_dur, workers, r=r, leaf_size=leaf_size, **kwargs)
        self.r = r
        self.leaf_size = leaf_size

    def pairs(self, histories: Histories) -> Pairs:
        logger.debug('Mapping location histories to coordinates...')
        locs = self.to_coordinates(histories)
        logger.debug('Indexing coordinates to map back to users...')
        idx = self.index(locs)
        logger.debug('Querying pairs using spatial indexing...')
        pairs = self.query_pairs(concatenate(locs))
        logger.debug('Filtering histories based on queried pairs...')
        return self.filter(pairs, idx, histories)

    def to_coordinates(self, hists: Histories) -> Histories:
        par = Parallel(self.workers)
        return par(delayed(self._to_coordinates)(h) for h in hists)

    @staticmethod
    def index(locations: Locations) -> ndarray:
        idx = arange(len(locations))
        repeats = [len(locs) for locs in locations]
        return repeat(idx, repeats)

    def query_pairs(self, locs: ndarray) -> ndarray:
        raise NotImplementedError

    def filter(self, pairs: ndarray, idx: ndarray, hists: Histories) -> Pairs:
        selected = unique(idx[pairs], axis=0)
        self._log_stats(len(hists), len(selected))
        return ((hists[h1], hists[h2]) for (h1, h2) in selected if h1 != h2)

    @staticmethod
    def _to_coordinates(hist: ndarray) -> ndarray:
        return vstack([to_coord(loc)['loc'] for loc in hist['locs']])

    def _log_stats(self, hists: int, pairs: int) -> NoReturn:
        combos = hists ** 2
        dec = self._percent_decrease(combos, pairs)
        logger.info(
            'Pairs (tree / brute): %d / %d (%.2f percent)', pairs, combos, dec)

    @staticmethod
    def _percent_decrease(org, new) -> float:
        return 0 if org == 0 else round(100 * (new - org) / org, 2)


class BallTreeContactSearch(TreeContactSearch):
    __slots__ = ()

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            **kwargs):
        super().__init__(min_dur, workers, r, leaf_size, **kwargs)

    def query_pairs(self, locs: ndarray) -> ndarray:
        logger.debug('Constructing a ball tree spatial index')
        ball_tree = BallTree(locs, self.leaf_size, 'haversine')
        logger.debug('Querying for pairs')
        points = ball_tree.query_radius(locs, self.r)
        idx = self.index(points)
        points = concatenate(points)
        # Sorting along the last axis ensures duplicate pairs are removed.
        return sort(column_stack((idx, points)))


class KdTreeContactSearch(TreeContactSearch):
    __slots__ = ('p', 'eps')

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            eps: int = 1e-5,
            p: float = 2,
            **kwargs):
        super().__init__(min_dur, workers, r, leaf_size, eps=eps, p=p, **kwargs)
        self.eps = eps
        self.p = p

    def query_pairs(self, locs: ndarray) -> ndarray:
        locs = self._project(locs)
        logger.debug('Constructing a k-d tree spatial index')
        kd_tree = KDTree(locs, self.leaf_size)
        logger.debug('Querying for pairs')
        return kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')

    @staticmethod
    def _project(coordinates: ndarray) -> ndarray:
        """Project from latitude-longitude to x-y Cartesian coordinates.

            References:
                https://stackoverflow.com/a/54039559

            Args:
                coordinates: A (N, 2) numpy array of latitude-longitude pairs.

            Returns:
                A (N, 2) numpy array of x-y Cartesian pairs.
        """
        lats, longs = coordinates.T
        ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        transformer = Transformer.from_proj(lla, ecef)
        return column_stack(transformer.transform(longs, lats, radians=False))
