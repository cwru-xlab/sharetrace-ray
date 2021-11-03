from itertools import chain
from typing import Iterable, NoReturn, Sequence, final

import joblib
import pyproj
from numpy import (
    arange, array, column_stack, ndarray, repeat, sort, unique, vstack
)
from scipy import spatial
from sklearn import neighbors

from sharetrace.model import to_coord
from sharetrace.search.base import Histories, Pairs, ZERO
from sharetrace.search.brute import BruteContactSearch
from sharetrace.util.types import TimeDelta

Locations = Sequence[ndarray]


class TreeContactSearch(BruteContactSearch):
    __slots__ = ('r', 'leaf_size')

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            n_workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            **kwargs):
        super().__init__(
            min_dur, n_workers, r=r, leaf_size=leaf_size, **kwargs)
        self.r = r
        self.leaf_size = leaf_size

    @final
    def pairs(self, histories: Histories) -> Pairs:
        self._logger.debug('Mapping location histories to coordinates...')
        locs = self.to_coordinates(histories)
        self._logger.debug('Indexing coordinates to map back to users...')
        idx = self.index(locs)
        self._logger.debug('Querying pairs using spatial indexing...')
        pairs = self.query_pairs(locs)
        self._logger.debug('Filtering histories based on queried pairs...')
        return self.filter(pairs, idx, histories)

    @final
    def to_coordinates(self, hists: Histories) -> Histories:
        par = joblib.Parallel(self.n_workers)
        return par(joblib.delayed(self._to_coordinates)(h) for h in hists)

    @staticmethod
    @final
    def index(locations: Locations) -> ndarray:
        idx = arange(len(locations))
        repeats = [len(locs) for locs in locations]
        return repeat(idx, repeats)

    def query_pairs(self, locations: Locations) -> ndarray:
        raise NotImplementedError

    @final
    def filter(self, pairs: ndarray, idx: ndarray, hists: Histories) -> Pairs:
        distinct = unique(idx[pairs], axis=0)
        self._log_stats(len(hists), len(distinct))
        return ((hists[h1], hists[h2]) for (h1, h2) in distinct if h1 != h2)

    @staticmethod
    def _to_coordinates(hist: ndarray) -> ndarray:
        return vstack([to_coord(loc)['loc'] for loc in hist['locs']])

    @staticmethod
    @final
    def flatten_ragged(ragged: Iterable[ndarray]) -> ndarray:
        return array([*chain.from_iterable(ragged)])

    def _log_stats(self, n_hists: int, n_pairs: int) -> NoReturn:
        n_combos = n_hists ** 2
        percent = self._percent_decrease(n_combos, n_pairs)
        self._logger.info(
            'Pairs (tree / brute): %d / %d (%.2f percent)',
            n_pairs, n_combos, percent)

    @staticmethod
    def _percent_decrease(org, new) -> float:
        return 0 if org == 0 else round(100 * (new - org) / org, 2)


class BallTreeContactSearch(TreeContactSearch):
    __slots__ = ()

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            n_workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            **kwargs):
        super().__init__(min_dur, n_workers, r, leaf_size, **kwargs)

    def query_pairs(self, locations: Locations) -> ndarray:
        locs = self.flatten_ragged(locations)
        self._logger.debug('Constructing a ball tree spatial index')
        tree = neighbors.BallTree(locs, self.leaf_size, 'haversine')
        self._logger.debug('Querying for pairs')
        points = tree.query_radius(locs, self.r)
        idx = self.index(points)
        points = self.flatten_ragged(points)
        # Sorting along the last axis ensures duplicate pairs are removed.
        return sort(column_stack((idx, points)))


class KdTreeContactSearch(TreeContactSearch):
    __slots__ = ('p', 'eps')

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            n_workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            eps: int = 1e-5,
            p: float = 2,
            **kwargs):
        super().__init__(
            min_dur, n_workers, r, leaf_size, eps=eps, p=p, **kwargs)
        self.eps = eps
        self.p = p

    def query_pairs(self, locations: Locations) -> ndarray:
        locs = self._project(self.flatten_ragged(locations))
        self._logger.debug('Constructing a k-d tree spatial index')
        kd_tree = spatial.KDTree(locs, self.leaf_size)
        self._logger.debug('Querying for pairs')
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
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        transformer = pyproj.Transformer.from_proj(lla, ecef)
        x, y = transformer.transform(longs, lats, radians=False)
        return column_stack((x, y))
