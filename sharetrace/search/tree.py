from typing import NoReturn, Sequence

from joblib import Parallel, delayed
from numpy import (arange, column_stack, concatenate, ndarray, repeat, sort,
                   unique, vstack)
from pyproj import Proj, Transformer
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree

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
            workers: int = 1,
            r: float = 1e-4,
            leaf_size: int = 10,
            **kwargs):
        super().__init__(
            min_dur, workers, r=r, leaf_size=leaf_size, **kwargs)
        self.r = r
        self.leaf_size = leaf_size

    def pairs(self, histories: Histories) -> Pairs:
        self._logger.debug('Mapping location histories to coordinates...')
        locs = self.to_coordinates(histories)
        self._logger.debug('Indexing coordinates to map back to users...')
        idx = self.index(locs)
        self._logger.debug('Querying pairs using spatial indexing...')
        pairs = self.query_pairs(concatenate(locs))
        self._logger.debug('Filtering histories based on queried pairs...')
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
        self._logger.info(
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
        self._logger.debug('Constructing a ball tree spatial index')
        ball_tree = BallTree(locs, self.leaf_size, 'haversine')
        self._logger.debug('Querying for pairs')
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
        self._logger.debug('Constructing a k-d tree spatial index')
        kd_tree = KDTree(locs, self.leaf_size)
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
        ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        transformer = Transformer.from_proj(lla, ecef)
        return column_stack(transformer.transform(longs, lats, radians=False))
