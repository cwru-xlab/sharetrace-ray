import itertools
from typing import Iterable, NoReturn, Sequence, final

import joblib
import numpy as np
from scipy import spatial
from sklearn import neighbors

from sharetrace import model
from sharetrace.search.base import Histories, Pairs, ZERO
from sharetrace.search.brute import BruteContactSearch
from sharetrace.util.types import TimeDelta

Locations = Sequence[np.ndarray]


class TreeContactSearch(BruteContactSearch):
    __slots__ = ('r', 'p', 'eps', 'leaf_size')

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            n_workers: int = 1,
            r: float = 0.01,
            p: float = 2,
            leaf_size: int = 10,
            **kwargs):
        super().__init__(min_dur, n_workers, **kwargs)
        self.r = r
        self.p = p
        self.leaf_size = leaf_size

    @final
    def pairs(self, histories: Histories) -> Pairs:
        locs = self.to_coordinates(histories)
        idx = self.index(locs)
        pairs = self.query_pairs(locs)
        return self.filter(pairs, idx, histories)

    @final
    def to_coordinates(self, hists: Histories) -> Histories:
        par = joblib.Parallel(self.n_workers)
        return par(joblib.delayed(self._map_history)(h) for h in hists)

    @staticmethod
    @final
    def index(locations: Locations) -> np.ndarray:
        idx = np.arange(len(locations))
        repeats = [len(locs) for locs in locations]
        return np.repeat(idx, repeats)

    def query_pairs(self, locations: Locations) -> np.ndarray:
        raise NotImplementedError

    @final
    def filter(
            self,
            pairs: np.ndarray,
            idx: np.ndarray,
            hists: Histories) -> Pairs:
        unique = np.unique(idx[pairs], axis=0)
        self._log_stats(len(hists), len(unique))
        return ((hists[h1], hists[h2]) for (h1, h2) in unique if h1 != h2)

    @staticmethod
    def _map_history(hist: np.ndarray) -> np.ndarray:
        return np.vstack([model.to_coord(loc)['loc'] for loc in hist['locs']])

    @staticmethod
    @final
    def flatten_ragged(ragged: Iterable[np.ndarray]) -> np.ndarray:
        return np.array([*itertools.chain.from_iterable(ragged)])

    def _log_stats(self, n_hists: int, n_pairs: int) -> NoReturn:
        n_combos = n_hists ** 2
        percent = TreeContactSearch._percent_decrease(n_combos, n_pairs)
        self.logger.info(
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
            r: float = 0.01,
            p: float = 2,
            leaf_size: int = 10,
            **kwargs):
        super().__init__(min_dur, n_workers, r, p, leaf_size, **kwargs)

    def query_pairs(self, locations: Locations) -> np.ndarray:
        locs = self.flatten_ragged(locations)
        tree = neighbors.BallTree(locs, self.leaf_size, 'haversine')
        points = tree.query_radius(locs, self.r)
        idx = self.index(points)
        flat = self.flatten_ragged(points)
        # Sorting along the last axis ensures duplicate pairs are removed.
        return np.sort(np.column_stack((idx, flat)))


class KdTreeContactSearch(TreeContactSearch):
    __slots__ = ('eps',)

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            n_workers: int = 1,
            r: float = 0.01,
            p: float = 2,
            leaf_size: int = 10,
            eps: int = 0.001,
            **kwargs):
        super().__init__(min_dur, n_workers, r, p, leaf_size, **kwargs)
        self.eps = eps

    def query_pairs(self, locations: Locations) -> np.ndarray:
        locs = self.flatten_ragged(locations)
        kd_tree = spatial.KDTree(locs, self.leaf_size)
        return kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')
