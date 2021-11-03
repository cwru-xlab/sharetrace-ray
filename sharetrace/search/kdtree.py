import itertools
from typing import Iterable, NoReturn, Sequence

import joblib
import numpy as np
from scipy import spatial

from sharetrace import model
from sharetrace.search.base import Histories, Pairs, ZERO
from sharetrace.search.brute import BruteContactSearch
from sharetrace.util.types import TimeDelta

Locations = Sequence[np.ndarray]


class KdTreeContactSearch(BruteContactSearch):
    __slots__ = ('r', 'p', 'eps')

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            r: float = 0.01,
            p: float = 2,
            eps: float = 0.001,
            n_workers: int = 1,
            **kwargs):
        super().__init__(min_dur, n_workers, **kwargs)
        self.r = r
        self.p = p
        self.eps = eps

    def pairs(self, histories: Histories) -> Pairs:
        locs = self._to_coordinates(histories)
        idx = self._to_index(locs)
        pairs = self._query_pairs(locs)
        return self._filter(pairs, idx, histories)

    def _to_coordinates(self, hists: Histories) -> Histories:
        self.logger.debug('Converting to coordinate pairs')
        par = joblib.Parallel(self.n_workers)
        return par(joblib.delayed(self._map_history)(h) for h in hists)

    def _to_index(self, locations: Locations) -> np.ndarray:
        self.logger.debug('Generating mapping index')
        idx = np.arange(len(locations))
        repeats = [len(locs) for locs in locations]
        return np.repeat(idx, repeats)

    def _query_pairs(self, locations: Locations) -> np.ndarray:
        self.logger.debug('Querying the k-d tree for pairs')
        kd_tree = spatial.KDTree(self._flatten_ragged(locations))
        return kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')

    def _filter(
            self,
            pairs: np.ndarray,
            idx: np.ndarray,
            hists: Histories) -> Pairs:
        self.logger.debug('Filtering queried pairs')
        unique = np.unique(idx[pairs], axis=0)
        self._log_stats(len(hists), len(unique))
        return ((hists[h1], hists[h2]) for (h1, h2) in unique if h1 != h2)

    @staticmethod
    def _map_history(hist: np.ndarray) -> np.ndarray:
        return np.vstack([model.to_coord(loc)['loc'] for loc in hist['locs']])

    @staticmethod
    def _flatten_ragged(ragged: Iterable[np.ndarray]) -> np.ndarray:
        return np.array([*itertools.chain.from_iterable(ragged)])

    def _log_stats(self, n_hists: int, n_pairs: int) -> NoReturn:
        n_combos = n_hists ** 2
        percent = KdTreeContactSearch._percent_decrease(n_combos, n_pairs)
        self.logger.info(
            'Pairs (k-d tree / brute): %d / %d (%.2f percent)',
            n_pairs, n_combos, percent)

    @staticmethod
    def _percent_decrease(org, new) -> float:
        return 0 if org == 0 else round(100 * (new - org) / org, 2)
