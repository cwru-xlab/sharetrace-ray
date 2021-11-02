import itertools
import logging.config
from typing import Iterable, NoReturn, Sequence

import joblib
import numpy as np
from scipy import spatial

from sharetrace import logging_config, model
from sharetrace.search.base import Histories, Pairs, ZERO
from sharetrace.search.naive import NaiveContactSearch
from sharetrace.util.types import TimeDelta

logging.config.dictConfig(logging_config.config)
logger = logging.getLogger(__name__)

Locations = Sequence[np.ndarray]


class KdTreeContactSearch(NaiveContactSearch):
    __slots__ = ('r', 'p', 'eps')

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            r: float = 1,
            p: float = 2,
            eps: float = 0,
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
        logger.debug('Converting to coordinate pairs')
        par = joblib.Parallel(self.n_workers)
        return par(joblib.delayed(self._map_history)(h) for h in hists)

    @staticmethod
    def _to_index(locations: Locations) -> np.ndarray:
        logger.debug('Generating mapping index')
        idx = np.arange(len(locations))
        repeats = [len(locs) for locs in locations]
        return np.repeat(idx, repeats)

    def _query_pairs(self, locations: Locations) -> np.ndarray:
        logger.debug('Querying the k-d tree for pairs')
        kd_tree = spatial.KDTree(self._flatten_ragged(locations))
        return kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')

    @staticmethod
    def _filter(pairs: np.ndarray, idx: np.ndarray, hists: Histories) -> Pairs:
        logger.debug('Filtering queried pairs')
        unique = np.unique(idx[pairs], axis=0)
        KdTreeContactSearch._log_percent_diff(len(hists), len(unique))
        return ((hists[h1], hists[h2]) for (h1, h2) in unique if h1 != h2)

    @staticmethod
    def _map_history(hist: np.ndarray) -> np.ndarray:
        return np.vstack([model.to_coord(loc)['loc'] for loc in hist['locs']])

    @staticmethod
    def _flatten_ragged(ragged: Iterable[np.ndarray]) -> np.ndarray:
        return np.array([*itertools.chain.from_iterable(ragged)])

    @staticmethod
    def _log_percent_diff(n_hists: int, n_pairs: int) -> NoReturn:
        n_combos = n_hists ** 2
        diff = n_combos - n_pairs
        percent = KdTreeContactSearch._percent_diff(n_combos, n_pairs)
        logger.info(
            'Number fewer pairs than naive: %d (%.2f percent)', diff, percent)

    @staticmethod
    def _percent_diff(x, y) -> float:
        return 0 if x == 0 else round(100 * (x - y) / x, 2)
