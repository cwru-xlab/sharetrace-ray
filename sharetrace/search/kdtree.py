import itertools
import logging
from typing import Iterable, NoReturn, Sequence, Tuple

import numpy as np
from scipy import spatial

from sharetrace.search.naive import NaiveContactSearch
from sharetrace.util import logging as log
from sharetrace.util.types import TimeDelta

logger = log.get_stdout_logger(__name__, logging.INFO)


class KdTreeContactSearch(NaiveContactSearch):
    __slots__ = ('r', 'p', 'eps', '_naive',)

    def __init__(
            self,
            min_duration: TimeDelta,
            r: float,
            p: float = 2,
            eps: float = 0,
            n_workers: int = 1):
        super().__init__(min_duration, n_workers)
        self.r = r
        self.p = p
        self.eps = eps

    def pairs(self, histories: Sequence[np.ndarray]) -> Iterable:
        names, locations = self._extract(histories)
        kd_tree = spatial.KDTree(self._flatten_ragged(locations))
        pairs = kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')
        idx = self._map(locations)
        return self._filter(pairs, histories, idx)

    @staticmethod
    def _extract(
            histories: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        names = np.array([h['name'] for h in histories])
        locations = [np.column_stack((h['lat'], h['long'])) for h in histories]
        return names, locations

    @staticmethod
    def _map(locations: Sequence[np.ndarray]) -> np.ndarray:
        idx = np.arange(len(locations))
        repeats = [len(locs) for locs in locations]
        return np.repeat(idx, repeats)

    @staticmethod
    def _filter(
            pairs: np.ndarray,
            histories: Sequence[np.ndarray],
            idx: np.ndarray
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        KdTreeContactSearch._log_percent_diff(len(histories), len(pairs))
        return ((histories[h1], histories[h2]) for (h1, h2) in idx[pairs])

    @staticmethod
    def _flatten_ragged(ragged: Iterable[np.ndarray]) -> np.ndarray:
        return np.array([*itertools.chain.from_iterable(ragged)])

    @staticmethod
    def _log_percent_diff(n_histories: int, n_pairs: int) -> NoReturn:
        n_combos = KdTreeContactSearch._n_combos(n_histories)
        percent_diff = KdTreeContactSearch._percent_diff(n_combos, n_pairs)
        logger.info('Percent fewer pairs than naive: %.2f', percent_diff)

    @staticmethod
    def _n_combos(n: int) -> int:
        return int(n * (n - 1) / 2)

    @staticmethod
    def _percent_diff(x, y, precision: int = 2) -> float:
        return 0 if x == 0 else round(100 * (x - y) / x, precision)
