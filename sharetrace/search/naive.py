import functools
import itertools
from typing import Any, Iterable, Iterator, Optional

import joblib
import numpy as np

from sharetrace import model
from sharetrace.search.base import BaseContactSearch

_rng = np.random.default_rng()


class NaiveContactSearch(BaseContactSearch):
    __slots__ = ()

    def __init__(self, min_duration: np.timedelta64, n_workers: int = 1):
        super().__init__(min_duration, n_workers)

    def search(self, histories: np.ndarray) -> np.ndarray:
        pairs = self.pairs(histories)
        par = joblib.Parallel(n_jobs=self.n_workers)
        contacts = par(joblib.delayed(self._find_contact)(*p) for p in pairs)
        return np.array([c for c in contacts if c is not None])

    def pairs(self, histories: np.ndarray) -> Iterable:
        return itertools.combinations(histories, 2)

    def _find_contact(self, h1, h2):
        if (occurrences := self._find_occurrences(h1, h2)) is None:
            contact = None
        else:
            names = np.array([h1['name'], h2['name']])
            contact = model.contact(names, occurrences)
        return contact

    # noinspection PyTypeChecker
    def _find_occurrences(self, h1, h2):
        def advance(it: Iterator, n: int = 1, default: Any = None):
            nxt = functools.partial(lambda it_: next(it_, default))
            return nxt(it) if n == 1 else tuple(nxt(it) for _ in range(n))

        name1, name2 = h1['name'], h2['name']
        locs1, locs2 = h1['locations'], h2['locations']
        if locs1.size < 1 or locs2.size < 1 or name1 == name2:
            return None
        occurrences = []
        iter1, iter2 = iter(locs1), iter(locs2)
        (loc1, next1), (loc2, next2) = advance(iter1, 2), advance(iter2, 2)
        started = False
        start = self._later(loc1, loc2)
        while next1 is not None and next2 is not None:
            if loc1['location'] == loc2['location']:
                if started:
                    loc1, next1 = next1, advance(iter1)
                    loc2, next2 = next2, advance(iter2)
                else:
                    started = True
                    start = self._later(loc1, loc2)
            elif started:
                started = False
                if (occ := self._occurrence(start, loc1, loc2)) is not None:
                    occurrences.append(occ)
            elif loc1['timestamp'] < loc2['timestamp']:
                loc1, next1 = next1, advance(iter1)
            elif loc2['timestamp'] < loc1['timestamp']:
                loc2, next2 = next2, advance(iter2)
            elif _rng.choice((True, False)):
                loc1, next1 = next1, advance(iter1)
            else:
                loc2, next2 = next2, advance(iter2)
        if started:
            if (occ := self._occurrence(start, loc1, loc2)) is not None:
                occurrences.append(occ)
        return np.array(occurrences) if len(occurrences) > 0 else None

    def _occurrence(self, start, loc1, loc2) -> Optional[np.ndarray]:
        end = self._earlier(loc1, loc2)
        if (duration := end - start) >= self.min_duration:
            occurrence = model.occurrence(start, duration)
        else:
            occurrence = None
        return occurrence

    @staticmethod
    def _later(loc1, loc2):
        t1, t2 = loc1['timestamp'], loc2['timestamp']
        return t1 if t1 > t2 else t2

    @staticmethod
    def _earlier(loc1, loc2):
        t1, t2 = loc1['timestamp'], loc2['timestamp']
        return t1 if t1 < t2 else t2
