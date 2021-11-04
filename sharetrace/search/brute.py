from itertools import combinations
from typing import Iterable, List, Optional

from joblib import Parallel, delayed
from numpy import array, ndarray
from numpy.random import default_rng

from sharetrace.model import contact, event
from sharetrace.search.base import (
    BaseContactSearch, Contacts, Histories, Pairs, ZERO
)
from sharetrace.util.timer import Timer
from sharetrace.util.types import TimeDelta

_rng = default_rng()
_EMPTY = ()


class BruteContactSearch(BaseContactSearch):
    __slots__ = ()

    def __init__(self, min_dur: TimeDelta = ZERO, workers: int = 1, **kwargs):
        super().__init__(min_dur, workers, **kwargs)

    def search(self, histories: Histories) -> Contacts:
        timer = Timer.time(lambda: self._search(histories))
        self._logger.info(
            '%s: %.2f seconds', self.__class__.__name__, timer.seconds)
        return timer.result

    def _search(self, histories: Histories) -> Contacts:
        self._logger.debug('Finding pairs to search...')
        pairs = self.pairs(histories)
        self._logger.debug('Initiating contact search...')
        par = Parallel(n_jobs=self.workers)
        contacts = par(delayed(self._find_contact)(*p) for p in pairs)
        contacts = array([c for c in contacts if c is not None])
        self._logger.debug('Contact search completed')
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
