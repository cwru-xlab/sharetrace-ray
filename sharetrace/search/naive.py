import itertools
import logging.config
from typing import Iterable, List, Optional

import joblib
import numpy as np

from sharetrace import logging_config, model
from sharetrace.search.base import (
    BaseContactSearch, Contacts, Histories, Pairs, ZERO
)
from sharetrace.util.timer import Timer
from sharetrace.util.types import TimeDelta

_rng = np.random.default_rng()

logging.config.dictConfig(logging_config.config)
logger = logging.getLogger(__name__)

_EMPTY = ()


class NaiveContactSearch(BaseContactSearch):
    __slots__ = ()

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            n_workers: int = 1,
            **kwargs):
        super().__init__(min_dur, n_workers, **kwargs)

    def search(self, histories: Histories) -> Contacts:
        timed = Timer.time(lambda: self._search(histories))
        secs = timed.duration.microseconds / 1e6
        logger.info('Contact search: %.3f seconds', secs)
        return timed.result

    def _search(self, histories: Histories) -> Contacts:
        pairs = self.pairs(histories)
        par = joblib.Parallel(n_jobs=self.n_workers)
        contacts = par(joblib.delayed(self._find_contact)(*p) for p in pairs)
        return np.array([c for c in contacts if c is not None])

    def pairs(self, histories: Histories) -> Pairs:
        return itertools.combinations(histories, 2)

    def _find_contact(self, h1, h2) -> Optional[np.ndarray]:
        contact = None
        name1, name2 = h1['name'], h2['name']
        if name1 != name2:
            events = self._find(h1['locs'], 0, h2['locs'], 0)
            if len(events) > 0:
                names = (name1, name2)
                contact = model.contact(names, events)
        return contact

    # noinspection PyTypeChecker
    def _find(self, locs1, i1, locs2, i2) -> List[np.ndarray]:
        events = []
        later, event, find, add_events = (
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
                add_events(event(start, loc1, loc2))
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
            add_events(event(start, loc1, loc2))
        return events

    def _event(self, start, loc1, loc2) -> Iterable[np.ndarray]:
        end = self._earlier(loc1, loc2)
        dur = end - start
        return [model.event(start, dur)] if dur >= self.min_dur else _EMPTY

    @staticmethod
    def _later(loc1, loc2):
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 > t2 else t2

    @staticmethod
    def _earlier(loc1, loc2):
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 < t2 else t2
