import functools
import itertools
import logging.config
from typing import Any, Iterator, Optional

import joblib
import numpy as np

from sharetrace import logging_config, model
from sharetrace.search.base import (
    BaseContactSearch, Contacts, Histories, Pairs, ZERO
)
from sharetrace.util.types import TimeDelta

_rng = np.random.default_rng()

logging.config.dictConfig(logging_config.config)
logger = logging.getLogger(__name__)


class NaiveContactSearch(BaseContactSearch):
    __slots__ = ()

    def __init__(
            self,
            min_dur: TimeDelta = ZERO,
            n_workers: int = 1,
            **kwargs):
        super().__init__(min_dur, n_workers, **kwargs)

    def search(self, histories: Histories) -> Contacts:
        # timed = Timer.time(lambda: self._search(histories))
        # secs = timed.duration.microseconds / 1e6
        # logger.info('Contact search: %.3f seconds', secs)
        return self._search(histories)

    def _search(self, histories: Histories) -> Contacts:
        pairs = self.pairs(histories)
        par = joblib.Parallel(n_jobs=self.n_workers)
        contacts = par(joblib.delayed(self._find_contact)(*p) for p in pairs)
        return np.array([c for c in contacts if c is not None])

    def pairs(self, histories: Histories) -> Pairs:
        return itertools.combinations(histories, 2)

    def _find_contact(self, h1, h2):
        events = self._find_events(h1, 0, h2, 0)
        if events is None:
            contact = None
        else:
            names = [h1['name'], h2['name']]
            contact = model.contact(names, events)
        return contact

    def _find(self, h1, i1, h2, i2):
        events = []
        len1, len2 = h1.size - 1, h2.size - 1
        while i1 < len1 and i2 < len2:
            pass

    # TODO Refactor to use indices and recursion
    # noinspection PyTypeChecker
    def _find_events(self, h1, h2):
        def advance(it: Iterator, n: int = 1, default: Any = None):
            nxt = functools.partial(lambda it_: next(it_, default))
            return nxt(it) if n == 1 else tuple(nxt(it) for _ in range(n))

        name1, name2 = h1['name'], h2['name']
        locs1, locs2 = h1['locs'], h2['locs']
        if locs1.size < 1 or locs2.size < 1 or name1 == name2:
            return None
        events = []
        iter1, iter2 = iter(locs1), iter(locs2)
        (loc1, next1), (loc2, next2) = advance(iter1, 2), advance(iter2, 2)
        started = False
        start = self._later(loc1, loc2)
        while next1 is not None and next2 is not None:
            if loc1['loc'] == loc2['loc']:
                if started:
                    loc1, next1 = next1, advance(iter1)
                    loc2, next2 = next2, advance(iter2)
                else:
                    started = True
                    start = self._later(loc1, loc2)
            elif started:
                started = False
                event = self._event(start, loc1, loc2)
                if event is not None:
                    events.append(event)
            elif loc1['time'] < loc2['time']:
                loc1, next1 = next1, advance(iter1)
            elif loc2['time'] < loc1['time']:
                loc2, next2 = next2, advance(iter2)
            elif _rng.choice((True, False)):
                print(name1, name2, 'incrementing 1 randomly')
                loc1, next1 = next1, advance(iter1)
            else:
                print(name1, name2, 'incrementing 2 randomly')
                loc2, next2 = next2, advance(iter2)
        if started:
            event = self._event(start, loc1, loc2)
            if event is not None:
                events.append(event)
        return events if len(events) > 0 else None

    def _event(self, start, loc1, loc2) -> Optional[np.ndarray]:
        end = self._earlier(loc1, loc2)
        dur = end - start
        if dur >= self.min_dur:
            event = model.event(start, dur)
        else:
            event = None
        return event

    @staticmethod
    def _later(loc1, loc2):
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 > t2 else t2

    @staticmethod
    def _earlier(loc1, loc2):
        t1, t2 = loc1['time'], loc2['time']
        return t1 if t1 < t2 else t2
