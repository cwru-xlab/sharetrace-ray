import functools
from collections import deque
from datetime import timedelta
from itertools import filterfalse
from timeit import default_timer
from typing import Hashable, Mapping, MutableMapping, NoReturn, Optional

import numpy as np
from numpy import (
    argmax, array, clip, datetime64, inf, log, ndarray, timedelta64
)

from lewicki.lewicki.actors import BaseActor
from sharetrace.model import message, risk_score
from sharetrace.util.types import TimeDelta

_CONTACT = 1
_USER = 2
_DAY = timedelta64(1, 'D')
_DEFAULT_MSG = risk_score(0, datetime64('0'))
TWO_DAYS = np.timedelta64(172_800, 's')


class Partition(BaseActor):
    __slots__ = (
        'graph',
        'nodes',
        'group'
        'send_thresh',
        'time_buffer',
        'time_const',
        'transmission',
        'timeout',
        'max_dur',
        'early_stop',
        '_local_inbox',
        '_start',
        '_since_update',
        '_timed_out')

    def __init__(
            self,
            graph: Mapping[Hashable, Mapping],
            nodes: MutableMapping[Hashable, ndarray],
            group: int,
            name: Optional[Hashable] = None,
            send_thresh: float = 0.75,
            time_buffer: TimeDelta = TWO_DAYS,
            time_const: float = 1,
            transmission: float = 0.8,
            timeout: Optional[TimeDelta] = None,
            max_dur: Optional[TimeDelta] = None,
            early_stop: int = inf):
        super().__init__(name)
        self.graph = graph
        self.nodes = nodes
        self.group = group
        self.send_thresh = send_thresh
        self.time_buffer = timedelta64(time_buffer)
        self.time_const = time_const
        self.transmission = transmission
        self.timeout = self._seconds(timeout)
        self.max_dur = self._seconds(max_dur)
        self.early_stop = early_stop
        self._local_inbox = deque()
        self._start = None
        self._since_update = 0
        self._timed_out = False

    @staticmethod
    def _seconds(delta: Optional[TimeDelta]) -> Optional[float]:
        if isinstance(delta, timedelta64):
            delta = np.float_(timedelta64(delta, 'us'))
        elif isinstance(delta, timedelta):
            delta = delta.microseconds
        return delta / 1e6

    def run(self) -> Mapping[Hashable, ndarray]:
        should_stop, receive, on_next = (
            self.should_stop, self.receive, self.on_next)
        self._start = default_timer()
        while not should_stop():
            on_next(receive())
        # TODO Can't return directly with lewicki - send in queue to actor sys
        return self.nodes

    def should_stop(self) -> bool:
        too_long = False
        if self.max_dur is not None:
            too_long = (default_timer() - self._start) >= self.max_dur
        no_updates = self._since_update >= self.early_stop
        # TODO Add logic to flip this timed_out flag
        return self._timed_out or too_long or no_updates

    def receive(self):
        # TODO Alternate with local or something
        return self.inbox.get(block=True, timeout=self.timeout)

    def on_next(self, msg: ndarray) -> NoReturn:
        if msg['kind'] == _USER:
            self._on_user_msg(msg)
        else:
            self._on_contact_msg(msg)

    def _on_contact_msg(self, msg: ndarray) -> NoReturn:
        # Wrap the value since there can be multiple initial scores.
        factor, var, vgroup, scores = (
            msg['src'], msg['dest'], msg['dgroup'], array([msg['val']]))
        self._update(var, scores[0])
        factors = self.graph[var]['ne']
        self.send(*(
            message(scores, var, vgroup, f, self.graph[f]['group'], _USER)
            for f in factors if f != factor))

    def _on_user_msg(self, msg: ndarray) -> NoReturn:
        var, factor, fgroup, scores = (
            msg['src'], msg['dest'], msg['dgroup'], msg['val'])
        variables = self.graph[factor]['ne']
        var = variables[var != variables][0]
        vgroup = self.graph[var]['group']
        recent = self.graph[factor]['data']['time']
        times = scores['time']
        scores = scores[times <= recent + self.time_buffer]
        if len(scores) > 0:
            diff = clip((times - recent) / _DAY, -inf, 0)
            weight = diff / self.time_const
            max_score = scores[argmax(log(scores['val']) + weight)]
            max_score['val'] *= self.transmission
            msg = message(max_score, factor, fgroup, var, vgroup, _CONTACT)
        else:
            msg = message(_DEFAULT_MSG, factor, fgroup, var, vgroup, _CONTACT)
        self.send(array([msg]))

    def _update(self, node: int, msg: np.ndarray) -> NoReturn:
        curr = self.nodes[node]
        self.nodes[node] = np.sort((curr, msg), order=('val', 'time'))[-1]

    def send(self, *msgs: ndarray) -> NoReturn:
        in_group = functools.partial(lambda msg: msg['dgroup'] == self.group)
        for m in filter(in_group, msgs):
            self._local_inbox.append(m)
        for m in filterfalse(in_group, msgs):
            self.outbox[m['dgroup']].put(m, block=True, timeout=self.timeout)
