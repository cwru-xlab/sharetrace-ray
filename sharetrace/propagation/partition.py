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
        'group_id'
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
            group_id: int,
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
        self.group_id = group_id
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
            delta = np.float_(timedelta64(delta, 'us')) / 1e6
        elif isinstance(delta, timedelta):
            delta = delta.microseconds / 1e6
        return delta

    def run(self) -> Mapping[Hashable, ndarray]:
        should_stop, receive, on_next = (
            self.should_stop, self.receive, self.on_next)
        self._start = default_timer()
        while not should_stop:
            on_next(receive())
        return self.nodes

    def should_stop(self) -> bool:
        too_long = False
        if self.max_dur is not None:
            too_long = (default_timer() - self._start) >= self.max_dur
        no_updates = self._since_update > self.early_stop
        return self._timed_out or too_long or no_updates

    def receive(self):
        # TODO Alternate with local or something
        return self.inbox.get(block=True, timeout=self.timeout)

    def on_next(self, msg: ndarray) -> NoReturn:
        kind = msg['kind']
        if kind == _USER:
            self._on_user_msg(msg)
        elif kind == _CONTACT:
            self._on_contact_msg(msg)
        else:
            # Log unknown message
            pass

    def _on_contact_msg(self, msg: ndarray) -> NoReturn:
        # Wrap the value since there can be multiple initial scores.
        contact, user, sgroup, scores = (
            msg['src'], msg['sgroup'], msg['dest'], array([msg['val']]))
        self._update(user, scores[0])
        contacts = self.graph[user]['ne']
        self.send(*(
            message(scores, user, sgroup, c, self.graph[c]['group'], _USER)
            for c in contacts if c != contact))

    def _on_user_msg(self, msg: ndarray) -> NoReturn:
        user, sgroup, contact, dgroup, scores = (
            msg['src'], msg['sgroup'], msg['dest'], msg['dgroup'], msg['val'])
        users = self.graph[contact]['ne']
        ne = users[user != users][0]
        ngroup = self.graph[ne]['group']
        recent = self.graph[contact]['data']['time']
        times = scores['time']
        scores = scores[times <= recent + self.time_buffer]
        if scores.size > 0:
            diff = (times - recent) / _DAY
            clip(diff, -inf, 0, out=diff)
            weight = diff / self.time_const
            max_score = scores[argmax(log(scores['val']) + weight)]
            max_score['val'] *= self.transmission
            msg = message(max_score, contact, dgroup, ne, ngroup, _CONTACT)
        else:
            msg = message(_DEFAULT_MSG, contact, dgroup, ne, ngroup, _CONTACT)
        self.send(array([msg]))

    def _update(self, node: int, msg: np.ndarray) -> NoReturn:
        curr = self.nodes[node]
        self.nodes[node] = np.sort((curr, msg), order=('val', 'time'))[-1]

    def send(self, *msgs: ndarray) -> NoReturn:
        in_group = functools.partial(lambda msg: msg['dgroup'] == self.group_id)
        for m in filter(in_group, msgs):
            self._local_inbox.append(m)
        for m in filterfalse(in_group, msgs):
            self.outbox[m['dgroup']].put(m, block=True, timeout=self.timeout)

    def handle_ack(self, msg):
        raise NotImplementedError

    def handle_call(self, msg):
        raise NotImplementedError

    def handle_return(self, msg):
        raise NotImplementedError

    def handle_set(self, msg):
        raise NotImplementedError

    def should_ignore(self, msg):
        raise NotImplementedError
