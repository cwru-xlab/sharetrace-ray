from collections import deque
from datetime import timedelta
from queue import Empty
from timeit import default_timer
from typing import Any, Hashable, Mapping, MutableMapping, NoReturn, Optional

from numpy import (all, argmax, array, clip, float_, inf, log, ndarray,
                   timedelta64, void)

from lewicki.lewicki.actors import BaseActor
from sharetrace.model import message
from sharetrace.util.types import TimeDelta

_ACTOR_SYSTEM = 0
_CONTACT = 1
_USER = 2
_DAY = timedelta64(1, 'D')
_TWO_DAYS = timedelta64(172_800, 's')


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
            time_buffer: TimeDelta = _TWO_DAYS,
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
            delta = float_(timedelta64(delta, 'us'))
        elif isinstance(delta, timedelta):
            delta = delta.microseconds
        return delta / 1e6

    def run(self) -> NoReturn:
        stop, receive, on_next = self.should_stop, self.receive, self.on_next
        self._start = default_timer()
        while not stop():
            if (msg := receive()) is not None:
                on_next(msg)
        self._push(self.nodes, _ACTOR_SYSTEM)

    def should_stop(self) -> bool:
        too_long = False
        if self.max_dur is not None:
            too_long = (default_timer() - self._start) >= self.max_dur
        no_updates = self._since_update >= self.early_stop
        return self._timed_out or too_long or no_updates

    def receive(self) -> Optional[void]:
        return self._pop_local() if len(self._local_inbox) > 0 else self._pop()

    def on_next(self, msg: void) -> NoReturn:
        if msg['kind'] == _USER:
            self._on_user_msg(msg)
        else:
            self._on_contact_msg(msg)

    def _on_contact_msg(self, msg: void) -> NoReturn:
        # Message value contains exactly 1 score.
        factor, var, vgroup, score = (
            msg['src'], msg['dest'], msg['dgroup'], msg['val'])
        nodes = self.nodes
        if score <= nodes[var]['val']:
            self._since_update += 1
        else:
            nodes[var] = msg
            self._since_update = 0
            graph = self.graph
            factors = graph[var]['ne']
            scores = array([score])
            send = self.send
            for f in factors[factor != factors]:
                send(message(scores, var, vgroup, f, graph[f]['group'], _USER))

    def _on_user_msg(self, msg: void) -> NoReturn:
        # Message value contains 1 or more scores.
        var, factor, fgroup, scores = (
            msg['src'], msg['dest'], msg['dgroup'], msg['val'])
        graph = self.graph
        variables = graph[factor]['ne']
        v_ne = variables[var != variables][0]
        vgroup = graph[v_ne]['group']
        recent = graph[factor]['data']['time']
        times = scores['time']
        scores = scores[times <= recent + self.time_buffer]
        if len(scores) > 0:
            diff = clip((times - recent) / _DAY, -inf, 0)
            weight = diff / self.time_const
            max_score = scores[argmax(log(scores['val']) + weight)]
            max_score['val'] *= self.transmission
            msg = message(max_score, factor, fgroup, v_ne, vgroup, _CONTACT)
            self.send(msg)

    def send(self, msg: void) -> NoReturn:
        if self._is_local(msg):
            self._push_local(msg)
        else:
            self._push(msg, msg['dgroup'])

    def _is_local(self, msg: void) -> bool:
        return all(msg['dgroup'] == self.group)

    def _push_local(self, msg: void) -> NoReturn:
        self._local_inbox.append(msg)

    def _pop_local(self) -> void:
        return self._local_inbox.popleft()

    def _pop(self) -> Optional[void]:
        try:
            msg = self.inbox.get(block=True, timeout=self.timeout)
        except Empty:
            msg, self._timed_out = None, True
        return msg

    def _push(self, msg: Any, key: Hashable) -> NoReturn:
        self.outbox[key].put(msg, block=True, timeout=self.timeout)
