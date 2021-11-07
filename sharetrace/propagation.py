from collections import deque
from datetime import timedelta
from queue import Empty
from timeit import default_timer
from typing import Any, Hashable, Mapping, MutableMapping, NoReturn, Optional

from numpy import (argmax, array, clip, float_, inf, log, ndarray, timedelta64,
                   void)

from lewicki.lewicki.actors import ActorSystem, BaseActor
from sharetrace.model import message
from sharetrace.util import TimeDelta

_ACTOR_SYSTEM = 0
_FACTOR = 1
_VAR = 2
_DAY = timedelta64(1, 'D')
_TWO_DAYS = timedelta64(172_800, 's')


class Partition(BaseActor):
    __slots__ = (
        'graph',
        'nodes',
        'group',
        'send_thresh',
        'time_buffer',
        'time_const',
        'transmission',
        'timeout',
        'max_dur',
        'early_stop',
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
        # noinspection PyTypeChecker
        self.outbox[group] = deque()
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
        self.send(self.nodes, _ACTOR_SYSTEM)

    def should_stop(self) -> bool:
        too_long = False
        if self.max_dur is not None:
            too_long = (default_timer() - self._start) >= self.max_dur
        no_updates = self._since_update >= self.early_stop
        return self._timed_out or too_long or no_updates

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def receive(self) -> Optional[void]:
        if len((local := self.outbox[self.group])) > 0:
            msg = local.popleft()
        else:
            try:
                msg = self.inbox.get(block=True, timeout=self.timeout)
            except Empty:
                msg, self._timed_out = None, True
        return msg

    def on_next(self, msg: void) -> NoReturn:
        if msg['kind'] == _VAR:
            self._on_variable_msg(msg)
        else:
            self._on_factor_msg(msg)

    def _on_factor_msg(self, msg: void) -> NoReturn:
        factor, var, vgroup, score = (
            msg['src'], msg['dest'], msg['dgroup'], msg['val'])
        graph, nodes, send = self.graph, self.nodes, self.send
        if (val := score['val']) <= nodes[var]:
            self._since_update += 1
        else:
            self._since_update, nodes[var] = 0, val
            factors = graph[var]['ne']
            scores = array([score])
            for f in factors[factor != factors]:
                send(message(scores, var, vgroup, f, graph[f]['group'], _VAR))

    def _on_variable_msg(self, msg: void) -> NoReturn:
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
            maximum = scores[argmax(log(scores['val']) + weight)]
            maximum['val'] *= self.transmission
            self.send(message(maximum, factor, fgroup, v_ne, vgroup, _FACTOR))

    def send(self, msg: Any, key: Hashable = None) -> NoReturn:
        key = key if key is not None else msg['dgroup']
        if key == self.group:
            # noinspection PyUnresolvedReferences
            self.outbox[key].append(msg)
        else:
            self.outbox[key].put(msg, block=True, timeout=self.timeout)


class RiskPropagation(ActorSystem):
    __slots__ = (
        'graph',
        'nodes',
        'send_thresh',
        'time_buffer',
        'time_const',
        'transmission',
        'timeout',
        'max_dur')

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
        super().__init__()

    def call(self, scores: ndarray, contacts: ndarray) -> ndarray:
        ...

    def create_graph(self, contacts: ndarray, n_parts: int = 1):
        ...
