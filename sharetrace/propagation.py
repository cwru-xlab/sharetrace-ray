from collections import deque
from queue import Empty
from timeit import default_timer
from typing import Any, Hashable, Mapping, NoReturn, Optional, Tuple

from numpy import argmax, array, log, ndarray, sort, timedelta64, void

from lewicki.lewicki.actors import ActorSystem, BaseActor
from sharetrace.model import message

ACTOR_SYSTEM = -1
FACTOR = 0
DAY = timedelta64(1, 'D')


def fkey(n1, n2) -> Tuple[Any, Any]:
    return min(n1, n2), max(n1, n2)


class Partition(BaseActor):
    __slots__ = (
        'graph',
        'nodes',
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
            graph: Mapping[Hashable, ndarray],
            name: Hashable,
            send_thresh: float = 0.75,
            time_buffer: float = 1.728e5,
            time_const: float = 1.,
            transmission: float = 0.8,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None):
        super().__init__(name)
        # noinspection PyTypeChecker
        self.outbox[name] = deque()
        self.graph = graph
        self.nodes = None
        self.send_thresh = send_thresh
        self.time_buffer = timedelta64(int(time_buffer * 1e6), 'us')
        self.time_const = time_const
        self.transmission = transmission
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._start = None
        self._since_update = 0
        self._timed_out = False

    def run(self) -> NoReturn:
        stop, receive, on_next = self.should_stop, self.receive, self.on_next
        self._start = default_timer()
        self._on_initial(receive())
        while not stop():
            if (msg := receive()) is not None:
                on_next(msg)
        self.send(self.nodes, ACTOR_SYSTEM)

    def should_stop(self) -> bool:
        too_long, no_updates = False, False
        if self.max_dur is not None:
            too_long = (default_timer() - self._start) >= self.max_dur
        if self.early_stop is not None:
            no_updates = self._since_update >= self.early_stop
        return self._timed_out or too_long or no_updates

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def receive(self) -> Optional[Any]:
        if len((local := self.outbox[self.name])) > 0:
            msg = local.popleft()
        else:
            try:
                msg = self.inbox.get(block=True, timeout=self.timeout)
            except Empty:
                msg, self._timed_out = None, True
        return msg

    def on_next(self, msg: void) -> NoReturn:
        factor, var, score = msg['src'], msg['dest'], msg['val']
        if (val := score['val']) > self.nodes[var]:
            self._since_update, self.nodes[var] = 0, val
            factors = self.graph[var]['ne']
            self._send(array([score]), var, factors[factor != factors])
        elif self.early_stop is not None:
            self._since_update += 1

    def _on_initial(self, scores: Mapping[Hashable, ndarray]) -> NoReturn:
        self.nodes = {
            var: sort(vscores, order=('val', 'time')[-1])['val']
            for var, vscores in scores.items()}
        graph, send = self.graph, self._send
        for var, values in scores.items():
            send(values, var, graph[var]['ne'])

    def _send(self, scores: ndarray, var: Hashable, factors: ndarray):
        graph, sgroup, send, time_buffer, time_const, transmission = (
            self.graph, self.name, self.send, self.time_buffer,
            self.time_const, self.transmission)
        for f in factors:
            recent = graph[fkey(var, f)]['data']['time']
            scores = scores[scores['time'] <= recent + time_buffer]
            if len(scores) > 0:
                diff = (scores['time'] - recent) / DAY
                weighted = log(scores['val']) + (diff / time_const)
                maximum = scores[argmax(weighted)]
                maximum['val'] *= transmission
                dgroup = graph[f]['group']
                send(message(maximum, var, sgroup, f, dgroup, FACTOR))

    def send(self, msg: Any, key: Hashable = None) -> NoReturn:
        key = key if key is not None else msg['dgroup']
        if key == self.name:
            # noinspection PyUnresolvedReferences
            self.outbox[key].append(msg)
        else:
            self.outbox[key].put(msg, block=True, timeout=self.timeout)


class RiskPropagation(ActorSystem):
    __slots__ = (
        'send_thresh',
        'time_buffer',
        'time_const',
        'transmission',
        'timeout',
        'max_dur',
        'early_stop')

    def __init__(
            self,
            send_thresh: float = 0.75,
            time_buffer: float = 1.728e5,
            time_const: float = 1.,
            transmission: float = 0.8,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None):
        super().__init__(ACTOR_SYSTEM)
        self.send_thresh = send_thresh
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop

    def compute(self, scores: ndarray, contacts: ndarray) -> ndarray:
        ...

    def setup(self):
        ...

    def create_graph(self, contacts: ndarray, n_parts: int = 1):
        ...
