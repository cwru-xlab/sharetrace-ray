import datetime
from collections import defaultdict, deque
from functools import reduce
from queue import Empty
from timeit import default_timer
from typing import (
    Any, Hashable, Iterable, Mapping, NoReturn, Optional, Sequence, Tuple
)

from numpy import (
    argmax, array, log, ndarray, sort, timedelta64, unique, void
)
from pymetis import part_graph

from lewicki.lewicki.actors import ActorSystem, BaseActor
from sharetrace.model import message, node, risk_score

NpSeq = Sequence[ndarray]
Graph = Mapping[Hashable, Any]

ACTOR_SYSTEM = -1
FACTOR = 0
DAY = timedelta64(1, 'D')
EMPTY = ()
DEFAULT_SCORE = risk_score(0, datetime.datetime.min)


def fkey(n1, n2) -> Tuple[Any, Any]:
    return min(n1, n2), max(n1, n2)


class Partition(BaseActor):
    __slots__ = (
        'graph',
        'send_thresh',
        'time_buffer',
        'time_const',
        'transmission',
        'timeout',
        'max_dur',
        'early_stop',
        '_nodes',
        '_start',
        '_since_update',
        '_timed_out')

    def __init__(
            self,
            graph: Graph,
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
        self.send_thresh = send_thresh
        self.time_buffer = timedelta64(int(time_buffer * 1e6), 'us')
        self.time_const = time_const
        self.transmission = transmission
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._nodes: Mapping[int, Mapping] = {}
        self._start: int = -1
        self._since_update: int = 0
        self._timed_out: bool = False

    def run(self) -> NoReturn:
        stop, receive, on_next = self.should_stop, self.receive, self.on_next
        self._start = default_timer()
        self._on_initial(receive())
        while not stop():
            if (msg := receive()) is not None:
                on_next(msg)
        results = {n: data['val'] for n, data in self._nodes.items()}
        self.send(results, ACTOR_SYSTEM)

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
        if (val := score['val']) > (curr := self._nodes[var])['val']:
            self._since_update, curr['val'] = 0, val
        elif self.early_stop is not None:
            self._since_update += 1
        factors = self.graph[var]['ne']
        self._send(array([score]), var, factors[factor != factors])

    def _on_initial(self, scores: NpSeq) -> NoReturn:
        self._nodes = {
            n: {
                'val': sort(nscores, order=('val', 'time')[-1])['val'],
                'prev': defaultdict(lambda: DEFAULT_SCORE)}
            for n, nscores in enumerate(scores)}
        graph, send = self.graph, self._send
        for var, score in self._nodes.items():
            send(score, var, graph[var]['ne'])

    def _send(self, scores: ndarray, var: int, factors: ndarray):
        graph, sgroup, send, time_buffer, time_const, transmission = (
            self.graph, self.name, self.send, self.time_buffer,
            self.time_const, self.transmission)
        prev = self._nodes[var]['prev']
        for f in factors:
            recent = graph[fkey(var, f)]
            scores = scores[scores['time'] <= recent + time_buffer]
            if len(scores) > 0:
                diff = (scores['time'] - recent) / DAY
                weighted = log(scores['val']) + (diff / time_const)
                maximum = scores[argmax(weighted)]
                maximum['val'] *= transmission
                newer = maximum['time'] > prev[f]['time']
                higher = maximum['val'] > prev[f]['val']
                if newer or higher:
                    dgroup = graph[f]['group']
                    send(message(maximum, var, sgroup, f, dgroup, FACTOR))
                    prev[f] = maximum

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
        'parts',
        'timeout',
        'max_dur',
        'early_stop',
        '_scores')

    def __init__(
            self,
            send_thresh: float = 0.75,
            time_buffer: float = 1.728e5,
            time_const: float = 1.,
            transmission: float = 0.8,
            parts: int = 1,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None):
        super().__init__(ACTOR_SYSTEM)
        self.send_thresh = send_thresh
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.parts = parts
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._scores: int = -1

    def setup(self, scores: NpSeq, contacts: NpSeq) -> NoReturn:
        self._scores = len(scores)
        graph, membership = self._create_graph(contacts)
        self._connect(graph)
        groups = self._group(scores, membership)
        self.send(*groups)

    def run(self) -> ndarray:
        super().run()
        return self._results()

    def _results(self) -> ndarray:
        results = reduce(
            lambda d1, d2: {**d1, **d2},
            (self.receive() for _ in range(self.parts)))
        return array([results[i] for i in range(self._scores)])

    def _connect(self, graph: Graph) -> NoReturn:
        self.connect(*(
            Partition(
                graph=graph,
                name=name,
                send_thresh=self.send_thresh,
                time_buffer=self.time_buffer,
                time_const=self.time_const,
                transmission=self.transmission,
                timeout=self.timeout,
                max_dur=self.max_dur,
                early_stop=self.early_stop)
            for name in range(self.parts)))

    def _group(self, scores: NpSeq, idx: Iterable[int]) -> Sequence[NpSeq]:
        groups = [list() for _ in range(self.parts)]
        for i, e in zip(idx, scores):
            # noinspection PyTypeChecker
            groups[i].append(e)
        return groups

    def send(self, *nodes: NpSeq) -> NoReturn:
        for p, pnodes in enumerate(nodes):
            self.outbox[p].put(pnodes, block=True)

    def _create_graph(self, contacts: NpSeq) -> Tuple[Graph, ndarray]:
        graph, adj = {}, defaultdict(list)
        # Assumes a name corresponds to an index in scores.
        for contact in contacts:
            n1, n2 = contact['names']
            adj[n1].append(n2)
            adj[n2].append(n1)
            graph[fkey(n1, n2)] = contact['time']
        # Keep indexing consistent in case of users that have no contacts.
        adj = [unique(adj.get(n, EMPTY)) for n in range(self._scores)]
        membership = self._partition(adj)
        # noinspection PyTypeChecker
        graph.update((n, node(ne, membership[n])) for n, ne in enumerate(adj))
        return graph, membership

    def _partition(self, adj: NpSeq) -> ndarray:
        cuts, membership = part_graph(self.parts, adj)
        return membership
