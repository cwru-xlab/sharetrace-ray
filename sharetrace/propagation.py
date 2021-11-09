import datetime
from collections import defaultdict, deque
from functools import reduce
from logging import getLogger
from logging.config import dictConfig
from queue import Empty
from timeit import default_timer
from typing import (
    Any, Hashable, Iterable, Mapping, MutableMapping, NoReturn, Optional,
    Sequence, Tuple
)

from numpy import (
    argmax, array, log, ndarray, sort, timedelta64, unique, void
)
from pymetis import part_graph

from lewicki.actors import ActorSystem, BaseActor
from sharetrace.model import message, node, risk_score
from sharetrace.util import LOGGING_CONFIG, Timer

NpSeq = Sequence[ndarray]
NpMap = Mapping[int, ndarray]
Graph = Mapping[Hashable, Any]
MutableGraph = MutableMapping[Hashable, Any]
Nodes = Mapping[int, void]

ACTOR_SYSTEM = -1
DAY = timedelta64(1, 'D')
EMPTY = ()
DEFAULT_SCORE = risk_score(0, datetime.datetime.min)

dictConfig(LOGGING_CONFIG)


def fkey(n1, n2) -> Tuple[Any, Any]:
    return min(n1, n2), max(n1, n2)


class Partition(BaseActor):
    __slots__ = (
        'graph',
        'time_buffer',
        'time_const',
        'transmission',
        'timeout',
        'max_dur',
        'early_stop',
        '_local',
        '_nodes',
        '_start',
        '_since_update',
        '_timed_out',
        '_total_msgs',
        '_logger')

    def __init__(
            self,
            name: Hashable,
            graph: Graph,
            time_buffer: int,
            time_const: float,
            transmission: float,
            timeout: Optional[float],
            max_dur: Optional[float],
            early_stop: Optional[int]):
        super().__init__(name)
        self.graph = graph
        self.time_buffer = timedelta64(time_buffer, 's')
        self.time_const = time_const
        self.transmission = transmission
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._local = deque()
        self._nodes: Nodes = {}
        self._start: int = -1
        self._since_update: int = 0
        self._timed_out: bool = False
        self._total_msgs = 0
        self._logger = getLogger(__name__)

    def run(self) -> NoReturn:
        self._logger.info('Partition %d - initializing...', self.name)
        timed = Timer.time(self._run)
        self._logger.info(
            'Partition %d - runtime: %.2f seconds', self.name, timed.seconds)
        self._logger.info(
            'Partition %d - messages processed: %d',
            self.name, self._total_msgs)
        results = {n: data['curr']['val'] for n, data in self._nodes.items()}
        # Must send the results of the child process via queue.
        self.send(results, ACTOR_SYSTEM)

    def _run(self):
        stop, receive, on_next = self.should_stop, self.receive, self.on_next
        self._start = default_timer()
        self._on_initial(receive())
        while not stop():
            if (msg := receive()) is not None:
                on_next(msg)

    def should_stop(self) -> bool:
        too_long, no_updates = False, False
        if self.max_dur is not None:
            if too_long := ((default_timer() - self._start) >= self.max_dur):
                self._logger.info(
                    'Partition %d - maximum duration of %.2f seconds reached.',
                    self.name, self.max_dur)
        if self.early_stop is not None:
            if no_updates := (self._since_update >= self.early_stop):
                self._logger.info(
                    'Partition %d - early stopping after %d messages.',
                    self.name, self.early_stop)
        if self._timed_out:
            self._logger.info(
                'Partition %d - timed out after %.2f seconds',
                self.name, self.timeout)
        return self._timed_out or too_long or no_updates

    def receive(self) -> Optional[Any]:
        # Prioritize local convergence over processing remote messages.
        if len((local := self._local)) > 0:
            msg = local.popleft()
        else:
            try:
                msg = self.inbox.get(block=True, timeout=self.timeout)
            except Empty:
                msg, self._timed_out = None, True
        self._total_msgs += msg is not None
        return msg

    def on_next(self, msg: void) -> NoReturn:
        """Update the variable node and send messages to neighbors."""
        factor, var, score = msg['src'], msg['dest'], msg['val']
        # As a variable node, update its current value.
        self._update(var, score)
        # As factor nodes, send a message to each neighboring variable node.
        factors = self.graph[var]['ne']
        self._send(array([score]), var, factors[factor != factors])

    def _update(self, var: int, score: void) -> NoReturn:
        """Update the exposure score of the current variable node."""
        updated, data = False, self._nodes[var]
        if (new := score['val']) > data['curr']['val']:
            self._since_update, data['curr']['val'] = 0, new
            updated = True
        if (new := score['time']) > data['curr']['time']:
            self._since_update, data['curr']['time'] = 0, new
            updated = True
        if self.early_stop is not None and not updated:
            self._since_update += 1

    def _on_initial(self, scores: NpMap) -> NoReturn:
        self._logger.info(
            'Partition %d - nodes: %d', self.name, len(scores))
        # Must be a mapping since indices may neither start at 0 nor be
        # contiguous, based on the original input scores.
        nodes = {}
        for n, nscores in scores.items():
            init = sort(nscores, order=('val', 'time'))[-1]
            nodes[n] = {
                'init': init,
                'curr': init,
                'prev': defaultdict(lambda: DEFAULT_SCORE)}
        self._nodes = nodes
        graph, send = self.graph, self._send
        # Send initial symptom score messages to all neighbors.
        for n, nscores in scores.items():
            send(nscores, n, graph[n]['ne'])

    def _send(self, scores: ndarray, var: int, factors: ndarray):
        """Compute a factor node message and send if it will be effective."""
        graph, init, prev, sgroup, send, buffer, const, transmission = (
            self.graph, self._nodes[var]['init'], self._nodes[var]['prev'],
            self.name, self.send, self.time_buffer, self.time_const,
            self.transmission)
        for f in factors:
            # Only consider scores that may have been transmitted from contact.
            recent = graph[fkey(var, f)]
            scores = scores[scores['time'] <= recent + buffer]
            if len(scores) > 0:
                # Scales time deltas in partial days.
                diff = (scores['time'] - recent) / DAY
                # Use the log transform to avoid overflow issues.
                weighted = log(scores['val']) + (diff / const)
                score = scores[argmax(weighted)]
                score['val'] *= transmission
                higher = score['val'] > prev[f]['val']
                newer = score['time'] > prev[f]['time']
                high_enough = score['val'] > init['val']
                if higher or (newer and high_enough):
                    send(message(score, var, sgroup, f, graph[f]['group']))
                    prev[f] = score

    def send(self, msg: Any, key: int = None) -> NoReturn:
        """Send a message either to the local inbox or an outbox."""
        # msg must be a message structured array if key is None
        key = key if key is not None else msg['dgroup']
        if key == self.name:
            self._local.append(msg)
        else:
            self.outbox[key].put(msg, block=True, timeout=self.timeout)


class RiskPropagation(ActorSystem):
    __slots__ = (
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
            time_buffer: int = 172_800,
            time_const: float = 1.,
            transmission: float = 0.8,
            parts: int = 1,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None):
        super().__init__(ACTOR_SYSTEM)
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
                name=name,
                graph=graph,
                time_buffer=self.time_buffer,
                time_const=self.time_const,
                transmission=self.transmission,
                timeout=self.timeout,
                max_dur=self.max_dur,
                early_stop=self.early_stop)
            for name in range(self.parts)))

    def _group(self, scores: NpSeq, idx: Iterable[int]) -> Sequence[NpMap]:
        groups = [dict() for _ in range(self.parts)]
        for n, (g, nscores) in enumerate(zip(idx, scores)):
            groups[g][n] = nscores
        return groups

    def send(self, *nodes: NpSeq) -> NoReturn:
        for p, pnodes in enumerate(nodes):
            self.outbox[p].put(pnodes, block=True)

    def _create_graph(self, contacts: NpSeq) -> Tuple[Graph, ndarray]:
        graph: MutableGraph = {}
        adj = defaultdict(list)
        # Assumes a name corresponds to an index in scores.
        for contact in contacts:
            n1, n2 = contact['names']
            adj[n1].append(n2)
            adj[n2].append(n1)
            graph[fkey(n1, n2)] = contact['time']
        # Keep indexing consistent in case of users that have no contacts.
        adj = [unique(adj.get(n, EMPTY)) for n in range(self._scores)]
        membership = self.partition(adj)
        graph.update((n, node(ne, membership[n])) for n, ne in enumerate(adj))
        return graph, membership

    def partition(self, adj: NpSeq) -> ndarray:
        cuts, membership = part_graph(self.parts, adj)
        return membership
