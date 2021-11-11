import datetime
from collections import defaultdict, deque
from enum import Enum
from functools import reduce
from json import dumps
from logging import getLogger
from logging.config import dictConfig
from queue import Empty
from time import sleep
from timeit import default_timer
from typing import (
    Any, Hashable, Iterable, Mapping, MutableMapping, NoReturn, Optional,
    Sequence, Tuple, Type
)

import ray
from numpy import (
    argmax, array, log, ndarray, sort, timedelta64, unique, void
)
from pymetis import part_graph
from ray import ObjectRef
from ray.util.queue import Empty as RayEmpty, Queue as RayQueue

from lewicki.actors import BaseActor, BaseActorSystem
from sharetrace.model import message, node, risk_score
from sharetrace.util import LOGGING_CONFIG, Timer, get_mb

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


class StopCondition(Enum):
    EARLY_STOP = 'early_stop'
    TIMED_OUT = 'timed_out'
    MAX_DURATION = 'max_duration'


class Partition(BaseActor):
    __slots__ = (
        'graph',
        'time_buffer',
        'time_const',
        'transmission',
        'empty_except',
        'timeout',
        'max_dur',
        'early_stop',
        '_local',
        '_nodes',
        '_start',
        '_since_update',
        '_timed_out',
        '_stop_condition',
        '_msgs',
        '_logger')

    def __init__(
            self,
            name: Hashable,
            graph: Graph,
            time_buffer: int,
            time_const: float,
            transmission: float,
            empty_except: Type,
            inbox: Optional[Any] = None,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None):
        super().__init__(name, inbox)
        self.graph = graph
        self.time_buffer = timedelta64(time_buffer, 's')
        self.time_const = time_const
        self.transmission = transmission
        self.timeout = timeout
        self.empty_except = empty_except
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._local = deque()
        self._nodes: Nodes = {}
        self._start = -1
        self._since_update = 0
        self._timed_out = False
        self._msgs = 0
        self._stop_condition: Tuple[StopCondition, float] = tuple()
        self._logger = getLogger(__name__)

    def run(self) -> Optional[Mapping[int, float]]:
        timed = Timer.time(self._run)
        results = {n: data['curr'] for n, data in self._nodes.items()}
        self._log_stats(timed.seconds, results)
        return self.on_complete(results)

    def _log_stats(self, runtime: float, results: Mapping[int, float]):
        name, nodes = self.name, self._nodes
        diffs = (data['init']['val'] - results[n] for n, data in nodes.items())
        updates = int(sum(map(lambda diff: diff != 0, diffs)))
        condition, data = self._stop_condition
        self._logger.info(
            dumps({
                'Partition': name,
                'RuntimeInSec': round(runtime, 4),
                'Messages': self._msgs,
                'Nodes': len(nodes),
                'NodeDataInMb': round(get_mb(nodes), 4),
                'NodeUpdates': updates,
                'StopCondition': condition.name,
                'StopData': data}))

    def _run(self):
        stop, receive, on_next = self.should_stop, self.receive, self.on_next
        self._start = default_timer()
        self._on_initial(receive())
        while not stop():
            if (msg := receive()) is not None:
                on_next(msg)

    def on_complete(self, results) -> Optional:
        # Must send the results of the child process via queue.
        self.send(results, ACTOR_SYSTEM)

    def should_stop(self) -> bool:
        too_long, no_updates = False, False
        if (max_dur := self.max_dur) is not None:
            if too_long := ((default_timer() - self._start) >= max_dur):
                self._stop_condition = (StopCondition.MAX_DURATION, max_dur)
        if (early_stop := self.early_stop) is not None:
            if no_updates := (self._since_update >= early_stop):
                self._stop_condition = (StopCondition.EARLY_STOP, early_stop)
        if self._timed_out:
            self._stop_condition = (StopCondition.TIMED_OUT, self.timeout)
        return self._timed_out or too_long or no_updates

    def receive(self) -> Optional[Any]:
        # Prioritize local convergence over processing remote messages.
        if len((local := self._local)) > 0:
            msg = local.popleft()
        else:
            try:
                msg = self.inbox.get(block=True, timeout=self.timeout)
            except self.empty_except:
                msg, self._timed_out = None, True
        self._msgs += msg is not None
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
        if (new := score['val']) > (data := self._nodes[var])['curr']:
            self._since_update, data['curr'] = 0, new
        elif self.early_stop is not None:
            self._since_update += 1

    def _on_initial(self, scores: NpMap) -> NoReturn:
        # Must be a mapping since indices may neither start at 0 nor be
        # contiguous, based on the original input scores.
        nodes = {}
        for var, vscores in scores.items():
            init = sort(vscores, order=('val', 'time'))[-1]
            nodes[var] = {
                'init': init,
                'curr': init['val'],
                'prev': defaultdict(lambda: DEFAULT_SCORE)}
        self._nodes = nodes
        graph, send = self.graph, self._send
        # Send initial symptom score messages to all neighbors.
        for var, vscores in scores.items():
            send(vscores, var, graph[var]['ne'])

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
                # Always send if the value is higher as it will always result
                # in an update to the neighbor value, but may not result in
                # them propagating the message, depending on the time.
                higher = score['val'] > prev[f]['val']
                # If the score is newer and is at least as high as what was
                # initially sent, then send the message. This will not result
                # in an update to the neighbor value, but may result in
                # propagating the message, depending on the value and time.
                newer = score['time'] > prev[f]['time']
                high_enough = score['val'] >= init['val']
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


class RiskPropagation(BaseActorSystem):
    __slots__ = (
        'time_buffer',
        'time_const',
        'transmission',
        'parts',
        'timeout',
        'max_dur',
        'early_stop',
        '_scores',
        '_logger')

    def __init__(
            self,
            time_buffer: int = 172_800,
            time_const: float = 1.,
            transmission: float = 0.8,
            parts: int = 1,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None,
            inbox: Optional[Any] = None):
        super().__init__(ACTOR_SYSTEM, inbox)
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.parts = parts
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._scores: int = -1
        self._logger = getLogger(__name__)

    def setup(self, scores: NpSeq, contacts: NpSeq) -> NoReturn:
        self._scores = len(scores)
        graph, membership = self.create_graph(contacts)
        self._connect(graph)
        groups = self._group(scores, membership)
        self.send(*groups)

    def run(self) -> ndarray:
        super().run()
        return self.results((self.receive() for _ in range(self.parts)))

    def results(self, results: Iterable) -> ndarray:
        results = reduce(lambda d1, d2: {**d1, **d2}, results)
        return array([results[i] for i in range(self._scores)])

    def _connect(self, graph: Graph) -> NoReturn:
        create = self.create_partition
        self.connect(*(create(name, graph) for name in range(self.parts)))

    def create_partition(self, name: Hashable, graph: Graph) -> Partition:
        return Partition(
            name=name,
            graph=graph,
            time_buffer=self.time_buffer,
            time_const=self.time_const,
            transmission=self.transmission,
            empty_except=Empty,
            timeout=self.timeout,
            max_dur=self.max_dur,
            early_stop=self.early_stop)

    def _group(self, scores: NpSeq, idx: Iterable[int]) -> Sequence[NpMap]:
        groups = [dict() for _ in range(self.parts)]
        for var, (g, vscores) in enumerate(zip(idx, scores)):
            groups[g][var] = vscores
        return groups

    def send(self, *nodes: NpSeq) -> NoReturn:
        for p, pnodes in enumerate(nodes):
            self.outbox[p].put(pnodes, block=True)

    def create_graph(self, contacts: NpSeq) -> Tuple[Graph, ndarray]:
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
        self._logger.info(dumps({'GraphSizeInMb': round(get_mb(graph), 4)}))
        return graph, membership

    def partition(self, adj: NpSeq) -> ndarray:
        cuts, membership = part_graph(self.parts, adj)
        return membership


class RayPartition(Partition):
    __slots__ = ('_actor',)

    def __init__(self, name: Hashable, inbox: RayQueue, **kwargs):
        super().__init__(name, inbox=inbox, **kwargs)
        self._actor = _RayPartition.remote(name, inbox=inbox, **kwargs)

    def run(self) -> ObjectRef:
        # Do not block to allow asynchronous invocation of actors.
        return self._actor.run.remote()

    def connect(self, *actors: BaseActor) -> NoReturn:
        # Block to ensure all actors get connected before running.
        ray.get(self._actor.connect.remote(*actors))


@ray.remote
class _RayPartition(Partition):
    __slots__ = ()

    def __init__(self, name: Hashable, inbox: RayQueue, **kwargs):
        super().__init__(name, inbox=inbox, **kwargs)

    def run(self) -> Optional[Mapping[int, float]]:
        result = super().run()
        # Allow the last partition output its log.
        sleep(0.1)
        return result

    def on_complete(self, results):
        # Returns the results, as opposed to using a queue to send them.
        return results


class RayRiskPropagation(RiskPropagation):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        # Actor system does not need an inbox None uses multiprocessing.Queue.
        super().__init__(*args, inbox=EMPTY, **kwargs)

    def create_partition(self, name: Hashable, graph: ObjectRef) -> Partition:
        # Ray Queue must be created and then passed as an object reference.
        return RayPartition(
            name=name,
            inbox=RayQueue(),
            graph=graph,
            time_buffer=self.time_buffer,
            time_const=self.time_const,
            transmission=self.transmission,
            empty_except=RayEmpty,
            timeout=self.timeout,
            max_dur=self.max_dur,
            early_stop=self.early_stop)

    def create_graph(self, contacts: NpSeq) -> Tuple[ObjectRef, ndarray]:
        graph, membership = super().create_graph(contacts)
        return ray.put(graph), membership

    def run(self) -> Any:
        # No need to use a queue since Ray actors can return results.
        results = self.results(ray.get([a.run() for a in self.actors]))
        ray.shutdown()
        return results

    def setup(self, scores: NpSeq, contacts: NpSeq) -> NoReturn:
        ray.init()
        super().setup(scores, contacts)

    def connect(self, *actors: BaseActor) -> NoReturn:
        # Connect in order to send the initial message.
        BaseActor.connect(self, *actors)
        # Remember all actors in order to get their results.
        self.actors.extend(actors)
        # Each partition may need to communicate with all other partitions.
        self._make_complete(*actors)
