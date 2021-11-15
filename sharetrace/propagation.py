import datetime
from collections import defaultdict, deque
from enum import Enum
from functools import reduce
from itertools import product
from json import dumps
from logging import getLogger
from logging.config import dictConfig
from queue import Empty
from time import sleep
from timeit import default_timer
from typing import (
    Any, Dict, Hashable, Iterable, Mapping, MutableMapping, NoReturn, Optional,
    Sequence, Tuple, Type
)

import ray
from numpy import (
    argmax, array, clip, inf, log, mean, ndarray, sort, std,
    timedelta64, unique, void
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
# This is only used for comparison. Cannot use exactly 0 with log.
DEFAULT_SCORE = risk_score(0, datetime.datetime.min)
dictConfig(LOGGING_CONFIG)


def ckey(n1, n2) -> Tuple[Any, Any]:
    return min(n1, n2), max(n1, n2)


def round_float(val):
    return val if val is None else round(float(val), 4)


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
        'tol',
        'empty_except',
        'timeout',
        'max_dur',
        'early_stop',
        '_local',
        '_nodes',
        '_start',
        '_since_update',
        '_init_done',
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
            tol: float,
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
        self.tol = tol
        self.timeout = timeout
        self.empty_except = empty_except
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._local = deque()
        self._nodes: Nodes = {}
        self._start = -1
        self._since_update = 0
        self._init_done = False
        self._timed_out = False
        self._msgs = 0
        self._stop_condition: Tuple[StopCondition, float] = tuple()
        self._logger = getLogger(__name__)

    def run(self) -> Optional[Mapping[int, float]]:
        timed = Timer.time(self._run)
        results = {n: props['curr'] for n, props in self._nodes.items()}
        self._log_stats(timed.seconds)
        return self.on_complete(results)

    def _log_stats(self, runtime: float):
        def safe_stat(func, values):
            return 0 if len(values) == 0 else round_float(func(values))

        nodes, (condition, data) = self._nodes, self._stop_condition
        update = array([
            props['curr'] - props['init_val'] for props in nodes.values()])
        update = update[update != 0]
        updates = array([props['updates'] for props in nodes.values()])
        updates = updates[updates != 0]
        logged = {
            'Partition': self.name,
            'RuntimeInSec': round_float(runtime),
            'Messages': self._msgs,
            'Nodes': len(nodes),
            'NodeDataInMb': round_float(get_mb(nodes)),
            'Updates': len(updates),
            'StopCondition': condition.name,
            'StopData': data}
        data = ((update, '{}Update'), (updates, '{}Updates'))
        funcs = ((min, 'Min'), (max, 'Max'), (mean, 'Avg'), (std, 'Std'))
        for (datum, name), (f, fname) in product(data, funcs):
            logged[name.format(fname)] = safe_stat(f, datum)
        self._logger.info(dumps(logged))

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
        if (new := score['val']) > (props := self._nodes[var])['curr']:
            self._since_update = 0
            props['curr'] = new
            props['updates'] += 1
        elif self.early_stop is not None and self._init_done:
            self._since_update += 1

    def _on_initial(self, scores: NpMap) -> NoReturn:
        # Must be a mapping since indices may neither start at 0 nor be
        # contiguous, based on the original input scores.
        nodes = {}
        for var, vscores in scores.items():
            init = sort(vscores, order=('val', 'time'))[-1]
            init_msg = init.copy()
            init_msg['val'] *= self.transmission
            nodes[var] = {
                'init_val': init['val'],
                'init_msg': init_msg,
                'curr': init['val'],
                'updates': 0}
        self._nodes = nodes
        graph, send = self.graph, self._send
        # Send initial symptom score messages to all neighbors.
        for var, vscores in scores.items():
            send(vscores, var, graph[var]['ne'])
        self._init_done = True

    def _send(self, scores: ndarray, var: int, factors: ndarray):
        """Compute a factor node message and send if it will be effective."""
        graph, init, sgroup, send, buffer, tol, const, transmission = (
            self.graph, self._nodes[var]['init_msg'], self.name, self.send,
            self.time_buffer, self.tol, self.time_const, self.transmission)
        for f in factors:
            # Only consider scores that may have been transmitted from contact.
            ctime = graph[ckey(var, f)]
            scores = scores[scores['time'] <= ctime + buffer]
            if len(scores) > 0:
                # Scales time deltas in partial days.
                diff = clip((scores['time'] - ctime) / DAY, -inf, 0)
                # Use the log transform to avoid overflow issues.
                weighted = log(scores['val']) + (diff / const)
                score = scores[argmax(weighted)]
                score['val'] *= transmission
                # This is a necessary, but not sufficient, condition for the
                # value of a neighbor to be updated. The transmission rate
                # causes the value of a score to monotonically decrease as it
                # propagates further from its source. Thus, this criterion
                # will converge. A higher tolerance results in faster
                # convergence at the cost of completeness.
                high_enough = score['val'] > init['val'] * tol
                # The older the score, the likelier it is to be propagated,
                # regardless of its value. A newer score with a lower value
                # will not result in an update to the neighbor. This
                # criterion will not converge as messages do not get older.
                # The conjunction of the first criterion allows for convergence.
                older = score['time'] < init['time']
                if high_enough and older:
                    send(message(score, var, sgroup, f, graph[f]['group']))

    def send(self, msg: Any, key: int = None) -> NoReturn:
        """Send a message either to the local inbox or an outbox."""
        # msg must be a message structured array if key is None
        if key is None:
            key = msg['dgroup']
        if key == self.name:
            self._local.append(msg)
        else:
            self.outbox[key].put(msg, block=True, timeout=self.timeout)


class RiskPropagation(BaseActorSystem):
    __slots__ = (
        'time_buffer',
        'time_const',
        'transmission',
        'tol',
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
            tol: float = 0.1,
            parts: int = 1,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None,
            inbox: Optional[Any] = None):
        super().__init__(ACTOR_SYSTEM, inbox)
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.tol = tol
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
        receive = self.receive
        return self._map((receive() for _ in range(self.parts)))

    def _map(self, results: Iterable[Dict]) -> ndarray:
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
            tol=self.tol,
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
        outbox = self.outbox
        for p, pnodes in enumerate(nodes):
            outbox[p].put(pnodes, block=True)

    def create_graph(self, contacts: NpSeq) -> Tuple[Graph, ndarray]:
        graph: MutableGraph = {}
        adj = defaultdict(list)
        # Assumes a name corresponds to an index in scores.
        for contact in contacts:
            n1, n2 = contact['names']
            adj[n1].append(n2)
            adj[n2].append(n1)
            graph[ckey(n1, n2)] = contact['time']
        # Keep indexing consistent in case of users that have no contacts.
        adj = [unique(adj.get(n, EMPTY)) for n in range(self._scores)]
        membership = self.partition(adj)
        graph.update((n, node(ne, membership[n])) for n, ne in enumerate(adj))
        self._logger.info(dumps({
            'GraphSizeInMb': round_float(get_mb(graph)),
            'TimeBufferInSec': self.time_buffer,
            'Transmission': round_float(self.transmission),
            'SendTolerance': round_float(self.tol),
            'Partitions': self.parts,
            'TimeoutInSec': round_float(self.timeout),
            'MaxDurationInSec': round_float(self.max_dur),
            'EarlyStop': self.early_stop
        }))
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
            tol=self.tol,
            empty_except=RayEmpty,
            timeout=self.timeout,
            max_dur=self.max_dur,
            early_stop=self.early_stop)

    def create_graph(self, contacts: NpSeq) -> Tuple[ObjectRef, ndarray]:
        graph, membership = super().create_graph(contacts)
        return ray.put(graph), membership

    def run(self) -> Any:
        # No need to use a queue since Ray actors can return results.
        results = self._map(ray.get([a.run() for a in self.actors]))
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
