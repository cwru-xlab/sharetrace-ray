import collections
import datetime
import functools
import itertools
import json
import logging
import queue
import time
import timeit
import uuid
from enum import Enum
from typing import (
    Any, Collection, Dict, Hashable, Iterable, Mapping, MutableMapping,
    NoReturn, Optional, Sequence, Tuple, Type
)

import numpy as np
import pymetis
import ray
from ray import ObjectRef
from ray.util.queue import Empty as RayEmpty, Queue as RayQueue

from lewicki.lewicki.actors import BaseActor, BaseActorSystem
from sharetrace import model, util

NpSeq = Sequence[np.ndarray]
NpMap = Mapping[int, np.ndarray]
Graph = Mapping[Hashable, Any]
MutableGraph = MutableMapping[Hashable, Any]
Nodes = Mapping[int, np.void]

ACTOR_SYSTEM = -1
DAY = np.timedelta64(1, 'D')
EMPTY = ()
# This is only used for comparison. Cannot use exactly 0 with log.
DEFAULT_SCORE = model.risk_score(0, datetime.datetime.min)


def ckey(n1, n2) -> Tuple:
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
        '_msgs')

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
        self.time_buffer = np.timedelta64(time_buffer, 's')
        self.time_const = time_const
        self.transmission = transmission
        self.tol = tol
        self.timeout = timeout
        self.empty_except = empty_except
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._local = collections.deque()
        self._nodes: Nodes = {}
        self._start = -1
        self._since_update = 0
        self._init_done = False
        self._timed_out = False
        self._msgs = 0
        self._stop_condition: Tuple[StopCondition, float] = tuple()

    def run(self) -> Optional[Mapping[int, float]]:
        timed = util.time(self._run)
        results = {n: props['curr'] for n, props in self._nodes.items()}
        log = self._log(timed.seconds)
        return self.on_complete(results, log)

    def _log(self, runtime: float) -> Dict[str, Any]:
        def safe_stat(func, values):
            return 0 if len(values) == 0 else util.approx(func(values))

        nodes, (condition, data) = self._nodes, self._stop_condition
        update = np.array([
            props['curr'] - props['init_val'] for props in nodes.values()])
        update = update[update != 0]
        updates = np.array([props['updates'] for props in nodes.values()])
        updates = updates[updates != 0]
        logged = {
            'Partition': self.name,
            'RuntimeInSec': util.approx(runtime),
            'Messages': self._msgs,
            'Nodes': len(nodes),
            'NodeDataInMb': util.approx(util.get_mb(nodes)),
            'Updates': len(updates),
            'StopCondition': condition.name,
            'StopData': data}
        data = ((update, '{}Update'), (updates, '{}Updates'))
        funcs = ((min, 'Min'), (max, 'Max'), (np.mean, 'Avg'), (np.std, 'Std'))
        for (datum, name), (f, fname) in itertools.product(data, funcs):
            logged[name.format(fname)] = safe_stat(f, datum)
        return logged

    def _run(self):
        stop, receive, on_next = self.should_stop, self.receive, self.on_next
        self._start = timeit.default_timer()
        self._on_initial(receive())
        while not stop():
            if (msg := receive()) is not None:
                on_next(msg)

    def on_complete(self, results, log) -> Optional:
        # Must send the data of the child process via queue.
        self.send((results, log), ACTOR_SYSTEM)

    def should_stop(self) -> bool:
        too_long, no_updates = False, False
        if (max_dur := self.max_dur) is not None:
            if too_long := ((timeit.default_timer() - self._start) >= max_dur):
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

    def on_next(self, msg: np.void) -> NoReturn:
        """Update the variable node and send messages to neighbors."""
        factor, var, score = msg['src'], msg['dest'], msg['val']
        # As a variable node, update its current value.
        self._update(var, score)
        # As factor nodes, send a message to each neighboring variable node.
        factors = self.graph[var]['ne']
        self._send(np.array([score]), var, factors[factor != factors])

    def _update(self, var: int, score: np.void) -> NoReturn:
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
            init = np.sort(vscores, order=('val', 'time'))[-1]
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

    def _send(self, scores: np.ndarray, var: int, factors: np.ndarray):
        """Compute a factor node message and send if it will be effective."""
        graph, init, sgroup, send, buffer, tol, const, transmission = (
            self.graph, self._nodes[var]['init_msg'], self.name, self.send,
            self.time_buffer, self.tol, self.time_const, self.transmission)
        message = model.message
        for f in factors:
            # Only consider scores that may have been transmitted from contact.
            ctime = graph[ckey(var, f)]
            scores = scores[scores['time'] <= ctime + buffer]
            if len(scores) > 0:
                # Scales time deltas in partial days.
                diff = np.clip((scores['time'] - ctime) / DAY, -np.inf, 0)
                # Use the log transform to avoid overflow issues.
                weighted = np.log(scores['val']) + (diff / const)
                score = scores[np.argmax(weighted)]
                score['val'] *= transmission
                # This is a necessary, but not sufficient, condition for the
                # value of a neighbor to be updated. The transmission rate
                # causes the value of a score to monotonically decrease as it
                # propagates further from its source. Thus, this criterion
                # will converge. A higher tolerance data in faster
                # convergence at the cost of completeness.
                high_enough = score['val'] >= init['val'] * tol
                # The older the score, the likelier it is to be propagated,
                # regardless of its value. A newer score with a lower value
                # will not result in an update to the neighbor. This
                # criterion will not converge as messages do not get older.
                # The conjunction of the first criterion allows for convergence.
                old_enough = score['time'] <= init['time']
                if high_enough and old_enough:
                    dgroup = graph[f]['group']
                    send(message(score, var, sgroup, f, dgroup))

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
        'logger',
        '_scores',
        '_log_id')

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
            inbox: Optional[Any] = None,
            logger: Optional[logging.Logger] = None):
        super().__init__(ACTOR_SYSTEM, inbox)
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.tol = tol
        self.parts = parts
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self.logger = logger
        self._scores: int = -1
        self._log_id = str(uuid.uuid4())

    def setup(self, scores: NpSeq, contacts: NpSeq) -> NoReturn:
        self._scores = len(scores)
        graph, membership = self.create_graph(contacts)
        self._connect(graph)
        groups = self._group(scores, membership)
        self.send(*groups)

    def run(self) -> np.ndarray:
        super().run()
        receive = self.receive
        return self._handle([receive() for _ in range(self.parts)])

    def _handle(self, data: Collection) -> np.ndarray:
        if (logger := self.logger) is not None:
            for log in (log for _, log in data):
                log.update({'LogId': self._log_id})
                logger.info(json.dumps(log))
        results = (res for res, _ in data)
        data = functools.reduce(lambda d1, d2: {**d1, **d2}, results)
        return np.array([data[i] for i in range(self._scores)])

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
            empty_except=queue.Empty,
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

    def create_graph(self, contacts: NpSeq) -> Tuple[Graph, np.ndarray]:
        graph: MutableGraph = {}
        adj = collections.defaultdict(list)
        # Assumes a name corresponds to an index in scores.
        for contact in contacts:
            n1, n2 = contact['names']
            adj[n1].append(n2)
            adj[n2].append(n1)
            graph[ckey(n1, n2)] = contact['time']
        # Keep indexing consistent in case of users that have no contacts.
        # TODO Re-index and don't add disconnected nodes; this may be
        #  negatively impacting METIS.
        #   Move all disconnected nodes to the end
        #   Shift nodes with neighbors to front
        with_ne = np.array([n for n in range(self._scores) if n in adj])
        no_ne = np.array([n for n in range(self._scores) if n not in adj])
        adj = [np.unique(adj[n]) for n in with_ne]
        labels = self.partition(adj)
        graph.update((n, model.node(ne, labels[n])) for n, ne in enumerate(adj))
        if (logger := self.logger) is not None:
            logger.info(json.dumps({
                'LogId': self._log_id,
                'GraphSizeInMb': util.approx(util.get_mb(graph)),
                'Nodes': self._scores,
                'Edges': len(contacts),
                'TimeBufferInSeconds': self.time_buffer,
                'Transmission': util.approx(self.transmission),
                'SendTolerance': util.approx(self.tol),
                'Partitions': self.parts,
                'TimeoutInSeconds': util.approx(self.timeout),
                'MaxDurationInSeconds': util.approx(self.max_dur),
                'EarlyStop': self.early_stop}))
        return graph, labels

    def partition(self, adj: NpSeq) -> np.ndarray:
        cuts, labels = pymetis.part_graph(self.parts, adj)
        return labels


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
        time.sleep(0.1)
        return result

    def on_complete(self, results, log):
        # Return, as opposed to using a queue.
        return results, log


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

    def create_graph(self, contacts: NpSeq) -> Tuple[ObjectRef, np.ndarray]:
        graph, membership = super().create_graph(contacts)
        return ray.put(graph), membership

    def run(self) -> Any:
        # No need to use a queue since Ray actors can return data.
        results = self._handle(ray.get([a.run() for a in self.actors]))
        ray.shutdown()
        return results

    def setup(self, scores: NpSeq, contacts: NpSeq) -> NoReturn:
        ray.init()
        super().setup(scores, contacts)

    def connect(self, *actors: BaseActor) -> NoReturn:
        # Connect in order to send the initial message.
        BaseActor.connect(self, *actors)
        # Remember all actors in order to get their data.
        self.actors.extend(actors)
        # Each partition may need to communicate with all other partitions.
        self._make_complete(*actors)
