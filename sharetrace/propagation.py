import collections
import copy
import itertools
import json
import logging
import queue
import time
import timeit
import uuid
from enum import Enum
from typing import (
    Any, Collection, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Type
)

import numpy as np
import pymetis
import ray
from ray import ObjectRef
from ray.util.queue import Empty as RayEmpty, Queue as RayQueue

from lewicki.lewicki.actors import BaseActor, BaseActorSystem
from sharetrace import model, util
from sharetrace.util import approx

Array = np.ndarray
NpSeq = Sequence[Array]
NpMap = Mapping[int, Array]
Graph = Mapping[Hashable, Any]
Nodes = Mapping[int, np.void]

ACTOR_SYSTEM = -1
DAY = np.timedelta64(1, 'D')


def ckey(n1, n2) -> Tuple:
    return min(n1, n2), max(n1, n2)


def initial(scores: Array) -> np.void:
    return np.sort(scores, order=('val', 'time'))[-1]


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
        'empty_error',
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
        self.empty_error = empty_except
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

    def _log(self, runtime: float) -> Mapping[str, Any]:
        def safe_stat(func, values):
            return 0 if len(values) == 0 else approx(func(values))

        nodes, (condition, data) = self._nodes, self._stop_condition
        update = np.array([
            props['curr'] - props['init_val'] for props in nodes.values()])
        update = update[update != 0]
        updates = np.array([props['updates'] for props in nodes.values()])
        updates = updates[updates != 0]
        logged = {
            'Partition': self.name,
            'RuntimeInSec': approx(runtime),
            'Messages': self._msgs,
            'Nodes': len(nodes),
            'NodeDataInMb': approx(util.get_mb(nodes)),
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
        elif (early_stop := self.early_stop) is not None:
            if no_updates := (self._since_update >= early_stop):
                self._stop_condition = (StopCondition.EARLY_STOP, early_stop)
        elif self._timed_out:
            self._stop_condition = (StopCondition.TIMED_OUT, self.timeout)
        return self._timed_out or too_long or no_updates

    def receive(self) -> Optional[np.void]:
        # Prioritize local convergence over processing remote messages.
        if len((local := self._local)) > 0:
            msg = local.popleft()
        else:
            try:
                msg = self.inbox.get(block=True, timeout=self.timeout)
            except self.empty_error:
                msg, self._timed_out = None, True
        self._msgs += msg is not None
        return msg

    def on_next(self, msg: np.void) -> None:
        """Update the variable node and send messages to its neighbors."""
        factor, var, score = msg['src'], msg['dest'], msg['val']
        # As a variable node, update its current value.
        self._update(var, score)
        # As factor nodes, send a message to each neighboring variable node.
        factors = self.graph[var]['ne']
        self._send(np.array([score]), var, factors[factor != factors])

    def _update(self, var: int, score: np.void) -> None:
        """Update the exposure score of the current variable node."""
        if (new := score['val']) > (props := self._nodes[var])['curr']:
            self._since_update = 0
            props['curr'] = new
            props['updates'] += 1
        elif self.early_stop is not None and self._init_done:
            self._since_update += 1

    def _on_initial(self, scores: NpMap) -> None:
        # Must be a mapping since indices may neither start at 0 nor be
        # contiguous, based on the original input scores.
        nodes = {}
        transmission = self.transmission
        for var, vscores in scores.items():
            init = initial(vscores)
            init_msg = init.copy()
            init_msg['val'] *= transmission
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

    def _send(self, scores: Array, var: int, factors: Array):
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

    def send(self, msg: Any, key: int = None) -> None:
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
        'workers',
        'timeout',
        'max_dur',
        'early_stop',
        'logger',
        '_nodes',
        '_u2i',
        '_init',
        '_log_id')

    def __init__(
            self,
            time_buffer: int = 172_800,
            time_const: float = 1.,
            transmission: float = 0.8,
            tol: float = 0.1,
            workers: int = 1,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None,
            inbox: Optional[Any] = None,
            logger: Optional[logging.Logger] = None):
        super().__init__(ACTOR_SYSTEM, inbox)
        assert time_buffer > 0 and isinstance(time_buffer, int)
        assert time_const > 0
        assert tol >= 0
        assert workers > 0 and isinstance(workers, int)
        assert timeout > 0 if timeout is not None else True
        assert max_dur > 0 if max_dur is not None else True
        assert early_stop > 0 if early_stop is not None else True
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.tol = tol
        self.workers = workers
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self.logger = logger
        self._nodes: int = -1
        self._u2i: Mapping[int, int] = {}
        self._init: Dict[int, float] = {}
        self._log_id: str = ""

    def setup(self, scores: NpSeq, contacts: Array) -> None:
        """Create the graph, log statistics, and send symptom scores."""
        assert len(scores) > 0 and len(contacts) > 0
        self._log_id = str(uuid.uuid4())
        graph, parts = self.create_graph(scores, contacts)
        self.send(parts)

    def run(self) -> Array:
        """Initiate message passing and return the exposure scores."""
        super().run()
        receive = self.receive
        return self._results([receive() for _ in range(self.workers)])

    def _results(self, data: Collection) -> Array:
        self._log_parts(data)
        return self._gather_results(data)

    def _log_parts(self, data: Collection) -> None:
        if (logger := self.logger) is not None:
            for log in (log for _, log in data):
                log.update({'LogId': self._log_id})
                logger.info(json.dumps(log))

    def _gather_results(self, data: Collection) -> Array:
        merged = copy.copy(self._init)
        for results, _ in data:
            merged.update(results)
        # Use the reverse mapping to preserve the original ordering.
        results = np.zeros(len(self._u2i))
        for u, i in self._u2i.items():
            results[u] = merged[i]
        return results

    def create_part(self, name: Hashable, graph: Graph) -> Partition:
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

    def send(self, parts: Sequence[NpMap]) -> None:
        outbox = self.outbox
        for p, pscores in enumerate(parts):
            outbox[p].put(pscores, block=True)

    def create_graph(
            self,
            scores: NpSeq,
            contacts: Array
    ) -> Tuple[Graph, Sequence[NpMap]]:
        self._index(scores, contacts)
        graph, adj, n2i = self._build_graph(contacts)
        n2p = self.partition(adj)
        graph.update(
            (n2i[n], model.node(ne, n2p[n])) for n, ne in enumerate(adj))
        self._connect(graph)
        parts = self._group(scores, {i: p for i, p in zip(n2i, n2p)})
        self._log_stats(len(contacts), graph)
        return graph, parts

    def _index(self, scores: NpSeq, contacts: Array) -> None:
        # Assumes a name corresponds to an index in scores.
        with_ne = set(contacts['names'].flatten().tolist())
        no_ne = set(range(len(scores))) - with_ne
        # METIS requires contiguous indexing.
        # Relies on consistent iteration ordering to index.
        u2i = {u: i for i, u in enumerate(itertools.chain(with_ne, no_ne))}
        self._u2i = u2i
        # Compute the exposure score for those without neighbors.
        self._init = {u2i[u]: initial(scores[u])['val'] for u in no_ne}

    def _build_graph(self, contacts: Array) -> Tuple:
        u2i = self._u2i
        graph, adj = {}, collections.defaultdict(list)
        for contact in contacts:
            u1, u2 = contact['names']
            i1, i2 = u2i[u1], u2i[u2]
            adj[i1].append(i2)
            adj[i2].append(i1)
            graph[ckey(i1, i2)] = contact['time']
        # Relies on consistent iteration ordering to index.
        n2i, adj = list(adj), list(adj.values())
        self._nodes = len(adj)
        return graph, adj, n2i

    def _connect(self, graph: Graph) -> None:
        self.connect(*(self.create_part(p, graph) for p in range(self.workers)))

    def _group(self, scores: NpSeq, i2p: Mapping[int, int]) -> Sequence[NpMap]:
        parts = [{} for _ in range(self.workers)]
        # Exclude those that correspond to users without neighbors.
        init, u2i = self._init, self._u2i
        for u, uscores in enumerate(scores):
            if (i := u2i[u]) not in init:
                parts[i2p[i]][i] = uscores
        return parts

    def partition(self, adj: Sequence) -> Array:
        _, labels = pymetis.part_graph(self.workers, adj)
        return labels

    def _log_stats(self, contacts: int, graph: Graph):
        if (logger := self.logger) is not None:
            logger.info(json.dumps({
                'LogId': self._log_id,
                'GraphSizeInMb': approx(util.get_mb(graph)),
                'Nodes': self._nodes,
                'Edges': contacts,
                'TimeBufferInSeconds': self.time_buffer,
                'Transmission': approx(self.transmission),
                'SendTolerance': approx(self.tol),
                'Workers': self.workers,
                'TimeoutInSeconds': approx(self.timeout),
                'MaxDurationInSeconds': approx(self.max_dur),
                'EarlyStop': self.early_stop}))


class RayPartition(Partition):
    __slots__ = ('_actor',)

    def __init__(self, name: Hashable, inbox: RayQueue, **kwargs):
        super().__init__(name, inbox=inbox, **kwargs)
        self._actor = _RayPartition.remote(name, inbox=inbox, **kwargs)

    def run(self) -> ObjectRef:
        # Do not block to allow asynchronous invocation of actors.
        return self._actor.run.remote()

    def connect(self, *actors: BaseActor) -> None:
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
        super().__init__(*args, inbox=None, **kwargs)

    def create_part(self, name: Hashable, graph: ObjectRef) -> Partition:
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

    def create_graph(
            self,
            scores: NpSeq,
            contacts: Array
    ) -> Tuple[ObjectRef, Sequence[NpMap]]:
        graph, parts = super().create_graph(scores, contacts)
        return ray.put(graph), parts

    def run(self) -> Any:
        # No need to use a queue since Ray actors can return data.
        results = self._results(ray.get([a.run() for a in self.actors]))
        ray.shutdown()
        return results

    def setup(self, scores: NpSeq, contacts: Array) -> None:
        ray.init()
        super().setup(scores, contacts)

    def connect(self, *actors: BaseActor) -> None:
        # Connect in order to send the initial message.
        BaseActor.connect(self, *actors)
        # Remember all actors in order to get their data.
        self.actors.extend(actors)
        # Each partition may need to communicate with all other partitions.
        self._make_complete(*actors)
