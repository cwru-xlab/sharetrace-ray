import collections
import itertools
import json
import logging
import time
import timeit
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any, Collection, Hashable, Iterable, Mapping, MutableMapping, Optional,
    Sequence, Set,
    Tuple,
    Type, Union
)

import numpy as np
import pymetis
import ray
from ray import ObjectRef
from scipy import sparse
from sklearn import cluster

from sharetrace import model, queue, util
from sharetrace.actors import Actor, ActorSystem

Array = np.ndarray
NpSeq = Sequence[np.ndarray]
NpMap = Mapping[int, np.ndarray]
Graph = MutableMapping[Hashable, Any]
Nodes = Mapping[int, np.void]
Index = Union[Mapping[int, int], Sequence[int], np.ndarray]
Log = MutableMapping[str, Any]
Real = (int, float)

ACTOR_SYSTEM = -1


def ckey(n1: int, n2: int) -> Tuple[int, int]:
    return min(n1, n2), max(n1, n2)


def initial(scores: Array) -> np.void:
    return np.sort(scores, order=('val', 'time'))[-1]


class StopCondition(Enum):
    EARLY_STOP = 'EarlyStop'
    TIMED_OUT = 'TimedOut'
    MAX_DURATION = 'MaxDuration'

    def data(self, x):
        return self, x


@dataclass(frozen=True)
class WorkerLog:
    name: Hashable
    runtime: float
    messages: int
    nodes: int
    node_data: float
    updates: int
    stop_condition: StopCondition
    stop_data: float
    min_update: float
    max_update: float
    avg_update: float
    std_update: float
    min_updates: float
    max_updates: float
    avg_updates: float
    std_updates: float

    @classmethod
    def summarize(cls, *logs: 'WorkerLog') -> Mapping[str, float]:
        approx, sdiv = util.approx, util.sdiv
        updates = sum(w.updates for w in logs)
        return {
            'RuntimeInSeconds': approx(max(w.runtime for w in logs)),
            'Messages': sum(w.messages for w in logs),
            'Nodes': sum(w.nodes for w in logs),
            'NodeDataInMb': approx(sum(w.node_data for w in logs)),
            'Updates': sum(w.updates for w in logs),
            'MinUpdate': approx(min(w.min_update for w in logs)),
            'MaxUpdate': approx(max(w.max_update for w in logs)),
            'AvgUpdate': approx(
                sdiv(sum(w.updates * w.avg_update for w in logs), updates)),
            'MinUpdates': min(w.min_updates for w in logs),
            'MaxUpdates': max(w.max_updates for w in logs),
            'AvgUpdates': approx(
                sdiv(sum(w.updates * w.avg_updates for w in logs), updates))}

    def format(self) -> Mapping[str, Any]:
        approx = util.approx
        return {
            'Name': self.name,
            'RuntimeInSeconds': approx(self.runtime),
            'Messages': int(self.messages),
            'Nodes': int(self.nodes),
            'NodeDataInMb': approx(self.node_data),
            'Updates': int(self.updates),
            'StopCondition': self.stop_condition.name,
            'StopData': approx(self.stop_data),
            'MinUpdate': approx(self.min_update),
            'MaxUpdate': approx(self.max_update),
            'AvgUpdate': approx(self.avg_update),
            'StdUpdate': approx(self.std_update),
            'MinUpdates': int(self.min_updates),
            'MaxUpdates': int(self.max_updates),
            'AvgUpdates': approx(self.avg_updates),
            'StdUpdates': approx(self.std_updates)}


class Result:
    __slots__ = ('data', 'log')

    def __init__(self, data, log: WorkerLog):
        self.data = data
        self.log = log


class Partition(Actor):
    __slots__ = ('_actor',)

    def __init__(self, name: int, mailbox: queue.Queue, **kwargs):
        super().__init__(name, mailbox)
        self._actor = _Partition.remote(name, mailbox=mailbox, **kwargs)

    def run(self) -> ObjectRef:
        # Do not block to allow asynchronous invocation of actors.
        return self._actor.run.remote()

    def connect(self, *actors: Actor, duplex: bool = False) -> None:
        # Block to ensure all actors get connected before running.
        ray.get(self._actor.connect.remote(*actors, duplex=duplex))


@ray.remote(max_retries=3, max_restarts=3)
class _Partition(Actor):
    __slots__ = (
        'graph',
        'time_buffer',
        'time_const',
        'transmission',
        'tol',
        'eps',
        'empty',
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
            name: int,
            mailbox: queue.Queue,
            graph: Graph,
            time_buffer: int,
            time_const: float,
            transmission: float,
            tol: float,
            empty: Type,
            eps: float,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None):
        super().__init__(name, mailbox)
        self.graph = graph
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.tol = tol
        self.eps = eps
        self.empty = empty
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self._local = collections.deque()
        self._nodes: Nodes = {}
        self._start = -1
        self._since_update = 0
        self._init_done = False
        self._timed_out = False
        self._msgs = 0
        self._stop_condition: Tuple[StopCondition, Any] = tuple()

    def run(self) -> Result:
        runtime = util.time(self._run).seconds
        result = Result(
            data={n: props['curr'] for n, props in self._nodes.items()},
            log=self._log(runtime))
        # Allow the last partition output its log.
        time.sleep(0.1)
        return result

    def _run(self) -> None:
        stop, receive, on_next = self.should_stop, self.receive, self.on_next
        self._start = timeit.default_timer()
        self.on_start(receive())
        while not stop():
            if (msg := receive()) is not None:
                on_next(msg)

    def should_stop(self) -> bool:
        too_long, no_updates = False, False
        if (max_dur := self.max_dur) is not None:
            if too_long := ((timeit.default_timer() - self._start) >= max_dur):
                self._stop_condition = StopCondition.MAX_DURATION.data(max_dur)
        elif (early_stop := self.early_stop) is not None:
            if no_updates := (self._since_update >= early_stop):
                self._stop_condition = StopCondition.EARLY_STOP.data(early_stop)
        elif self._timed_out:
            self._stop_condition = StopCondition.TIMED_OUT.data(self.timeout)
        return self._timed_out or too_long or no_updates

    def receive(self) -> Optional:
        # Prioritize local convergence over processing remote messages.
        if len((local := self._local)) > 0:
            msg = local.popleft()
        else:
            try:
                msg = self.mailbox.get(block=True, timeout=self.timeout)
            except self.empty:
                msg, self._timed_out = None, True
        if isinstance(msg, np.void):
            self._msgs += 1
        elif isinstance(msg, Collection):
            self._msgs = len(msg)
        return msg

    def on_next(self, msg: np.void, **kwargs) -> None:
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

    def on_start(self, scores: NpMap) -> None:
        """Assign node attributes and send symptom scores."""
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

    def _send(self, scores: Array, var: int, factors: Array) -> None:
        """Compute a factor node message and send if it will be effective."""
        graph, init, sgroup, send, buffer, tol, eps, const, transmission = (
            self.graph, self._nodes[var]['init_msg'], self.name, self.send,
            self.time_buffer, self.tol, self.eps, self.time_const,
            self.transmission)
        inf, sec_per_day = np.inf, 86400
        message, log, argmax, minimum, maximum = (
            model.message, np.log, np.argmax, np.minimum, np.maximum)
        for f in factors:
            # Only consider scores that may have been transmitted from contact.
            ctime = graph[ckey(var, f)]
            if len(scores := scores[scores['time'] <= ctime + buffer]) > 0:
                # Scales time deltas in partial days.
                diff = minimum((scores['time'] - ctime) / sec_per_day, 0)
                # Use the log transform to avoid overflow issues.
                weighted = log(maximum(scores['val'], eps)) + (diff / const)
                score = scores[argmax(weighted)]
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
                    # noinspection PyTypeChecker
                    send(message(score, var, sgroup, f, graph[f]['group']))

    def send(self, msg: np.void) -> None:
        if (key := msg['dgroup']) == self.name:
            self._local.append(msg)
        else:
            self.neighbors[key].put_nowait(msg)

    def _log(self, runtime: float) -> WorkerLog:
        def safe_stat(func, values):
            return 0 if len(values) == 0 else float(func(values))

        nodes, (condition, data) = self._nodes, self._stop_condition
        props = nodes.values()
        update = np.array([node['curr'] - node['init_val'] for node in props])
        update = update[update != 0]
        updates = np.array([node['updates'] for node in props])
        updates = updates[updates != 0]
        return WorkerLog(
            name=self.name,
            runtime=runtime,
            messages=self._msgs,
            nodes=len(nodes),
            node_data=util.get_mb(nodes),
            updates=len(updates),
            stop_condition=condition,
            stop_data=data,
            min_update=safe_stat(min, update),
            max_update=safe_stat(max, update),
            avg_update=safe_stat(np.mean, update),
            std_update=safe_stat(np.std, update),
            min_updates=safe_stat(min, updates),
            max_updates=safe_stat(max, updates),
            avg_updates=safe_stat(np.mean, updates),
            std_updates=safe_stat(np.std, updates))


class RiskPropagation(ActorSystem):
    __slots__ = (
        'time_buffer',
        'time_const',
        'transmission',
        'tol',
        'eps',
        'workers',
        'timeout',
        'max_dur',
        'early_stop',
        'partitioning',
        'auto',
        'logger',
        'nodes',
        'edges',
        'log')

    def __init__(
            self,
            time_buffer: int = 172_800,
            time_const: float = 1.,
            transmission: float = 0.8,
            tol: float = 0.1,
            eps: float = 1e-7,
            workers: int = 1,
            timeout: Optional[float] = None,
            max_dur: Optional[float] = None,
            early_stop: Optional[int] = None,
            partitioning: str = 'spectral',
            auto: bool = True,
            logger: Optional[logging.Logger] = None):
        super().__init__(ACTOR_SYSTEM)
        assert isinstance(time_buffer, int) and time_buffer > 0
        assert isinstance(time_const, Real) and time_const > 0
        assert isinstance(tol, Real) and tol >= 0
        assert isinstance(workers, int) and workers > 0
        if timeout is not None:
            assert isinstance(timeout, Real) and timeout >= 0
        if max_dur is not None:
            assert isinstance(max_dur, Real) and max_dur > 0
        if early_stop is not None:
            assert isinstance(early_stop, int) and early_stop > 0
        assert partitioning in ('metis', 'spectral')
        self.time_buffer = time_buffer
        self.time_const = time_const
        self.transmission = transmission
        self.tol = tol
        self.eps = eps
        self.workers = workers
        self.timeout = timeout
        self.max_dur = max_dur
        self.early_stop = early_stop
        self.partitioning = partitioning
        self.auto = auto
        self.logger = logger
        self.nodes: int = -1
        self.edges: int = -1
        self.log: Log = {}

    def send(self, parts: Sequence[NpMap]) -> None:
        neighbors = self.neighbors
        for p, pscores in enumerate(parts):
            neighbors[p].put_nowait(pscores)

    def run(self, scores: NpSeq, contacts: Array) -> Array:
        if len(scores) == 0 or len(contacts) == 0:
            result = np.array([])
        else:
            if self.auto:
                self.on_start()
            result = self._run(scores, contacts)
            if self.auto:
                self.on_stop()
        return result

    def on_start(self):
        ray.init(ignore_reinit_error=True)

    def on_stop(self):
        ray.shutdown()

    def _run(self, scores: NpSeq, contacts: Array) -> Array:
        self.log.clear()
        timed = util.time(lambda: self.create_graph(scores, contacts))
        graph, parts, u2i, n2p, no_ne, partition_runtime = timed.result
        build_time = timed.seconds
        self.send(parts)
        results = [a.run() for a in self.actors]
        # Compute the exposure score for those without neighbors.
        no_ne = {u2i[u]: initial(scores[u])['val'] for u in no_ne}
        results = ray.get(results)
        exposures = self._gather(results, u2i, no_ne)
        # noinspection PyTypeChecker
        self._log(
            graph=graph,
            build_time=build_time,
            partition_runtime=partition_runtime,
            results=results,
            membership=n2p.tolist(),
            symptoms=[float(initial(s)['val']) for s in scores],
            exposures=exposures.tolist())
        self._save_log()
        return exposures

    def create_graph(
            self,
            scores: NpSeq,
            contacts: Array
    ) -> Tuple[Graph, Sequence[NpMap], Index, Index, Set[int], float]:
        u2i, no_ne = self._index(scores, contacts)
        graph, adjlist, n2i = self._add_factors(contacts, u2i)
        n2p, partition_runtime = self._add_vars(graph, adjlist, n2i)
        self._connect(ray.put(graph))
        parts = self._group(scores, u2i, n2i, n2p, no_ne)
        return graph, parts, u2i, n2p, no_ne, partition_runtime

    @staticmethod
    def _index(scores: NpSeq, contacts: Array) -> Tuple[Index, Set[int]]:
        # Assumes a name corresponds to an index in scores.
        with_ne = set(contacts['names'].flatten().tolist())
        no_ne = set(range(len(scores))) - with_ne
        # METIS requires 0-based contiguous indexing.
        u2i = {u: i for i, u in enumerate(itertools.chain(with_ne, no_ne))}
        return u2i, no_ne

    def _add_factors(
            self,
            contacts: Array,
            u2i: Index
    ) -> Tuple[Graph, Sequence, Index]:
        graph, adjlist = {}, collections.defaultdict(list)
        for contact in contacts:
            u1, u2 = contact['names']
            i1, i2 = u2i[u1], u2i[u2]
            adjlist[i1].append(i2)
            adjlist[i2].append(i1)
            graph[ckey(i1, i2)] = contact['time']
        n2i = np.array(list(adjlist))
        adjlist = [np.array(ne) for ne in adjlist.values()]
        self.nodes, self.edges = len(adjlist), len(graph)
        return graph, adjlist, n2i

    def _add_vars(
            self,
            graph: Graph,
            adjlist: Sequence,
            n2i: Index
    ) -> Tuple[Index, float]:
        timed = util.time(lambda: self.partition(adjlist, n2i))
        n2p = timed.result
        node = model.node
        graph.update((n2i[n], node(ne, n2p[n])) for n, ne in enumerate(adjlist))
        return n2p, timed.seconds

    def partition(self, adjlist: Sequence, n2i: Index) -> Array:
        if self.partitioning == 'spectral':
            labels = self._spectral_partition(adjlist, n2i)
        else:
            labels = self._metis_partition(adjlist)
        return labels

    def _metis_partition(self, adjlist: Sequence) -> Array:
        _, labels = pymetis.part_graph(self.workers, adjlist)
        return labels

    def _spectral_partition(self, adjlist: Sequence, n2i: Index) -> Array:
        # Ignore warning regarding disconnected graph.
        warnings.filterwarnings('ignore')
        spectral = cluster.SpectralClustering(
            self.workers,
            n_init=100,
            affinity='precomputed',
            assign_labels='discretize')
        adjmat = self._adjmat(adjlist, n2i)
        return spectral.fit_predict(adjmat)

    def _adjmat(self, adjlist: Sequence, n2i: Index) -> sparse.spmatrix:
        adjmat = sparse.dok_matrix((self.nodes, self.nodes), dtype=np.int8)
        for n, ne in enumerate(adjlist):
            adjmat[n2i[n], ne] = 1
        return adjmat.tocsr()

    def _connect(self, graph: ObjectRef) -> None:
        create_part, workers = self._create_part, self.workers
        parts = (create_part(w, graph) for w in range(workers))
        pairs = parts if workers == 1 else itertools.combinations(parts, 2)
        self.connect(*pairs, duplex=True)

    def _create_part(self, name: int, graph: ObjectRef) -> Partition:
        # Ray Queue must be created and then passed as an object reference.
        return Partition(
            name=name,
            mailbox=queue.Queue(),
            graph=graph,
            time_buffer=self.time_buffer,
            time_const=self.time_const,
            transmission=self.transmission,
            tol=self.tol,
            eps=self.eps,
            empty=queue.Empty,
            timeout=self.timeout,
            max_dur=self.max_dur,
            early_stop=self.early_stop)

    def _group(
            self,
            scores: NpSeq,
            u2i: Index,
            n2i: Index,
            n2p: Index,
            no_ne: Set[int]
    ) -> Sequence[NpMap]:
        parts = [{} for _ in range(self.workers)]
        i2p = {i: p for i, p in zip(n2i, n2p)}
        for u, uscores in enumerate(scores):
            if u not in no_ne:
                i = u2i[u]
                parts[i2p[i]][i] = uscores
        return parts

    @staticmethod
    def _gather(
            results: Iterable[Result],
            u2i: Index,
            no_ne: MutableMapping[int, float]
    ) -> Array:
        merged = no_ne
        for result in results:
            merged.update(result.data)
        result = np.zeros(len(u2i))
        for u, i in u2i.items():
            result[u] = merged[i]
        return result

    def _log(
            self,
            symptoms: Sequence[float],
            exposures: Sequence[float],
            graph: Graph,
            membership: Sequence[int],
            build_time: float,
            partition_runtime: float,
            results: Iterable[Result]):
        approx = util.approx
        self.log.update({
            'GraphSizeInMb': approx(util.get_mb(graph)),
            'GraphBuildTimeInSeconds': approx(build_time),
            'PartitionTimeInSeconds': approx(partition_runtime),
            'Nodes': int(self.nodes),
            'Edges': int(self.edges),
            'TimeBufferInSeconds': float(self.time_buffer),
            'Transmission': approx(self.transmission),
            'ZeroEpsilon': self.eps,
            'SendTolerance': approx(self.tol),
            'Workers': int(self.workers),
            'TimeoutInSeconds': approx(self.timeout),
            'MaxDurationInSeconds': approx(self.max_dur),
            'EarlyStop': self.early_stop,
            'Membership': membership,
            'SymptomScores': symptoms,
            'ExposureScores': exposures})
        logs = [r.log for r in results]
        self.log.update(WorkerLog.summarize(*logs))
        self.log['WorkerLogs'] = [log.format() for log in logs]

    def _save_log(self) -> None:
        if (logger := self.logger) is not None:
            logger.info(json.dumps(self.log))
