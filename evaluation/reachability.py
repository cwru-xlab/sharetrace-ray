from __future__ import annotations

import collections
import functools
import heapq
import igraph as ig
import joblib
import numpy as np
from typing import (
    Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union)

from sharetrace import propagation

ContactMap = Mapping[Tuple[int, int], float]
AdjList = Mapping[int, Union[Sequence, np.ndarray]]
Reached = Mapping[int, Mapping[int, int]]
NpSeq = Sequence[np.ndarray]


def ratio(reached: Reached, graph: ig.Graph) -> Tuple[float, int, int]:
    ckey = propagation.ckey
    actual = len(set(ckey(n, ne) for n, nes in reached.items() for ne in nes))
    ideal = np.count_nonzero(graph.shortest_paths())
    return actual / ideal, actual, ideal


class Node:
    """A node used by MessageReachability.

    Attributes:
        name: An integer identifier.
        init: The initial messages sent during risk propagation.
        msg: The risk score message to send to its neighbors.
        dist: The shortest-path distance from the source node.
    """
    __slots__ = ("name", "init", "msg", "dist")

    def __init__(
            self,
            name: int,
            init: np.void,
            msg: Optional[np.void] = None,
            dist: float = np.inf):
        self.name = name
        self.init = init
        self.msg = msg
        self.dist = dist

    def __eq__(self, other) -> bool:
        self._check_comparable(other)
        return self.dist == other.dist

    def __le__(self, other) -> bool:
        self._check_comparable(other)
        return self.dist <= other.dist

    def __lt__(self, other) -> bool:
        self._check_comparable(other)
        return self.dist < other.dist

    def __ge__(self, other) -> bool:
        self._check_comparable(other)
        return self.dist >= other.dist

    def __gt__(self, other) -> bool:
        self._check_comparable(other)
        return self.dist > other.dist

    def _check_comparable(self, other):
        if not hasattr(other, "dist"):
            raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self):
        cls = self.__class__.__name__
        name, dist, init, msg = self.name, self.dist, self.init, self.msg
        return f"{cls}(name={name}, dist={dist}, init={init}, msg={msg})"


class MessageReachability:
    """Equivalent to the set of influence of a node in a temporal graph."""
    __slots__ = ("transmission", "tol", "buffer", "workers", "verbose")

    def __init__(
            self,
            transmission: float,
            tol: float,
            buffer: float,
            workers: int = 1,
            verbose: int = 0):
        self.transmission = transmission
        self.tol = tol
        self.buffer = buffer
        self.workers = workers
        self.verbose = verbose

    def run_all(self, scores: NpSeq, contacts: np.ndarray) -> Reached:
        """Computes message reachability for all users."""
        par = joblib.Parallel(self.workers, batch_size=1, verbose=self.verbose)
        contacts = self._to_map(contacts)
        adjlist = self._adjlist(contacts)
        run = functools.partial(
            lambda s: self.copy().run(
                s, scores, contacts, adjlist, precomputed=True))
        run = joblib.delayed(run)
        sources = np.arange(users := len(scores))
        ranges = np.array_split(sources, np.ceil(users / 100))
        results = par(run(rng) for rng in ranges)
        merged = {}
        for r in results:
            merged.update(r)
        return merged

    def run(
            self,
            sources: Iterable[int],
            scores: NpSeq,
            contacts: Union[np.ndarray, ContactMap],
            adjlist: Optional[AdjList] = None,
            precomputed: bool = False
    ) -> Reached:
        """Computes message reachability for the given sources."""
        if not precomputed:
            contacts = self._to_map(contacts)
            adjlist = self._adjlist(contacts)
        initialize, dijkstra = self._initialize, self._dijkstra
        results = {}
        for s in sources:
            nodes = initialize(s, scores)
            reached = dijkstra(nodes, adjlist, contacts)
            results[s] = {n.name: n.dist for n in reached}
        return results

    @staticmethod
    def _to_map(contacts: np.ndarray) -> ContactMap:
        ckey = propagation.ckey
        return {ckey(*c["names"]): c["time"] for c in contacts}

    @staticmethod
    def _adjlist(contacts: ContactMap) -> AdjList:
        adjlist = collections.defaultdict(list)
        for n1, n2 in contacts:
            adjlist[n1].append(n2)
            adjlist[n2].append(n1)
        unique = {}
        for n, ne in adjlist.items():
            unique[n] = np.unique(ne)
        return unique

    def _initialize(self, source: int, scores: NpSeq) -> List[Node]:
        initial = propagation.initial
        inits = (scores := np.array([initial(s) for s in scores])).copy()
        inits["val"] *= self.transmission
        nodes = []
        nodes.extend(Node(n, inits[n]) for n in range(source))
        nodes.append(Node(source, inits[source], dist=0, msg=scores[source]))
        nodes.extend(Node(n, inits[n]) for n in range(source + 1, len(scores)))
        return nodes

    def _dijkstra(
            self,
            nodes: List[Node],
            adjlist: AdjList,
            contacts: ContactMap
    ) -> Set[Node]:
        pop = heapq.heappop
        transmission, tol, buffer = self.transmission, self.tol, self.buffer
        ckey = propagation.ckey
        reached = set()
        add, get_ne = reached.add, adjlist.get
        heapq.heapify(heap := nodes.copy())
        while heap:
            node = pop(heap)
            if (msg := node.msg) is not None:
                name, dist, init = node.name, node.dist, node.init
                for n in get_ne(name, []):
                    ne = nodes[n]
                    if msg["time"] <= contacts[ckey(name, ne.name)] + buffer:
                        send = msg.copy()
                        send["val"] *= transmission
                        high_enough = send["val"] >= init["val"] * tol
                        old_enough = send["time"] <= init["time"]
                        closer = ne.dist > (new := dist + 1)
                        if high_enough and old_enough and closer:
                            ne.dist, ne.msg = new, send
                            add(ne)
        return reached

    def copy(self) -> MessageReachability:
        return self.__copy__()

    def __copy__(self) -> MessageReachability:
        return MessageReachability(
            transmission=self.transmission,
            tol=self.tol,
            buffer=self.buffer,
            workers=self.workers,
            verbose=self.verbose)
