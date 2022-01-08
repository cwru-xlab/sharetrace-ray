from __future__ import annotations

import collections
import copy
import functools
import heapq
from typing import (
    Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union)

import joblib
import numpy as np

from sharetrace import propagation

ContactMap = Mapping[Tuple[int, int], float]
AdjList = Mapping[int, Union[Sequence, np.ndarray]]
Reached = Mapping[int, Mapping[int, int]]


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

    def run_all(self, scores: np.ndarray, contacts: np.ndarray) -> Reached:
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
            scores: np.ndarray,
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
            nodes, heap, reached = initialize(s, scores)
            dijkstra(nodes, heap, reached, adjlist, contacts)
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

    def _initialize(
            self,
            source: int,
            scores: np.ndarray
    ) -> Tuple[List[Node], List[Node], Set[Node]]:
        initial = propagation.initial
        inits = (scores := np.array([initial(s) for s in scores])).copy()
        inits["val"] *= self.transmission
        nodes = []
        append = nodes.append
        for n in range(source):
            append(Node(n, init=inits[n]))
        append(Node(source, dist=0, msg=scores[source], init=inits[source]))
        for n in range(source + 1, len(scores)):
            append(Node(n, init=inits[n]))
        heapq.heapify(heap := nodes.copy())
        reached = set()
        return nodes, heap, reached

    def _dijkstra(
            self,
            nodes: List[Node],
            heap: List[Node],
            reached: Set[Node],
            adjlist: AdjList,
            contacts: ContactMap
    ) -> None:
        pop = heapq.heappop
        transmission, tol, buffer = self.transmission, self.tol, self.buffer
        ckey = propagation.ckey
        add = reached.add
        while heap:
            node = pop(heap)
            if (msg := node.msg) is not None:
                name, dist, init = node.name, node.dist, node.init
                for n in adjlist[name]:
                    ne = nodes[n]
                    if msg["time"] <= contacts[ckey(name, ne.name)] + buffer:
                        send = msg.copy()
                        send["val"] *= transmission
                        high_enough = send["val"] >= init["val"] * tol
                        old_enough = send["time"] <= init["time"]
                        if high_enough and old_enough:
                            if ne.dist > (new := dist + 1):
                                ne.dist = new
                            ne.msg = send
                            add(ne)

    def copy(self) -> MessageReachability:
        return copy.copy(self)

    def __copy__(self) -> MessageReachability:
        return MessageReachability(
            transmission=self.transmission,
            tol=self.tol,
            buffer=self.buffer,
            workers=self.workers,
            verbose=self.verbose)
