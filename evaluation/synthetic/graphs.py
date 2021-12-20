from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

import igraph as ig
import networkx as nx

from evaluation.synthetic.data import DataFactory


class Graph(ABC):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        pass

    @abstractmethod
    def nodes(self) -> Iterable[int]:
        pass

    @abstractmethod
    def edges(self) -> Iterable[Tuple[int, int]]:
        pass


class IGraph(Graph):
    __slots__ = ('graph',)

    def __init__(self, graph: Union[ig.Graph, nx.Graph]):
        super().__init__()
        if isinstance(graph, nx.Graph):
            graph = ig.Graph.from_networkx(graph)
        self.graph = graph

    @property
    def num_nodes(self) -> int:
        return len(self.graph.vs)

    @property
    def num_edges(self) -> int:
        return len(self.graph.es)

    def nodes(self) -> Iterable[int]:
        return iter(self.graph.vs.indices)

    def edges(self) -> Iterable[Tuple[int, int]]:
        return (e.tuple for e in self.graph.es)


class GraphFactory(DataFactory):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, n: int) -> Graph:
        pass


class ConnectedCavemanGraphFactory(GraphFactory):
    __slots__ = ('cliques',)

    def __init__(self, cliques: int):
        super().__init__()
        self.cliques = cliques

    def __call__(self, n: int) -> Graph:
        clique = math.floor(n / self.cliques)
        graph = nx.generators.connected_caveman_graph(self.cliques, clique)
        return IGraph(graph)
