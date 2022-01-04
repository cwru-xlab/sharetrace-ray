import argparse
import logging
import os.path
import pathlib
import pprint
import time
from typing import Optional

import igraph as ig
import networkx as nx
import numpy as np
import tqdm

from evaluation import synthetic
from sharetrace import model, propagation
from synthetic import (ContactFactory, DataFactory, Dataset, DatasetFactory,
                       Graph, ScoreFactory, SocioPatternsContactFactory,
                       SocioPatternsDatasetFactory, SocioPatternsScoreFactory,
                       TimeFactory, UniformBernoulliValueFactory)

SCALABILITY_DIR = './/logs//scalability'
pathlib.Path(SCALABILITY_DIR).mkdir(parents=True, exist_ok=True)


def filter_isolated(g: ig.Graph) -> ig.Graph:
    return g.subgraph(g.vs.select(_degree_gt=0))


def create_sociopatterns_data(
        path: str,
        sep: str = ' ',
        users: int = 1000,
        days: int = 15,
        p: float = 0.2,
        seed=None
) -> Dataset:
    dataset_factory = SocioPatternsDatasetFactory(
        score_factory=SocioPatternsScoreFactory(
            value_factory=UniformBernoulliValueFactory(
                per_user=days, p=p, seed=seed),
            time_factory=TimeFactory(days=days, per_day=1, seed=seed)),
        contact_factory=SocioPatternsContactFactory(path=path, sep=sep))
    return dataset_factory(users)


def create_synthetic_data(
        users: int,
        graph_factory: DataFactory,
        days: int = 15,
        p: float = 0.2,
        graph_path: Optional[str] = None,
        seed=None,
) -> Dataset:
    dataset_factory = DatasetFactory(
        score_factory=ScoreFactory(
            value_factory=UniformBernoulliValueFactory(
                per_user=days, p=p, seed=seed),
            time_factory=TimeFactory(days=days, per_day=1, seed=seed)),
        contact_factory=ContactFactory(
            graph_factory=graph_factory,
            time_factory=TimeFactory(
                days=15, per_day=1, random_first=True, seed=seed),
            graph_path=graph_path))
    return dataset_factory(users)


# noinspection PyTypeChecker
class ScalabilityExperiments:
    __slots__ = ('seed',)

    def __init__(self, seed=None):
        self.seed = seed

    def benchmark(self, graph: str):
        if graph == 'geometric':
            self.benchmark_geometric()
        elif graph == 'power':
            self.benchmark_power_law_cluster()
        elif graph == 'lfr':
            self.benchmark_lfr()
        else:
            raise ValueError(
                f"'graph' must be one of ('geometric', 'power', 'lfr'), "
                f"not {graph}")

    def benchmark_geometric(self) -> None:
        self._benchmark(self._create_geometric, 'geometric')

    def benchmark_power_law_cluster(self) -> None:
        self._benchmark(self._create_power_law_cluster, 'power')

    def benchmark_lfr(self) -> None:
        self._benchmark(self._create_lfr, 'lfr')

    def _create_geometric(self, n: int) -> Graph:
        graph = nx.generators.random_geometric_graph(
            n, radius=self._geo_rad(n), seed=self.seed)
        graph = filter_isolated(ig.Graph.from_networkx(graph))
        return synthetic.IGraph(graph)

    @staticmethod
    def _geo_rad(n: int) -> float:
        return min(1, 0.25 ** (np.log10(n) - 1))

    def _create_power_law_cluster(self, n: int) -> Graph:
        graph = nx.generators.powerlaw_cluster_graph(
            n, m=2, p=0.95, seed=self.seed)
        graph = filter_isolated(ig.Graph.from_networkx(graph))
        return synthetic.IGraph(graph)

    def _create_lfr(self, n: int) -> Graph:
        graph = nx.generators.LFR_benchmark_graph(
            n,
            tau1=3,
            tau2=2,
            mu=0.1,
            min_degree=3,
            max_degree=50,
            min_community=10,
            max_community=100,
            seed=self.seed)
        # Wrap with filter_isolated() if min_degree < 2
        return synthetic.IGraph(ig.Graph.from_networkx(graph))

    def _benchmark(self, graph_factory: DataFactory, graph: str) -> None:
        workers = (1, 2, 4, 8)
        users = (
            200,
            400,
            800,
            1_000,
            20_000,
            40_000,
            80_000,
            100_000,
            200_000,
            400_000,
            800_000,
            1_000_000)
        logger = self._configure_logger(graph)
        for u in tqdm.tqdm(users):
            if u in (1_000, 20_000):
                graph_path = os.path.join(
                    SCALABILITY_DIR, self._filename(graph, 'graphml', u))
            else:
                graph_path = None
            dataset = create_synthetic_data(
                users=u,
                graph_factory=graph_factory,
                graph_path=graph_path,
                days=15,
                p=0.2,
                seed=self.seed)
            for w in workers:
                risk_prop = propagation.RiskPropagation(
                    tol=0.3,
                    workers=w,
                    timeout=0 if w == 1 else 5,
                    logger=logger)
                risk_prop.run(dataset.scores, dataset.contacts)

    def _configure_logger(self, graph: str):
        name = self._filename(graph, 'log')
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            filename=os.path.join(SCALABILITY_DIR, name), mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        return logger

    @staticmethod
    def _filename(graph: str, ext: str, users: Optional[int] = None) -> str:
        if users is None:
            name = f'g-{graph}_{round(time.time())}.{ext}'
        else:
            name = f'g-{graph}_u-{users}_{round(time.time())}.{ext}'
        return name


def model_object_sizes():
    score = model.risk_score(1, 1)
    min_geohash = model.temporal_loc('a', 1)
    max_geohash = model.temporal_loc('abcdefghijkl', 1)
    coord = model.temporal_loc((0, 0), 1)
    objects = {
        'risk score': score.nbytes,
        'min geohash': min_geohash.nbytes,
        'max geohash': max_geohash.nbytes,
        'coord': coord.nbytes,
        'contact': model.contact((0, 0), 1).nbytes,
        'msg': model.message([], 1, 1, 1, 1).nbytes,
        'min geohash history': model.history([min_geohash], 0).nbytes,
        'max geohash history': model.history([max_geohash], 0).nbytes,
        'coord history': model.history([coord], 0).nbytes,
        'node': model.node([1], 1).nbytes}
    pprint.pprint(objects, indent=1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    scalability = subparsers.add_parser('scalability')
    scalability.add_argument(
        '--graph', choices=('lfr', 'power', 'geometric'), required=True)
    scalability.add_argument('--seed', type=int, default=None)
    scalability.set_defaults(func=parse_scalability)

    args = parser.parse_args()
    args.func(args)


def parse_scalability(args: argparse.Namespace):
    ScalabilityExperiments(args.seed).benchmark(args.graph)


if __name__ == '__main__':
    main()
