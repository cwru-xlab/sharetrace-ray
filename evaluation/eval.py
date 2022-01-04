import argparse
import functools
import itertools
import logging
import os.path
import pathlib
import pprint
import time
from abc import ABC, abstractmethod
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

PARAMETER_DIR = './/logs//parameters'
pathlib.Path(PARAMETER_DIR).mkdir(parents=True, exist_ok=True)

REAL_WORLD_DIR = './/logs//real-world'
pathlib.Path(REAL_WORLD_DIR).mkdir(parents=True, exist_ok=True)


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


def filter_isolated(g: ig.Graph) -> ig.Graph:
    return g.subgraph(g.vs.select(_degree_gt=0))


def create_sociopatterns_data(
        path: str,
        sep: str = ' ',
        p: float = 0.2,
        seed=None
) -> Dataset:
    dataset_factory = SocioPatternsDatasetFactory(
        score_factory=SocioPatternsScoreFactory(
            value_factory=UniformBernoulliValueFactory(
                per_user=1, p=p, seed=seed),
            time_factory=TimeFactory(days=1, per_day=1, seed=seed)),
        contact_factory=SocioPatternsContactFactory(path=path, sep=sep))
    return dataset_factory()


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


def create_geometric_graph(n: int, seed=None) -> Graph:
    graph = nx.generators.random_geometric_graph(
        n, radius=geometric_radius(n), seed=seed)
    graph = filter_isolated(ig.Graph.from_networkx(graph))
    return synthetic.IGraph(graph)


def geometric_radius(n: int) -> float:
    return min(1, 0.25 ** (np.log10(n) - 1))


def create_power_law_cluster_graph(n: int, seed=None) -> Graph:
    graph = nx.generators.powerlaw_cluster_graph(
        n, m=2, p=0.95, seed=seed)
    graph = filter_isolated(ig.Graph.from_networkx(graph))
    return synthetic.IGraph(graph)


def create_lfr_graph(n: int, seed=None) -> Graph:
    graph = nx.generators.LFR_benchmark_graph(
        n,
        tau1=3,
        tau2=2,
        mu=0.1,
        min_degree=3,
        max_degree=50,
        min_community=10,
        max_community=100,
        seed=seed)
    # Wrap with filter_isolated() if min_degree < 2
    return synthetic.IGraph(ig.Graph.from_networkx(graph))


def get_logger(directory: str, logfile: str) -> logging.Logger:
    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        filename=os.path.join(directory, logfile), mode='w')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    return logger


# noinspection PyTypeChecker
class SyntheticExperiments(ABC):
    __slots__ = ('seed',)

    def __init__(self, seed=None):
        super().__init__()
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
        graph_factory = functools.partial(
            lambda n: create_geometric_graph(n, seed=self.seed))
        self._benchmark(graph_factory, 'geometric')

    def benchmark_power_law_cluster(self) -> None:
        graph_factory = functools.partial(
            lambda n: create_power_law_cluster_graph(n, seed=self.seed))
        self._benchmark(graph_factory, 'power')

    def benchmark_lfr(self) -> None:
        graph_factory = functools.partial(
            lambda n: create_lfr_graph(n, seed=self.seed))
        self._benchmark(graph_factory, 'lfr')

    @abstractmethod
    def _benchmark(self, graph_factory: DataFactory, graph: str) -> None:
        pass


# noinspection PyTypeChecker
class ScalabilityExperiments(SyntheticExperiments):
    __slots__ = ()

    def __init__(self, seed=None):
        super().__init__(seed)

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
        get_logfile, seed = self._logfile, self.seed
        logger = get_logger(SCALABILITY_DIR, get_logfile(graph, 'log'))
        for u in tqdm.tqdm(users):
            if u in (1_000, 20_000):
                graph_path = os.path.join(
                    SCALABILITY_DIR, get_logfile(graph, 'graphml', u))
            else:
                graph_path = None
            dataset = create_synthetic_data(
                users=u,
                graph_factory=graph_factory,
                graph_path=graph_path,
                days=15,
                p=0.2,
                seed=seed)
            for w in workers:
                risk_prop = propagation.RiskPropagation(
                    tol=0.3,
                    workers=w,
                    timeout=0 if w == 1 else 5,
                    logger=logger)
                risk_prop.run(dataset.scores, dataset.contacts)

    @staticmethod
    def _logfile(graph: str, ext: str, users: Optional[int] = None) -> str:
        if users is None:
            name = f'g-{graph}_{round(time.time())}.{ext}'
        else:
            name = f'g-{graph}_u-{users}_{round(time.time())}.{ext}'
        return name


class ParameterExperiments(SyntheticExperiments):
    __slots__ = ('seed',)

    def __init__(self, seed=None):
        super().__init__(seed)

    def _benchmark(self, graph_factory: DataFactory, graph: str) -> None:
        dataset = create_synthetic_data(
            users=10_000,
            graph_factory=graph_factory,
            days=15,
            p=0.2,
            seed=self.seed)
        get_logfile = self._logfile
        loop = list(itertools.product(range(1, 11), range(1, 11)))
        for tol, transmission in tqdm.tqdm(loop):
            logfile = get_logfile(graph, tol, transmission)
            logger = get_logger(PARAMETER_DIR, logfile)
            risk_prop = propagation.RiskPropagation(
                tol=tol / 10,
                transmission=transmission / 10,
                workers=4,
                timeout=5,
                logger=logger)
            risk_prop.run(dataset.scores, dataset.contacts)

    @staticmethod
    def _logfile(graph: str, tol: int, transmission: int) -> str:
        return f'g-{graph}_to-{tol}_tr-{transmission}.log'


class RealWorldExperiments:
    __slots__ = ('seed',)

    def __init__(self, seed=None):
        self.seed = seed

    def benchmark(self, setting: str, path: str) -> None:
        if setting == 'high-school11':
            self.benchmark_high_school11(path)
        elif setting == 'high-school12':
            self.benchmark_high_school12(path)
        elif setting == 'conference':
            self.benchmark_conference(path)
        elif setting == 'workplace':
            self.benchmark_workplace(path)
        else:
            raise ValueError(
                f"'setting' must be one of ('high-school11', 'high-school12', "
                f"'conference', 'workplace'), not {setting}")

    def benchmark_high_school11(self, path: str) -> None:
        self._benchmark(setting='high-school11', path=path, sep='\t')

    def benchmark_high_school12(self, path: str) -> None:
        self._benchmark(setting='high-school12', path=path, sep='\t')

    def benchmark_conference(self, path: str) -> None:
        self._benchmark(setting='conference', path=path, sep=' ')

    def benchmark_workplace(self, path: str) -> None:
        self._benchmark(setting='workplace', path=path, sep=' ')

    def _benchmark(self, setting: str, path: str, sep: str) -> None:
        logger = get_logger(REAL_WORLD_DIR, self._logfile(setting))
        seed = self.seed
        for _ in tqdm.trange(10):
            dataset = create_sociopatterns_data(
                path=path, sep=sep, p=0.2, seed=seed)
            risk_prop = propagation.RiskPropagation(
                tol=0.3, workers=4, timeout=5, logger=logger)
            risk_prop.run(dataset.scores, dataset.contacts)

    @staticmethod
    def _logfile(setting: str) -> str:
        return f'{setting}.log'


def parse_scalability_exps(args: argparse.Namespace) -> None:
    ScalabilityExperiments(args.seed).benchmark(args.graph)


def parse_parameter_exps(args: argparse.Namespace) -> None:
    ParameterExperiments(args.seed).benchmark(args.graph)


def parse_real_world_exps(args: argparse.Namespace) -> None:
    RealWorldExperiments(args.seed).benchmark(args.setting, args.path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    scalability = subparsers.add_parser('scalability')
    scalability.add_argument(
        '--graph', choices=('lfr', 'power', 'geometric'), required=True)
    scalability.add_argument('--seed', type=int, default=None)
    scalability.set_defaults(func=parse_scalability_exps)

    parameters = subparsers.add_parser('parameters')
    parameters.add_argument(
        '--graph', choices=('lfr', 'power', 'geometric'), required=True)
    parameters.add_argument('--seed', type=int, default=None)
    parameters.set_defaults(func=parse_parameter_exps)

    real_world = subparsers.add_parser('real-world')
    real_world.add_argument(
        '--setting',
        choices=('workplace', 'high-school11', 'high-school12', 'conference'),
        required=True)
    real_world.add_argument('--path', required=True)
    real_world.add_argument('--seed', type=int, default=None)
    real_world.set_defaults(func=parse_real_world_exps)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
