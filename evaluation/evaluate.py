import argparse
import functools
import itertools
import logging
import os.path
import pathlib
import pprint
import time
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import igraph as ig
import networkx as nx
import numpy as np
import tqdm

from evaluation import reachability
from sharetrace import model, propagation, util
from sharetrace.propagation import RiskPropagation
from synthetic import (CachedGraphFactory, ContactFactory, DataFactory, Dataset,
                       DatasetFactory, ScoreFactory,
                       SocioPatternsContactFactory, SocioPatternsDatasetFactory,
                       SocioPatternsGraphFactory, TimeFactory,
                       UniformBernoulliValueFactory)

SCALABILITY_DIR = ".//logs//scalability"
pathlib.Path(SCALABILITY_DIR).mkdir(parents=True, exist_ok=True)

PARAMS_DIR = ".//logs//parameters"
pathlib.Path(PARAMS_DIR).mkdir(parents=True, exist_ok=True)

REAL_WORLD_DIR = ".//logs//real"
pathlib.Path(REAL_WORLD_DIR).mkdir(parents=True, exist_ok=True)


def model_object_sizes():
    score = model.risk_score(1, 1)
    min_geohash = model.temporal_loc("a", 1)
    max_geohash = model.temporal_loc("abcdefghijkl", 1)
    coord = model.temporal_loc((0, 0), 1)
    objects = {
        "risk score": score.nbytes,
        "min geohash": min_geohash.nbytes,
        "max geohash": max_geohash.nbytes,
        "coord": coord.nbytes,
        "contact": model.contact((0, 0), 1).nbytes,
        "msg": model.message([], 1, 1, 1, 1).nbytes,
        "min geohash history": model.history([min_geohash], 0).nbytes,
        "max geohash history": model.history([max_geohash], 0).nbytes,
        "coord history": model.history([coord], 0).nbytes,
        "node": model.node([1], 1).nbytes}
    pprint.pprint(objects, indent=1)


def prune(g: ig.Graph) -> ig.Graph:
    """Remove isolated nodes and simplify the graph."""
    pruned = g.subgraph(g.vs.select(_degree_gt=0))
    pruned.simplify(combine_edges="max")
    return pruned


def create_sociopatterns_data(
        path: str,
        sep: str = " ",
        p: float = 0.2,
        graph_path: Optional[str] = None,
        seed=None
) -> Dataset:
    graph_factory = CachedGraphFactory(SocioPatternsGraphFactory(path, sep))
    dataset_factory = SocioPatternsDatasetFactory(
        score_factory=ScoreFactory(
            value_factory=UniformBernoulliValueFactory(
                per_user=1, p=p, seed=seed),
            time_factory=TimeFactory(days=1, per_day=1, seed=seed)),
        graph_factory=graph_factory,
        contact_factory=SocioPatternsContactFactory(graph_factory, graph_path))
    return dataset_factory()


def create_synthetic_data(
        users: int,
        graph_factory: DataFactory,
        days: int = 15,
        p: float = 0.2,
        graph_path: Optional[str] = None,
        seed=None,
) -> Dataset:
    graph_factory = CachedGraphFactory(graph_factory)
    dataset_factory = DatasetFactory(
        score_factory=ScoreFactory(
            value_factory=UniformBernoulliValueFactory(
                per_user=days, p=p, seed=seed),
            time_factory=TimeFactory(days=days, per_day=1, seed=seed)),
        graph_factory=graph_factory,
        contact_factory=ContactFactory(
            graph_factory=graph_factory,
            time_factory=TimeFactory(
                days=15, per_day=1, random_first=True, seed=seed),
            graph_path=graph_path))
    return dataset_factory(users)


def geometric_graph(n: int, seed=None) -> ig.Graph:
    graph = nx.generators.random_geometric_graph(
        n, radius=geometric_radius(n), seed=seed)
    return prune(ig.Graph.from_networkx(graph))


def geometric_radius(n: int) -> float:
    return min(1, 0.25 ** (np.log10(n) - 1))


def scale_free_cluster_graph(n: int, seed=None) -> ig.Graph:
    graph = nx.generators.powerlaw_cluster_graph(n, m=2, p=0.95, seed=seed)
    return prune(ig.Graph.from_networkx(graph))


def lfr_graph(n: int, seed=None) -> ig.Graph:
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
    return prune(ig.Graph.from_networkx(graph))


def get_logger(directory: str, logfile: str) -> logging.Logger:
    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        filename=os.path.join(directory, logfile), mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


# noinspection PyTypeChecker
class SyntheticExperiments(ABC):
    __slots__ = ("seed",)

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def benchmark(self, graph: str):
        if graph == "geometric":
            self.benchmark_geometric()
        elif graph == "power":
            self.benchmark_scale_free_cluster()
        elif graph == "lfr":
            self.benchmark_lfr()
        else:
            raise ValueError(
                f"'graph' must be one of ('geometric', 'power', 'lfr'), "
                f"not {graph}")

    def benchmark_geometric(self) -> None:
        factory = functools.partial(lambda n: geometric_graph(n, self.seed))
        self._benchmark(factory, "geometric")

    def benchmark_scale_free_cluster(self) -> None:
        factory = functools.partial(
            lambda n: scale_free_cluster_graph(n, self.seed))
        self._benchmark(factory, "power")

    def benchmark_lfr(self) -> None:
        factory = functools.partial(lambda n: lfr_graph(n, self.seed))
        self._benchmark(factory, "lfr")

    @abstractmethod
    def _benchmark(self, graph_factory: DataFactory, graph_name: str) -> None:
        pass


# noinspection PyTypeChecker
class ScalabilityExperiments(SyntheticExperiments):
    __slots__ = ()

    def __init__(self, seed=None):
        super().__init__(seed)

    def _benchmark(self, graph_factory: DataFactory, graph_name: str) -> None:
        logger = get_logger(SCALABILITY_DIR, self._logfile(graph_name, "log"))
        rng = np.arange(1, 11)
        users = np.concatenate([rng * (10 ** p) for p in range(2, 5)])
        for u in tqdm.tqdm(users):
            if u in (100, 1000, 10000):
                graph_path = os.path.join(
                    SCALABILITY_DIR, self._logfile(graph_name, "graphml", u))
            else:
                graph_path = None
            dataset = create_synthetic_data(
                users=u,
                graph_factory=graph_factory,
                graph_path=graph_path,
                days=15,
                p=0.2,
                seed=self.seed)
            risk_prop = GraphMetricsRiskPropagation(
                dataset.graph,
                tol=0.6,
                workers=(w := self._workers(u)),
                timeout=0 if w == 1 else 5,
                early_stop=u * 10,
                logger=logger)
            risk_prop.run(dataset.scores, dataset.contacts)

    @staticmethod
    def _workers(users: int) -> int:
        if 100 <= users < 1000:
            workers = 1
        elif 1000 <= users < 10000:
            workers = 2
        else:
            workers = 4
        return workers

    @staticmethod
    def _logfile(graph: str, ext: str, users: Optional[int] = None) -> str:
        if users is None:
            name = f"{graph}-{round(time.time())}.{ext}"
        else:
            name = f"{graph}-{users}-{round(time.time())}.{ext}"
        return name


class ParameterExperiments(SyntheticExperiments):
    __slots__ = ("seed",)

    def __init__(self, seed=None):
        super().__init__(seed)

    def _benchmark(self, graph_factory: DataFactory, graph_name: str) -> None:
        logger = get_logger(PARAMS_DIR, self._logfile(graph_name, "log"))
        # transmission = 1 never terminates because of no decay.
        loop = list(itertools.product(range(1, 11), range(1, 10)))
        users = 5000
        dataset = create_synthetic_data(
            users=users,
            graph_factory=graph_factory,
            days=15,
            p=0.2,
            seed=self.seed)
        for tol, transmission in tqdm.tqdm(loop):
            risk_prop = propagation.RiskPropagation(
                tol=tol / 10,
                transmission=transmission / 10,
                workers=2,
                timeout=5,
                early_stop=10 * users,
                logger=logger)
            risk_prop.run(dataset.scores, dataset.contacts)

    @staticmethod
    def _logfile(graph: str, ext: str) -> str:
        return f"{graph}-{round(time.time())}.{ext}"


class RealWorldExperiments:
    __slots__ = ("seed",)

    def __init__(self, seed=None):
        self.seed = seed

    def benchmark(self, setting: str, path: str) -> None:
        if setting == "highschool11":
            self.benchmark_high_school11(path)
        elif setting == "highschool12":
            self.benchmark_high_school12(path)
        elif setting == "conference":
            self.benchmark_conference(path)
        elif setting == "workplace":
            self.benchmark_workplace(path)
        else:
            raise ValueError(
                f"'setting' must be one of ('highschool11', 'highschool12', "
                f"'conference', 'workplace'), not {setting}")

    def benchmark_high_school11(self, path: str) -> None:
        self._benchmark(setting="highschool11", path=path, sep="\t")

    def benchmark_high_school12(self, path: str) -> None:
        self._benchmark(setting="highschool12", path=path, sep="\t")

    def benchmark_conference(self, path: str) -> None:
        self._benchmark(setting="conference", path=path, sep=" ")

    def benchmark_workplace(self, path: str) -> None:
        self._benchmark(setting="workplace", path=path, sep=" ")

    def _benchmark(self, setting: str, path: str, sep: str) -> None:
        logger = get_logger(REAL_WORLD_DIR, self._logfile(setting, "log"))
        for i in tqdm.trange(10):
            if i == 0:
                graph_path = os.path.join(
                    REAL_WORLD_DIR, self._logfile(setting, "graphml"))
            else:
                graph_path = None
            dataset = create_sociopatterns_data(
                path=path,
                sep=sep,
                p=0.2,
                graph_path=graph_path,
                seed=self.seed)
            risk_prop = GraphMetricsRiskPropagation(
                dataset.graph, tol=0.6, workers=1, timeout=0, logger=logger)
            risk_prop.run(dataset.scores, dataset.contacts)

    @staticmethod
    def _logfile(setting: str, ext: str) -> str:
        return f"{setting}-{round(time.time())}.{ext}"


class GraphMetricsRiskPropagation(RiskPropagation):
    __slots__ = ("graph",)

    def __init__(self, graph: ig.Graph, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph

    def _run(self, scores, contacts):
        result = super()._run(scores, contacts)
        self._log_graph_stats(scores, contacts)
        return result

    def _log_graph_stats(self, scores, contacts):
        graph = self.graph
        approx = util.approx
        reach = reachability.MessageReachability(
            transmission=self.transmission,
            tol=self.tol,
            buffer=self.time_buffer,
            workers=4)
        reached = reach.run_all(scores, contacts)
        ratio, _, ideal = reachability.ratio(reached, graph)
        reaches = itertools.chain(*(n.values() for n in reached.values()))
        reaches = np.array(list(reaches), dtype=np.int8)
        unreached = np.zeros(ideal - len(reaches), dtype=np.int8)
        reaches = np.concatenate([reaches, unreached])
        effs = graph.harmonic_centrality(mode="all", normalized=True)
        reach_stats = self._basic_stats(reaches)
        deg_stats = self._basic_stats(degrees := graph.degree())
        eff_stats = self._basic_stats(effs)
        min_reached, max_reached, avg_reached, std_reached = reach_stats
        min_deg, max_deg, avg_deg, std_deg = deg_stats
        min_eff, max_eff, avg_eff, std_eff = eff_stats
        self.log["Statistics"].update({
            "MinMessageReachability": min_reached,
            "MaxMessageReachability": max_reached,
            "AvgMessageReachability": avg_reached,
            "StdMessageReachability": std_reached,
            "MessagesReachabilities": reaches.tolist(),
            "ReachabilityRatio": approx(ratio),
            "Radius": graph.radius(mode="all"),
            "Diameter": graph.diameter(directed=False, unconn=True),
            "MinDegree": min_deg,
            "MaxDegree": max_deg,
            "AvgDegree": avg_deg,
            "StdDegree": std_deg,
            "Degrees": degrees,
            "MinEfficiency": min_eff,
            "MaxEfficiency": max_eff,
            "AvgEfficiency": avg_eff,
            "StdEfficiency": std_eff,
            "Efficiencies": [approx(e) for e in effs]})

    @staticmethod
    def _basic_stats(data) -> Sequence[float]:
        ops = (np.min, np.max, np.mean, np.std)
        return [util.approx(op(data)) for op in ops]


def parse_scalability_exps(args: argparse.Namespace) -> None:
    ScalabilityExperiments(args.seed).benchmark(args.graph)


def parse_parameter_exps(args: argparse.Namespace) -> None:
    ParameterExperiments(args.seed).benchmark(args.graph)


def parse_real_world_exps(args: argparse.Namespace) -> None:
    RealWorldExperiments(args.seed).benchmark(args.setting, args.path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    scalability = subparsers.add_parser("scalability")
    scalability.add_argument(
        "--graph", choices=("lfr", "power", "geometric"), required=True)
    scalability.add_argument("--seed", type=int, default=None)
    scalability.set_defaults(func=parse_scalability_exps)

    parameters = subparsers.add_parser("parameters")
    parameters.add_argument(
        "--graph", choices=("lfr", "power", "geometric"), required=True)
    parameters.add_argument("--seed", type=int, default=None)
    parameters.set_defaults(func=parse_parameter_exps)

    real_world = subparsers.add_parser("real")
    real_world.add_argument(
        "--setting",
        choices=("workplace", "highschool11", "highschool12", "conference"),
        required=True)
    real_world.add_argument("--path", required=True)
    real_world.add_argument("--seed", type=int, default=None)
    real_world.set_defaults(func=parse_real_world_exps)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
