import argparse
import json
import logging
from logging import config

import numpy as np

import synthetic
from sharetrace import propagation, search, util
from sharetrace.propagation import Array

logging.config.dictConfig(util.logging_config())


class LogExposuresRiskPropagation(propagation.RayRiskPropagation):
    __slots__ = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> Array:
        results = super().run()
        self.logger.info(json.dumps({'ExposureScores': results.tolist()}))
        return results


def parse():
    parser = argparse.ArgumentParser(
        description='''
            Experiment 5 measures the effect of the send tolerance.

            EXPERIMENTS

                Abbreviations
                    CS:  contact search
                    RP:  risk propagation (implementation)
                    P:   number of processes
                    U:   number of users
                    S:   step size

                1: CS               P = 8       U = 1K – 10K    S = 1K
                2: RP (serial)      P = 1       U = 1K - 10K    S = 1K
                3: RP (lewicki)     P = 2 – 4   U = 1K – 10K    S = 1K
                4: RP (ray)         P = 2 – 4   U = 1K – 10K    S = 1K
                5: RP (ray)         P = 4       U = 1K - 10K    S = 1K
            ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('experiment', type=int)
    return parser.parse_args()


def exp1():
    contact_search(1000, 10100, 1000)


def exp2():
    risk_prop(
        timeout=0.01,
        impl='serial',
        ustart=1000,
        ustop=10100,
        ustep=1000,
        wstart=1,
        wstop=1,
        wstep=1)


def exp3():
    exp34('lewicki')


def exp4():
    exp34('ray')


def exp34(impl: str):
    risk_prop(
        timeout=3,
        impl=impl,
        ustart=1000,
        ustop=10100,
        ustep=1000,
        wstart=2,
        wstop=5,
        wstep=1)


def contact_search(start: int, stop: int, step: int) -> None:
    logger = get_logger('contact-search')
    data = synthetic.create_data(stop - step)
    hists = data.geohashes()
    cs = search.ContactSearch(logger=logger, min_dur=15, workers=-1)
    for n in range(start, stop, step):
        cs.search(hists[:n])


def risk_prop(
        timeout: float,
        impl: str,
        ustart: int,
        ustop: int,
        ustep: int,
        wstart: int,
        wstop: int,
        wstep: int):
    data = synthetic.create_data()
    scores, hists = data.scores, data.geohashes()
    logger = logging.getLogger(f'risk-propagation:{impl}')
    cs = search.ContactSearch(min_dur=15, workers=-1)
    if impl == 'ray':
        rp = LogExposuresRiskPropagation
    else:
        rp = propagation.RiskPropagation
    for n in range(ustart, ustop, ustep):
        contacts = cs.search(hists[:n])
        for w in range(wstart, wstop, wstep):
            rp = rp(timeout=timeout, logger=logger, workers=w, tol=0.3)
            rp.setup(scores[:n], contacts)
            rp.run()


def exp5():
    data = synthetic.create_data()
    scores, hists = data.scores, data.geohashes()
    logger = get_logger('risk-propagation:tolerance')
    cs = search.ContactSearch(min_dur=15, workers=-1)
    for n, t in np.arange(0.1, 1.1, 0.1):
        contacts = cs.search(hists[:n])
        rp = propagation.RayRiskPropagation(tol=t, timeout=3, logger=logger)
        rp.setup(scores[:n], contacts)
        rp.run()


def get_logger(name: str):
    return logging.getLogger(name)


def main():
    eval(f'exp{parse().experiment}()')


def cs_main():
    logger = logging.getLogger()
    dataset = synthetic.create_data(5000, low=-2, high=2, save=False)
    cs = search.ContactSearch(
        logger=logger, min_dur=15, tol=200, workers=-1, verbose=2)
    contacts = cs.search(dataset.geohashes())


def rp_main():
    logger = logging.getLogger()
    dataset = synthetic.create_data(1000, low=-1, high=1, save=False)
    cs = search.ContactSearch(
        logger=logger, min_dur=15, tol=200, workers=-1, verbose=1)
    contacts = cs.search(dataset.geohashes())
    rp = propagation.RayRiskPropagation(
        logger=logger, workers=4, timeout=0.01, tol=0.3)
    rp.setup(dataset.scores, contacts)
    rp.run()


if __name__ == '__main__':
    rp_main()
