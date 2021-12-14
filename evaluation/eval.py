import argparse
import itertools
import json
import logging
from logging import config

import tqdm

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
    contact_search(1000, 11000, 1000)


def exp2():
    risk_prop(
        timeout=0.01,
        impl='serial',
        ustart=1000,
        ustop=11000,
        ustep=1000,
        wstart=1,
        wstop=2,
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
        ustop=11000,
        ustep=1000,
        wstart=2,
        wstop=5,
        wstep=1)


def contact_search(start: int, stop: int, step: int) -> None:
    data = synthetic.create_data(stop - step, low=-2, high=2)
    scores, hists = data.scores, data.geohashes()
    synthetic.save_scores(scores)
    synthetic.save_geohashes(hists)
    cs = new_contact_search(get_logger('contact-search'))
    for n in tqdm.tqdm(range(start, stop, step)):
        contacts = cs.search(hists[:n])
        synthetic.save_contacts(contacts, n)


def risk_prop(
        timeout: float,
        impl: str,
        ustart: int,
        ustop: int,
        ustep: int,
        wstart: int,
        wstop: int,
        wstep: int):
    logger = logging.getLogger(f'risk-propagation:{impl}')
    scores = synthetic.load_scores()
    loop = list(itertools.product(
        range(ustart, ustop, ustep),
        range(wstart, wstop, wstep)))
    for n, w in tqdm.tqdm(loop):
        if impl == 'ray':
            rp = LogExposuresRiskPropagation(
                timeout=timeout, logger=logger, workers=w, tol=0.3)
        else:
            rp = propagation.RiskPropagation(
                timeout=timeout, logger=logger, workers=w, tol=0.3)
        contacts = synthetic.load_contacts(n)
        rp.setup(scores[:n], contacts)
        rp.run()


def exp5():
    scores = synthetic.load_scores()
    contacts = synthetic.load_contacts(10_000)
    logger = get_logger('risk-propagation:tolerance')
    for t in tqdm.tqdm(range(1, 11, 1)):
        rp = propagation.RayRiskPropagation(
            tol=round(t / 10, 1), timeout=3, logger=logger)
        rp.setup(scores, contacts)
        rp.run()


def new_contact_search(logger=None):
    return search.ContactSearch(min_dur=15, tol=200, workers=-1, logger=logger)


def get_logger(name: str):
    return logging.getLogger(name)


def main(e=None):
    if e is None:
        e = parse().experiment
    eval(f'exp{e}()')


def cs_main():
    dataset = synthetic.create_data(5000, low=-2, high=2, save=False)
    cs = new_contact_search(logging.getLogger())
    contacts = cs.search(dataset.geohashes())


def rp_main():
    logger = logging.getLogger()
    dataset = synthetic.create_data(1000, low=-2, high=2, save=False)
    cs = new_contact_search()
    contacts = cs.search(dataset.geohashes())
    rp = propagation.RayRiskPropagation(
        logger=logger, workers=2, timeout=3, tol=0.3)
    rp.setup(dataset.scores, contacts)
    rp.run()


if __name__ == '__main__':
    main()
