import datetime
import logging.config
import pathlib
from typing import Callable, Tuple

import joblib
import numpy as np
from scipy import stats

import sharetrace.util
from sharetrace import model, propagation, search

logging.config.dictConfig(sharetrace.util.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

NOW = np.datetime64(datetime.datetime.utcnow(), 's')
rng = np.random.default_rng()
DAY_IN_SECONDS = 86_400
HISTORIES_FILENAME_FORMAT = 'histories_%s'
HISTORIES_FILE_FORMAT = 'histories_%s.npy'
SCORES_FILENAME_FORMAT = 'scores_%s'
SCORES_FILE_FORMAT = 'scores_%s.npy'


def random_time(n: int = 1, lookback: int = 14):
    lookback *= DAY_IN_SECONDS
    return NOW - rng.integers(0, lookback, size=n).astype('timedelta64[s]')


def random_coordinate(
        n: int = 1,
        lookback: int = 14,
        lats: Tuple = (-90, 90),
        longs: Tuple = (-180, 180)):
    lats = rng.uniform(*lats, size=n)
    longs = rng.uniform(*longs, size=n)
    times = random_time(n, lookback)
    return np.array([
        model.temporal_loc((lat, long), t)
        for (lat, long, t) in zip(lats, longs, times)])


def random_geohash(
        n: int = 1,
        lookback: int = 14,
        lats: Tuple = (-90, 90),
        longs: Tuple = (-180, 180),
        precision: int = 12):
    locs = random_coordinate(n, lookback, lats, longs)
    return np.array([model.to_geohash(loc, precision) for loc in locs])


def _random_locs(
        name: int,
        events: int = 10,
        lookback: int = 14,
        precision: int = 12,
        lats: Tuple = (-90, 90),
        longs: Tuple = (-180, 180),
        geohash: bool = True):
    if geohash:
        locs = random_geohash(events, lookback, lats, longs, precision)
    else:
        locs = random_coordinate(events, lookback, lats, longs)
    return model.history(locs, name)


def random_history(
        n: int = 1,
        events: int = 10,
        lookback: int = 14,
        precision: int = 12,
        lats: Tuple = (-90, 90),
        longs: Tuple = (-180, 180),
        geohash: bool = True,
        workers: int = 1):
    logger.debug('Generating %d location histories', n)
    par = joblib.Parallel(workers)
    kwargs = {
        'events': events,
        'lookback': lookback,
        'precision': precision,
        'lats': lats,
        'longs': longs,
        'geohash': geohash}
    return par(joblib.delayed(_random_locs)(h, **kwargs) for h in range(n))


def random_score(
        n: int = 1,
        scores: int = 3,
        lookback: int = 14,
        alpha: float = 0.5,
        beta: float = 0.5):
    logger.debug('Generating %d scores', n)
    risk_scores = []
    for _ in range(n):
        values = stats.beta.rvs(alpha, beta, size=scores)
        times = random_time(scores, lookback)
        risk_scores.append([
            model.risk_score(v, t) for v, t in zip(values, times)])
    return np.array(risk_scores)


def generate_histories(
        n: int = 1,
        events: int = 10,
        lookback: int = 14,
        precision: int = 8,
        lats: Tuple = (-90, 90),
        longs: Tuple = (-180, 180),
        geohash: bool = True,
        workers: int = 1):
    return generate(
        lambda: random_history(
            n=n,
            events=events,
            lookback=lookback,
            precision=precision,
            lats=lats,
            longs=longs,
            geohash=geohash,
            workers=workers),
        HISTORIES_FILE_FORMAT % n,
        HISTORIES_FILENAME_FORMAT % n)


def generate_scores(
        n: int = 1,
        scores: int = 3,
        lookback: int = 14,
        alpha: float = 0.5,
        beta: float = 0.5):
    return generate(
        lambda: random_score(n, scores, lookback, alpha, beta),
        SCORES_FILE_FORMAT % n,
        SCORES_FILENAME_FORMAT % n)


def generate(func: Callable, file: str, filename: str):
    data = func()
    if not pathlib.Path(file).exists():
        np.save(filename, data)
    return data


def load_histories(n: int):
    return load(n, 'histories', HISTORIES_FILE_FORMAT % n)


def load_scores(n: int):
    return load(n, 'scores', SCORES_FILE_FORMAT % n)


def load(n: int, name: str, file: str):
    logger.debug('Loading %d %s', n, name)
    return np.load(file, allow_pickle=True)


def generate_data():
    lookback = 14
    for n in range(1, 6):
        generate_histories(
            10 ** n,
            events=10,
            lookback=lookback,
            lats=(0, 10),
            longs=(0, 10),
            geohash=True,
            precision=8,
            workers=-1)
        generate_scores(10 ** n, lookback=lookback, alpha=0.5, beta=0.5)


def main():
    # generate_data()
    n = 1000
    histories = generate_histories(
        n,
        events=25,
        lookback=1,
        lats=(0, 10),
        longs=(0, 10),
        geohash=True,
        precision=8,
        workers=-1)
    scores = generate_scores(n)
    initial = np.array([
        np.sort(s, order=('val', 'time'))[-1]['val'] for s in scores])
    contact_search = search.KdTreeContactSearch(min_dur=900, workers=-1)
    contacts = contact_search.search(histories)
    risk_propagation = propagation.RiskPropagation(parts=4, timeout=1)
    risk_propagation.setup(scores, contacts)
    final = risk_propagation.run()
    print(np.count_nonzero(final - initial))


if __name__ == '__main__':
    main()
