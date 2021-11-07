import datetime
import logging.config
import pathlib
from typing import Tuple

import joblib
import numpy as np
from scipy import stats

from sharetrace import logging_config, model, search

logging.config.dictConfig(logging_config.config)
logger = logging.getLogger(__name__)

NOW = np.datetime64(datetime.datetime.utcnow(), 's')
rng = np.random.default_rng()
HISTORIES_FILENAME_FORMAT = 'histories_{}'
HISTORIES_FILE_FORMAT = 'histories_{}.npy'


def random_time(n: int = 1, n_days_back: int = 14):
    return NOW - rng.integers(0, n_days_back, size=n).astype('timedelta64[D]')


def random_coordinate(
        n: int = 1,
        n_days_back: int = 14,
        lat_range: Tuple = (-90, 90),
        long_range: Tuple = (-180, 180)):
    lats = rng.uniform(*lat_range, size=n)
    longs = rng.uniform(*long_range, size=n)
    times = random_time(n, n_days_back)
    return np.array([
        model.temporal_loc((lat, long), t)
        for (lat, long, t) in zip(lats, longs, times)])


def random_geohash(
        n: int = 1,
        n_days_back: int = 14,
        lat_range: Tuple = (-90, 90),
        long_range: Tuple = (-180, 180),
        precision: int = 12):
    locs = random_coordinate(n, n_days_back, lat_range, long_range)
    return np.array([model.to_geohash(loc, precision) for loc in locs])


def _random_locs(
        name: int,
        n_events: int = 10,
        n_days_back: int = 14,
        precision: int = 12,
        lat_range: Tuple = (-90, 90),
        long_range: Tuple = (-180, 180),
        geohash: bool = True):
    if geohash:
        locs = random_geohash(
            n_events, n_days_back, lat_range, long_range, precision)
    else:
        locs = random_coordinate(
            n_events, n_days_back, lat_range, long_range)
    return model.history(locs, name)


def random_history(
        n: int = 1,
        n_events: int = 10,
        n_days_back: int = 14,
        precision: int = 12,
        lat_range: Tuple = (-90, 90),
        long_range: Tuple = (-180, 180),
        geohash: bool = True,
        n_jobs: int = 1):
    logger.debug('Generating %d location histories', n)
    par = joblib.Parallel(n_jobs=n_jobs)
    kwargs = {
        'n_events': n_events,
        'n_days_back': n_days_back,
        'precision': precision,
        'lat_range': lat_range,
        'long_range': long_range,
        'geohash': geohash}
    return par(joblib.delayed(_random_locs)(h, **kwargs) for h in range(n))


def random_score(
        n: int = 1,
        n_days_back: int = 14,
        alpha: float = 0.5,
        beta: float = 0.5):
    logger.debug('Generating %d scores', n)
    values = stats.beta.rvs(alpha, beta, size=n)
    times = random_time(n, n_days_back)
    return np.array([model.risk_score(v, t) for v, t in zip(values, times)])


def generate_histories(
        n: int = 1,
        n_events: int = 10,
        n_days_back: int = 14,
        precision: int = 12,
        lat_range: Tuple = (-90, 90),
        long_range: Tuple = (-180, 180),
        geohash: bool = True,
        n_jobs: int = 1):
    file = HISTORIES_FILE_FORMAT.format(n)
    filename = HISTORIES_FILENAME_FORMAT.format(n)
    if not pathlib.Path(file).exists():
        np.save(filename, random_history(
            n,
            n_events,
            n_days_back,
            precision,
            lat_range,
            long_range,
            geohash,
            n_jobs))


def load_histories(n: int):
    logger.debug('Loading %d location histories', n)
    return np.load(HISTORIES_FILE_FORMAT.format(n), allow_pickle=True)


def generate_data():
    for n in range(1, 6):
        generate_histories(
            10 ** n,
            n_events=10,
            n_days_back=14,
            lat_range=(0, 10),
            long_range=(0, 10),
            geohash=True,
            precision=8,
            n_jobs=-1)


def main():
    generate_data()
    histories = load_histories(1000)
    kwargs = {'min_dur': np.timedelta64(15, 'm'), 'workers': -1}
    searches = (
        search.KdTreeContactSearch(**kwargs),
        # search.BallTreeContactSearch(**kwargs),
        # search.BruteContactSearch(**kwargs)
    )
    for contact_search in searches:
        contact_search.search(histories)


if __name__ == '__main__':
    main()
