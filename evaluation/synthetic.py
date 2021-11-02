import datetime
from typing import Tuple

import numpy as np
from scipy import stats

from sharetrace import model
from sharetrace.search import kdtree, naive

NOW = np.datetime64(datetime.datetime.utcnow(), 's')
rng = np.random.default_rng()


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


def random_history(
        n: int = 1,
        n_events: int = 10,
        n_days_back: int = 14,
        precision: int = 12,
        lat_range: Tuple = (-90, 90),
        long_range: Tuple = (-180, 180),
        geohash: bool = True):
    histories = []
    for h in range(n):
        if geohash:
            locs = random_geohash(
                n_events, n_days_back, lat_range, long_range, precision)
        else:
            locs = random_coordinate(
                n_events, n_days_back, lat_range, long_range)
        times = random_time(n_events, n_days_back)
        histories.append(model.history(locs, h))
    return histories


def random_score(
        n: int = 1,
        n_days_back: int = 14,
        alpha: float = 0.5,
        beta: float = 0.5):
    values = stats.beta.rvs(alpha, beta, size=n)
    times = random_time(n, n_days_back)
    return np.array([model.score(v, t) for v, t in zip(values, times)])


def main():
    min_dur = np.timedelta64(15, 'm')
    n_workers = -1
    n_search = naive.NaiveContactSearch(min_dur=min_dur, n_workers=n_workers)
    k_search = kdtree.KdTreeContactSearch(
        min_dur=min_dur, n_workers=n_workers, eps=0, r=0.01)
    histories = random_history(
        100, lat_range=(0, 1), long_range=(0, 1), precision=5)
    n_contacts = n_search.search(histories)
    k_contacts = k_search.search(histories)
    print(f'SIZE: naive {n_contacts.size} kd-tree {k_contacts.size}')


if __name__ == '__main__':
    main()
