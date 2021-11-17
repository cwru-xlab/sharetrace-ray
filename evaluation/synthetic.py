import datetime
import pathlib
from typing import Callable, Tuple

import joblib
import numpy as np
from numpy import arange, clip, concatenate, datetime64, vstack, zeros
from scipy import stats

from sharetrace import model, propagation, search

NOW = np.datetime64(datetime.datetime.utcnow(), 's')
rng = np.random.default_rng()
SEC_PER_DAY = 86400
MIN_PER_HR = 3600
HISTORIES_FILENAME_FORMAT = 'histories_%s'
HISTORIES_FILE_FORMAT = 'histories_%s.npy'
SCORES_FILENAME_FORMAT = 'scores_%s'
SCORES_FILE_FORMAT = 'scores_%s.npy'

LATS = (-90, 90)
LONGS = (-180, 180)


def walk(
        n: int,
        low: float = -1,
        high: float = 1,
        step_low: float = -0.1,
        step_high: float = 0.1,
        xinit: float = 0,
        yinit: float = 0):
    def step(pos, idx, dim):
        delta = rng.uniform(step_low, step_high)
        pos[dim][idx] = clip(pos[dim][idx - 1] + delta, low, high)

    xy = zeros((2, n))
    xy[0][0] = clip(xinit, low, high)
    xy[1][0] = clip(yinit, low, high)
    for i in range(1, n):
        step(xy, i, 0)
        step(xy, i, 1)
    return xy


def create_locs(n: int, steps: int, dist, **kwargs):
    locs = zeros((n, 2, steps))
    for i in range(n):
        xinit, yinit = dist.rvs(size=2)
        locs[i, ::] = walk(steps, xinit=xinit, yinit=yinit, **kwargs)
    return locs


def create_times(n: int, now: datetime64, days: int = 14, per_day: int = 16):
    dt_type, td_type = 'datetime64[s]', 'timedelta64[s]'
    now = datetime64(now).astype(dt_type)
    days += 1
    times = zeros((n, days * per_day), dtype=dt_type)
    offsets = (arange(days) * SEC_PER_DAY).astype(td_type)
    starts = now - offsets
    for i in range(n):
        delta = rng.uniform(0, MIN_PER_HR, size=per_day).cumsum()
        delta = delta.astype(td_type)
        times[i, :] = concatenate([start + delta for start in starts])
    return times


def take(a, idx, axis):
    cut = [slice(None)] * a.ndim
    cut[axis] = idx
    return a[tuple(cut)]


def create_scores(locs, dist, z, days_back: int = 14, per_day: int = 16):
    idx = arange((days_back + 1) * per_day, step=per_day)
    selected = take(locs, idx, -1)
    return vstack([dist.pdf(x.T) / z for x in selected])


def random_time(n: int = 1, lookback: int = 14):
    lookback *= SEC_PER_DAY
    return NOW - rng.integers(0, lookback, size=n).astype('timedelta64[s]')


def random_coordinate(
        n: int = 1,
        lookback: int = 14,
        lats: Tuple = LATS,
        longs: Tuple = LONGS):
    lats = rng.uniform(*lats, size=n)
    longs = rng.uniform(*longs, size=n)
    times = random_time(n, lookback)
    return np.array([
        model.temporal_loc((lat, long), t)
        for (lat, long, t) in zip(lats, longs, times)])


def random_geohash(
        n: int = 1,
        lookback: int = 14,
        lats: Tuple = LATS,
        longs: Tuple = LONGS,
        precision: int = 12):
    locs = random_coordinate(n, lookback, lats, longs)
    return np.array([model.to_geohash(loc, precision) for loc in locs])


def _random_locs(
        name: int,
        events: int = 10,
        lookback: int = 14,
        precision: int = 12,
        lats: Tuple = LATS,
        longs: Tuple = LONGS,
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
        lats: Tuple = LATS,
        longs: Tuple = LONGS,
        geohash: bool = True,
        workers: int = 1):
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
        lats: Tuple = LATS,
        longs: Tuple = LONGS,
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
    return load(HISTORIES_FILE_FORMAT % n)


def load_scores(n: int):
    return load(SCORES_FILE_FORMAT % n)


def load(file: str):
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
    contact_search = search.KdTreeContactSearch(min_dur=900, workers=-1)
    contacts = contact_search.search(histories)
    for t in np.arange(0.1, 1, 0.1):
        risk_propagation = propagation.RayRiskPropagation(
            parts=4, early_stop=10_000, tol=t, timeout=3)
        risk_propagation.setup(scores, contacts)
        risk_propagation.run()


if __name__ == '__main__':
    main()
