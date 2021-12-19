import logging
import os
import random
import warnings
from datetime import datetime
from logging import config

import numpy as np
from scipy import stats

from sharetrace import model, util
from sharetrace.util import DateTime

logging.config.dictConfig(util.logging_config())

rng = np.random.default_rng()
SEC_PER_DAY = 86400
MIN_PER_HR = 3600
TIMES_FILENAME = 'data//times.npy'
VALUES_FILENAME = 'data//values.npy'
SCORES_FILENAME = 'data//scores.npy'
LOCATIONS_FILENAME = 'data//locations.npy'
HISTORIES_FILENAME = 'data//histories.npy'
GEOHASHES_FILENAME = 'data//geohashes.npy'
CONTACTS_FILE_FORMAT = 'data//contacts:{}.npy'

try:
    os.mkdir('./data')
except FileExistsError:
    pass


def walk(
        n: int,
        low: float = -1,
        high: float = 1,
        step_low: float = -0.1,
        step_high: float = 0.1,
        xinit: float = 0,
        yinit: float = 0):
    xy = np.zeros((2, n))
    xy[:, 0] = np.clip((xinit, yinit), low, high)
    delta = rng.uniform(step_low, step_high, size=(2, n))
    for i in range(1, n):
        xy[:, i] = np.clip(xy[:, i - 1] + delta[:, i], low, high)
    return xy


def create_locs(n: int, days: int, per_day: int, dist, **kwargs):
    steps = num_points(days, per_day)
    locs = np.zeros((n, 2, steps))
    for i in range(n):
        xinit, yinit = dist(2)
        locs[i, ::] = walk(steps, xinit=xinit, yinit=yinit, **kwargs)
    return locs


def create_times(n: int, now: DateTime, days: int = 14, per_day: int = 16):
    dt_type, td_type = 'datetime64[s]', 'timedelta64[s]'
    now = np.datetime64(now).astype(dt_type)
    times = np.zeros((n, num_points(days, per_day)), dtype=dt_type)
    offsets = (np.arange(days + 1) * SEC_PER_DAY).astype(td_type)
    starts = now - offsets
    for i in range(n):
        delta = rng.uniform(0, MIN_PER_HR, size=per_day).cumsum()
        delta = delta.astype(td_type)
        times[i, :] = np.concatenate([start + delta for start in starts])
    return times


def take(a, idx, axis):
    cut = [slice(None)] * a.ndim
    cut[axis] = idx
    return a[tuple(cut)]


def create_values(users: int, per_user: int = 15, p: float = 0.5):
    indicator = stats.bernoulli.rvs(p, size=users)
    replace = np.flatnonzero(indicator)
    values = rng.uniform(0, 0.5, size=(users, per_user))
    values[replace] = rng.uniform(0.5, 1, size=(len(replace), per_user))
    return values


def num_points(days, per_day):
    return (days + 1) * per_day


def to_scores(values, times):
    scores = []
    for uvals, utimes in zip(values, times):
        scores.append([model.risk_score(v, t) for v, t in zip(uvals, utimes)])
    return np.array(scores)


def to_histories(locs, times):
    histories = []
    for u, (utimes, ulocs) in enumerate(zip(times, locs)):
        tlocs = [
            model.temporal_loc(loc, time) for time, loc in zip(utimes, ulocs.T)]
        histories.append(model.history(tlocs, u))
    return np.array(histories)


def to_geohashes(histories, prec: int = 8):
    convert = model.to_geohashes
    return [convert(h, prec) for h in histories]


class Dataset:
    __slots__ = ('times', 'values', 'locs', 'scores', 'histories')

    def __init__(self, times, values, locs, scores, histories):
        self.times = times
        self.values = values
        self.locs = locs
        self.scores = scores
        self.histories = histories

    def geohashes(self, prec: int = 8):
        return to_geohashes(self.histories, prec)


class GaussianMixture:
    __slots__ = ('locs', 'scales', 'weights', 'components')

    def __init__(self, locs, scales, weights):
        self.locs = locs
        self.scales = scales
        self.weights = weights
        self.components = [
            stats.norm(loc, scale) for loc, scale in zip(locs, scales)]

    def __call__(self, n):
        component = random.choices(self.components, weights=self.weights)[0]
        return component.rvs(size=n)


def create_data(
        users: int = 10_000,
        days: int = 14,
        per_day: int = 16,
        low: float = -1,
        high: float = 1,
        step_low: float = -0.01,
        step_high: float = 0.01,
        p: float = 0.2,
        save: bool = False):
    times = create_times(users, datetime.now(), days, per_day)
    loc = abs(high) - abs(low)
    scale = np.sqrt((high - low) / 2)
    dist = GaussianMixture([-3, 3], [0.1, 0.1], [0.5, 0.5])
    locs = create_locs(
        users,
        days=days,
        per_day=per_day,
        dist=dist,
        low=-5,
        high=5,
        step_low=step_low,
        step_high=step_high)
    values = create_values(users, per_user=days + 1, p=p)
    scores = to_scores(values, times)
    histories = to_histories(locs, times)
    if save:
        save_times(times)
        save_values(values)
        save_locations(locs)
        save_scores(scores)
        save_histories(histories)
    return Dataset(times, values, locs, scores, histories)


def save_contacts(contacts, n: int):
    save_data(CONTACTS_FILE_FORMAT.format(n), contacts)


def load_contacts(n: int):
    return load(CONTACTS_FILE_FORMAT.format(n))


def save_times(times):
    save_data(TIMES_FILENAME, times)


def load_times(n=None):
    return load(TIMES_FILENAME, n)


def save_values(values):
    save_data(VALUES_FILENAME, values)


def load_values(n=None):
    return load(VALUES_FILENAME, n)


def save_scores(scores):
    save_data(SCORES_FILENAME, scores)


def load_scores(n=None):
    return load(SCORES_FILENAME, n)


def save_locations(locations):
    save_data(LOCATIONS_FILENAME, locations)


def load_locations(n=None):
    return load(LOCATIONS_FILENAME, n)


def save_histories(histories):
    save_data(HISTORIES_FILENAME, histories)


def load_histories(n=None, geohashes: bool = True, prec: int = 8):
    histories = load(HISTORIES_FILENAME, n)
    if geohashes:
        histories = to_geohashes(histories, prec)
    return histories


def save_geohashes(geohashes):
    save_data(GEOHASHES_FILENAME, geohashes)


def load_geohashes(n=None):
    load(GEOHASHES_FILENAME, n)


def save_data(filename: str, arr: np.ndarray):
    warnings.filterwarnings("ignore")
    np.save(filename, arr)


def load(filename, n=None):
    return np.load(filename, allow_pickle=True)[:n]


if __name__ == '__main__':
    create_data(1000, low=-0.5, high=0.5, p=0.3, save=True)
