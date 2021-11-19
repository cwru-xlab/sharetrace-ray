from datetime import datetime

import numpy as np
from scipy import stats

from sharetrace import model
from sharetrace.util import DateTime

rng = np.random.default_rng()
SEC_PER_DAY = 86400
MIN_PER_HR = 3600
TIMES_FILENAME = 'data//times.npy'
VALUES_FILENAME = 'data//values.npy'
SCORES_FILENAME = 'data//scores.npy'
LOCATIONS_FILENAME = 'data//locations.npy'
HISTORIES_FILENAME = 'data//histories.npy'


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
        xinit, yinit = dist.rvs(size=2)
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


def create_values(locs, dist, z, days: int = 14, per_day: int = 16):
    idx = np.arange(num_points(days, per_day), step=per_day)
    selected = take(locs, idx, -1)
    return np.vstack([dist.pdf(x.T) / z for x in selected])


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
    return np.array([model.to_geohashes(h, prec) for h in histories])


def create_data(
        users: int = 10_000,
        days: int = 14,
        per_day: int = 16,
        low: float = -10,
        high: float = 10,
        step_low: float = -0.1,
        step_high: float = 0.1):
    times = create_times(users, datetime.now(), days, per_day)
    loc = abs(high) - abs(low)
    scale = np.sqrt((high - low) / 2)
    locs = create_locs(
        users,
        days=days,
        per_day=per_day,
        dist=stats.norm(loc, scale),
        low=low,
        high=high,
        step_low=step_low,
        step_high=step_high)
    mu = (loc, loc)
    normal = stats.multivariate_normal(mu, np.eye(2) * scale)
    values = create_values(locs, normal, normal.pdf(mu), days, per_day)
    save_times(times)
    save_values(values)
    save_locations(locs)
    save_scores(to_scores(values, times))
    save_histories(to_histories(locs, times))


def save_times(times):
    np.save(TIMES_FILENAME, times)


def load_times(n=None):
    return load(TIMES_FILENAME, n)


def save_values(values):
    np.save(VALUES_FILENAME, values)


def load_values(n=None):
    return load(VALUES_FILENAME, n)


def save_scores(scores):
    np.save(SCORES_FILENAME, scores)


def load_scores(n=None):
    return load(SCORES_FILENAME, n)


def save_locations(locations):
    np.save(LOCATIONS_FILENAME, locations)


def load_locations(n=None):
    return load(LOCATIONS_FILENAME, n)


def save_histories(histories):
    np.save(HISTORIES_FILENAME, histories)


def load_histories(n=None):
    return load(HISTORIES_FILENAME, n)


def load(filename, n=None):
    return np.load(filename, allow_pickle=True)[:n]
