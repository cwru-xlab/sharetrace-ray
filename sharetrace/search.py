import datetime
import json
import logging
from typing import NoReturn, Optional, Sequence, Tuple

import joblib
import numpy as np
import pyproj
from scipy import interpolate, spatial

from sharetrace import model, util

Locations = Histories = Sequence[np.void]
Contacts = np.ndarray


class ContactSearch:
    __slots__ = (
        'min_dur',
        'r',
        'leaf_size',
        'p',
        'eps',
        'tol',
        'workers',
        'logger')

    def __init__(
            self,
            min_dur: float = 0,
            r: float = 1e-4,
            leaf_size: int = 10,
            p: float = 2,
            eps: float = 1e-5,
            tol: float = 0.01,
            workers: int = 1,
            logger: Optional[logging.Logger] = None):
        self.min_dur = min_dur
        self.r = r
        self.leaf_size = leaf_size
        self.p = p
        self.eps = eps
        self.tol = tol
        self.workers = workers
        self.logger = logger

    def search(self, histories: Histories) -> Contacts:
        timer = util.time(lambda: self._search(histories))
        result = timer.result
        self._log_stats(len(histories), len(result), timer.seconds)
        return result

    def _search(self, histories: Histories) -> Contacts:
        pairs, locs = self._selected(histories)
        print(len(pairs))
        par = joblib.Parallel(self.workers, batch_size=500)
        find_contact = joblib.delayed(self._find_contact)
        contacts = par(find_contact(p, histories, locs) for p in pairs)
        return np.array([c for c in contacts if c is not None])

    def _find_contact(
            self,
            pair: np.ndarray,
            histories: Histories,
            locs: np.ndarray
    ) -> Optional[np.void]:
        u1, u2 = pair
        hist1, hist2 = histories[u1], histories[u2]
        times1, locs1 = _resample(hist1['locs']['time'], locs[u1])
        times2, locs2 = _resample(hist2['locs']['time'], locs[u2])
        times1, locs1, times2, locs2 = _pad(times1, locs1, times2, locs2)
        close = self._proximal(locs1, locs2)
        ints = _intervals(close)
        contact = None
        if len(ints) > 0:
            options = np.flatnonzero(
                ints[:, 1] - ints[:, 0] >= self.min_dur / 60)
            if len(options) > 0:
                names = (hist1['name'], hist2['name'])
                most_recent = ints[options[-1]]
                start, end = most_recent
                duration = np.timedelta64(end - start, 'm')
                time = datetime.datetime.utcfromtimestamp(times1[start])
                contact = model.contact(names, time, duration)
        return contact

    def _selected(self, histories: Histories) -> Tuple:
        locs = self._coordinates(histories)
        idx = _index(locs)
        pairs = self._query_pairs(np.concatenate(locs))
        return _select(pairs, idx), locs

    def _coordinates(self, hists: Histories) -> Histories:
        par = joblib.Parallel(self.workers)
        return par(joblib.delayed(_coords)(h) for h in hists)

    def _query_pairs(self, locs: np.ndarray) -> np.ndarray:
        locs = _project(locs)
        kd_tree = spatial.KDTree(locs, self.leaf_size)
        return kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')

    def _proximal(self, locs1: np.ndarray, locs2: np.ndarray) -> np.ndarray:
        diff = np.linalg.norm(locs1.T - locs2.T, axis=1)
        return np.flatnonzero(diff <= self.tol)

    def _log_stats(self, n: int, contacts: int, runtime: float) -> NoReturn:
        if self.logger is not None:
            util.info(self.logger, json.dumps({
                'RuntimeInSec': util.approx(runtime),
                'Workers': self.workers,
                'MinDurationInSec': self.min_dur,
                'InputSize': n,
                'Contacts': contacts,
                'Radius': self.r,
                'LeafSize': self.leaf_size,
                'MinkowskiNorm': self.p,
                'Epsilon': self.eps,
                'Tolerance': self.tol}))


def _coords(hist: np.void) -> np.ndarray:
    return model.to_coords(hist)['locs']['loc']


def _index(locations: Locations) -> np.ndarray:
    users = np.arange(len(locations))
    repeats = [len(locs) for locs in locations]
    return np.repeat(users, repeats)


def _project(coordinates: np.ndarray) -> np.ndarray:
    """Projects from latitude-longitude to x-y Cartesian coordinates."""
    lats, longs = coordinates.T
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(lla, ecef)
    projected = transformer.transform(longs, lats, radians=False)
    return np.column_stack(projected)


def _select(queried: np.ndarray, idx: np.ndarray) -> np.ndarray:
    selected = np.unique(idx[queried], axis=0)
    return selected[~(selected[:, 0] == selected[:, 1])]


def _resample(times: np.ndarray, locs: np.ndarray) -> Tuple:
    # Second resolution results in really slow performance.
    times = np.int64(times.astype('datetime64[m]'))
    # Prefer interp1d over np.interp to use 'previous' interpolation.
    interp = interpolate.interp1d(
        times, locs.T, kind='previous', assume_sorted=True)
    new_times = np.arange(times[0], times[-1])
    new_locs = interp(new_times)
    return new_times, new_locs


def _intervals(a: np.ndarray) -> np.ndarray:
    split_at = np.flatnonzero(np.diff(a) != 1)
    chunks = np.split(a, split_at + 1)
    has_splits = not (len(chunks) == 1 and len(chunks[0]) == 0)
    intervals = [(ch[0], ch[-1] + 1) for ch in chunks] if has_splits else []
    return np.array(intervals, dtype=np.int64)


def _pad(
        times1: np.ndarray,
        locs1: np.ndarray,
        times2: np.ndarray,
        locs2: np.ndarray
) -> Tuple:
    new_start = min(times1[0], times2[0])
    new_end = max(times1[-1], times2[-1])
    (new_times1, new_locs1) = _expand(times1, locs1, new_start, new_end)
    (new_times2, new_locs2) = _expand(times2, locs2, new_start, new_end)
    return new_times1, new_locs1, new_times2, new_locs2


def _expand(
        times: np.ndarray, locs: np.ndarray, new_start: int, new_end: int
) -> Tuple:
    prepend = np.arange(new_start, times[0])
    append = np.arange(times[-1] + 1, new_end + 1)
    new_times = np.concatenate((prepend, times, append))
    prepend = np.ones((2, len(prepend))) * np.inf
    append = np.ones((2, len(append))) * np.inf
    new_locs = np.column_stack((prepend, locs, append))
    return new_times, new_locs
