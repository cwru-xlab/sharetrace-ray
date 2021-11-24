import datetime
import json
import logging
from typing import NoReturn, Optional, Sequence, Tuple

import joblib
import numpy as np
import pyproj
from scipy import interpolate, spatial

from sharetrace import model, util


class ContactSearch:
    """Algorithm for finding contacts in location histories using a KD tree."""
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
            tol: float = 1,
            workers: int = 1,
            logger: Optional[logging.Logger] = None):
        """Configures contact search.

        Args:
            min_dur: Minimum duration (in seconds) for a contact.
            r: Radius used by the KD tree to find nearest neighbors.
            leaf_size: Number of points in a KD leaf before using brute force.
            p: Minkowski norm used by KD tree to measure distance.
            eps: If nonzero, used for approximate nearest-neighbor search.
            tol: Minimum distance between two locations to be "in contact."
            workers: Number of concurrent workers. -1 uses all processes.
            logger: Logger for logging contact search statistics.
        """
        self.min_dur = min_dur
        self.r = r
        self.leaf_size = leaf_size
        self.p = p
        self.eps = eps
        self.tol = tol
        self.workers = workers
        self.logger = logger

    def search(self, histories: Sequence[np.void]) -> np.ndarray:
        """Searches the location histories for contacts."""
        timer = util.time(lambda: self._search(histories))
        result = timer.result
        self._log(len(histories), len(result), timer.seconds)
        return result

    def _search(self, histories: Sequence[np.void]) -> np.ndarray:
        pairs, locs = self._selected(histories)
        print(f'Number of selected pairs: {len(pairs)}')
        # 'Auto' batch size results in slow performance.
        par = joblib.Parallel(self.workers, batch_size=500, verbose=2)
        find_contact = joblib.delayed(self._find_contact)
        # Memmapping the arguments does not result in a speedup.
        contacts = par(find_contact(p, histories, locs) for p in pairs)
        return np.array([c for c in contacts if c is not None])

    def _find_contact(
            self,
            pair: np.ndarray,
            histories: Sequence[np.void],
            locs: Sequence[np.ndarray]
    ) -> Optional[np.void]:
        u1, u2 = pair
        hist1, hist2 = histories[u1], histories[u2]
        times1, locs1 = _resample(hist1['locs']['time'], locs[u1])
        times2, locs2 = _resample(hist2['locs']['time'], locs[u2])
        times1, locs1, times2, locs2 = _pad(times1, locs1, times2, locs2)
        contact = None
        if len(close := self._proximal(locs1, locs2)) > 0:
            intervals = _intervals(close)
            options = np.flatnonzero(
                intervals[:, 1] - intervals[:, 0] >= self.min_dur / 60)
            if len(options) > 0:
                names = (hist1['name'], hist2['name'])
                start, end = intervals[options[-1]]
                duration = np.timedelta64(end - start, 'm')
                time = datetime.datetime.utcfromtimestamp(times1[start] * 60)
                contact = model.contact(names, time, duration)
        return contact

    def _selected(self, histories: Sequence[np.void]) -> Tuple:
        """Returns the unique user pairs and grouped Cartesian coordinates."""
        latlongs = [_latlongs(h) for h in histories]
        idx = _index(latlongs)
        xy = _project(np.concatenate(latlongs))
        pairs = self._query_pairs(xy)
        return _select(pairs, idx), _partition(xy, idx)

    def _query_pairs(self, points: np.ndarray) -> np.ndarray:
        """Returns all sufficiently-near neighbor pairs for each location."""
        kd_tree = spatial.KDTree(points, self.leaf_size)
        return kd_tree.query_pairs(self.r, self.p, self.eps, 'ndarray')

    def _proximal(self, locs1: np.ndarray, locs2: np.ndarray) -> np.ndarray:
        """Returns the (time) indices the locations that are close."""
        # Uses the default L2 norm.
        diff = np.linalg.norm(locs1.T - locs2.T, axis=1)
        return np.flatnonzero(diff <= self.tol)

    def _log(self, inputs: int, contacts: int, runtime: float) -> NoReturn:
        if self.logger is not None:
            util.info(self.logger, json.dumps({
                'RuntimeInSec': util.approx(runtime),
                'Workers': self.workers,
                'MinDurationInSec': self.min_dur,
                'InputSize': inputs,
                'Contacts': contacts,
                'Radius': self.r,
                'LeafSize': self.leaf_size,
                'MinkowskiNorm': self.p,
                'Epsilon': self.eps,
                'Tolerance': self.tol}))


def _latlongs(history: np.void) -> np.ndarray:
    """Maps a location history to lat-long coordinate pairs."""
    return model.to_coords(history)['locs']['loc']


def _index(locations: Sequence[np.ndarray]) -> np.ndarray:
    """Returns a mapping from (flattened) locations to users."""
    users = np.arange(len(locations))
    # Must flatten after indexing to know number of locations for each user.
    repeats = [len(locs) for locs in locations]
    return np.repeat(users, repeats)


def _project(latlongs: np.ndarray) -> np.ndarray:
    """Projects from latitude-longitude to x-y Cartesian coordinates."""
    lats, longs = latlongs.T
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(lla, ecef)
    projected = transformer.transform(longs, lats, radians=False)
    return np.column_stack(projected)


def _select(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Selects the unique users pairs that correspond to the queried points."""
    selected = np.unique(idx[points], axis=0)
    return selected[~(selected[:, 0] == selected[:, 1])]


def _partition(a: np.ndarray, idx: np.ndarray) -> Sequence[np.ndarray]:
    """Groups the values in an array by the index."""
    return [a[idx == u] for u in np.unique(idx)]


def _resample(times: np.ndarray, locs: np.ndarray) -> Tuple:
    """Resamples the times and locations to be at the minute-resolution."""
    # Second resolution results in really slow performance.
    times = np.int64(times.astype('datetime64[m]'))
    # Prefer interp1d over np.interp to use 'previous' interpolation.
    interp = interpolate.interp1d(
        times, locs.T, kind='previous', assume_sorted=True)
    new_times = np.arange(times[0], times[-1])
    new_locs = interp(new_times)
    return new_times, new_locs


def _intervals(a: np.ndarray) -> np.ndarray:
    """Returns an array of start-end contiguous interval pairs."""
    split_at = np.flatnonzero(np.diff(a) != 1)
    chunks = np.split(a, split_at + 1)
    return np.array([(ch[0], ch[-1] + 1) for ch in chunks], dtype=np.int64)


def _pad(
        times1: np.ndarray,
        locs1: np.ndarray,
        times2: np.ndarray,
        locs2: np.ndarray
) -> Tuple:
    """Pads the times and locations based on the union of the time ranges."""
    start = min(times1[0], times2[0])
    end = max(times1[-1], times2[-1])
    (new_times1, new_locs1) = _expand(times1, locs1, start, end)
    (new_times2, new_locs2) = _expand(times2, locs2, start, end)
    return new_times1, new_locs1, new_times2, new_locs2


def _expand(times: np.ndarray, locs: np.ndarray, start: int, end: int) -> Tuple:
    """Expands the times and locations to the new start/end, fills with inf. """
    prepend = np.arange(start, times[0])
    append = np.arange(times[-1] + 1, end + 1)
    if len(prepend) > 0 or len(append) > 0:
        new_times = np.concatenate((prepend, times, append))
        # Use inf as dummy value; used when finding small differences, so these
        # new values will never be selected. Assumes lat-long coordinates.
        prepend = np.ones((2, len(prepend))) * np.inf
        append = np.ones((2, len(append))) * np.inf
        new_locs = np.column_stack((prepend, locs, append))
    else:
        new_times, new_locs = times, locs
    return new_times, new_locs
