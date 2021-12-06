import datetime
import json
import logging
from typing import NoReturn, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from scipy import interpolate
from sklearn import neighbors

from sharetrace import model, util

Array = np.ndarray
NDArrays = Sequence[Array]
Histories = Sequence[np.void]

# Source: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
_EARTH_RADIUS_METERS = 6378137


class ContactSearch:
    """Algorithm for finding contacts in trajectories using a ball tree."""
    __slots__ = (
        'min_dur',
        'r',
        'leaf_size',
        'tol',
        'workers',
        'verbose',
        'logger')

    def __init__(
            self,
            min_dur: float = 0,
            r: float = 1e-4,
            leaf_size: int = 10,
            tol: float = 1,
            workers: int = 1,
            verbose: int = 0,
            logger: Optional[logging.Logger] = None):
        """Configures contact search.

        Args:
            min_dur: Minimum duration (minutes) for a contact.
            r: Radius (meters) used by the ball tree to find nearest neighbors.
            leaf_size: Number of points in a leaf before using brute force.
            tol: Minimum distance (meters) b/t two locations to be "in contact."
            workers: Number of concurrent workers. -1 uses all processes.
            verbose: Level of verbosity used when printing joblib updates.
            logger: Logger for logging contact search statistics.
        """
        self.min_dur = min_dur
        self.r = r
        self.leaf_size = leaf_size
        self.tol = tol
        self.workers = workers
        self.verbose = verbose
        self.logger = logger

    def search(
            self,
            histories: Histories,
            return_pairs: bool = False
    ) -> Union[Array, Tuple]:
        """Searches the location histories for contacts."""
        timer = util.time(lambda: self._search(histories))
        contacts, pairs = timer.result
        self.log(len(histories), len(contacts), timer.seconds)
        return (contacts, pairs) if return_pairs else contacts

    def _search(self, histories: Histories) -> NDArrays:
        pairs, locs = self.select(histories)
        # 'Auto' batch size results in slow performance.
        par = joblib.Parallel(
            self.workers, batch_size=500, verbose=self.verbose)
        find_contact = joblib.delayed(self.find_contact)
        # Memmapping the arguments does not result in a speedup.
        contacts = par(find_contact(p, histories, locs) for p in pairs)
        return np.array([c for c in contacts if c is not None]), pairs

    def select(self, histories: Histories) -> Tuple:
        """Returns the unique user pairs and grouped Cartesian coordinates."""
        points, pidx = flatten([to_latlongs(h) for h in histories])
        tree = neighbors.BallTree(points, self.leaf_size, metric='haversine')
        queried, qidx = flatten(
            tree.query_radius(points, self.r / _EARTH_RADIUS_METERS))
        # Sorting along the last axis ensures duplicate pairs are removed.
        pairs = np.sort(np.column_stack((qidx, queried)))
        pairs = np.unique(pidx[pairs], axis=0)
        # Only include pairs that correspond to two distinct users.
        pairs = pairs[~(pairs[:, 0] == pairs[:, 1])]
        # Use the user point index for both selecting and grouping.
        locs = [points[pidx == u] for u in range(len(histories))]
        return pairs, locs

    def find_contact(
            self,
            pair: Array,
            histories: Histories,
            locs: NDArrays
    ) -> Optional[np.void]:
        u1, u2 = pair
        hist1, hist2 = histories[u1], histories[u2]
        times1, locs1 = resample(hist1['locs']['time'], locs[u1])
        times2, locs2 = resample(hist2['locs']['time'], locs[u2])
        times1, locs1, times2, locs2 = pad(times1, locs1, times2, locs2)
        contact = None
        if len(close := self.proximal(locs1, locs2)) > 0:
            ints = get_intervals(close)
            durations = ints[:, 1] - ints[:, 0]
            if len(options := np.flatnonzero(durations >= self.min_dur)) > 0:
                names = (hist1['name'], hist2['name'])
                start, end = ints[options[-1]]
                duration = np.timedelta64(end - start, 'm')
                time = datetime.datetime.utcfromtimestamp(times1[start] * 60)
                contact = model.contact(names, time, duration)
        return contact

    def proximal(self, locs1: Array, locs2: Array) -> Array:
        """Returns the (time) indices the locations that are close."""
        # Uses the default L2 norm.
        diff = np.linalg.norm(locs1.T - locs2.T, axis=1)
        return np.flatnonzero(diff <= self.tol / _EARTH_RADIUS_METERS)

    def log(self, inputs: int, contacts: int, runtime: float) -> NoReturn:
        if self.logger is not None:
            self.logger.info(json.dumps({
                'RuntimeInSeconds': util.approx(runtime),
                'Workers': self.workers,
                'MinDurationInSeconds': self.min_dur,
                'Inputs': inputs,
                'Contacts': contacts,
                'LeafSize': self.leaf_size,
                'RadiusInMeters': self.r,
                'ToleranceInMeters': self.tol}))


def to_latlongs(history: np.void) -> Array:
    """Maps a location history to radian lat-long coordinate pairs."""
    return np.radians(model.to_coords(history)['locs']['loc'])


def flatten(arrays: NDArrays) -> NDArrays:
    """Return a flat concatenation and an index to map back to seq indices. """
    idx = np.repeat(np.arange(len(arrays)), repeats=[len(a) for a in arrays])
    return np.concatenate(arrays), idx


def get_intervals(a: Array) -> Array:
    """Returns an array of start-end contiguous interval pairs."""
    split_at = np.flatnonzero(np.diff(a) != 1)
    chunks = np.split(a, split_at + 1)
    return np.array([(c[0], c[-1] + 1) for c in chunks], dtype=np.int64)


def resample(times: Array, locs: Array) -> NDArrays:
    """Resamples the times and locations to be at the minute-resolution."""
    # Second resolution results in really slow performance.
    times = np.int64(times.astype('datetime64[m]'))
    # Prefer interp1d over np.interp to use 'previous' interpolation.
    # Use the transpose of locs so its shape is (2, n_samples), where each
    # row is latitude and longitude.
    interp = interpolate.interp1d(
        times, locs.T, kind='previous', assume_sorted=True)
    new_times = np.arange(times[0], times[-1])
    new_locs = interp(new_times)
    return new_times, new_locs


def pad(times1: Array, locs1: Array, times2: Array, locs2: Array) -> NDArrays:
    """Pads the times and locations based on the union of the time ranges."""
    start = min(times1[0], times2[0])
    end = max(times1[-1], times2[-1])
    new_times1, new_locs1 = expand(times1, locs1, start, end)
    new_times2, new_locs2 = expand(times2, locs2, start, end)
    return new_times1, new_locs1, new_times2, new_locs2


def expand(times: Array, locs: Array, start: int, end: int) -> Tuple:
    """Expands the times and locations to the new start/end, fills with inf. """
    prepend = np.arange(start, times[0])
    append = np.arange(times[-1] + 1, end + 1)
    psize, asize = prepend.size, append.size
    if psize > 0 or asize > 0:
        new_times = np.concatenate((prepend, times, append))
        # Use inf as dummy value; used when finding small differences, so these
        # new values will never be selected. Assumes lat-long coordinates.
        prepend = np.full((2, psize), np.inf)
        append = np.full((2, asize), np.inf)
        new_locs = np.hstack((prepend, locs, append))
    else:
        new_times, new_locs = times, locs
    return new_times, new_locs
