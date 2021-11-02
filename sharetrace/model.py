from typing import List, Tuple, Union, overload

import numpy as np
import pygeohash

from sharetrace.util.types import DateTime, TimeDelta

Coordinate = Tuple[float, float]
Location = Union[str, Coordinate]
ArrayLike = Union[np.ndarray, List, Tuple]


def score(val: float, time: DateTime) -> np.ndarray:
    """Creates a timestamped probability.

    Args:
        val: A 32-bit float between 0 and 1, inclusive.
        time: A datetime datetime or numpy datetime64.

    Returns:
        A structured array with attributes 'val' and 'time'.
    """
    time = np.datetime64(time)
    dt = [('val', np.float32), ('time', time.dtype)]
    return np.array([(val, time)], dtype=dt)[0]


@overload
def temporal_loc(loc: Coordinate, time: DateTime) -> np.ndarray: ...


@overload
def temporal_loc(loc: str, time: DateTime) -> np.ndarray: ...


def temporal_loc(loc: Location, time: DateTime) -> np.ndarray:
    """Creates a temporal location.

        Args:
            loc: A string geohash or a (latitude, longitude) tuple.
            time: A datetime datetime or numpy datetime64.

        Returns:
            A structured array with attributes 'time' and 'loc'.
        """
    time = np.datetime64(time)
    if isinstance(loc, str):
        dt = [('loc', f'<U{len(loc)}'), ('time', time.dtype)]
    else:
        dt = [('loc', np.float32, (2,)), ('time', time.dtype)]
    return np.array([(loc, time)], dtype=dt)[0]


def to_geohash(coord: np.ndarray, prec: int = 12) -> np.ndarray:
    lat, long = coord['loc']
    geohash = pygeohash.encode(lat, long, prec)
    return temporal_loc(geohash, coord['time'])


def to_coord(geohash: np.ndarray) -> np.ndarray:
    lat, long = pygeohash.decode(geohash['loc'])
    return temporal_loc((lat, long), geohash['time'])


def event(time: DateTime, dur: TimeDelta) -> np.ndarray:
    """Creates a timestamped duration, where the timestamp indicates the start.

    Args:
        time: A datetime datetime or numpy datetime64.
        dur: A datetime timedelta or numpy timedelta64

    Returns:
        A structured array with attributes 'time' and 'dur'.
    """
    time, dur = np.datetime64(time), np.timedelta64(dur)
    dt = [('time', time.dtype), ('dur', dur.dtype)]
    return np.array([(time, dur)], dtype=dt)[0]


def contact(names: ArrayLike, events: ArrayLike) -> np.ndarray:
    """Creates a named set of events.

    Args:
        names: An iterable, typically of length 2.
        events: An iterable of event numpy structured arrays.

    Returns:
        A structured array with attributes 'names', 'time', and 'dur'.
    """
    names, events = np.array(names), np.array(events)
    most_recent = np.sort(events, order=('time', 'dur'), kind='stable')[-1]
    time, dur = most_recent['time'], most_recent['dur']
    dt = [
        ('names', names.dtype, names.shape),
        ('time', time.dtype),
        ('dur', dur.dtype)]
    return np.array([(names, time, dur)], dtype=dt)[0]


def history(locs: ArrayLike, name: int) -> np.ndarray:
    """Creates a named and sorted location history.

    Args:
        locs: An iterable of location numpy structured arrays.
        name: A 32-bit int that labels the history.

    Returns:
        A structured array with attributes 'name' and 'locs'.
    """
    locs = np.sort(locs, order=('time', 'loc'), kind='stable')
    dt = [('locs', locs.dtype, locs.shape), ('name', np.int32)]
    return np.array([(locs, name)], dtype=dt)[0]


def message(val: np.ndarray, src: int, dest: int, kind: int) -> np.ndarray:
    """Creates a message used for passing information between objects.

    Args:
        val: A numpy structured array.
        src: A 32-bit int that represents the source of the message.
        dest: A 32-bit int that represents the destination of the message.
        kind: An 8-bit int that represents the type of message.

    Returns:
       A structured array with attributes 'val', 'src', 'dest', and 'kind'.
    """
    dt = [
        ('val', val.dtype, val.shape),
        ('src', np.int32),
        ('dest', np.int32),
        ('kind', np.int8)]
    return np.array([(val, src, dest, kind)], dtype=dt)[0]
