from typing import List, Tuple, Union, overload

from numpy import (array, datetime64, float32, int32, int8, ndarray, sort,
                   timedelta64)
from pygeohash import decode, encode

from sharetrace.util.types import DateTime, TimeDelta

Coordinate = Tuple[float, float]
Location = Union[str, Coordinate]
ArrayLike = Union[ndarray, List, Tuple]


def risk_score(val: float, time: DateTime) -> ndarray:
    """Creates a timestamped risk probability.

    Args:
        val: A 32-bit float between 0 and 1, inclusive.
        time: A datetime datetime or numpy datetime64.

    Returns:
        A structured array with attributes 'val' and 'time'.
    """
    time = datetime64(time)
    dt = [('val', float32), ('time', time.dtype)]
    return array([(val, time)], dtype=dt)[0]


@overload
def temporal_loc(loc: Coordinate, time: DateTime) -> ndarray: ...


@overload
def temporal_loc(loc: str, time: DateTime) -> ndarray: ...


def temporal_loc(loc: Location, time: DateTime) -> ndarray:
    """Creates a temporal location.

        Args:
            loc: A string geohash or a (latitude, longitude) tuple.
            time: A datetime datetime or numpy datetime64.

        Returns:
            A structured array with attributes 'time' and 'loc'.
        """
    time = datetime64(time)
    if isinstance(loc, str):
        dt = [('loc', f'<U{len(loc)}'), ('time', time.dtype)]
    else:
        dt = [('loc', float32, (2,)), ('time', time.dtype)]
    return array([(loc, time)], dtype=dt)[0]


def to_geohash(coord: ndarray, prec: int = 12) -> ndarray:
    lat, long = coord['loc']
    geohash = encode(lat, long, prec)
    return temporal_loc(geohash, coord['time'])


def to_coord(geohash: ndarray) -> ndarray:
    lat, long = decode(geohash['loc'])
    return temporal_loc((lat, long), geohash['time'])


def event(time: DateTime, dur: TimeDelta) -> ndarray:
    """Creates a timestamped duration, where the timestamp indicates the start.

    Args:
        time: A datetime datetime or numpy datetime64.
        dur: A datetime timedelta or numpy timedelta64

    Returns:
        A structured array with attributes 'time' and 'delta'.
    """
    time, dur = datetime64(time), timedelta64(dur)
    dt = [('time', time.dtype), ('delta', dur.dtype)]
    return array([(time, dur)], dtype=dt)[0]


def contact(names: ArrayLike, events: ArrayLike) -> ndarray:
    """Creates a named set of events.

    Args:
        names: An iterable, typically of length 2.
        events: An iterable of event numpy structured arrays.

    Returns:
        A structured array with attributes 'names', 'time', and 'delta'.
    """
    names, events = array(names), array(events)
    most_recent = sort(events, order=('time', 'delta'), kind='stable')[-1]
    time, dur = most_recent['time'], most_recent['delta']
    dt = [
        ('names', names.dtype, names.shape),
        ('time', time.dtype),
        ('delta', dur.dtype)]
    return array([(names, time, dur)], dtype=dt)[0]


def history(locs: ArrayLike, name: int) -> ndarray:
    """Creates a named and sorted location history.

    Args:
        locs: An iterable of location numpy structured arrays.
        name: A 32-bit int that labels the history.

    Returns:
        A structured array with attributes 'name' and 'locs'.
    """
    locs = sort(locs, order=('time', 'loc'), kind='stable')
    dt = [('locs', locs.dtype, locs.shape), ('name', int32)]
    return array([(locs, name)], dtype=dt)[0]


def message(
        val: ndarray,
        src: int,
        sgroup: int,
        dest: int,
        dgroup: int,
        kind: int) -> ndarray:
    """Creates a message used for passing information between objects.

    Args:
        val: A numpy structured array.
        src: A 32-bit int that represents the source of the message.
        sgroup: An 8-bit int that represents the group of the source.
        dest: A 32-bit int that represents the destination of the message.
        dgroup: An 8-bit int that represents the group of the destination.
        kind: An 8-bit int that represents the type of message.

    Returns:
       An array with attributes val, src, sgroup, dest, dgroup, and kind.
    """
    dt = [
        ('val', val.dtype, val.shape),
        ('src', int32),
        ('sgroup', int8),
        ('dest', int32),
        ('dgroup', int8),
        ('kind', int8)]
    return array([(val, src, sgroup, dest, dgroup, kind)], dtype=dt)[0]
