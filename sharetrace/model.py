from typing import Sequence, Tuple, Union, overload

import numpy as np
import pygeohash

from sharetrace.util import DateTime

ArrayLike = Union[Sequence, Tuple, np.ndarray]
LatLong = Tuple[float, float]


def risk_score(val: float, time: int) -> np.void:
    """Creates a timestamped risk probability.

    Args:
        val: A 64-bit float between 0 and 1, inclusive.
        time: A 64-bit integer timestamp.

    Returns:
        A structured array with attributes 'val' and 'time'.
    """
    time = np.int64(time)
    dt = [('val', np.float64), ('time', time.dtype)]
    return np.array([(val, time)], dtype=dt)[0]


@overload
def temporal_loc(loc: LatLong, time: DateTime) -> np.void: ...


@overload
def temporal_loc(loc: str, time: DateTime) -> np.void: ...


def temporal_loc(loc: Union[str, LatLong], time: DateTime) -> np.void:
    """Creates a temporal location.

        Args:
            loc: A string geohash or a (latitude, longitude) tuple.
            time: A 64-bit integer timestamp.

        Returns:
            A structured array with attributes 'time' and 'loc'.
        """
    time = np.int64(time)
    if isinstance(loc, str):
        dt = [('loc', f'<U{len(loc)}'), ('time', time.dtype)]
    else:
        dt = [('loc', np.float64, (2,)), ('time', time.dtype)]
    return np.array([(loc, time)], dtype=dt)[0]


def to_geohash(coord: np.void, prec: int = 12) -> np.void:
    """Converts the coordinates of the temporal location into a geohash."""
    lat, long = coord['loc']
    geohash = pygeohash.encode(lat, long, prec)
    return temporal_loc(geohash, coord['time'])


def to_geohashes(*hists: np.void, prec: int = 12) -> Union[np.void, Sequence]:
    """Converts the coordinates of the location history into geohashes."""
    assert 0 < prec < 13
    converted = [
        history([to_geohash(loc, prec) for loc in hist['locs']], hist['name'])
        for hist in hists]
    if len(hists) == 1:
        converted = converted[0]
    return converted


def to_coords(hist: np.void) -> np.void:
    """Converts the geohashes of the location history into coordinates."""
    coords = [to_coord(geohash) for geohash in hist['locs']]
    return history(coords, hist['name'])


def to_coord(geohash: np.void) -> np.void:
    """Converts the geohash of the temporal location into coordinates."""
    lat, long = pygeohash.decode(geohash['loc'])
    return temporal_loc((lat, long), geohash['time'])


def contact(names: ArrayLike, time: int) -> np.void:
    """Creates a named event.

    Args:
        names: An array-like object, typically of length 2.
        time: A 64-bit integer timestamp.
    Returns:
        A structured array with attributes 'names' and 'time'.
    """
    names, time = np.asarray(names), np.int64(time)
    dt = [('names', names.dtype, names.shape), ('time', time.dtype)]
    return np.array([(names, time)], dtype=dt)[0]


def history(locs: ArrayLike, name: int) -> np.void:
    """Creates a named and sorted location history.

    Args:
        locs: An iterable of location numpy structured arrays.
        name: A 64-bit int that labels the history.

    Returns:
        A structured array with attributes 'name' and 'locs'.
    """
    locs = np.sort(locs, order=('time', 'loc'), kind='stable')
    dt = [('locs', locs.dtype, locs.shape), ('name', np.int64)]
    return np.array([(locs, name)], dtype=dt)[0]


def message(
        val: np.ndarray,
        src: int,
        sgroup: int,
        dest: int,
        dgroup: int) -> np.void:
    """Creates a message used for passing information between objects.

    Args:
        val: A numpy structured array.
        src: A 64-bit int that represents the source of the message.
        sgroup: An 8-bit int that represents the group of the source.
        dest: A 64-bit int that represents the destination of the message.
        dgroup: An 8-bit int that represents the group of the destination.

    Returns:
       An array with attributes 'val', 'src', 'sgroup', 'dest', and 'dgroup'.
    """
    dt = [
        ('val', val.dtype, val.shape),
        ('src', np.int64),
        ('sgroup', np.int8),
        ('dest', np.int64),
        ('dgroup', np.int8)]
    return np.array([(val, src, sgroup, dest, dgroup)], dtype=dt)[0]


def node(ne: ArrayLike, group: int) -> np.void:
    """Creates a graph node.

    Args:
        ne: An array-like object of neighbor identifiers.
        group: An 8-bit int that indicates the graph partition.

    Returns:
        A numpy structured array with attributes 'ne' and 'group'.
    """
    ne = np.array(ne)
    dt = [('ne', ne.dtype, ne.shape), ('group', np.int8)]
    return np.array([(ne, group)], dtype=dt)[0]
