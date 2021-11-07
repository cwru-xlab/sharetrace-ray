from typing import Tuple, Union, overload

from nptyping import (Datetime64, Float32, Int32, Int8, NDArray, StructuredType,
                      Timedelta64)
from numpy import array, datetime64, float32, int32, int8, sort, timedelta64
from pygeohash import decode, encode

from sharetrace.util import DateTime, TimeDelta

RiskScore = NDArray[(), StructuredType[Float32, Datetime64]]
Coordinate = Tuple[Float32, Float32]
Coordinate_ = NDArray[(2,), Float32]
Location = Union[str, Coordinate]
TemporalLoc = NDArray[(), StructuredType[Union[str, Coordinate_], Datetime64]]
CoordTemporalLoc = NDArray[(), StructuredType[Coordinate_, Datetime64]]
GeohashTemporalLoc = NDArray[(), StructuredType[str, Datetime64]]
Event = NDArray[(), StructuredType[Datetime64, Timedelta64]]
Names = NDArray[(1,), str]
Contact = NDArray[(), StructuredType[Names, Event]]
History = NDArray[(), StructuredType[NDArray[(1,), TemporalLoc], Int32]]
Message = NDArray[(), StructuredType[NDArray, Int32, Int8, Int32, Int8, Int8]]
Neighbors = NDArray[(1,), Int32]
Node = NDArray[(), StructuredType[Neighbors, Int8, NDArray]]


def risk_score(val: Float32, time: DateTime) -> RiskScore:
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
def temporal_loc(loc: Coordinate, time: DateTime) -> CoordTemporalLoc: ...


@overload
def temporal_loc(loc: str, time: DateTime) -> GeohashTemporalLoc: ...


def temporal_loc(loc: Location, time: DateTime) -> TemporalLoc:
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


def to_geohash(coord: CoordTemporalLoc, prec: int = 12) -> GeohashTemporalLoc:
    lat, long = coord['loc']
    geohash = encode(lat, long, prec)
    return temporal_loc(geohash, coord['time'])


def to_coord(geohash: GeohashTemporalLoc) -> CoordTemporalLoc:
    lat, long = decode(geohash['loc'])
    return temporal_loc((lat, long), geohash['time'])


def event(time: DateTime, dur: TimeDelta) -> Event:
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


def contact(names: Names, events: NDArray[(1,), Event]) -> Contact:
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


def history(locs: NDArray[(1,), TemporalLoc], name: Int32) -> History:
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
        val: NDArray,
        src: Int32,
        sgroup: Int8,
        dest: Int32,
        dgroup: Int8,
        kind: Int8) -> Message:
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


def node(ne: Neighbors, group: Int8, data: NDArray) -> Node:
    ne = array(ne)
    data = array(data)
    dt = [
        ('ne', ne.dtype, ne.shape),
        ('group', int8),
        ('data', data.dtype, data.shape)]
    return array([(ne, group, data)], dtype=dt)[0]
