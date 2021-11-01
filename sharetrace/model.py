from typing import Iterable, Tuple, overload

import numpy as np

from sharetrace.util.types import DateTime, TimeDelta

Coordinate = Tuple[float, float]


def score(value: float, timestamp: DateTime) -> np.ndarray:
    """Creates a timestamped probability of a certain event.

    Args:
        value: A 32-bit float between 0 and 1, inclusive.
        timestamp: A datetime datetime or numpy datetime64.

    Returns:
        A structured array with attributes 'value' and 'timestamp'.
    """
    timestamp = np.datetime64(timestamp)
    dt = [('value', np.float32), ('timestamp', timestamp.dtype)]
    return np.array([(value, timestamp)], dtype=dt)[0]


@overload
def location(coordinate: Coordinate, timestamp: DateTime) -> np.ndarray:
    """Creates a temporal coordinate-based location.

    Args:
        coordinate: A (latitude, longitude) tuple.
        timestamp: A datetime datetime or numpy datetime64.

    Returns:
        A structured array with attributes 'lat', 'long', and 'timestamp'.
    """
    """Returns a temporal location of a given time and location."""
    lat, long = coordinate
    timestamp = np.datetime64(timestamp)
    dt = [
        ('lat', np.float32),
        ('long', np.float32),
        ('timestamp', timestamp.dtype)]
    return np.array([(lat, long, timestamp)], dtype=dt)[0]


def location(geohash: str, timestamp: DateTime) -> np.ndarray:
    """Creates a temporal geohash-based location.

    Args:
        geohash: A string geohash. Precision is based on its length.
        timestamp: A datetime datetime or numpy datetime64.

    Returns:
        A structured array with attributes 'geohash' and 'timestamp'.
    """
    timestamp = np.datetime64(timestamp)
    dt = [('geohash', f'<U{len(geohash)}'), ('timestamp', timestamp.dtype)]
    return np.array([(geohash, timestamp)], dtype=dt)[0]


def occurrence(timestamp: DateTime, duration: TimeDelta) -> np.ndarray:
    """Creates a timestamped duration, where the timestamp indicates the start.

    Args:
        timestamp: A datetime datetime or numpy datetime64.
        duration: A datetime timedelta or numpy timedelta64

    Returns:
        A structured array with attributes 'timestamp' and 'duration'.
    """
    timestamp, duration = np.datetime64(timestamp), np.timedelta64(duration)
    dt = [('timestamp', timestamp.dtype), ('duration', duration.dtype)]
    return np.array([(timestamp, duration)], dtype=dt)[0]


def contact(names: Iterable, occurrences: Iterable) -> np.ndarray:
    """Creates a labeled set of occurrences.

    Args:
        names: An iterable of ints, typically of length 2.
        occurrences: An iterable of occurrence numpy structured arrays.

    Returns:
        A structured array with attributes 'names', 'timestamp', and 'duration'.
    """
    names, occurrences = np.array(names), np.array(occurrences)
    most_recent = np.sort(
        occurrences, order=('timestamp', 'duration'), kind='stable')[-1]
    timestamp, duration = most_recent['timestamp'], most_recent['duration']
    dt = [
        ('names', names.dtype, names.shape),
        ('timestamp', timestamp.dtype),
        ('duration', duration.dtype)]
    return np.array([(names, timestamp, duration)], dtype=dt)[0]


def history(locations: Iterable, name: int) -> np.ndarray:
    """Creates a labeled location history.

    Args:
        locations: An iterable of location numpy structured arrays.
        name: A 32-bit int that labels the history.

    Returns:
        A structured array with attributes 'name' and 'locations'.
    """
    """Returns a sorted location history with a name."""
    locations = np.sort(
        locations, order=('timestamp', 'location'), kind='stable')
    dt = [('locations', locations.dtype, locations.shape), ('name', np.int32)]
    return np.array([(locations, name)], dtype=dt)[0]


def message(value: np.ndarray, src: int, dest: int, kind: int) -> np.ndarray:
    """Creates a message used for passing information between objects.

    Args:
        value: A numpy structured array.
        src: A 32-bit int that represents the source of the message.
        dest: A 32-bit int that represents the destination of the message.
        kind: An 8-bit int that represents the type of message.

    Returns:
       A structured array with attributes 'value', 'src', 'dest', and 'kind'.
    """
    """Returns a message of a given source, destination, kind, and value."""
    dt = [
        ('value', value.dtype, value.shape),
        ('src', np.int32),
        ('dest', np.int32),
        ('kind', np.int8)]
    return np.array([(value, src, dest, kind)], dtype=dt)[0]
