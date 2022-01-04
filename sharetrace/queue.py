import queue
from abc import ABC, abstractmethod
# noinspection PyUnresolvedReferences
# Empty and Full are used to allow for easy import swapping.
from queue import Empty, Full
from typing import Any, Optional

import ray


class BaseQueue(ABC):
    """A data structure with optimized FIFO access.

    Attributes:
        maxsize: An integer that indicates the maximum number of elements
            allowed in the queue. If 0 (default), then the queue is only
            bounded by the amount of system memory.
    """

    __slots__ = ('maxsize',)

    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize

    @abstractmethod
    def empty(self) -> bool:
        pass

    @abstractmethod
    def full(self) -> bool:
        pass

    @abstractmethod
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        pass

    @abstractmethod
    def get_nowait(self) -> Any:
        pass

    @abstractmethod
    def put(
            self,
            item,
            block: bool = True,
            timeout: Optional[float] = None
    ) -> None:
        pass

    @abstractmethod
    def put_nowait(self, item) -> None:
        pass

    @abstractmethod
    def qsize(self) -> int:
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}(maxsize={self.maxsize})'


class Queue(BaseQueue):
    """A queue actor that is compatible with Ray local mode."""

    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        self._actor = _QueueActor.remote(maxsize)

    def empty(self) -> bool:
        return ray.get(self._actor.empty.remote())

    def full(self) -> bool:
        return ray.get(self._actor.full.remote())

    def get(self, block: bool = True, timeout: Optional[float] = None):
        return ray.get(self._actor.get.remote(block, timeout))

    def get_nowait(self) -> Any:
        return ray.get(self._actor.get_nowait.remote())

    def put(
            self,
            item,
            block: bool = True,
            timeout: Optional[float] = None
    ) -> None:
        return ray.get(self._actor.put.remote(item, block, timeout))

    def put_nowait(self, item) -> None:
        return ray.get(self._actor.put_nowait.remote(item))

    def qsize(self) -> int:
        return ray.get(self._actor.qsize.remote())


@ray.remote(max_retries=3, max_restarts=3)
class _QueueActor(BaseQueue):

    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        self._actor = queue.Queue(maxsize)

    def empty(self) -> bool:
        return self._actor.empty()

    def full(self) -> bool:
        return self._actor.full()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        return self._actor.get(block, timeout)

    def get_nowait(self) -> Any:
        return self._actor.get_nowait()

    def put(
            self,
            item,
            block: bool = True,
            timeout: Optional[float] = None
    ) -> None:
        self._actor.put(item, block, timeout)

    def put_nowait(self, item) -> None:
        self._actor.put_nowait(item)

    def qsize(self) -> int:
        return self._actor.qsize()
