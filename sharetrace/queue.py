import queue
from abc import ABC, abstractmethod
# noinspection PyUnresolvedReferences
# Empty and Full are used to allow for easy import swapping.
from queue import Empty, Full
from typing import Optional

import ray


class BaseQueue(ABC):
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
    def get(self, block: bool = True, timeout: Optional[float] = None):
        pass

    @abstractmethod
    def get_nowait(self):
        pass

    @abstractmethod
    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        pass

    @abstractmethod
    def put_nowait(self, item):
        pass

    @abstractmethod
    def qsize(self) -> int:
        pass


class Queue(BaseQueue):

    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        self._actor = _QueueActor.remote(maxsize)

    def empty(self) -> bool:
        return ray.get(self._actor.empty.remote())

    def full(self) -> bool:
        return ray.get(self._actor.full.remote())

    def get(self, block: bool = True, timeout: Optional[float] = None):
        return ray.get(self._actor.get.remote(block, timeout))

    def get_nowait(self):
        return ray.get(self._actor.get_nowait.remote())

    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        ray.get(self._actor.put.remote(item, block, timeout))

    def put_nowait(self, item):
        ray.get(self._actor.put_nowait.remote(item))

    def qsize(self) -> int:
        return ray.get(self._actor.qsize.remote())


@ray.remote
class _QueueActor(BaseQueue):

    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        self._actor = queue.Queue()

    def empty(self) -> bool:
        return self._actor.empty()

    def full(self) -> bool:
        return self._actor.full()

    def get(self, block: bool = True, timeout: Optional[float] = None):
        return self._actor.get(block, timeout)

    def get_nowait(self):
        return self._actor.get_nowait()

    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        self._actor.put(item, block, timeout)

    def put_nowait(self, item):
        self._actor.put_nowait(item)

    def qsize(self) -> int:
        return self._actor.qsize()
