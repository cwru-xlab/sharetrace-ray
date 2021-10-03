import collections
from typing import Iterable, NoReturn, Optional

from messaging.base import BaseMailbox


class Inbox(BaseMailbox):
	"""A local mailbox for storing messages belonging to a single process."""
	__slots__ = ()

	def __init__(self, max_size: Optional[int] = None):
		super().__init__(max_size)
		self.inbox = collections.deque(maxlen=max_size)

	def put(
			self,
			msgs: Iterable,
			block: bool = True,
			timeout: Optional[float] = None
	) -> NoReturn:
		for msg in msgs:
			self.inbox.append(msg)

	def get(
			self,
			n: int = 1,
			block: bool = True,
			timeout: Optional[float] = None
	) -> Iterable:
		m = min(len(self.inbox), n)
		return tuple(self.inbox.popleft() for _ in range(m))

	def empty(self) -> bool:
		return len(self.inbox) == 0

	def full(self) -> bool:
		return len(self.inbox) == self.max_size

	def __len__(self) -> int:
		return len(self.inbox)
