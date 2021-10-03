from multiprocessing import queues as queue
from typing import Any, Iterable, NoReturn, Optional

from messaging.base import BaseMailbox


class Mailbox(BaseMailbox):
	"""A mailbox for storing messages sent by other processes."""
	__slots__ = ()

	def __init__(self, max_size: Optional[int] = None):
		super().__init__(max_size)
		self.mailbox = queue.Queue(max_size or 0)

	def get(
			self,
			n: int = 1,
			block: bool = True,
			timeout: Optional[float] = None
	) -> Iterable:
		return tuple(self.mailbox.get(block, timeout) for _ in range(n))

	def put(
			self,
			*msgs: Any,
			block: bool = True,
			timeout: Optional[float] = None
	) -> NoReturn:
		for msg in msgs:
			self.mailbox.put(msg, block, timeout)

	def empty(self) -> bool:
		# Queue docs mention to use this over empty()
		return len(self) == 0

	def full(self) -> bool:
		# Queue docs mention to use this over full()
		return len(self) >= self.max_size

	def __len__(self) -> int:
		return self.mailbox.qsize()
