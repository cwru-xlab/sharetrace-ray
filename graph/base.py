from abc import ABC, abstractmethod
from typing import Any, Hashable, Mapping, NoReturn, Sequence

from messaging.base import BaseMailbox


class BasePartition(ABC):
	"""A subgraph that processes messages involving its nodes.

	Attributes:
		mailbox: Where new messaging from other partitions arrive.
		inbox: Where new messaging from nodes within the partition arrive.
		nodes: The nodes belonging to this partition of the graph.
	"""

	__slots__ = ("mailbox", "inbox", "nodes")

	def __init__(
			self,
			mailbox: BaseMailbox,
			inbox: BaseMailbox,
			nodes: Mapping[Hashable, Any]
	):
		super().__init__()
		self.mailbox = mailbox
		self.inbox = inbox
		self.nodes = nodes

	@abstractmethod
	def call(self, *args, **kwargs) -> Any:
		pass

	@abstractmethod
	def on_next(self, msg: Any) -> NoReturn:
		pass

	@abstractmethod
	def on_complete(self) -> NoReturn:
		pass

	@abstractmethod
	def on_error(self, error: BaseException) -> NoReturn:
		pass


class BaseGraph(ABC):
	"""A data structure containing nodes and connecting edges."""

	__slots__ = ()

	def __init__(self):
		super().__init__()

	@classmethod
	@abstractmethod
	def load(cls) -> "BaseGraph":
		pass

	@abstractmethod
	def save(self, dest: str) -> NoReturn:
		pass

	@abstractmethod
	def partition(self, n: int) -> Sequence[BasePartition]:
		pass
