from abc import ABC, abstractmethod
from typing import Any, Hashable, Mapping, NoReturn, Sequence

from messaging.base import BaseMailbox


class BasePartition(ABC):
	"""A subgraph that processes messages involving its nodes.

	Attributes:
		inbox: Where new messages from nodes within the partition arrive.
			This allows for efficient in-process communication.
		address: An identifier for the partition that corresponds to its
			mailbox. This should be included in messages sent to other
			mailboxes if a response is required.
		mailboxes: Where new messages from partitions arrive. Each mailbox
			has a corresponding address to which messages can be sent. This
			allows a partition to communicate with other partitions by sending
			messages to their mailbox.
		nodes: The nodes belonging to this partition of the graph. Each node
			has an identifier and zero or more attributes.
	"""

	__slots__ = ("inbox", "address", "mailboxes", "nodes")

	def __init__(
			self,
			inbox: BaseMailbox,
			address: Hashable,
			mailboxes: Mapping[Hashable, BaseMailbox],
			nodes: Mapping[Hashable, Any]
	):
		super().__init__()
		self.inbox = inbox
		self.address = address
		self.mailboxes = mailboxes
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
	def load(cls, src: str) -> "BaseGraph":
		pass

	@abstractmethod
	def save(self, dest: str) -> NoReturn:
		pass

	@abstractmethod
	def partition(self, n: int) -> Sequence[BasePartition]:
		pass
