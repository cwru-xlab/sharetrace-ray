from abc import ABC, abstractmethod
from collections import Sized
from typing import Any, Iterable, NoReturn, Optional


class BaseMailbox(ABC, Sized):
	__slots__ = ("max_size",)

	def __init__(self, max_size: Optional[int] = None):
		super().__init__()
		self.max_size = max_size

	@abstractmethod
	def put(
			self,
			*msgs: Any,
			block: bool = True,
			timeout: Optional[float] = None
	) -> NoReturn:
		pass

	@abstractmethod
	def get(
			self,
			n: int = 1,
			block: bool = True,
			timeout: Optional[float] = None
	) -> Iterable:
		pass

	@abstractmethod
	def empty(self) -> bool:
		pass

	@abstractmethod
	def full(self) -> bool:
		pass
