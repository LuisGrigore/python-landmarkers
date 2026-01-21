from collections import deque
from collections.abc import Sequence, MutableSequence
from typing import Generic, Self, TypeVar, Optional, Callable, overload, Iterable

T = TypeVar('T')

class BaseSequence(Sequence[tuple[T, int]], Generic[T]):
	_elements: MutableSequence[T]
	_time_stamps_ms: MutableSequence[int]

	def __init__(self, fixed_buffer_length: Optional[int] = None):
		self._fixed_buffer_length = fixed_buffer_length

		if fixed_buffer_length is not None:
			self._elements = deque(maxlen=fixed_buffer_length)
			self._time_stamps_ms = deque(maxlen=fixed_buffer_length)
		else:
			self._elements = []
			self._time_stamps_ms = []

	@classmethod
	def from_list(
		cls,
		elements: list[T],
		time_stamps_ms: list[int],
		fixed_buffer_length: Optional[int] = None
	) -> Self:
		if len(elements) != len(time_stamps_ms):
			raise ValueError("Lengths must match")

		obj = cls(fixed_buffer_length)
		for e, ts in zip(elements, time_stamps_ms):
			obj.append(e, ts)
		return obj

	def append(self, element: T, time_stamp_ms: int) -> None:
		self._elements.append(element)
		self._time_stamps_ms.append(time_stamp_ms)

	def __len__(self) -> int:
		return len(self._elements)

	@overload
	def __getitem__(self, index: int) -> tuple[T, int]: ...

	@overload
	def __getitem__(self, index: slice) -> Self: ...

	def __getitem__(self, index: int | slice) -> tuple[T, int] | Self:
		if isinstance(index, slice):
			return self.from_list(
				list(self._elements)[index],
				list(self._time_stamps_ms)[index],
				self._fixed_buffer_length
			)
		return self._elements[index], self._time_stamps_ms[index]

	@property
	def time_stamps_ms(self) -> list[int]:
		return list(self._time_stamps_ms)

	def map(self, transform_func: Callable[[T], T]) -> Self:
		return self.from_list(
			list(transform_func(e) for e in self._elements),
			list(self._time_stamps_ms),
			self._fixed_buffer_length
		)
