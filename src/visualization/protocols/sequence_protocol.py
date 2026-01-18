from abc import abstractmethod
from typing import Iterable, Iterator, Protocol, TypeVar, runtime_checkable


T_SequenceElement = TypeVar("T_SequenceElement", covariant=True)


@runtime_checkable
class HasSequence(Protocol[T_SequenceElement]):
    @abstractmethod
    def sequence(self) -> Iterator[T_SequenceElement]:
        pass

    @abstractmethod
    def time_stamps_ms(self) -> Iterable[int]:
        pass
