from abc import abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class HasBBox(Protocol):
    @abstractmethod
    def bbox(self) -> tuple[tuple[int,int], tuple[int,int]]:
        pass
