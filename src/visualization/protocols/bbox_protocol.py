from abc import abstractmethod
from typing import Protocol, runtime_checkable

from ...types import NormalizedCoordinate2D

@runtime_checkable
class HasBBox(Protocol):
    @abstractmethod
    def bbox(self) -> tuple[NormalizedCoordinate2D, NormalizedCoordinate2D]:
        pass
