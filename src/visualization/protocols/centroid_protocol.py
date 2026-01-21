from abc import abstractmethod
from typing import Protocol, runtime_checkable

from ...types import NormalizedCoordinate2D


@runtime_checkable
class HasCentroid(Protocol):
    @abstractmethod
    def centroid(self) -> NormalizedCoordinate2D:
        pass
