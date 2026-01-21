from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from ...types import NormalizedCoordinate2D

@runtime_checkable
class HasPoints(Protocol):
    @abstractmethod
    def points(self) -> list[NormalizedCoordinate2D]:
        pass
