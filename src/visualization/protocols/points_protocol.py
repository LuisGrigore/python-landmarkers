from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class HasPoints(Protocol):
    @abstractmethod
    def points(self) -> np.ndarray:
        pass
