from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class HasCentroid(Protocol):
    @abstractmethod
    def centroid(self) -> np.ndarray:
        pass
