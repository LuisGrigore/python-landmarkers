from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class HasBBox(Protocol):
    @abstractmethod
    def bbox(self) -> np.ndarray:
        pass
