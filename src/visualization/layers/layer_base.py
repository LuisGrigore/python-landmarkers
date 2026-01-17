from typing import Protocol, TypeVar, runtime_checkable
import numpy as np

T = TypeVar("T", contravariant=True)

@runtime_checkable
class Layer(Protocol[T]):
    def draw(self, data: T, image: np.ndarray) -> None:
        pass
