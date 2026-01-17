
from abc import abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from ..draw_utils import draw_circle

@runtime_checkable
class HasPoints(Protocol):
    @abstractmethod
    def points(self) -> np.ndarray:
        pass
    
T_Points = TypeVar("T_Points", bound=HasPoints)


class PointsLayer(Generic[T_Points]):
    def __init__(
        self,
        color: tuple[int, int, int] = (0, 255, 0),
        radius: int = 3,
    ):
        self.color = color
        self.radius = radius

    def draw(self, data: T_Points, image: np.ndarray) -> None:
        for x, y in data.points().astype(int):
            draw_circle(image, (x, y), self.radius, self.color)