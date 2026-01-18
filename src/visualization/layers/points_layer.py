
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
		h, w = image.shape[:2]

		for point in data.points():
			x_norm, y_norm = point[:2]

			x_px = int(np.clip(x_norm * w, 0, w - 1))
			y_px = int(np.clip(y_norm * h, 0, h - 1))

			draw_circle(image, (x_px, y_px), self.radius, self.color)