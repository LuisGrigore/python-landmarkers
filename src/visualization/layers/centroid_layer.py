
from abc import abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from ..draw_utils import draw_circle

@runtime_checkable
class HasCentroid(Protocol):
	@abstractmethod
	def centroid(self) -> np.ndarray:
		pass


T_Centroid = TypeVar("T_Centroid", bound=HasCentroid)

class CentroidLayer(Generic[T_Centroid]):
	def __init__(
		self,
		color: tuple[int, int, int] = (0, 0, 255),
		radius: int = 5,
	):
		self.color = color
		self.radius = radius

	def draw(self, data: T_Centroid, image: np.ndarray) -> None:
		x, y = data.centroid().astype(int)
		draw_circle(image, (x, y), self.radius, self.color)
