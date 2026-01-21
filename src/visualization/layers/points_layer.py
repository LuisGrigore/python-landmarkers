from typing import Generic, TypeVar
import numpy as np

from ...types import PixelCoordinate2D

from ..protocols import HasPoints

from ..draw_utils import draw_circle


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

            px_coord = PixelCoordinate2D(
                int(np.clip(point.x * w, 0, w - 1)), int(np.clip(point.y * h, 0, h - 1))
            )
            draw_circle(image, px_coord, self.radius, self.color)
