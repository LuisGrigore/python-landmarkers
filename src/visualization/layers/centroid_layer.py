from typing import Generic, TypeVar

import numpy as np

from ...types import PixelCoordinate2D

from ..protocols import HasCentroid

from ..draw_utils import draw_circle

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
        h, w = image.shape[:2]
        centroid = data.centroid()
        px_centroid = PixelCoordinate2D(
                int(np.clip(centroid.x * w, 0, w - 1)), int(np.clip(centroid.y * h, 0, h - 1))
            )
        x, y = centroid[:2]
        draw_circle(image, px_centroid, self.radius, self.color)
