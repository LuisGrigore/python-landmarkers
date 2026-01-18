from typing import Generic, TypeVar
import numpy as np

from ..protocols import HasBBox

from ..draw_utils import draw_rectangle

T_BBox = TypeVar("T_BBox", bound=HasBBox)


class BBoxLayer(Generic[T_BBox]):
    def __init__(
        self,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ):
        self.color = color
        self.thickness = thickness

    def draw(self, data: T_BBox, image: np.ndarray) -> None:
        bbox = data.bbox()
        draw_rectangle(image, bbox[0], bbox[1], self.color, self.thickness)
