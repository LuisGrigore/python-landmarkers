from typing import Generic, TypeVar
import numpy as np

from ...types import PixelCoordinate2D, RgbColor

from ..protocols import HasBBox

from ..draw_utils import draw_rectangle

T_BBox = TypeVar("T_BBox", bound=HasBBox)


class BBoxLayer(Generic[T_BBox]):
    def __init__(
        self,
        color = RgbColor(255, 0, 0),
        thickness: int = 2,
    ):
        self.color = color
        self.thickness = thickness

    def draw(self, data: T_BBox, image: np.ndarray) -> None:
        """
        Dibuja un bounding box sobre la imagen.
        Convierte coordenadas normalizadas (0..1) a píxeles.
        Se espera que `data` cumpla con el protocolo HasBBox:
        bbox() -> ((x_min, y_min), (x_max, y_max)), normalizado
        """
        top_left, bottom_right = data.bbox()
        height, width = image.shape[:2]

        # Convertir coordenadas normalizadas a píxeles
        pixel_top_left = PixelCoordinate2D(int(round(top_left.x * width)),int(round(top_left.y * height)))
        pixel_bottom_right =  PixelCoordinate2D(int(round(bottom_right.x * width)),int(round(bottom_right.y * height)))

        # Dibujar el rectángulo usando coordenadas de píxel
        draw_rectangle(
            image, pixel_top_left, pixel_bottom_right, self.color, self.thickness
        )
