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
        """
        Dibuja un bounding box sobre la imagen.
        Convierte coordenadas normalizadas (0..1) a píxeles.
        Se espera que `data` cumpla con el protocolo HasBBox:
        bbox() -> ((x_min, y_min), (x_max, y_max)), normalizado
        """
        (x_min_norm, y_min_norm), (x_max_norm, y_max_norm) = data.bbox()

        # Obtener tamaño de la imagen
        height, width = image.shape[:2]

        # Convertir coordenadas normalizadas a píxeles
        x_min = int(round(x_min_norm * width))
        y_min = int(round(y_min_norm * height))
        x_max = int(round(x_max_norm * width))
        y_max = int(round(y_max_norm * height))

        # Dibujar el rectángulo usando coordenadas de píxel
        draw_rectangle(
            image, (x_min, y_min), (x_max, y_max), self.color, self.thickness
        )
