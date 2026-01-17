import numpy as np
from typing import Generic, List, TypeVar

from .layers.layer_base import Layer

T = TypeVar("T")


class Viewer(Generic[T]):
    def __init__(
        self,
        layers: List[Layer[T]],
        image_size: tuple[int, int] | None = None,
    ):
        self._layers = layers
        self._image_size = image_size

    def render(
        self,
        data: T,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        if background is not None:
            image = background.copy()
        else:
            if self._image_size is None:
                raise ValueError("image_size required if no background is provided")
            h, w = self._image_size
            image = np.zeros((h, w, 3), dtype=np.uint8)

        for layer in self._layers:
            layer.draw(data, image)

        return image

class ViewerBuilder(Generic[T]):
    def __init__(self):
        self._layers: List[Layer[T]] = []
        self._image_size: tuple[int, int] | None = None

    def add(self, layer: Layer[T]) -> "ViewerBuilder[T]":
        self._layers.append(layer)
        return self

    def image_size(self, height: int, width: int) -> "ViewerBuilder[T]":
        self._image_size = (height, width)
        return self

    def build(self) -> Viewer[T]:
        return Viewer(self._layers, self._image_size)