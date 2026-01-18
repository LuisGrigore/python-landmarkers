from typing import Generic, List, TypeVar

import numpy as np

from ..protocols import HasSequence

from ..layers.layer_base import Layer

T_Sequence = TypeVar("T_Sequence", bound=HasSequence)
T_SequenceElement = TypeVar("T_SequenceElement", contravariant=True)


class SequenceLayer(Generic[T_Sequence, T_SequenceElement]):
    def __init__(
        self,
        layers: List[Layer[T_SequenceElement]],
        *,
        step: int = 1,
        time_fade: bool = False,
        min_weight: float = 0.2,
        time_scale_ms: float = 1000.0,
    ):
        self._layers = layers
        self._step = step
        self._time_fade = time_fade
        self._min_weight = min_weight
        self._time_scale_ms = time_scale_ms

    def draw(self, data: T_Sequence, image: np.ndarray) -> None:
        sequence = data.sequence()
        timestamps = data.time_stamps_ms()

        if not self._time_fade:
            # 🔥 FAST PATH: sin buffers, sin overlays
            for i, element in enumerate(sequence):
                if i % self._step != 0:
                    continue
                for layer in self._layers:
                    layer.draw(element, image)
            return

        # 🕒 PATH CON TIME FADE
        h, w, c = image.shape

        # Necesitamos conocer el timestamp más reciente
        try:
            latest_ts = max(timestamps)
        except ValueError:
            return

        for i, (element, ts) in enumerate(zip(sequence, timestamps)):
            if i % self._step != 0:
                continue

            # 1. buffer negro
            buffer = np.zeros((h, w, c), dtype=np.uint8)

            # 2. dibujar capas en el buffer
            for layer in self._layers:
                layer.draw(element, buffer)

            # 3. aplicar fade temporal (monótono)
            delta_t = latest_ts - ts
            weight = 1.0 - (delta_t / self._time_scale_ms)
            weight = float(np.clip(weight, self._min_weight, 1.0))
            np.multiply(buffer, weight, out=buffer, casting="unsafe")

            # 4. overlay sobre imagen final
            mask = buffer.any(axis=-1)
            image[mask] = buffer[mask]
