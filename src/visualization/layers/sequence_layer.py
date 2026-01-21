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
			for i, element in enumerate(sequence):
				if i % self._step != 0:
					continue
				for layer in self._layers:
					layer.draw(element, image)
			return

		h, w, c = image.shape

		try:
			latest_ts = max(timestamps)
		except ValueError:
			return

		for i, (element, ts) in enumerate(zip(sequence, timestamps)):
			if i % self._step != 0:
				continue

			buffer = np.zeros((h, w, c), dtype=np.uint8)

			for layer in self._layers:
				layer.draw(element, buffer)

			delta_t = latest_ts - ts
			weight = 1.0 - (delta_t / self._time_scale_ms)
			weight = float(np.clip(weight, self._min_weight, 1.0))
			np.multiply(buffer, weight, out=buffer, casting="unsafe")

			mask = buffer.any(axis=-1)
			image[mask] = buffer[mask]

	@classmethod
	def builder(cls):
		return SequenceLayerBuilder[T_SequenceElement]()


class SequenceLayerBuilder(Generic[T_SequenceElement]):
	def __init__(self) -> None:
		super().__init__()
		self._step: int = 1
		self._time_fade: bool = False
		self._min_weight: float = 0.2
		self._time_scale_ms: float = 1000.0
		self._layers: list[Layer[T_SequenceElement]] = []
		
		
	def add_layer(self, layer: Layer[T_SequenceElement]) -> "SequenceLayerBuilder[T_SequenceElement]":
		self._layers.append(layer)
		return self

	def step(self, step: int) -> "SequenceLayerBuilder[T_SequenceElement]":
		self._step = step
		return self

	def time_fade(self, time_fade: bool) -> "SequenceLayerBuilder[T_SequenceElement]":
		self._time_fade = time_fade
		return self

	def min_weight(self, min_weight: float) -> "SequenceLayerBuilder[T_SequenceElement]":
		self._min_weight = min_weight
		return self

	def time_scale_ms(self, time_scale_ms: float) -> "SequenceLayerBuilder[T_SequenceElement]":
		self._time_scale_ms = time_scale_ms
		return self

	def build(self) -> SequenceLayer:
		return SequenceLayer(
			self._layers,
			step=self._step,
			time_fade=self._time_fade,
			min_weight=self._min_weight,
			time_scale_ms=self._time_scale_ms,
		)
