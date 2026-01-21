from typing import Iterable, Iterator

from ..layers.sequence_layer import HasSequence
from .landmarks_visualizer import LandmarksVisualizer
from ...landmarks.landmarks import LandmarksSequence


class LandmarksSequenceVisualizer(HasSequence):
	def __init__(self, landmarks_sequence: LandmarksSequence) -> None:
		self._landmarks_sequence = landmarks_sequence

	def sequence(self) -> Iterator[LandmarksVisualizer]:
		for landmarks in self._landmarks_sequence.landmarks:
			yield LandmarksVisualizer(landmarks)
   
	def time_stamps_ms(self) -> Iterable[int]:
		return self._landmarks_sequence.time_stamps_ms