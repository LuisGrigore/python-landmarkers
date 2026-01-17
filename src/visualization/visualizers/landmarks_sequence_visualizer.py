from typing import Iterable, Iterator
import numpy as np

from src.visualization.layers.sequence_layer import HasSequence
from src.visualization.visualizers.landmarks_visualizer import LandmarksVisualizer
from ...landmarks.landmarks import LandmarksSequence
from ..layers.points_layer import HasPoints
from ..layers.centroid_layer import HasCentroid
from ..layers.bbox_layer import HasBBox

class LandmarksSequenceVisualizer(HasBBox, HasCentroid, HasPoints, HasSequence):
	def __init__(self, landmarks_sequence: LandmarksSequence) -> None:
		self._landmarks_sequence = landmarks_sequence
  
	def bbox(self) -> np.ndarray:
		return self._landmarks_sequence.bounding_box_2d()[-1]

	def centroid(self) -> np.ndarray:
		return self._landmarks_sequence.centroid()[-1]

	def points(self) -> np.ndarray:
		return self._landmarks_sequence.array[-1]

	def sequence(self) -> Iterator[LandmarksVisualizer]:
		for landmarks in self._landmarks_sequence.landmarks:
			yield LandmarksVisualizer(landmarks)
   
	def time_stamps_ms(self) -> Iterable[int]:
		return self._landmarks_sequence.time_stamps_ms