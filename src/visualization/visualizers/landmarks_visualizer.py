import numpy as np
from ...landmarks.landmarks import Landmarks
from ..layers.points_layer import HasPoints
from ..layers.centroid_layer import HasCentroid
from ..layers.bbox_layer import HasBBox


class LandmarksVisualizer(HasBBox, HasCentroid, HasPoints):
	def __init__(self, landmarks: Landmarks) -> None:
		self._landmarks = landmarks
  
	def bbox(self) -> np.ndarray:
		return self._landmarks.bounding_box_2d()

	def centroid(self) -> np.ndarray:
		return self._landmarks.centroid()

	def points(self) -> np.ndarray:
		return self._landmarks.array