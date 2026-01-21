
from ...types import NormalizedCoordinate2D
from ...landmarks.landmarks import Landmarks
from ..layers.points_layer import HasPoints
from ..layers.centroid_layer import HasCentroid
from ..layers.bbox_layer import HasBBox


class LandmarksVisualizer(HasBBox, HasCentroid, HasPoints):
	def __init__(self, landmarks: Landmarks) -> None:
		self._landmarks = landmarks
  
	def bbox(self) -> tuple[NormalizedCoordinate2D, NormalizedCoordinate2D]:
		bbox = self._landmarks.bounding_box_2d()
		return (NormalizedCoordinate2D(bbox[0][0],bbox[0][1]),NormalizedCoordinate2D(bbox[1][0],bbox[1][1]))

	def centroid(self) -> NormalizedCoordinate2D:
		centroid_3d = self._landmarks.centroid()
		return NormalizedCoordinate2D(centroid_3d.x, centroid_3d.y)

	def points(self) -> list[NormalizedCoordinate2D]:
		return [NormalizedCoordinate2D(landmark[0], landmark[1]) for landmark in  self._landmarks.array]