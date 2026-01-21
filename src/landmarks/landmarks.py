from copy import deepcopy
from typing import Optional, Self, TypeAlias, Union
import numpy as np

from ..base_sequence import BaseSequence

from ..types import LandmarkArray, LandmarkSequenceArray, NormalizedCoordinate3D


Reference: TypeAlias = Union[int, NormalizedCoordinate3D]


class Landmarks:
	def __init__(self, landmarks: LandmarkArray) -> None:
		self._landmarks: LandmarkArray = landmarks

	@property
	def array(self) -> LandmarkArray:
		return self._landmarks.copy()

	@property
	def n_points(self) -> int:
		return self._landmarks.shape[0]

	def _resolve_reference(
		self, reference: Optional[Reference]
	) -> NormalizedCoordinate3D:
		if isinstance(reference, int):
			if not (0 <= reference < self.n_points):
				raise ValueError(f"Reference index {reference} out of bounds")
			return self._landmarks[reference]
		elif isinstance(reference, NormalizedCoordinate3D):
			return reference
		else:
			return self.centroid()

	def centered(self, reference: Optional[Reference] = None) -> "Landmarks":
		ref_point = self._resolve_reference(reference)
		return Landmarks(self._landmarks - ref_point)

	def normalized(self, reference: Optional[Reference] = None) -> "Landmarks":
		ref_point = self._resolve_reference(reference)
		diff = self._landmarks - ref_point
		scale = np.linalg.norm(diff, axis=1).max()
		return Landmarks(diff / scale)

	def centroid(self) -> NormalizedCoordinate3D:
		return self._landmarks.mean(axis=0)

	def distance(self, a: int, b: int) -> float:
		return float(np.linalg.norm(self._landmarks[a] - self._landmarks[b]))

	def bounding_box(self) -> np.ndarray:
		return np.stack(
			[self._landmarks.min(axis=0), self._landmarks.max(axis=0)], axis=0
		)

	def bounding_box_2d(self) -> np.ndarray:
		xy = self._landmarks[:, :2]
		return np.stack([xy.min(axis=0), xy.max(axis=0)], axis=0)

	def extent(self) -> np.ndarray:
		mn, mx = self.bounding_box()
		return mx - mn

	def distance_to(self, point: np.ndarray) -> np.ndarray:
		return np.linalg.norm(self._landmarks - point, axis=1)

	def pairwise_distances(self) -> np.ndarray:
		diff = self._landmarks[:, None, :] - self._landmarks[None, :, :]
		return np.linalg.norm(diff, axis=-1)

	def angle(self, a: int, b: int, c: int) -> float:
		ba = self._landmarks[a] - self._landmarks[b]
		bc = self._landmarks[c] - self._landmarks[b]

		cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
		return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

	def scaled(self, factor: float) -> "Landmarks":
		return Landmarks(self._landmarks * factor)

	def rotated_2d(self, angle_rad: float) -> "Landmarks":
		R = np.array(
			[
				[np.cos(angle_rad), -np.sin(angle_rad)],
				[np.sin(angle_rad), np.cos(angle_rad)],
			]
		)
		return Landmarks(self._landmarks[:, :2] @ R.T)

	def variance(self) -> np.ndarray:
		return self._landmarks.var(axis=0)

	def subset(self, indices: list[int]) -> "Landmarks":
		return Landmarks(self._landmarks[indices])


class LandmarksSequence(BaseSequence[Landmarks]):
    def __init__(self, fixed_buffer_length: Optional[int] = None):
        super().__init__(fixed_buffer_length)

    # -------- Acceso a datos --------
    @property
    def landmarks(self) -> list[Landmarks]:
        return deepcopy(list(self._elements))

    @property
    def array(self) -> LandmarkSequenceArray:
        return np.stack([lm.array for lm in self._elements], axis=0)

    @property
    def n_frames(self) -> int:
        return len(self._elements)

    @property
    def n_points(self) -> int:
        return self._elements[0].n_points if self.n_frames > 0 else 0

    # -------- Transformaciones (usando map del padre) --------
    def centered(self, reference: Optional[Reference] = None) -> Self:
        return self.map(lambda lm: lm.centered(reference))

    def normalized(self, reference: Optional[Reference] = None) -> Self:
        return self.map(lambda lm: lm.normalized(reference))

    def scaled(self, factor: float) -> Self:
        return self.map(lambda lm: lm.scaled(factor))

    def rotated_2d(self, angle_rad: float) -> Self:
        return self.map(lambda lm: lm.rotated_2d(angle_rad))

    def subset(self, indices: list[int]) -> Self:
        return self.map(lambda lm: lm.subset(indices))

    # -------- Operaciones que no transforman la secuencia --------
    def centroid(self) -> list[NormalizedCoordinate3D]:
        return [lm.centroid() for lm in self._elements]

    def pairwise_distances(self) -> np.ndarray:
        return np.stack([lm.pairwise_distances() for lm in self._elements], axis=0)

    def variance(self) -> np.ndarray:
        return self.array.var(axis=0)

    def distance(self, a: int, b: int) -> np.ndarray:
        return np.array([lm.distance(a, b) for lm in self._elements])

    def bounding_box(self) -> np.ndarray:
        return np.stack([lm.bounding_box() for lm in self._elements], axis=0)

    def bounding_box_2d(self) -> np.ndarray:
        return np.stack([lm.bounding_box_2d() for lm in self._elements], axis=0)

    def extent(self) -> np.ndarray:
        return np.stack([lm.extent() for lm in self._elements], axis=0)

    def distance_to(self, point: np.ndarray) -> np.ndarray:
        return np.stack([lm.distance_to(point) for lm in self._elements], axis=0)

    def angle(self, a: int, b: int, c: int) -> np.ndarray:
        return np.stack([lm.angle(a, b, c) for lm in self._elements], axis=0)