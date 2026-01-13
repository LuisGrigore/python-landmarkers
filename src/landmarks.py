from typing import Optional, Union
import numpy as np


class Landmarks:
	def __init__(self, landmarks: np.ndarray) -> None:
		self._landmarks: np.ndarray = landmarks

	@property
	def array(self) -> np.ndarray:
		return self._landmarks.copy()

	@property
	def n_points(self) -> int:
		return self._landmarks.shape[0]

	def _resolve_reference(
		self, reference: Optional[Union[int, np.ndarray]]
	) -> np.ndarray:
		if isinstance(reference, int):
			if not (0 <= reference < self.n_points):
				raise ValueError(f"Reference index {reference} out of bounds")
			return self._landmarks[reference]
		elif isinstance(reference, np.ndarray):
			if reference.shape != self._landmarks.shape[1:]:
				raise ValueError(
					f"Reference point must have shape {self._landmarks.shape[1:]}"
				)
			return reference
		else:
			return self.centroid()

	def centered(
		self, reference: Optional[Union[int, np.ndarray]] = None
	) -> "Landmarks":
		ref_point = self._resolve_reference(reference)
		return Landmarks(self._landmarks - ref_point)

	def normalized(
		self, reference: Optional[Union[int, np.ndarray]] = None
	) -> "Landmarks":
		ref_point = self._resolve_reference(reference)
		diff = self._landmarks - ref_point
		scale = np.linalg.norm(diff, axis=1).max()
		return Landmarks(diff / scale)

	def centroid(self) -> np.ndarray:
		return self._landmarks.mean(axis=0)

	def distance(self, a: int, b: int) -> float:
		return float(np.linalg.norm(self._landmarks[a] - self._landmarks[b]))

	def bounding_box(self) -> np.ndarray:
		return np.stack([self._landmarks.min(axis=0), self._landmarks.max(axis=0)], axis=0)

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


class LandmarkSequence:
	def __init__(self, landmarks: list[Landmarks]):
		self._landmarks = landmarks

	@property
	def array(self) -> np.ndarray:
		return np.stack([lm.array for lm in self._landmarks], axis=0)

	@property
	def n_frames(self) -> int:
		return len(self._landmarks)

	@property
	def n_points(self) -> int:
		return self._landmarks[0].n_points if self.n_frames > 0 else 0

	def centered(self, reference: Optional[Union[int, np.ndarray]] = None) -> "LandmarkSequence":
		return LandmarkSequence([lm.centered(reference) for lm in self._landmarks])

	def normalized(self, reference: Optional[Union[int, np.ndarray]] = None) -> "LandmarkSequence":
		return LandmarkSequence([lm.normalized(reference) for lm in self._landmarks])

	def scaled(self, factor: float) -> "LandmarkSequence":
		return LandmarkSequence([lm.scaled(factor) for lm in self._landmarks])

	def rotated_2d(self, angle_rad: float) -> "LandmarkSequence":
		return LandmarkSequence([lm.rotated_2d(angle_rad) for lm in self._landmarks])

	def centroid(self) -> np.ndarray:
		return np.stack([lm.centroid() for lm in self._landmarks], axis=0)

	def pairwise_distances(self) -> np.ndarray:
		return np.stack([lm.pairwise_distances() for lm in self._landmarks], axis=0)

	def subset(self, indices: list[int]) -> "LandmarkSequence":
		return LandmarkSequence([lm.subset(indices) for lm in self._landmarks])

	def variance(self) -> np.ndarray:
		return self.array.var(axis=0)

	def distance(self, a: int, b: int) -> np.ndarray:
		return np.array([lm.distance(a, b) for lm in self._landmarks])

	def bounding_box(self) -> np.ndarray:
		return np.stack([lm.bounding_box() for lm in self._landmarks], axis=0)

	def bounding_box_2d(self) -> np.ndarray:
		return np.stack([lm.bounding_box_2d() for lm in self._landmarks], axis=0)
	
	def extent(self) -> np.ndarray:
		return np.stack([lm.extent() for lm in self._landmarks], axis=0)
	
	def distance_to(self, point: np.ndarray) -> np.ndarray:
		return np.stack([lm.distance_to(point) for lm in self._landmarks], axis=0)

	def angle(self, a: int, b: int, c: int) -> np.ndarray:
		return np.stack([lm.angle(a, b, c) for lm in self._landmarks], axis=0)