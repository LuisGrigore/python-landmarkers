from copy import deepcopy
from typing import Any, Callable, Optional, Self, TypeAlias, Union, TypeVar
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


BaseSequenceLike = TypeVar("BaseSequenceLike", bound= BaseSequence[Any])

def sequence_transform(fn: Callable[..., Any]):
	def wrapper(self: BaseSequenceLike, *args, **kwargs) -> BaseSequenceLike:
		return self.map(lambda lm: fn(lm, *args, **kwargs))
	return wrapper

def sequence_list(fn: Callable[..., Any]):
	def wrapper(self: "BaseSequence[Any]", *args, **kwargs):
		return [fn(lm, *args, **kwargs) for lm in self._elements]
	return wrapper

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

	def resample(self, step: Optional[int] = None) -> "LandmarksSequence":
		"""
		Resamplea la secuencia a timestamps uniformes.
		
		Parámetros
		----------
		step : int | None
			Si se pasa un valor, se generan timestamps cada `step` ms.
			Si es None, se generan n_frames uniformes a lo largo de toda la duración.
		
		Devuelve
		-------
		LandmarksSequence
			Nueva secuencia resampleada.
		"""
		if self.n_frames == 0:
			return LandmarksSequence()

		# Original timestamps y array (n_frames, n_points, 3)
		ts = np.array(self._time_stamps_ms)
		arr = self.array
		n_frames, n_points, n_dims = arr.shape

		# Calcular timestamps nuevos
		if step is not None:
			ts_new = np.arange(ts[0], ts[-1] + 1, step)
		else:
			# Mantener misma cantidad de frames, distribución uniforme
			ts_new = np.linspace(ts[0], ts[-1], n_frames)

		# Interpolación lineal para cada punto y dimensión
		arr_new = np.zeros((len(ts_new), n_points, n_dims))
		for p in range(n_points):
			for d in range(n_dims):
				arr_new[:, p, d] = np.interp(ts_new, ts, arr[:, p, d])

		# Crear nueva secuencia
		landmarks_new = [Landmarks(arr_new[i]) for i in range(len(ts_new))]
		return LandmarksSequence.from_list(landmarks_new, list(ts_new))

	centered = sequence_transform(Landmarks.centered)
	normalized = sequence_transform(Landmarks.normalized)
	scaled = sequence_transform(Landmarks.scaled)
	rotated_2d = sequence_transform(Landmarks.rotated_2d)
	subset = sequence_transform(Landmarks.subset)

	centroid = sequence_list(Landmarks.centroid)
	distance = sequence_list(Landmarks.distance)
	bounding_box = sequence_list(Landmarks.bounding_box)
	bounding_box_2d = sequence_list(Landmarks.bounding_box_2d)
	extent = sequence_list(Landmarks.extent)
	distance_to = sequence_list(Landmarks.distance_to)
	pairwise_distances = sequence_list(Landmarks.pairwise_distances)
	angle = sequence_list(Landmarks.angle)
	variance = sequence_list(Landmarks.variance)