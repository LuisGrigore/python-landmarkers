from collections import deque
from typing import Generic, Iterable, Optional, TypeVar
import numpy as np
from ..landmarks.landmarks import LandmarksSequence, Landmarks


def validate_shape(
	arr: np.ndarray,
	expected_shape: Iterable[int | None],
	*,
	name: str = "array",
) -> None:
	"""
	Validate that a numpy array has the expected shape.

	Parameters
	----------
	arr : np.ndarray
		Array to validate.
	expected_shape : Iterable[int | None]
		Expected shape. Use None for flexible dimensions.
		Example: (None, 3)
	name : str
		Name used in error messages.

	Raises
	------
	TypeError
		If arr is not a numpy array.
	ValueError
		If shape does not match.
	"""
	if not isinstance(arr, np.ndarray):
		raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")

	expected_shape = tuple(expected_shape)

	if arr.ndim != len(expected_shape):
		raise ValueError(
			f"{name} must have {len(expected_shape)} dimensions, "
			f"got {arr.ndim} with shape {arr.shape}"
		)

	for axis, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
		if expected is not None and actual != expected:
			raise ValueError(
				f"{name} has invalid shape {arr.shape}: "
				f"dimension {axis} must be {expected}, got {actual}"
			)


def validate_array(
	arr: np.ndarray,
	*,
	shape: Iterable[int | None] | None = None,
	dtype: np.dtype | None = None,
	name: str = "array",
) -> None:
	if not isinstance(arr, np.ndarray):
		raise TypeError(f"{name} must be a numpy array")

	if dtype is not None and arr.dtype != dtype:
		raise ValueError(f"{name} must have dtype {dtype}, got {arr.dtype}")

	if shape is not None:
		validate_shape(arr, shape, name=name)


M = TypeVar("M")


class Inference(Generic[M]):
	def __init__(self, landmarks: np.ndarray, world_landmarks: np.ndarray, metadata: M):
		self._landmarks: np.ndarray = landmarks
		self._world_landmarks: np.ndarray = world_landmarks
		validate_shape(landmarks, (None, 3), name="landmarks")
		validate_shape(world_landmarks, (None, 3), name="world_landmarks")
		if landmarks.shape[0] != world_landmarks.shape[0]:
			raise ValueError(
				"landmarks and world_landmarks must have the same number of points, "
				f"got {landmarks.shape[0]} and {world_landmarks.shape[0]}"
			)
		self._metadata: M = metadata

	@property
	def landmarks(self) -> Landmarks:
		return Landmarks(self._landmarks.copy())

	@property
	def world_landmarks(self) -> Landmarks:
		return Landmarks(self._world_landmarks.copy())

	@property
	def metadata(self) -> M:
		return self._metadata


class InferenceSequence(Generic[M]):
    def __init__(self, fixed_buffer_length: Optional[int] = None):
        if fixed_buffer_length is not None and fixed_buffer_length < 1:
            raise ValueError("fixed_buffer_length must be > 0")

        self._fixed_buffer_length = fixed_buffer_length
        self._inferences = deque(maxlen=fixed_buffer_length)
        self._time_stamps_ms = deque(maxlen=fixed_buffer_length)
        self._metadata: Optional[M] = None

    @classmethod
    def from_lists(
        cls,
        inferences: list[Inference[M]],
        time_stamps_ms: list[int],
        fixed_buffer_length: Optional[int] = None,
    ) -> "InferenceSequence[M]":
        if len(inferences) != len(time_stamps_ms):
            raise ValueError("Lengths must match")

        obj = cls(fixed_buffer_length)
        for inf, ts in zip(inferences, time_stamps_ms):
            obj.append(inf, ts)

        return obj

    def append(self, hand: Inference[M], time_stamp_ms: int) -> None:
        if self._metadata is None:
            self._metadata = hand.metadata

        self._inferences.append(hand)
        self._time_stamps_ms.append(time_stamp_ms)

    @property
    def time_stamps_ms(self) -> list[int]:
        return list(self._time_stamps_ms)

    @property
    def landmarks_sequence(self) -> LandmarksSequence:
        return LandmarksSequence.from_lists(
            [inf.landmarks for inf in self._inferences],
            list(self._time_stamps_ms),
        )

    @property
    def world_landmarks_sequence(self) -> LandmarksSequence:
        return LandmarksSequence.from_lists(
            [inf.world_landmarks for inf in self._inferences],
            list(self._time_stamps_ms),
        )

    @property
    def metadata(self) -> Optional[M]:
        return self._metadata