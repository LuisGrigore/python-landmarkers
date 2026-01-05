from copy import deepcopy
from functools import cache
from typing import Literal, Optional, Protocol, Sequence, Tuple, runtime_checkable

import cv2
import numpy as np

from ..types import ImageArray

HandednessLabel = Literal["Left", "Right"]


@runtime_checkable
class Category(Protocol):
	index: int
	score: float
	category_name: HandednessLabel


@runtime_checkable
class Landmark(Protocol):
	x: float
	y: float
	z: float


LandmarkList = Sequence[Landmark]


@runtime_checkable
class HandLandmarkerResultProtocol(Protocol):
	handedness: Sequence[Sequence[Category]]
	hand_landmarks: Sequence[LandmarkList]
	hand_world_landmarks: Sequence[LandmarkList]


def cached_method(method):
	attr_name = f"_cached_{method.__name__}"

	def wrapper(self, *args, **kwargs):
		if not hasattr(self, attr_name):
			bound_method = lambda *a, **kw: method(self, *a, **kw)
			setattr(self, attr_name, cache(bound_method))

		result = getattr(self, attr_name)(*args, **kwargs)
		if isinstance(result, np.ndarray):
			return result.copy()
		return deepcopy(result)

	return wrapper


class HandLandmarkerResult:
	"""
	Hand Landmarker Result Wrapper.

	Wraps the raw MediaPipe hand landmarker result with helper methods for easy access to landmarks, handedness, and drawing.

	Args:
		data (HandLandmarkerResultProtocol): Raw MediaPipe result data.
		num_landmarks (int): Expected number of landmarks per hand.
		num_world_landmarks (int): Expected number of world landmarks per hand.
		time_stamp_ms (Optional[int]): Timestamp in milliseconds.

	Example:
		>>> result = detector.detect(frame)
		>>> print("Hands detected:", result.hands_count)
		>>> image_with_landmarks = result.draw(frame)
	"""

	def __init__(
		self,
		data: HandLandmarkerResultProtocol,
		num_landmarks: int,
		num_world_landmarks: int,
		time_stamp_ms: Optional[int] = None,
	) -> None:

		self._data: HandLandmarkerResultProtocol = deepcopy(data)
		self._num_landmarks: int = num_landmarks
		self._num_world_landmarks: int = num_world_landmarks
		self._time_stamp_ms = time_stamp_ms

	@property
	def data(self) -> HandLandmarkerResultProtocol:
		"""Raw MediaPipe result data."""
		return deepcopy(self._data)

	@property
	def hands_count(self) -> int:
		"""Number of detected hands."""
		return len(self._data.hand_landmarks)

	@property
	def time_stamp_ms(self) -> Optional[int]:
		"""Timestamp of the result in milliseconds."""
		return self._time_stamp_ms

	def draw(self, image: ImageArray, hand_index: Optional[int] = None) -> ImageArray:
		"""
		Draw hand landmarks on the image.

		Args:
			image (ImageArray): Input image as numpy array.
			hand_index (Optional[int]): Index of the hand to draw. If None, draws all hands.

		Returns:
				ImageArray: Image with landmarks drawn.
		"""
		new_image = image.copy()
		hands = self._select_hands(self._data.hand_landmarks, hand_index)
		h, w, _ = new_image.shape
		for hand in hands:
			for lm in hand:
				cx, cy = int(lm.x * w), int(lm.y * h)
				cv2.circle(new_image, (cx, cy), 4, (0, 255, 0), -1)
		return new_image

	def _pad_or_truncate(
		self, landmarks: Sequence[Landmark], target_length: int
	) -> np.ndarray:
		landmark_array = np.asarray(
			[(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
		)
		if len(landmark_array) == target_length:
			return landmark_array
		padded_array = np.zeros((target_length, 3), dtype=np.float32)
		elements_to_copy = min(len(landmark_array), target_length)
		padded_array[:elements_to_copy] = landmark_array[:elements_to_copy]
		return padded_array

	def _select_hands(
		self, hands_source: Sequence[LandmarkList], hand_index: Optional[int]
	) -> Sequence[LandmarkList]:
		if not hands_source:
			return []
		if hand_index is not None:
			if 0 <= hand_index < len(hands_source):
				return [hands_source[hand_index]]
			return []
		return list(hands_source)

	def _landmarks_array(
		self,
		hands_landmarks: Sequence[LandmarkList],
		fill_value: Optional[float] = None,
	) -> Tuple[np.ndarray, bool]:
		if not hands_landmarks:
			if fill_value is None:
				return (np.zeros((0, self._num_landmarks, 3), dtype=np.float32), False)
			return (
				np.full((1, self._num_landmarks, 3), fill_value, dtype=np.float32),
				False,
			)
		return (
			np.stack(
				[
					self._pad_or_truncate(hand, self._num_landmarks)
					for hand in hands_landmarks
				],
				axis=0,
			),
			True,
		)

	@cached_method
	def landmarks_array(
		self, hand_index: Optional[int] = None, fill_value: Optional[float] = None
	) -> np.ndarray:
		"""
		Get hand landmarks as a numpy array.

		Args:
			hand_index (Optional[int]): Index of the hand. If None, returns all hands.
			fill_value (Optional[float]): If no hands are detected and fill_value is provided, returns an array of shape (1, num_landmarks, 3) filled with this value. If None, returns an empty array of shape (0, num_landmarks, 3).

		Returns:
			np.ndarray: Array of shape (num_hands, num_landmarks, 3) with (x, y, z) coordinates, or (1, num_landmarks, 3) if fill_value is used and no hands detected.
		"""
		return self._landmarks_array(
			self._select_hands(self._data.hand_landmarks, hand_index), fill_value
		)[0]

	@cached_method
	def world_landmarks_array(
		self, hand_index: Optional[int] = None, fill_value: Optional[float] = None
	) -> np.ndarray:
		"""
		Get world hand landmarks as a numpy array.

		Args:
			hand_index (Optional[int]): Index of the hand. If None, returns all hands.
			fill_value (Optional[float]): If no hands are detected and fill_value is provided, returns an array of shape (1, num_world_landmarks, 3) filled with this value. If None, returns an empty array of shape (0, num_world_landmarks, 3).

		Returns:
			np.ndarray: Array of shape (num_hands, num_world_landmarks, 3) with (x, y, z) coordinates, or (1, num_world_landmarks, 3) if fill_value is used and no hands detected.
		"""
		return self._landmarks_array(
			self._select_hands(self._data.hand_world_landmarks, hand_index), fill_value
		)[0]

	@cached_method
	def landmarks_array_relative_to_wrist(
		self, hand_index: Optional[int] = None, fill_value: Optional[float] = None
	) -> np.ndarray:
		"""
		Get hand landmarks relative to the wrist as a numpy array.

		Args:
			hand_index (Optional[int]): Index of the hand. If None, returns all hands.
			fill_value (Optional[float]): If no hands are detected and fill_value is provided, returns an array of shape (1, num_landmarks, 3) filled with this value. If None, returns an empty array of shape (0, num_landmarks, 3).

		Returns:
			np.ndarray: Array of shape (num_hands, num_landmarks, 3) with relative (x, y, z) coordinates, or (1, num_landmarks, 3) if fill_value is used and no hands detected.
		"""

		(landmarks, hands_present) = self._landmarks_array(
			self._select_hands(self._data.hand_landmarks, hand_index), fill_value
		)

		return landmarks - landmarks[:, 0:1, :] if hands_present else landmarks

	@cached_method
	def handedness(self, hand_index: Optional[int] = None) -> np.ndarray:
		"""
		Get handedness information as a numpy array.

		Args:
			hand_index (Optional[int]): Index of the hand. If None, returns all hands.

		Returns:
			np.ndarray: Array of shape (num_hands, 3) with (index, score, label) where label is 0 for Left, 1 for Right.
		"""
		if not self._data.handedness:
			return np.zeros((0, 3), dtype=np.float32)

		hands_info = [
			hand[0] for hand in self._data.handedness if hand and len(hand) > 0
		]
		result = np.array(
			[
				[hand.index, hand.score, 0 if hand.category_name == "Left" else 1]
				for hand in hands_info
			],
			dtype=np.float32,
		)
		if hand_index is not None:
			if 0 <= hand_index < len(result):
				result = result[hand_index : hand_index + 1, :]
			else:
				result = np.zeros((0, 3), dtype=np.float32)
		return result
