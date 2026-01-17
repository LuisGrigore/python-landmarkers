from typing import Callable, List, Optional
import numpy as np
from ...landmarkers.landmarkers import ImageLandmarker, LiveStreamLandmarker, VideoLandmarker
import mediapipe as mp

from ...inferences.inference import Inference
from .hands_results import landmark_list_to_ndarray, HandLandmarkerResultProtocol
from .metadata import MediapipeHandsMetadata


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def get_mp_image(image: np.ndarray) -> mp.Image:
	return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)


def get_inferences_from_raw(
	raw_inferences: HandLandmarkerResultProtocol,
) -> List[Inference[MediapipeHandsMetadata]]:
	inferences = []
	for landmarks, world_landmarks, handedness in zip(
		raw_inferences.hand_landmarks,
		raw_inferences.hand_world_landmarks,
		raw_inferences.handedness,
	):
		inference = Inference(
			landmarks=landmark_list_to_ndarray(landmarks),
			world_landmarks=landmark_list_to_ndarray(world_landmarks),
			metadata=MediapipeHandsMetadata(
				index=handedness[0].index,
				score=handedness[0].score,
				category_name=handedness[0].category_name,
			),
		)
		inferences.append(inference)
	return inferences


class MPImageLandmarker(ImageLandmarker):
	def __init__(
		self,
		model_path: str,
		num_hands: int = 1,
		min_detection_confidence: float = 0.6,
		min_presence_confidence: float = 0.6,
		min_tracking_confidence: float = 0.6,
	) -> None:
		super().__init__()

		options = HandLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=model_path),
			running_mode=VisionRunningMode.IMAGE,
			num_hands=num_hands,
			min_hand_detection_confidence=min_detection_confidence,
			min_hand_presence_confidence=min_presence_confidence,
			min_tracking_confidence=min_tracking_confidence,
		)
		self._landmarker = HandLandmarker.create_from_options(options)

	def infer(
		self, image: np.ndarray
	) -> Optional[List[Inference[MediapipeHandsMetadata]]]:
		if self._landmarker is None:
			raise RuntimeError("Landmarker is closed")
		return get_inferences_from_raw(self._landmarker.detect(get_mp_image(image)))

	def close(self) -> None:
		if self._landmarker is not None:
			self._landmarker.close()
			self._landmarker = None


class MPVideoLandmarker(VideoLandmarker):
	def __init__(
		self,
		model_path: str,
		num_hands: int = 1,
		min_detection_confidence: float = 0.6,
		min_presence_confidence: float = 0.6,
		min_tracking_confidence: float = 0.6,
	) -> None:
		super().__init__()

		options = HandLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=model_path),
			running_mode=VisionRunningMode.VIDEO,
			num_hands=num_hands,
			min_hand_detection_confidence=min_detection_confidence,
			min_hand_presence_confidence=min_presence_confidence,
			min_tracking_confidence=min_tracking_confidence,
		)
		self._landmarker = HandLandmarker.create_from_options(options)

	def infer(
		self, image: np.ndarray, timestamp_ms: int
	) -> Optional[List[Inference[MediapipeHandsMetadata]]]:
		if self._landmarker is None:
			raise RuntimeError("Landmarker is closed")
		return get_inferences_from_raw(
			self._landmarker.detect_for_video(get_mp_image(image), timestamp_ms)
		)

	def close(self) -> None:
		if self._landmarker is not None:
			self._landmarker.close()
			self._landmarker = None


class MPLiveStreamLandmarker(LiveStreamLandmarker):
	def __init__(
		self,
		model_path: str,
		callback: Callable[
			[List[Inference[MediapipeHandsMetadata]], np.ndarray, int], None
		],
		num_hands: int = 1,
		min_detection_confidence: float = 0.6,
		min_presence_confidence: float = 0.6,
		min_tracking_confidence: float = 0.6,
	) -> None:
		super().__init__()
		self._callback = callback
		options = HandLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=model_path),
			running_mode=VisionRunningMode.LIVE_STREAM,
			num_hands=num_hands,
			min_hand_detection_confidence=min_detection_confidence,
			min_hand_presence_confidence=min_presence_confidence,
			min_tracking_confidence=min_tracking_confidence,
			result_callback=self._internal_callback,
		)
		self._landmarker = HandLandmarker.create_from_options(options)

	def _internal_callback(
		self,
		raw_result: HandLandmarkerResultProtocol,
		mp_image: mp.Image,
		timestamp_ms: int,
	):
		self._callback(
			get_inferences_from_raw(raw_result),
			mp_image.numpy_view().copy(),
			timestamp_ms,
		)

	def infer(self, image: np.ndarray, timestamp_ms: int) -> None:
		if self._landmarker is None:
			raise RuntimeError("Landmarker is closed")
		self._landmarker.detect_async(get_mp_image(image), timestamp_ms)

	def close(self) -> None:
		if self._landmarker is not None:
			self._landmarker.close()
			self._landmarker = None
