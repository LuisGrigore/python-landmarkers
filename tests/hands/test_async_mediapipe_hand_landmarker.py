import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from landmarkers.hands import (
	RawStreamMediapipeLandmarker,
	WrappedStreamMediapipeLandmarker,
	HandLandmarkerResult,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def dummy_frame():
	return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def mock_landmarker():
	return MagicMock()


@pytest.fixture
def patch_landmarker(mock_landmarker):
	with patch(
		"landmarkers.hands.async_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
		return_value=mock_landmarker,
	):
		yield mock_landmarker


# ============================================================
# RawStreamMediapipeLandmarker
# ============================================================


class TestRawStreamMediapipeLandmarker:
	def test_send_calls_detect_async(self, dummy_frame, patch_landmarker):
		callback = MagicMock()

		detector = RawStreamMediapipeLandmarker(
			"fake_model.task",
			callback,
		)

		detector.send(dummy_frame, timestamp_ms=123)

		patch_landmarker.detect_async.assert_called_once()

	def test_internal_callback_passes_raw_result(self, patch_landmarker):
		callback = MagicMock()
		raw_result = MagicMock()
		mp_image = MagicMock()

		detector = RawStreamMediapipeLandmarker(
			"fake_model.task",
			callback,
		)

		detector._internal_callback(raw_result, mp_image, 456)

		callback.assert_called_once_with(raw_result, mp_image, 456)

	def test_close_releases_resources(self, patch_landmarker):
		callback = MagicMock()

		detector = RawStreamMediapipeLandmarker(
			"fake_model.task",
			callback,
		)

		detector.close()

		patch_landmarker.close.assert_called_once()
		assert detector._landmarker is None


# ============================================================
# WrappedStreamMediapipeLandmarker
# ============================================================


class TestWrappedStreamMediapipeLandmarker:
	def test_send_calls_detect_async(self, dummy_frame, patch_landmarker):
		callback = MagicMock()

		detector = WrappedStreamMediapipeLandmarker(
			"fake_model.task",
			callback,
		)

		detector.send(dummy_frame, timestamp_ms=789)

		patch_landmarker.detect_async.assert_called_once()

	def test_internal_callback_wraps_result(self, patch_landmarker):
		callback = MagicMock()
		raw_result = MagicMock()
		mp_image = MagicMock()
		timestamp = 321

		detector = WrappedStreamMediapipeLandmarker(
			"fake_model.task",
			callback,
		)

		detector._internal_callback(raw_result, mp_image, timestamp)

		callback.assert_called_once()
		wrapped_result, image_arg, ts_arg = callback.call_args[0]

		assert isinstance(wrapped_result, HandLandmarkerResult)
		assert image_arg is mp_image
		assert ts_arg == timestamp

	def test_close_releases_resources(self, patch_landmarker):
		callback = MagicMock()

		detector = WrappedStreamMediapipeLandmarker(
			"fake_model.task",
			callback,
		)

		detector.close()

		patch_landmarker.close.assert_called_once()
		assert detector._landmarker is None
