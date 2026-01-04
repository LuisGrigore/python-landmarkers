import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from landmarkers.hands.sync_mediapipe_hand_landmarker import (
    SyncMediapipeHandLandmarker,
    MediapipeHandLandmarkerRunningMode,
)
from landmarkers.hands.hands_results import HandLandmarkerResult


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def dummy_frame():
    return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def mock_landmarker():
    landmarker = MagicMock()
    landmarker.detect.return_value = "IMAGE_RESULT"
    landmarker.detect_for_video.return_value = "VIDEO_RESULT"
    return landmarker


# -------------------------
# Tests
# -------------------------

def test_init_creates_landmarker(mock_landmarker):
    with patch(
        "landmarkers.hands.sync_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
        return_value=mock_landmarker,
    ):
        detector = SyncMediapipeHandLandmarker("fake_model.task")

        assert detector._landmarker is mock_landmarker


def test_detect_raw_image_mode(dummy_frame, mock_landmarker):
    with patch(
        "landmarkers.hands.sync_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
        return_value=mock_landmarker,
    ):
        detector = SyncMediapipeHandLandmarker(
            model_path="fake_model.task",
            running_mode=MediapipeHandLandmarkerRunningMode.IMAGE,
        )

        result = detector.detect_raw(dummy_frame)

        mock_landmarker.detect.assert_called_once()
        assert result == "IMAGE_RESULT"


def test_detect_raw_video_mode_with_timestamp(dummy_frame, mock_landmarker):
    with patch(
        "landmarkers.hands.sync_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
        return_value=mock_landmarker,
    ):
        detector = SyncMediapipeHandLandmarker(
            model_path="fake_model.task",
            running_mode=MediapipeHandLandmarkerRunningMode.VIDEO,
        )

        result = detector.detect_raw(dummy_frame, timestamp_ms=123)

        mock_landmarker.detect_for_video.assert_called_once()
        assert result == "VIDEO_RESULT"


def test_detect_raw_video_mode_without_timestamp_raises(dummy_frame, mock_landmarker):
    with patch(
        "landmarkers.hands.sync_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
        return_value=mock_landmarker,
    ):
        detector = SyncMediapipeHandLandmarker(
            model_path="fake_model.task",
            running_mode=MediapipeHandLandmarkerRunningMode.VIDEO,
        )

        with pytest.raises(ValueError, match="timestamp_ms required"):
            detector.detect_raw(dummy_frame)


def test_detect_raw_after_close_raises(dummy_frame, mock_landmarker):
    with patch(
        "landmarkers.hands.sync_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
        return_value=mock_landmarker,
    ):
        detector = SyncMediapipeHandLandmarker("fake_model.task")
        detector.close()

        with pytest.raises(RuntimeError, match="Landmarker is closed"):
            detector.detect_raw(dummy_frame)


def test_detect_returns_handlandmarkerresult(dummy_frame, mock_landmarker):
    with patch(
        "landmarkers.hands.sync_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
        return_value=mock_landmarker,
    ):
        detector = SyncMediapipeHandLandmarker("fake_model.task")

        result = detector.detect(dummy_frame)

        assert isinstance(result, HandLandmarkerResult)


def test_close_calls_landmarker_close(mock_landmarker):
    with patch(
        "landmarkers.hands.sync_mediapipe_hand_landmarker.HandLandmarker.create_from_options",
        return_value=mock_landmarker,
    ):
        detector = SyncMediapipeHandLandmarker("fake_model.task")

        detector.close()

        mock_landmarker.close.assert_called_once()
        assert detector._landmarker is None
