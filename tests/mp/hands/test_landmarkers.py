import numpy as np
import pytest
from unittest.mock import Mock, patch
from src.mp.hands.landmarkers import (
    get_mp_image,
    get_inferences_from_raw,
    MPImageLandmarker,
    MPVideoLandmarker,
    MPLiveStreamLandmarker
)
from src.mp.hands.metadata import MediapipeHandsMetadata
import mediapipe as mp


class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockCategory:
    def __init__(self, index, score, category_name):
        self.index = index
        self.score = score
        self.category_name = category_name


class MockHandLandmarkerResult:
    def __init__(self, hand_landmarks, hand_world_landmarks, handedness):
        self.hand_landmarks = hand_landmarks
        self.hand_world_landmarks = hand_world_landmarks
        self.handedness = handedness


class TestGetMpImage:
    def test_get_mp_image(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mp_image = get_mp_image(image)
        assert mp_image.image_format == mp.ImageFormat.SRGB
        np.testing.assert_array_equal(mp_image.numpy_view(), image)


class TestGetInferencesFromRaw:
    def test_empty_result(self):
        result = MockHandLandmarkerResult([], [], [])
        inferences = get_inferences_from_raw(result)
        assert inferences == []

    def test_single_hand(self):
        landmarks = [MockLandmark(0.1, 0.2, 0.3), MockLandmark(0.4, 0.5, 0.6)]
        world_landmarks = [MockLandmark(0.0, 0.0, 0.0), MockLandmark(0.1, 0.1, 0.1)]
        handedness = [MockCategory(0, 0.95, "Left")]
        result = MockHandLandmarkerResult([landmarks], [world_landmarks], handedness)
        inferences = get_inferences_from_raw(result)
        assert len(inferences) == 1
        inf = inferences[0]
        expected_lm = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.testing.assert_array_equal(inf.landmarks.array, expected_lm)
        expected_wlm = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
        np.testing.assert_array_equal(inf.world_landmarks.array, expected_wlm)
        assert inf.metadata == MediapipeHandsMetadata(0, 0.95, "Left")

    def test_multiple_hands(self):
        # Similar to single hand, but with multiple
        landmarks1 = [MockLandmark(0.1, 0.2, 0.3)]
        world_landmarks1 = [MockLandmark(0.0, 0.0, 0.0)]
        handedness1 = [MockCategory(0, 0.9, "Left")]

        landmarks2 = [MockLandmark(0.5, 0.6, 0.7)]
        world_landmarks2 = [MockLandmark(0.2, 0.2, 0.2)]
        handedness2 = [MockCategory(1, 0.85, "Right")]

        result = MockHandLandmarkerResult(
            [landmarks1, landmarks2],
            [world_landmarks1, world_landmarks2],
            handedness1 + handedness2
        )
        inferences = get_inferences_from_raw(result)
        assert len(inferences) == 2


@patch('src.mp.hands.landmarkers.HandLandmarker')
@patch('src.mp.hands.landmarkers.HandLandmarkerOptions')
@patch('src.mp.hands.landmarkers.BaseOptions')
class TestMPImageLandmarker:
    def test_init(self, mock_base_options, mock_options, mock_landmarker):
        mock_landmarker.create_from_options.return_value = Mock()
        lm = MPImageLandmarker("fake_path")
        mock_base_options.assert_called_once_with(model_asset_path="fake_path")
        mock_options.assert_called_once()
        mock_landmarker.create_from_options.assert_called_once()

    def test_infer(self, mock_base_options, mock_options, mock_landmarker):
        mock_lm_instance = Mock()
        mock_landmarker.create_from_options.return_value = mock_lm_instance
        mock_lm_instance.detect.return_value = MockHandLandmarkerResult([], [], [])

        lm = MPImageLandmarker("fake_path")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = lm.infer(image)
        assert result == []

    def test_infer_closed(self, mock_base_options, mock_options, mock_landmarker):
        mock_lm_instance = Mock()
        mock_landmarker.create_from_options.return_value = mock_lm_instance

        lm = MPImageLandmarker("fake_path")
        lm.close()
        with pytest.raises(RuntimeError, match="Landmarker is closed"):
            lm.infer(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_close(self, mock_base_options, mock_options, mock_landmarker):
        mock_lm_instance = Mock()
        mock_landmarker.create_from_options.return_value = mock_lm_instance

        lm = MPImageLandmarker("fake_path")
        lm.close()
        mock_lm_instance.close.assert_called_once()


@patch('src.mp.hands.landmarkers.HandLandmarker')
@patch('src.mp.hands.landmarkers.HandLandmarkerOptions')
@patch('src.mp.hands.landmarkers.BaseOptions')
class TestMPVideoLandmarker:
    def test_init(self, mock_base_options, mock_options, mock_landmarker):
        mock_landmarker.create_from_options.return_value = Mock()
        lm = MPVideoLandmarker("fake_path")
        assert mock_options.called

    def test_infer(self, mock_base_options, mock_options, mock_landmarker):
        mock_lm_instance = Mock()
        mock_landmarker.create_from_options.return_value = mock_lm_instance
        mock_lm_instance.detect_for_video.return_value = MockHandLandmarkerResult([], [], [])

        lm = MPVideoLandmarker("fake_path")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = lm.infer(image, 100)
        assert result == []

    def test_close(self, mock_base_options, mock_options, mock_landmarker):
        mock_lm_instance = Mock()
        mock_landmarker.create_from_options.return_value = mock_lm_instance

        lm = MPVideoLandmarker("fake_path")
        lm.close()
        mock_lm_instance.close.assert_called_once()


@patch('src.mp.hands.landmarkers.HandLandmarker')
@patch('src.mp.hands.landmarkers.HandLandmarkerOptions')
@patch('src.mp.hands.landmarkers.BaseOptions')
class TestMPLiveStreamLandmarker:
    def test_init(self, mock_base_options, mock_options, mock_landmarker):
        callback = Mock()
        mock_landmarker.create_from_options.return_value = Mock()
        lm = MPLiveStreamLandmarker("fake_path", callback)
        assert mock_options.called

    def test_infer(self, mock_base_options, mock_options, mock_landmarker):
        callback = Mock()
        mock_lm_instance = Mock()
        mock_landmarker.create_from_options.return_value = mock_lm_instance

        lm = MPLiveStreamLandmarker("fake_path", callback)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        lm.infer(image, 100)
        mock_lm_instance.detect_async.assert_called_once()

    def test_close(self, mock_base_options, mock_options, mock_landmarker):
        callback = Mock()
        mock_lm_instance = Mock()
        mock_landmarker.create_from_options.return_value = mock_lm_instance

        lm = MPLiveStreamLandmarker("fake_path", callback)
        lm.close()
        mock_lm_instance.close.assert_called_once()