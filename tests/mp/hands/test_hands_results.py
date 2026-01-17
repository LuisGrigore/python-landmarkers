import numpy as np
import pytest
from src.mp.hands.hands_results import landmark_list_to_ndarray


class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class TestLandmarkListToNdarray:
    def test_empty_list(self):
        result = landmark_list_to_ndarray([])
        expected = np.array([], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_single_landmark(self):
        landmarks = [MockLandmark(0.1, 0.2, 0.3)]
        result = landmark_list_to_ndarray(landmarks)
        expected = np.array([[0.1, 0.2, 0.3]])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_landmarks(self):
        landmarks = [
            MockLandmark(0.0, 0.0, 0.0),
            MockLandmark(1.0, 1.0, 1.0),
            MockLandmark(0.5, 0.5, 0.5)
        ]
        result = landmark_list_to_ndarray(landmarks)
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5]
        ])
        np.testing.assert_array_equal(result, expected)