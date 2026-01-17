import numpy as np
from src.visualization.visualizers.landmarks_visualizer import LandmarksVisualizer
from src.landmarks.landmarks import Landmarks


class TestLandmarksVisualizer:
    def test_init(self):
        landmarks = Landmarks(np.array([[0, 0, 0], [1, 1, 1]]))
        visualizer = LandmarksVisualizer(landmarks)
        assert visualizer._landmarks is landmarks

    def test_bbox(self):
        landmarks = Landmarks(np.array([[0, 0, 0], [2, 2, 0]]))
        visualizer = LandmarksVisualizer(landmarks)
        bbox = visualizer.bbox()
        expected = np.array([[0, 0], [2, 2]])
        np.testing.assert_array_equal(bbox, expected)

    def test_centroid(self):
        landmarks = Landmarks(np.array([[0, 0, 0], [2, 0, 0]]))
        visualizer = LandmarksVisualizer(landmarks)
        centroid = visualizer.centroid()
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(centroid, expected)

    def test_points(self):
        points = np.array([[0, 0, 0], [1, 1, 1]])
        landmarks = Landmarks(points)
        visualizer = LandmarksVisualizer(landmarks)
        result_points = visualizer.points()
        np.testing.assert_array_equal(result_points, points)