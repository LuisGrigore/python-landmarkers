import numpy as np
from src.visualization.visualizers.landmarks_sequence_visualizer import LandmarksSequenceVisualizer
from src.landmarks.landmarks import Landmarks, LandmarksSequence


class TestLandmarksSequenceVisualizer:
    def test_init(self):
        lm1 = Landmarks(np.array([[0, 0, 0]]))
        lm2 = Landmarks(np.array([[1, 1, 1]]))
        seq = LandmarksSequence.from_lists([lm1, lm2], [100, 200])
        visualizer = LandmarksSequenceVisualizer(seq)
        assert visualizer._landmarks_sequence is seq

    def test_bbox(self):
        lm1 = Landmarks(np.array([[0, 0, 0]]))
        lm2 = Landmarks(np.array([[2, 2, 0]]))
        seq = LandmarksSequence.from_lists([lm1, lm2], [100, 200])
        visualizer = LandmarksSequenceVisualizer(seq)
        bbox = visualizer.bbox()
        # Should return the bbox of the last frame
        expected = np.array([[2, 2], [2, 2]])  # Single point bbox
        np.testing.assert_array_equal(bbox, expected)

    def test_centroid(self):
        lm1 = Landmarks(np.array([[0, 0, 0]]))
        lm2 = Landmarks(np.array([[2, 0, 0]]))
        seq = LandmarksSequence.from_lists([lm1, lm2], [100, 200])
        visualizer = LandmarksSequenceVisualizer(seq)
        centroid = visualizer.centroid()
        expected = np.array([2.0, 0.0, 0.0])
        np.testing.assert_array_equal(centroid, expected)

    def test_points(self):
        lm1 = Landmarks(np.array([[0, 0, 0]]))
        lm2 = Landmarks(np.array([[1, 1, 1]]))
        seq = LandmarksSequence.from_lists([lm1, lm2], [100, 200])
        visualizer = LandmarksSequenceVisualizer(seq)
        points = visualizer.points()
        expected = np.array([[1, 1, 1]])  # Last frame
        np.testing.assert_array_equal(points, expected)

    def test_sequence(self):
        lm1 = Landmarks(np.array([[0, 0, 0]]))
        lm2 = Landmarks(np.array([[1, 1, 1]]))
        seq = LandmarksSequence.from_lists([lm1, lm2], [100, 200])
        visualizer = LandmarksSequenceVisualizer(seq)
        sequence = list(visualizer.sequence())
        assert len(sequence) == 2
        assert isinstance(sequence[0], type(visualizer).__bases__[0].__bases__[0])  # LandmarksVisualizer

    def test_time_stamps_ms(self):
        lm1 = Landmarks(np.array([[0, 0, 0]]))
        lm2 = Landmarks(np.array([[1, 1, 1]]))
        seq = LandmarksSequence.from_lists([lm1, lm2], [100, 200])
        visualizer = LandmarksSequenceVisualizer(seq)
        timestamps = list(visualizer.time_stamps_ms())
        assert timestamps == [100, 200]