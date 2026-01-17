import numpy as np
import pytest
from src.landmarks.landmarks import Landmarks, LandmarksSequence


class TestLandmarks:
    @pytest.fixture
    def sample_landmarks(self):
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ])

    @pytest.fixture
    def landmarks(self, sample_landmarks):
        return Landmarks(sample_landmarks)

    def test_init(self, landmarks):
        assert landmarks.n_points == 3

    def test_array_property(self, landmarks, sample_landmarks):
        arr = landmarks.array
        np.testing.assert_array_equal(arr, sample_landmarks)
        # Check it's a copy
        arr[0, 0] = 999
        assert landmarks._landmarks[0, 0] != 999

    def test_n_points(self, landmarks):
        assert landmarks.n_points == 3

    def test_resolve_reference_int(self, landmarks):
        ref = landmarks._resolve_reference(0)
        np.testing.assert_array_equal(ref, [0.0, 0.0, 0.0])

    def test_resolve_reference_int_out_of_bounds(self, landmarks):
        with pytest.raises(ValueError, match="out of bounds"):
            landmarks._resolve_reference(3)

    def test_resolve_reference_ndarray(self, landmarks):
        ref = landmarks._resolve_reference(np.array([0.5, 0.5, 0.5]))
        np.testing.assert_array_equal(ref, [0.5, 0.5, 0.5])

    def test_resolve_reference_ndarray_wrong_shape(self, landmarks):
        with pytest.raises(ValueError, match="must have shape"):
            landmarks._resolve_reference(np.array([0.5, 0.5]))

    def test_resolve_reference_none_centroid(self, landmarks):
        ref = landmarks._resolve_reference(None)
        expected = np.array([0.5, 1/3, 0.0])
        np.testing.assert_array_almost_equal(ref, expected)

    def test_centered(self, landmarks):
        centered = landmarks.centered(0)
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ])
        np.testing.assert_array_equal(centered.array, expected)

    def test_normalized(self, landmarks):
        normalized = landmarks.normalized(0)
        # Distance from point 0 to others
        dist1 = np.linalg.norm([1.0, 0.0, 0.0])
        dist2 = np.linalg.norm([0.5, 1.0, 0.0])
        scale = max(dist1, dist2)
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0/scale, 0.0, 0.0],
            [0.5/scale, 1.0/scale, 0.0]
        ])
        np.testing.assert_array_almost_equal(normalized.array, expected)

    def test_centroid(self, landmarks):
        centroid = landmarks.centroid()
        expected = np.array([0.5, 1/3, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_distance(self, landmarks):
        dist = landmarks.distance(0, 1)
        expected = 1.0
        assert dist == pytest.approx(expected)

    def test_bounding_box(self, landmarks):
        bbox = landmarks.bounding_box()
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        np.testing.assert_array_equal(bbox, expected)

    def test_bounding_box_2d(self, landmarks):
        bbox = landmarks.bounding_box_2d()
        expected = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ])
        np.testing.assert_array_equal(bbox, expected)

    def test_extent(self, landmarks):
        extent = landmarks.extent()
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(extent, expected)

    def test_distance_to(self, landmarks):
        point = np.array([0.5, 0.5, 0.0])
        distances = landmarks.distance_to(point)
        expected = [
            np.linalg.norm([0.5, 0.5, 0.0]),
            np.linalg.norm([0.5, 0.5, 0.0]),
            np.linalg.norm([0.0, 0.5, 0.0])
        ]
        np.testing.assert_array_almost_equal(distances, expected)

    def test_pairwise_distances(self, landmarks):
        pwd = landmarks.pairwise_distances()
        assert pwd.shape == (3, 3)
        assert pwd[0, 1] == pytest.approx(1.0)
        assert pwd[0, 0] == 0.0

    def test_angle(self, landmarks):
        angle = landmarks.angle(1, 0, 2)
        # Points: 0 at (0,0,0), 1 at (1,0,0), 2 at (0.5,1,0)
        # Vectors from 0: to1 (1,0,0), to2 (0.5,1,0)
        # Angle at 0 between 1 and 2
        expected = np.arccos(np.dot([1,0,0], [0.5,1,0]) / (1 * np.linalg.norm([0.5,1,0])))
        assert angle == pytest.approx(expected)

    def test_scaled(self, landmarks):
        scaled = landmarks.scaled(2.0)
        expected = landmarks.array * 2.0
        np.testing.assert_array_equal(scaled.array, expected)

    def test_rotated_2d(self, landmarks):
        rotated = landmarks.rotated_2d(np.pi/2)
        # Rotation by 90 degrees, only x,y rotated
        expected = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.5]
        ])
        np.testing.assert_array_almost_equal(rotated.array, expected)

    def test_variance(self, landmarks):
        var = landmarks.variance()
        expected = np.var(landmarks.array, axis=0)
        np.testing.assert_array_equal(var, expected)

    def test_subset(self, landmarks):
        subset = landmarks.subset([0, 2])
        expected = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ])
        np.testing.assert_array_equal(subset.array, expected)


class TestLandmarksSequence:
    @pytest.fixture
    def sample_landmarks_list(self):
        lm1 = Landmarks(np.array([[0, 0, 0], [1, 0, 0]]))
        lm2 = Landmarks(np.array([[0, 1, 0], [1, 1, 0]]))
        return [lm1, lm2]

    @pytest.fixture
    def sample_timestamps(self):
        return [100, 200]

    @pytest.fixture
    def landmarks_sequence(self, sample_landmarks_list, sample_timestamps):
        return LandmarksSequence.from_lists(sample_landmarks_list, sample_timestamps)

    def test_from_lists_valid(self, landmarks_sequence):
        assert landmarks_sequence.n_frames == 2
        assert landmarks_sequence.n_points == 2

    def test_from_lists_mismatched_lengths(self, sample_landmarks_list):
        with pytest.raises(ValueError, match="Lengths must match"):
            LandmarksSequence.from_lists(sample_landmarks_list, [100])

    def test_landmarks_property(self, landmarks_sequence, sample_landmarks_list):
        lms = landmarks_sequence.landmarks
        assert len(lms) == 2
        # Check it's a deep copy
        lms[0].array[0, 0] = 999
        assert landmarks_sequence._landmarks[0].array[0, 0] != 999

    def test_time_stamps_ms_property(self, landmarks_sequence):
        stamps = landmarks_sequence.time_stamps_ms
        assert stamps == [100, 200]
        # Check it's a copy
        stamps[0] = 999
        assert landmarks_sequence._time_stamps_ms[0] != 999

    def test_array_property(self, landmarks_sequence):
        arr = landmarks_sequence.array
        assert arr.shape == (2, 2, 3)

    def test_n_frames(self, landmarks_sequence):
        assert landmarks_sequence.n_frames == 2

    def test_n_points(self, landmarks_sequence):
        assert landmarks_sequence.n_points == 2

    def test_append(self):
        seq = LandmarksSequence()
        lm = Landmarks(np.array([[0, 0, 0]]))
        seq.append(lm, 100)
        assert seq.n_frames == 1

    def test_centered(self, landmarks_sequence):
        centered = landmarks_sequence.centered(0)
        assert centered.n_frames == 2

    def test_normalized(self, landmarks_sequence):
        normalized = landmarks_sequence.normalized(0)
        assert normalized.n_frames == 2

    def test_scaled(self, landmarks_sequence):
        scaled = landmarks_sequence.scaled(2.0)
        assert scaled.n_frames == 2

    def test_rotated_2d(self, landmarks_sequence):
        rotated = landmarks_sequence.rotated_2d(np.pi/2)
        assert rotated.n_frames == 2

    def test_centroid(self, landmarks_sequence):
        centroids = landmarks_sequence.centroid()
        assert centroids.shape == (2, 3)

    def test_pairwise_distances(self, landmarks_sequence):
        pwd = landmarks_sequence.pairwise_distances()
        assert pwd.shape == (2, 2, 2)

    def test_subset(self, landmarks_sequence):
        subset = landmarks_sequence.subset([0])
        assert subset.n_points == 1

    def test_variance(self, landmarks_sequence):
        var = landmarks_sequence.variance()
        assert var.shape == (2, 3)

    def test_distance(self, landmarks_sequence):
        distances = landmarks_sequence.distance(0, 1)
        assert len(distances) == 2

    def test_bounding_box(self, landmarks_sequence):
        bbox = landmarks_sequence.bounding_box()
        assert bbox.shape == (2, 2, 3)

    def test_bounding_box_2d(self, landmarks_sequence):
        bbox = landmarks_sequence.bounding_box_2d()
        assert bbox.shape == (2, 2, 2)

    def test_extent(self, landmarks_sequence):
        extent = landmarks_sequence.extent()
        assert extent.shape == (2, 3)

    def test_distance_to(self, landmarks_sequence):
        point = np.array([0.5, 0.5, 0.0])
        distances = landmarks_sequence.distance_to(point)
        assert distances.shape == (2, 2)

    def test_angle(self, landmarks_sequence):
        angles = landmarks_sequence.angle(0, 1, 0)  # Note: this might not be meaningful, but test the method
        assert len(angles) == 2