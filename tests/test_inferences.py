import numpy as np
import pytest
from src.inferences.inference import validate_shape, validate_array, Inference, InferenceSequence


class TestValidateShape:
    def test_valid_shape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        validate_shape(arr, (None, 3))  # Should not raise

    def test_invalid_ndim(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must have 2 dimensions"):
            validate_shape(arr, (None, 3))

    def test_invalid_dimension(self):
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="dimension 1 must be 3"):
            validate_shape(arr, (None, 3))

    def test_not_numpy_array(self):
        with pytest.raises(TypeError, match="must be a numpy array"):
            validate_shape([1, 2, 3], (3,))


class TestValidateArray:
    def test_valid_array(self):
        arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        validate_array(arr, shape=(None, 3), dtype=np.float32)  # Should not raise

    def test_invalid_dtype(self):
        arr = np.array([[1, 2, 3]], dtype=np.int32)
        with pytest.raises(ValueError, match="must have dtype"):
            validate_array(arr, dtype=np.float32)

    def test_invalid_shape(self):
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            validate_array(arr, shape=(None, 3))


class TestInference:
    @pytest.fixture
    def sample_inference(self):
        landmarks = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        world_landmarks = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
        metadata = "test_metadata"
        return Inference(landmarks, world_landmarks, metadata)

    def test_init_valid(self, sample_inference):
        assert isinstance(sample_inference, Inference)

    def test_init_invalid_shape_landmarks(self):
        with pytest.raises(ValueError):
            Inference(np.array([[1, 2]]), np.array([[0, 0, 0]]), None)

    def test_init_invalid_shape_world_landmarks(self):
        with pytest.raises(ValueError):
            Inference(np.array([[0, 0, 0]]), np.array([[1, 2]]), None)

    def test_init_mismatched_lengths(self):
        landmarks = np.array([[0, 0, 0], [1, 1, 1]])
        world_landmarks = np.array([[0, 0, 0]])
        with pytest.raises(ValueError, match="same number of points"):
            Inference(landmarks, world_landmarks, None)

    def test_landmarks_property(self, sample_inference):
        lm = sample_inference.landmarks
        assert lm.array.shape == (2, 3)
        # Check it's a copy
        lm.array[0, 0] = 999
        assert sample_inference._landmarks[0, 0] != 999

    def test_world_landmarks_property(self, sample_inference):
        wlm = sample_inference.world_landmarks
        assert wlm.array.shape == (2, 3)

    def test_metadata_property(self, sample_inference):
        assert sample_inference.metadata == "test_metadata"


class TestInferenceSequence:
    @pytest.fixture
    def sample_inference(self):
        landmarks = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        world_landmarks = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
        metadata = "test_metadata"
        return Inference(landmarks, world_landmarks, metadata)

    @pytest.fixture
    def sample_inferences(self, sample_inference):
        inf2 = Inference(
            np.array([[0.2, 0.3, 0.4]]),
            np.array([[0.0, 0.0, 0.0]]),
            "meta2"
        )
        return [sample_inference, inf2]

    @pytest.fixture
    def sample_timestamps(self):
        return [100, 200]

    def test_from_lists_valid(self, sample_inferences, sample_timestamps):
        seq = InferenceSequence.from_lists(sample_inferences, sample_timestamps)
        assert len(seq._inferences) == 2
        assert seq._time_stamps_ms == [100, 200]

    def test_from_lists_mismatched_lengths(self, sample_inferences):
        with pytest.raises(ValueError, match="Lengths must match"):
            InferenceSequence.from_lists(sample_inferences, [100])

    def test_append(self):
        seq = InferenceSequence()
        inf = Inference(np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), "meta")
        seq.append(inf, 100)
        assert len(seq._inferences) == 1
        assert seq._time_stamps_ms == [100]
        assert seq.metadata == "meta"

    def test_append_multiple_same_metadata(self):
        seq = InferenceSequence()
        inf1 = Inference(np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), "meta")
        inf2 = Inference(np.array([[1, 1, 1]]), np.array([[1, 1, 1]]), "meta")
        seq.append(inf1, 100)
        seq.append(inf2, 200)
        assert seq.metadata == "meta"

    def test_time_stamps_ms_property(self):
        seq = InferenceSequence()
        seq._time_stamps_ms = [100, 200]
        assert seq.time_stamps_ms == [100, 200]
        # Check it's a copy
        stamps = seq.time_stamps_ms
        stamps[0] = 999
        assert seq._time_stamps_ms[0] != 999

    def test_landmarks_sequence_property(self, sample_inferences, sample_timestamps):
        seq = InferenceSequence.from_lists(sample_inferences, sample_timestamps)
        lm_seq = seq.landmarks_sequence
        assert lm_seq.n_frames == 2
        assert lm_seq.n_points == 2

    def test_world_landmarks_sequence_property(self, sample_inferences, sample_timestamps):
        seq = InferenceSequence.from_lists(sample_inferences, sample_timestamps)
        wlm_seq = seq.world_landmarks_sequence
        assert wlm_seq.n_frames == 2