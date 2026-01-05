import numpy as np
import pytest

from landmarkers.hands import HandLandmarkerResult

# ============================================================
# Helpers / Fixtures
# ============================================================


class DummyLandmark:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z


class DummyCategory:
	def __init__(self, index, score, category_name):
		self.index = index
		self.score = score
		self.category_name = category_name


class DummyResult:
	def __init__(self, handedness, hand_landmarks, hand_world_landmarks):
		self.handedness = handedness
		self.hand_landmarks = hand_landmarks
		self.hand_world_landmarks = hand_world_landmarks


def make_hand(n=21):
	return [DummyLandmark(i / n, i / n, i / n) for i in range(n)]


@pytest.fixture
def empty_result():
	return DummyResult([], [], [])


@pytest.fixture
def one_hand_result():
	hand = make_hand()
	return DummyResult(
		handedness=[[DummyCategory(0, 0.9, "Right")]],
		hand_landmarks=[hand],
		hand_world_landmarks=[hand],
	)


@pytest.fixture
def two_hands_result():
	hand1 = make_hand()
	hand2 = make_hand()
	return DummyResult(
		handedness=[
			[DummyCategory(0, 0.9, "Left")],
			[DummyCategory(1, 0.8, "Right")],
		],
		hand_landmarks=[hand1, hand2],
		hand_world_landmarks=[hand1, hand2],
	)


# ============================================================
# Timestamp behaviour
# ============================================================


class TestTimestampBehaviour:
	def test_timestamp_none_returns_none(self, empty_result):
		result = HandLandmarkerResult(
			empty_result, num_landmarks=21, num_world_landmarks=21, time_stamp_ms=None
		)
		assert result.time_stamp_ms is None

	def test_timestamp_present(self, one_hand_result):
		result = HandLandmarkerResult(one_hand_result, 21, 21, time_stamp_ms=1234)
		assert result.time_stamp_ms == 1234


# ============================================================
# Hands count
# ============================================================


class TestHandsCount:
	def test_hands_count_empty(self, empty_result):
		result = HandLandmarkerResult(empty_result, 21, 21, None)
		assert result.hands_count == 0

	def test_hands_count_one(self, one_hand_result):
		result = HandLandmarkerResult(one_hand_result, 21, 21, None)
		assert result.hands_count == 1

	def test_hands_count_two(self, two_hands_result):
		result = HandLandmarkerResult(two_hands_result, 21, 21, None)
		assert result.hands_count == 2


# ============================================================
# Landmarks array
# ============================================================


class TestLandmarksArray:
	def test_landmarks_array_no_hands(self, empty_result):
		result = HandLandmarkerResult(empty_result, 21, 21, None)
		arr = result.landmarks_array()
		assert arr.shape == (0, 21, 3)

	def test_landmarks_array_fill_value_no_hands(self, empty_result):
		result = HandLandmarkerResult(empty_result, 21, 21, None)
		arr = result.landmarks_array(fill_value=0.5)
		assert arr.shape == (1, 21, 3)
		assert np.all(arr == 0.5)

	def test_landmarks_array_one_hand(self, one_hand_result):
		result = HandLandmarkerResult(one_hand_result, 21, 21, None)
		arr = result.landmarks_array()
		assert arr.shape == (1, 21, 3)

	def test_landmarks_array_two_hands(self, two_hands_result):
		result = HandLandmarkerResult(two_hands_result, 21, 21, None)
		arr = result.landmarks_array()
		assert arr.shape == (2, 21, 3)

	def test_landmarks_array_with_hand_index(self, two_hands_result):
		result = HandLandmarkerResult(two_hands_result, 21, 21, None)
		arr = result.landmarks_array(hand_index=1)
		assert arr.shape == (1, 21, 3)

	def test_landmarks_array_invalid_hand_index(self, two_hands_result):
		result = HandLandmarkerResult(two_hands_result, 21, 21, None)
		arr = result.landmarks_array(hand_index=99)
		assert arr.shape == (0, 21, 3)


# ============================================================
# World landmarks array
# ============================================================


class TestWorldLandmarksArray:
	def test_world_landmarks_array(self, two_hands_result):
		result = HandLandmarkerResult(two_hands_result, 21, 21, None)
		arr = result.world_landmarks_array()
		assert arr.shape == (2, 21, 3)

	def test_world_landmarks_array_fill_value_no_hands(self, empty_result):
		result = HandLandmarkerResult(empty_result, 21, 21, None)
		arr = result.world_landmarks_array(fill_value=0.5)
		assert arr.shape == (1, 21, 3)
		assert np.all(arr == 0.5)


# ============================================================
# Relative landmarks
# ============================================================


class TestRelativeLandmarks:
	def test_landmarks_relative_to_wrist(self, one_hand_result):
		result = HandLandmarkerResult(one_hand_result, 21, 21, None)
		rel = result.landmarks_array_relative_to_wrist()
		assert np.allclose(rel[:, 0, :], 0.0)

	def test_landmarks_relative_to_wrist_fill_value_no_hands(self, empty_result):
		result = HandLandmarkerResult(empty_result, 21, 21, None)
		arr = result.landmarks_array_relative_to_wrist(fill_value=0.5)
		assert arr.shape == (1, 21, 3)
		assert np.all(arr == 0.5)


# ============================================================
# Handedness
# ============================================================


class TestHandedness:
	def test_handedness_empty(self, empty_result):
		result = HandLandmarkerResult(empty_result, 21, 21, None)
		arr = result.handedness()
		assert arr.shape == (0, 3)

	def test_handedness_one_hand(self, one_hand_result):
		result = HandLandmarkerResult(one_hand_result, 21, 21, None)
		arr = result.handedness()
		assert arr.shape == (1, 3)
		assert arr[0, 2] == 1

	def test_handedness_two_hands_with_index(self, two_hands_result):
		result = HandLandmarkerResult(two_hands_result, 21, 21, None)
		arr = result.handedness(hand_index=0)
		assert arr.shape == (1, 3)
		assert arr[0, 2] == 0


# ============================================================
# Cache behaviour
# ============================================================


class TestCacheBehaviour:
	def test_cached_methods_are_cached(self, one_hand_result):
		result = HandLandmarkerResult(one_hand_result, 21, 21, None)

		a1 = result.landmarks_array()
		a2 = result.landmarks_array()

		assert np.array_equal(a1, a2)

	def test_cached_methods_return_new_instance(self, one_hand_result):
		result = HandLandmarkerResult(one_hand_result, 21, 21, None)

		a1 = result.landmarks_array()
		a2 = result.landmarks_array()

		a1[0] = [666, 666, 666]
		a4 = result.landmarks_array()

		assert a1 is not a2
		assert not np.array_equal(a1, a4)
