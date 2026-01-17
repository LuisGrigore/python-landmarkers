import pytest
from src.landmarkers.landmarkers import BaseLandmarker, ImageLandmarker, VideoLandmarker, LiveStreamLandmarker


class MockLandmarker(BaseLandmarker):
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class TestBaseLandmarker:
    def test_context_manager(self):
        lm = MockLandmarker()
        with lm as ctx:
            assert ctx is lm
            assert not lm.closed
        assert lm.closed

    def test_close(self):
        lm = MockLandmarker()
        lm.close()
        assert lm.closed


class MockImageLandmarker(ImageLandmarker):
    def __init__(self):
        super().__init__()
        self.closed = False

    def infer(self, image):
        if self.closed:
            raise RuntimeError("Closed")
        return None

    def close(self):
        self.closed = True


class TestImageLandmarker:
    def test_infer_abstract(self):
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            ImageLandmarker()

    def test_mock_implementation(self):
        lm = MockImageLandmarker()
        result = lm.infer(None)
        assert result is None


class MockVideoLandmarker(VideoLandmarker):
    def __init__(self):
        super().__init__()
        self.closed = False

    def infer(self, image, timestamp_ms):
        if self.closed:
            raise RuntimeError("Closed")
        return None

    def close(self):
        self.closed = True


class TestVideoLandmarker:
    def test_infer_abstract(self):
        with pytest.raises(TypeError):
            VideoLandmarker()

    def test_mock_implementation(self):
        lm = MockVideoLandmarker()
        result = lm.infer(None, 100)
        assert result is None


class MockLiveStreamLandmarker(LiveStreamLandmarker):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.closed = False

    def infer(self, image, timestamp_ms):
        if self.closed:
            raise RuntimeError("Closed")
        pass

    def close(self):
        self.closed = True


class TestLiveStreamLandmarker:
    def test_infer_abstract(self):
        with pytest.raises(TypeError):
            LiveStreamLandmarker(None)

    def test_mock_implementation(self):
        callback_called = False
        def callback(*args):
            nonlocal callback_called
            callback_called = True

        lm = MockLiveStreamLandmarker(callback)
        lm.infer(None, 100)
        # For live stream, infer doesn't call callback directly
        assert not callback_called