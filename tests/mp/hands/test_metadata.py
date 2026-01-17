import pytest
from src.mp.hands.metadata import MediapipeHandsMetadata


class TestMediapipeHandsMetadata:
    def test_init(self):
        meta = MediapipeHandsMetadata(index=0, score=0.95, category_name="Left")
        assert meta.index == 0
        assert meta.score == 0.95
        assert meta.category_name == "Left"

    def test_frozen(self):
        meta = MediapipeHandsMetadata(index=1, score=0.8, category_name="Right")
        with pytest.raises(AttributeError):
            meta.index = 2

    def test_equality(self):
        meta1 = MediapipeHandsMetadata(index=0, score=1.0, category_name="Left")
        meta2 = MediapipeHandsMetadata(index=0, score=1.0, category_name="Left")
        assert meta1 == meta2

    def test_inequality(self):
        meta1 = MediapipeHandsMetadata(index=0, score=1.0, category_name="Left")
        meta2 = MediapipeHandsMetadata(index=1, score=1.0, category_name="Left")
        assert meta1 != meta2