import numpy as np
from unittest.mock import Mock
from src.visualization.layers.bbox_layer import BBoxLayer, HasBBox


class MockHasBBox(HasBBox):
    def __init__(self, bbox):
        self._bbox = bbox

    def bbox(self):
        return self._bbox


class TestBBoxLayer:
    def test_init(self):
        layer = BBoxLayer(color=(255, 0, 0), thickness=2)
        assert layer.color == (255, 0, 0)
        assert layer.thickness == 2

    def test_draw(self):
        layer = BBoxLayer(color=(0, 255, 0), thickness=1)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        bbox = np.array([[2, 2], [18, 18]])
        data = MockHasBBox(bbox)
        layer.draw(data, image)
        # Check that some border pixels are set
        assert np.any(image[2, :] != 0)  # Top border
        assert np.any(image[:, 2] != 0)  # Left border
        assert np.any(image[18, :] != 0)  # Bottom border
        assert np.any(image[:, 18] != 0)  # Right border