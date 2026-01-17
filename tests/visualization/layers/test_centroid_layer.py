import numpy as np
from unittest.mock import Mock
from src.visualization.layers.centroid_layer import CentroidLayer, HasCentroid


class MockHasCentroid(HasCentroid):
    def __init__(self, centroid):
        self._centroid = centroid

    def centroid(self):
        return self._centroid


class TestCentroidLayer:
    def test_init(self):
        layer = CentroidLayer(color=(0, 0, 255), radius=10)
        assert layer.color == (0, 0, 255)
        assert layer.radius == 10

    def test_draw(self):
        layer = CentroidLayer(color=(255, 255, 0), radius=2)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        centroid = np.array([3.0, 3.0])
        data = MockHasCentroid(centroid)
        layer.draw(data, image)
        # Check that pixels around the centroid are set
        assert np.any(image[3, 3] != 0)