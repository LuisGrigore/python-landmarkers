import numpy as np
from unittest.mock import Mock
from src.visualization.layers.points_layer import PointsLayer, HasPoints


class MockHasPoints(HasPoints):
    def __init__(self, points):
        self._points = points

    def points(self):
        return self._points


class TestPointsLayer:
    def test_init(self):
        layer = PointsLayer(color=(0, 255, 0), radius=5)
        assert layer.color == (0, 255, 0)
        assert layer.radius == 5

    def test_draw(self):
        layer = PointsLayer(color=(255, 0, 0), radius=1)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        points = np.array([[5.0, 5.0], [2.0, 2.0]])
        data = MockHasPoints(points)
        layer.draw(data, image)
        # Check that pixels around the points are set
        assert np.any(image[5, 5] != 0)
        assert np.any(image[2, 2] != 0)