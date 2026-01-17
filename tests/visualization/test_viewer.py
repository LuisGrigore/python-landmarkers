import numpy as np
import pytest
from unittest.mock import Mock
from src.visualization.viewer import Viewer, ViewerBuilder
from src.visualization.layers.layer_base import Layer


class MockLayer(Layer):
    def __init__(self, color=(255, 0, 0)):
        self.color = color

    def draw(self, data, image):
        # Simple mock: set a pixel
        if image.size > 0:
            image[0, 0] = self.color


class TestViewer:
    def test_init_with_image_size(self):
        layers = [MockLayer()]
        viewer = Viewer(layers, image_size=(100, 100))
        assert viewer._image_size == (100, 100)

    def test_render_with_background(self):
        layers = [MockLayer((0, 255, 0))]
        viewer = Viewer(layers)
        background = np.zeros((50, 50, 3), dtype=np.uint8)
        result = viewer.render(None, background)
        assert result.shape == (50, 50, 3)
        # Check that layer was applied
        assert np.array_equal(result[0, 0], [0, 255, 0])

    def test_render_without_background(self):
        layers = [MockLayer()]
        viewer = Viewer(layers, image_size=(50, 50))
        result = viewer.render(None)
        assert result.shape == (50, 50, 3)

    def test_render_without_background_no_size(self):
        layers = [MockLayer()]
        viewer = Viewer(layers)
        with pytest.raises(ValueError, match="image_size required"):
            viewer.render(None)


class TestViewerBuilder:
    def test_init(self):
        builder = ViewerBuilder()
        assert builder._layers == []
        assert builder._image_size is None

    def test_add_layer(self):
        builder = ViewerBuilder()
        layer = MockLayer()
        result = builder.add(layer)
        assert result is builder
        assert builder._layers == [layer]

    def test_image_size(self):
        builder = ViewerBuilder()
        result = builder.image_size(100, 200)
        assert result is builder
        assert builder._image_size == (100, 200)

    def test_build(self):
        builder = ViewerBuilder()
        layer = MockLayer()
        builder.add(layer).image_size(50, 50)
        viewer = builder.build()
        assert isinstance(viewer, Viewer)
        assert viewer._layers == [layer]
        assert viewer._image_size == (50, 50)