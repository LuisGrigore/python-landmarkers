import numpy as np
from unittest.mock import Mock
from src.visualization.layers.sequence_layer import SequenceLayer, HasSequence


class MockHasSequence(HasSequence):
    def __init__(self, sequence_elements, timestamps):
        self._sequence = sequence_elements
        self._timestamps = timestamps

    def sequence(self):
        return iter(self._sequence)

    def time_stamps_ms(self):
        return self._timestamps


class MockSequenceElement:
    def __init__(self, id):
        self.id = id


class MockLayer:
    def __init__(self):
        self.draw_calls = []

    def draw(self, data, image):
        self.draw_calls.append((data, image.copy()))


class TestSequenceLayer:
    def test_init(self):
        layers = [MockLayer()]
        layer = SequenceLayer(layers, step=2, time_fade=True)
        assert layer._layers == layers
        assert layer._step == 2
        assert layer._time_fade == True

    def test_draw_no_time_fade(self):
        mock_layer = MockLayer()
        layers = [mock_layer]
        layer = SequenceLayer(layers, step=1, time_fade=False)

        elements = [MockSequenceElement(0), MockSequenceElement(1)]
        timestamps = [100, 200]
        data = MockHasSequence(elements, timestamps)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        layer.draw(data, image)

        assert len(mock_layer.draw_calls) == 2
        assert mock_layer.draw_calls[0][0] == elements[0]
        assert mock_layer.draw_calls[1][0] == elements[1]

    def test_draw_with_time_fade(self):
        mock_layer = MockLayer()
        layers = [mock_layer]
        layer = SequenceLayer(layers, step=1, time_fade=True, time_scale_ms=1000.0)

        elements = [MockSequenceElement(0)]
        timestamps = [100]
        data = MockHasSequence(elements, timestamps)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        layer.draw(data, image)

        assert len(mock_layer.draw_calls) == 1

    def test_draw_step_filtering(self):
        mock_layer = MockLayer()
        layers = [mock_layer]
        layer = SequenceLayer(layers, step=2, time_fade=False)

        elements = [MockSequenceElement(0), MockSequenceElement(1), MockSequenceElement(2)]
        timestamps = [100, 200, 300]
        data = MockHasSequence(elements, timestamps)

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        layer.draw(data, image)

        # Should draw every 2nd element: 0 and 2
        assert len(mock_layer.draw_calls) == 2
        assert mock_layer.draw_calls[0][0] == elements[0]
        assert mock_layer.draw_calls[1][0] == elements[2]