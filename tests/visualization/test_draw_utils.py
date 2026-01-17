import numpy as np
import cv2
from src.visualization.draw_utils import draw_circle, draw_line, draw_rectangle, draw_text


class TestDrawCircle:
    def test_draw_circle(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_circle(img, (50, 50), 10, (255, 0, 0))
        # Check that some pixels are set
        assert np.any(img != 0)


class TestDrawLine:
    def test_draw_line(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_line(img, (0, 0), (99, 99), (0, 255, 0))
        assert np.any(img != 0)


class TestDrawRectangle:
    def test_draw_rectangle(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_rectangle(img, (10, 10), (90, 90), (0, 0, 255))
        assert np.any(img != 0)


class TestDrawText:
    def test_draw_text(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_text(img, "Test", (10, 50), (255, 255, 255))
        assert np.any(img != 0)