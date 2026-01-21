import numpy as np
import cv2

from ..types import PixelCoordinate2D, RgbColor


def draw_circle(
    img: np.ndarray,
    center: PixelCoordinate2D,
    radius: int,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    cv2.circle(img, (center.x, center.y), radius, color, thickness)


def draw_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.line(img, p1, p2, color, thickness)



def draw_rectangle(
    img: np.ndarray,
    top_left: PixelCoordinate2D,
    bottom_right: PixelCoordinate2D,
    color: RgbColor,
    thickness: int = 1,
) -> None:
    """
    Draw a rectangle defined by its top-left and bottom-right corners in pixel coordinates.

    Args:
        img: The target image (numpy ndarray).
        top_left: Top-left corner in pixel coordinates.
        bottom_right: Bottom-right corner in pixel coordinates.
        color: Rectangle color in BGR format.
        thickness: Thickness of the rectangle border.
    """
    cv2.rectangle(img, (top_left.x, top_left.y), (bottom_right.x, bottom_right.y), color, thickness)



def draw_text(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    cv2.putText(
        img,
        text,
        origin,
        font,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )
