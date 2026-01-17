import numpy as np
import cv2


def draw_circle(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    thickness: int = -1,
) -> None:
    cv2.circle(img, center, radius, color, thickness)


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
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.rectangle(img, top_left, bottom_right, color, thickness)


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
