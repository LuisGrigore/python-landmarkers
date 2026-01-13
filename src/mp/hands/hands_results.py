from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np





HandednessLabel = Literal["Left", "Right"]


@runtime_checkable
class Category(Protocol):
	index: int
	score: float
	category_name: HandednessLabel


@runtime_checkable
class Landmark(Protocol):
	x: float
	y: float
	z: float



@runtime_checkable
class HandLandmarkerResultProtocol(Protocol):
	handedness: Sequence[Category]
	hand_landmarks: Sequence[Sequence[Landmark]]
	hand_world_landmarks: Sequence[Sequence[Landmark]]


def landmark_list_to_ndarray(
    landmarks: Sequence[Landmark],
) -> np.ndarray:
    return np.asarray(
        [(lm.x, lm.y, lm.z) for lm in landmarks],
    )