from typing import Annotated, Literal, NamedTuple
from numpy.typing import NDArray
import numpy as np

class WorldCoordinate3D(NamedTuple):
    x: float
    y: float
    z: float


class NormalizedCoordinate3D(NamedTuple):
    x: float
    y: float
    z: float


class PixelCoordinate3D(NamedTuple):
    x: int
    y: int
    z: int


class WorldCoordinate2D(NamedTuple):
    x: float
    y: float


class NormalizedCoordinate2D(NamedTuple):
    x: float
    y: float


class PixelCoordinate2D(NamedTuple):
    x: int
    y: int


class RgbColor(NamedTuple):
    red: int
    green: int
    blue: int
    
    
LandmarkArray = Annotated[NDArray[np.float64], Literal[("n_points", 3)]]
LandmarkSequenceArray = Annotated[NDArray[np.float64], Literal[("n_sequence_elements","n_points", 3)]]