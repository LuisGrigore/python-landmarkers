from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MediapipeHandsMetadata():
	index: int
	score: float
	category_name: Literal["Left", "Right"]