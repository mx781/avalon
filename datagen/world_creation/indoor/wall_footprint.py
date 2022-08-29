from typing import NamedTuple
from typing import Tuple

import numpy as np

from datagen.world_creation.indoor.constants import WallType


class WallFootprint(NamedTuple):
    top_left: Tuple[int, int]  # x, z
    bottom_right: Tuple[int, int]  # x, z
    wall_type: WallType
    is_vertical: bool

    @property
    def wall_thickness(self):
        return self.footprint_width if self.is_vertical else self.footprint_length

    @property
    def wall_length(self):
        return self.footprint_length if self.is_vertical else self.footprint_width

    @property
    def footprint_width(self):
        return self.bottom_right[0] - self.top_left[0]

    @property
    def footprint_length(self):
        return self.bottom_right[1] - self.top_left[1]

    @property
    def centroid(self):
        return np.array(self.top_left) + (np.array(self.bottom_right) - np.array(self.top_left)) / 2
