from enum import Enum
from typing import Dict

import numpy as np

from datagen.world_creation.geometry import Axis
from datagen.world_creation.types import TileConvKernel

TILE_SIZE = 1
DEFAULT_STORY_HEIGHT = 5  # does not include floor thickness
DEFAULT_FLOOR_THICKNESS = 1
SLOPE_THICKNESS = 0.1
LADDER_THICKNESS = 0.1
CEILING_THICKNESS = 1
HIGH_POLY = "high_poly"
LOW_POLY = "low_poly"

# Rooms need at least 3 tiles to place a ladder and 1 tile on each side to have a wall (so 5 tiles for a building)
# We could have smaller one-story buildings, but it's nicer to keep things consistent and extensible
MIN_ROOM_SIZE = 3
MIN_BUILDING_SIZE = 5


class ExportMode(Enum):
    PRETTY = "PRETTY"
    DEBUG = "DEBUG"


EXPORT_MODE = ExportMode.PRETTY


class TileIdentity(Enum):
    FULL = 0
    ROOM = -1
    HALLWAY = -2
    LINK = -3
    LINK_BOTTOM_LANDING = -4
    LINK_TOP_LANDING = -5
    VOID = -6

    @property
    def pretty_name(self):
        return " ".join([section.lower() for section in self.name.split("_")])


class CornerType(Enum):
    NE = "NE"
    SE = "SE"
    SW = "SW"
    NW = "NW"

    @property
    def convolution_kernel(self) -> TileConvKernel:
        kernels_by_outside_corner: Dict[CornerType, TileConvKernel] = {
            CornerType.SW: np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
            CornerType.SE: np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
            CornerType.NW: np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]]),
            CornerType.NE: np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        }
        return kernels_by_outside_corner[self]


class FootprintType(Enum):
    RECTANGLE = "RECTANGLE"
    L_SHAPE = "L_SHAPE"
    T_SHAPE = "T_SHAPE"
    IRREGULAR = "IRREGULAR"


class Orientation(Enum):
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"

    def other(self):
        return Orientation.HORIZONTAL if self == Orientation.VERTICAL else Orientation.VERTICAL


class Azimuth(Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"

    @property
    def convolution_kernel(self) -> TileConvKernel:
        kernels_by_azimuth = {
            Azimuth.NORTH: np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            Azimuth.EAST: np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
            Azimuth.SOUTH: np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
            Azimuth.WEST: np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        }
        return kernels_by_azimuth.get(self)

    @property
    def aligned_axis(self) -> Axis:
        if self in {Azimuth.NORTH, Azimuth.SOUTH}:
            return Axis.X
        else:
            return Axis.Z

    @property
    def angle_from_positive_x(self):
        if self == Azimuth.NORTH:
            return -90
        elif self == Azimuth.EAST:
            return 0
        elif self == Azimuth.SOUTH:
            return 90
        elif self == Azimuth.WEST:
            return 180

    @property
    def opposite(self):
        if self == Azimuth.NORTH:
            return Azimuth.SOUTH
        elif self == Azimuth.EAST:
            return Azimuth.WEST
        elif self == Azimuth.SOUTH:
            return Azimuth.NORTH
        elif self == Azimuth.WEST:
            return Azimuth.EAST


class WallType(Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"

    @property
    def azimuth(self):
        return Azimuth(self.value)

    @property
    def is_vertical(self) -> bool:
        return self in {WallType.EAST, WallType.WEST}

    @property
    def convolution_kernel(self) -> TileConvKernel:
        kernels_by_wall = {
            WallType.NORTH: np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]),
            WallType.EAST: np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
            WallType.SOUTH: np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
            WallType.WEST: np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
        }
        return kernels_by_wall[self]

    @property
    def corner_types(self):
        corner_types_by_wall_type = {
            WallType.NORTH: (CornerType.NW, CornerType.NE),
            WallType.EAST: (CornerType.NE, CornerType.SE),
            WallType.SOUTH: (CornerType.SE, CornerType.SW),
            WallType.WEST: (CornerType.NW, CornerType.SW),
        }
        return corner_types_by_wall_type[self]
