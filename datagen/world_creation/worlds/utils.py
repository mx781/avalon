import math
from typing import List
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from datagen.world_creation.entities.tools.tool import Tool


def get_random_position_along_path(
    visible_locations: np.ndarray, start: np.ndarray, end: np.ndarray, difficulty: float, rand: np.random.Generator
) -> np.ndarray:
    """Difficulty scales with how far away the point is from the straight line path between start and end"""
    path_length = np.linalg.norm(start - end)
    desired_distance = rand.uniform() * difficulty * path_length * 2
    target_location_distribution = stats.norm(desired_distance, 0.5)
    start_point = (start[0], start[2])
    end_point = (end[0], end[2])
    location_weights = np.array(
        [
            target_location_distribution.pdf(signed_line_distance((x[0], x[2]), start_point, end_point, path_length))
            for x in visible_locations
        ]
    )
    location_weights /= location_weights.sum()
    return rand.choice(visible_locations, p=location_weights)


def add_offsets(items: List[Tool]):
    new_items = []
    for item in items:
        new_position = item.position.copy().astype(np.float)
        new_position[1] = new_position[1] + item.get_offset()
        new_items.append(attr.evolve(item, position=new_position))
    return new_items


def convenient_signed_line_distance(point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]):
    ab_dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return (((b[0] - a[0]) * (a[1] - point[1])) - ((a[0] - point[0]) * (b[1] - a[1]))) / ab_dist


def signed_line_distance(point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float], ab_dist: float):
    return (((b[0] - a[0]) * (a[1] - point[1])) - ((a[0] - point[0]) * (b[1] - a[1]))) / ab_dist
