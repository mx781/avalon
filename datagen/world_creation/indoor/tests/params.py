from itertools import permutations
from itertools import product
from typing import Dict
from typing import Tuple

import numpy as np

from common.utils import float_to_str
from datagen.world_creation.configs.building import BuildingConfig
from datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from datagen.world_creation.indoor.builders import DefaultStoryLinker
from datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from datagen.world_creation.indoor.builders import IrregularFootprintBuilder
from datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from datagen.world_creation.indoor.builders import TLShapeFootprintBuilder
from datagen.world_creation.indoor.builders import WindowBuilder
from datagen.world_creation.indoor.building import BuildingTask
from datagen.world_creation.types import Point3DNP
from datagen.world_creation.worlds.compositional import get_radius_for_building_task

CANONICAL_SEED = 42
BuildingCatalog = Dict[str, BuildingConfig]
IndoorWorldParams = Tuple[np.random.Generator, float, BuildingTask, float, Point3DNP]
IndoorWorldCatalog = Dict[str, IndoorWorldParams]


def canonical_rand() -> np.random.Generator:
    return np.random.default_rng(CANONICAL_SEED)


def _create_building_catalogs() -> Tuple[BuildingCatalog, BuildingCatalog]:
    size_options = {
        "tiny": dict(width=5, length=5),
        "small": dict(width=8, length=8),
        "regular": dict(width=16, length=16),
        "narrow": dict(width=5, length=17),
        "wide": dict(width=16, length=6),
        "large": dict(width=32, length=32),
    }

    height_options = {
        "1": dict(story_count=1),
        "2_decorative": dict(story_count=2, story_linker=None),
        "2_functional": dict(story_count=2, story_linker=DefaultStoryLinker()),
        "5_decorative": dict(story_count=5, story_linker=None),
        "5_functional": dict(story_count=5, story_linker=DefaultStoryLinker()),
    }
    footprint_options = {
        "rectangle": dict(footprint_builder=RectangleFootprintBuilder()),
        "tl_shape": dict(footprint_builder=TLShapeFootprintBuilder()),
        "irregular": dict(footprint_builder=IrregularFootprintBuilder()),
    }

    other_params = dict(
        room_builder=HouseLikeRoomBuilder(), hallway_builder=DefaultHallwayBuilder(), window_builder=WindowBuilder()
    )

    incompatible_combos = {
        ("tl_shape", "tiny"),
        ("irregular", "tiny"),
    }

    valid_building_catalog = {}
    invalid_building_catalog = {}
    for (size, size_params), (height, height_params), (footprint, footprint_params), rest in product(
        size_options.items(), height_options.items(), footprint_options.items(), [other_params]
    ):
        param_type_permutations = {
            *permutations([size, height, footprint], 2),
            *permutations([size, height, footprint], 3),
        }
        is_combination_valid = len(param_type_permutations.intersection(incompatible_combos)) == 0
        params_id = "__".join([size, height, footprint])
        params = {**size_params, **height_params, **footprint_params, **other_params}
        if is_combination_valid:
            valid_building_catalog[params_id] = BuildingConfig(**params)
        else:
            invalid_building_catalog[params_id] = BuildingConfig(**params)
    return valid_building_catalog, invalid_building_catalog


def _create_indoor_world_catalog() -> IndoorWorldCatalog:
    indoor_tasks = list(BuildingTask)
    indoor_task_difficulties = [round(d, 1) for d in np.arange(0.0, 1.2, 0.2)]
    indoor_world_catalog: IndoorWorldCatalog = {}

    param_rand = canonical_rand()
    for task, difficulty in product(indoor_tasks, indoor_task_difficulties):
        seed = param_rand.integers(0, np.iinfo(np.int64).max)
        task_rand = np.random.default_rng(seed)

        radius = get_radius_for_building_task(task_rand, task, difficulty)
        location = task_rand.uniform(-100, 100, (3,))
        params = task_rand, difficulty, task, radius, location
        world_name = "__".join(["indoor_world", task.value.lower(), float_to_str(difficulty)])
        indoor_world_catalog[world_name] = params
    return indoor_world_catalog


VALID_BUILDING_CATALOG, INVALID_BUILDING_CATALOG = _create_building_catalogs()
BUILDING_CATALOG = {**VALID_BUILDING_CATALOG, **INVALID_BUILDING_CATALOG}
INDOOR_WORLD_CATALOG = _create_indoor_world_catalog()
