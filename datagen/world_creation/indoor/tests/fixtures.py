import json
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import TypedDict

import numpy as np
from loguru import logger

from datagen.world_creation.configs.building import BuildingConfig
from datagen.world_creation.indoor.building import Building
from datagen.world_creation.indoor.tests.params import CANONICAL_SEED
from datagen.world_creation.indoor.tests.params import INDOOR_WORLD_CATALOG
from datagen.world_creation.indoor.tests.params import INVALID_BUILDING_CATALOG
from datagen.world_creation.indoor.tests.params import VALID_BUILDING_CATALOG
from datagen.world_creation.indoor.tests.params import IndoorWorldParams
from datagen.world_creation.worlds.world import build_building
from science.common.testing_utils import RequestFixture
from science.common.testing_utils import fixture
from science.common.testing_utils import use

EMPTY_FILE_CHECKSUM = "d41d8cd98f00b204e9800998ecf8427e"


@fixture
def seed_() -> int:
    return CANONICAL_SEED


@fixture(params=VALID_BUILDING_CATALOG.keys())
def building_catalog_id_(request: RequestFixture) -> str:
    return request.param


@fixture(params=VALID_BUILDING_CATALOG.items())
@use(seed_)
def building_(request: RequestFixture, seed: int) -> Building:
    building_catalog_id, building_config = request.param
    logger.info(f"Generating building {building_catalog_id}")
    rand = np.random.default_rng(seed)
    return build_building(building_config, building_catalog_id, rand)


@fixture(params=INVALID_BUILDING_CATALOG.keys())
def incompatible_building_catalog_id_(request: RequestFixture) -> BuildingConfig:
    return request.param


@fixture(params=INDOOR_WORLD_CATALOG.keys())
def indoor_world_catalog_id_(request: RequestFixture) -> IndoorWorldParams:
    return request.param


class ChecksumManifest(TypedDict):
    snapshot_commit: str
    checksums: Dict[str, str]


def get_current_reference_manifest(reference_name: str, data_path: Optional[Path] = None) -> ChecksumManifest:
    if data_path is None:
        data_path = Path(__file__).parent / "data"
    return json.load(open(data_path / f"{reference_name}_manifest.json", "r"))


@fixture(scope="session")
def buildings_manifest_() -> ChecksumManifest:
    return get_current_reference_manifest("buildings")


@fixture(scope="session")
def indoor_worlds_manifest_() -> Path:
    return get_current_reference_manifest("indoor_worlds")
