from typing import Optional

import attr

from datagen.world_creation.indoor.builders import EntranceBuilder
from datagen.world_creation.indoor.builders import FootprintBuilder
from datagen.world_creation.indoor.builders import HallwayBuilder
from datagen.world_creation.indoor.builders import RoomBuilder
from datagen.world_creation.indoor.builders import StoryLinker
from datagen.world_creation.indoor.builders import WindowBuilder
from datagen.world_creation.indoor.building import BuildingAestheticsConfig


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BuildingConfig:
    width: int  # x
    length: int  # z
    story_count: int
    footprint_builder: FootprintBuilder
    room_builder: RoomBuilder
    hallway_builder: HallwayBuilder
    story_linker: Optional[StoryLinker] = None
    entrance_builder: Optional[EntranceBuilder] = None
    window_builder: Optional[WindowBuilder] = None
    is_climbable: bool = True
    aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig()


def generate_aesthetics_config() -> BuildingAestheticsConfig:
    return BuildingAestheticsConfig()
