from typing import List

import attr
import numpy as np

from datagen.world_creation.constants import IDENTITY_BASIS
from datagen.world_creation.entities.doors.locks.door_lock import DoorLock
from datagen.world_creation.entities.item import Item
from datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Door(Item):
    entity_id: int = -1

    # for doors, position is the door's centroid - the middle point of the plane we're trying to close using the door
    is_dynamic: bool = True
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = IDENTITY_BASIS
    locks: List[DoorLock] = attr.field(default=attr.Factory(list))
