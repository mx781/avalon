from typing import Optional

import attr
import numpy as np

from common.utils import to_immutable_array
from datagen.world_creation.entities.item import InstancedDynamicItem
from datagen.world_creation.types import MapBoolNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Tool(InstancedDynamicItem):
    position: np.ndarray = attr.ib(
        default=np.array([0.0, 0.0, 0.0]), converter=to_immutable_array, eq=attr.cmp_using(eq=np.array_equal)
    )
    solution_mask: Optional[MapBoolNP] = attr.ib(default=None, eq=False)
