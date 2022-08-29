from pathlib import Path
from typing import Optional

import attr
import numpy as np

from datagen.world_creation.configs.export import ExportConfig
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.worlds.compositional import CompositeTaskConfig
from datagen.world_creation.worlds.compositional import ForcedComposition
from datagen.world_creation.worlds.compositional import create_compositional_task
from datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class NavigateTaskConfig(CompositeTaskConfig):
    task: AvalonTask = AvalonTask.NAVIGATE


def generate_navigate_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    _FORCED: Optional[ForcedComposition] = None,
    task_config: NavigateTaskConfig = NavigateTaskConfig(),
):
    world, locations = create_compositional_task(rand, difficulty, task_config, export_config, _FORCED=_FORCED)
    export_world(output_path, rand, world)
