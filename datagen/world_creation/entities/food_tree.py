from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Tuple
from typing import TypeVar

import attr
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from scipy.spatial.transform import Rotation

from datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from datagen.world_creation.entities.constants import FOOD_TREE_BASE_HEIGHT
from datagen.world_creation.entities.item import InstancedDynamicItem
from datagen.world_creation.entities.utils import facing_2d
from datagen.world_creation.entities.utils import get_random_ground_points
from datagen.world_creation.entities.utils import normalized
from datagen.world_creation.types import GodotScene
from datagen.world_creation.types import MapBoolNP
from datagen.world_creation.types import Point3DListNP
from datagen.world_creation.types import Point3DNP
from datagen.world_creation.worlds.height_map import HeightMap

_FoodT = TypeVar("_FoodT", bound="Food")
GetHeightAt = Callable[[Tuple[float, float]], float]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FoodTreeBase(InstancedDynamicItem):
    resource_file: str = "res://scenery/tree_base.tscn"
    rotation: Rotation = Rotation.identity()

    def get_offset(self) -> float:
        return -0.25


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FoodTree(InstancedDynamicItem):
    ROTATION_TO_SUITABLE_OFFSET: ClassVar = {
        (0,): (1.25, CANONICAL_FOOD_HEIGHT_ON_TREE),
        # TODO add mechanism for multiple rotations
        # (45,): (0.875, 3.25),
    }

    @property
    def food_offset(self):
        return self.ROTATION_TO_SUITABLE_OFFSET[(0,)]

    def _relative_position(self, offset: np.array) -> np.ndarray:
        return self.rotation.apply(offset) * self.scale

    def get_food_height(self, food: _FoodT) -> float:
        xz, y = self.food_offset
        return y * self.scale[1]

    resource_file: str = "res://scenery/trees/fruit_tree_normal.tscn"
    is_tool_helpful: bool = True
    scale: np.ndarray = np.array([1.0, 1.0, 1.0])
    rotation: Rotation = Rotation.identity()

    is_food_on_tree: bool = False

    @property
    def height(self):
        scale_y = self.scale[1]
        return FOOD_TREE_BASE_HEIGHT * scale_y

    @staticmethod
    def build(tree_height: float, is_food_on_tree: bool = False) -> "FoodTree":
        scale_factor = tree_height / FOOD_TREE_BASE_HEIGHT
        return FoodTree(
            scale=scale_factor * np.array([1.0, 1.0, 1.0]),
            position=np.array([0.0, 0.0, 0.0]),
            is_food_on_tree=is_food_on_tree,
        )

    def _resolve_trunk_placement(
        self, spawn: Point3DNP, primary_food: Point3DNP, get_height_at: GetHeightAt
    ) -> Tuple[Point3DNP, float]:
        goal_vector: Point3DNP = primary_food - spawn
        goal_vector[1] = 0
        goal_vector = normalized(goal_vector)
        if self.is_food_on_tree:
            xz_offset, y_offset = self.food_offset
            away_from_spawn_and_down_from_food = goal_vector * xz_offset
            away_from_spawn_and_down_from_food[1] = -y_offset
            away_from_spawn_and_down_from_food *= self.scale
            position = primary_food + away_from_spawn_and_down_from_food
            return position, get_height_at((position[0], position[2]))

        # TODO: could use a bit more variety in how the tree is place relative to the food
        x, _, z = primary_food + 2.0 * goal_vector
        ground_level = get_height_at((x, z))
        ground_level_behind_food = np.array([x, ground_level, z])
        return ground_level_behind_food, ground_level

    def place(
        self, spawn: Point3DNP, primary_food: Point3DNP, get_height_at: GetHeightAt
    ) -> Tuple["FoodTree", Optional[FoodTreeBase]]:
        trunk_placement, ground_height = self._resolve_trunk_placement(spawn, primary_food, get_height_at)
        rotation = facing_2d(trunk_placement, spawn)
        root_affordance = 0.25 * self.scale[1]
        is_boulder_needed = self.is_food_on_tree and ground_height < (trunk_placement[1] - root_affordance)
        tree = attr.evolve(self, position=trunk_placement, rotation=rotation)
        if not is_boulder_needed:
            return (tree, None)

        return tree, FoodTreeBase(position=trunk_placement, rotation=rotation)

    def get_food_locations(
        self,
        rand: np.random.Generator,
        center: Point3DNP,
        count: int,
        map: HeightMap,
        min_radius: float,
        max_radius: float,
        offset: float,
        island_mask: MapBoolNP,
    ) -> Point3DListNP:
        # we originally intended that some food could sometimes be at different places on the tree and some on the
        # ground, but didn't get to finish this feature before the deadline
        ground_count = count
        ground_points = get_random_ground_points(
            rand, center, ground_count, map, min_radius, max_radius, offset, island_mask
        )
        tree_point_count = count - ground_count
        if tree_point_count == 0:
            return ground_points
        random_heights = rand.uniform(0.5, 0.9, size=tree_point_count) * self.height
        zeros = np.zeros_like(random_heights)
        height_vectors = np.stack([zeros, random_heights, zeros])
        tree_points = self.position + height_vectors
        return tree_points

    def get_offset(self) -> float:
        return 0.0

    def get_node(self, scene: GodotScene) -> GDNode:
        node = super().get_node(scene)
        node.properties["scale"] = GDObject("Vector3", *self.scale)
        return node
