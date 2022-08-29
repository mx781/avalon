from typing import Optional
from typing import Tuple
from typing import Type

import attr
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode

from common.utils import to_immutable_array
from datagen.world_creation.constants import IDENTITY_BASIS
from datagen.world_creation.entities.food_tree import _FoodT
from datagen.world_creation.entities.item import InstancedDynamicItem
from datagen.world_creation.entities.tools.tool import Tool
from datagen.world_creation.entities.tools.weapons import LargeRock
from datagen.world_creation.entities.tools.weapons import Rock
from datagen.world_creation.entities.tools.weapons import Stick
from datagen.world_creation.types import GodotScene
from datagen.world_creation.utils import scale_basis


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Food(InstancedDynamicItem):
    mass: float = 1.0
    resource_file: str = "res://items/food.tscn"
    energy: float = 1.0
    plant_path: Optional[str] = None

    position: np.ndarray = attr.ib(
        default=np.array([0.0, 0.0, 0.0]), converter=to_immutable_array, eq=attr.cmp_using(eq=np.array_equal)
    )
    scale: np.ndarray = np.array([1.0, 1.0, 1.0])
    is_grown_on_trees: bool = True
    is_found_on_ground: bool = True
    is_openable: bool = False
    count_distribution: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((0.0, 1.0), (1.0, 1.7))

    @property
    def additional_offset(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    def get_stem_joint(self, food_name: str) -> GDNode:
        basis = scale_basis(IDENTITY_BASIS, self.scale)
        point_five_over_parent = GDObject("Transform", *basis, 0, 0.5 * self.scale[1], 0)
        return GDNode(
            f"{food_name}_stem_joint",
            type="Generic6DOFJoint",
            properties={
                "transform": point_five_over_parent,
                "nodes/node_a": f"../../{food_name}",
                "nodes/node_b": f"../../{self.plant_path}",
                "linear_limit_x/upper_distance": 0.5,
                "linear_limit_y/upper_distance": 0.0,
                "linear_limit_z/upper_distance": 0.5,
            },
        )

    def attached_to(self: _FoodT, tree: "FoodTree") -> _FoodT:
        assert tree.entity_id > -1, f"Food must be attached_to the result of world.add_item({tree})"
        assert tree.is_food_on_tree, f"Food shouldn't be attached_too tree {tree} without is_food_on_tree=True"
        return attr.evolve(self, plant_path=tree.node_name)

    def get_node(self, scene: GodotScene) -> GDNode:
        food_node = super().get_node(scene)
        food_node.properties["energy"] = self.energy

        if self.plant_path is None:
            return food_node

        food_node.add_child(self.get_stem_joint(food_node.name))
        return food_node

    def get_count(self, rand: np.random.Generator) -> int:
        return round(np.interp(rand.uniform(), self.count_distribution[0], self.count_distribution[1]))

    def get_opened_version(self) -> "Food":
        if not self.is_openable:
            return self
        raise NotImplementedError("Openable food must implement get_opened_version")

    def get_tool_options(self) -> Tuple[float, Tuple[Type[Tool], ...]]:
        return 0, tuple()

    @property
    def is_tool_required(self) -> bool:
        return self.get_tool_options()[0] == 1

    @property
    def is_tool_useful(self) -> bool:
        return self.get_tool_options()[0] > 0

    def is_always_multiple(self):
        return False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Banana(Food):
    resource_file: str = "res://items/food/banana.tscn"
    energy: float = 0.5

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, 0.3, 0.0])
        return super().additional_offset


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Apple(Food):
    resource_file: str = "res://items/food/apple.tscn"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Fig(Food):
    resource_file: str = "res://items/food/fig.tscn"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class OpenableFood(Food):
    is_openable: bool = True

    @property
    def is_open(self):
        return not self.is_openable

    def get_opened_version(self) -> "Food":
        if self.is_open:
            return self
        with self.mutable_clone() as opened:
            opened.resource_file = opened.resource_file.replace(".tscn", "_open.tscn")
            opened.is_openable = False
            return opened


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Orange(OpenableFood):
    resource_file: str = "res://items/food/orange.tscn"

    def get_tool_options(self):
        return 0.1, (Rock, Stick, LargeRock)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Avocado(OpenableFood):
    resource_file: str = "res://items/food/avocado.tscn"

    def get_tool_options(self):
        return 1, (Rock, LargeRock)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Coconut(OpenableFood):
    resource_file: str = "res://items/food/coconut.tscn"

    def get_tool_options(self):
        return 0.25, (Rock, Stick, LargeRock)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Honeycomb(Food):
    resource_file: str = "res://items/food/honeycomb.tscn"
    is_found_on_ground: bool = False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Cherry(Food):
    resource_file: str = "res://items/food/cherry.tscn"

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, -0.5, 0.0])
        return super().additional_offset

    def is_always_multiple(self):
        return True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Mulberry(Food):
    resource_file: str = "res://items/food/mulberry.tscn"

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, 0.3, 0.0])
        return super().additional_offset


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Carrot(Food):
    is_grown_on_trees: bool = False
    resource_file: str = "res://items/food/carrot.tscn"

    def get_offset(self) -> float:
        return 0.0


FOODS = (
    Apple(),
    Banana(),
    Cherry(),
    Honeycomb(),
    Mulberry(),
    Fig(),
    Orange(),
    Avocado(),
    Coconut(),
    Carrot(),
)
NON_TREE_FOODS = [x for x in FOODS if not x.is_grown_on_trees]
CANONICAL_FOOD = FOODS[0]
CANONICAL_FOOD_CLASS = FOODS[0].__class__
CANONICAL_FOOD_HEIGHT = CANONICAL_FOOD.get_offset()
