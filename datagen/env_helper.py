from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import attr
import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange

from agent.godot.godot_gym import create_base_benchmark_config
from common.errors import SwitchError
from common.visual_utils import visualize_tensor_as_video
from datagen.godot_env import AttrsAction
from datagen.godot_env import AttrsObservation
from datagen.godot_env import AvalonObservationType
from datagen.godot_env import GoalEvaluator
from datagen.godot_env import GoalProgressResult
from datagen.godot_env import GodotEnv
from datagen.godot_env import MouseKeyboardActionType
from datagen.godot_env import VRActionType
from datagen.godot_generated_types import AgentPlayerSpec
from datagen.godot_generated_types import AvalonSimSpec
from datagen.godot_generated_types import MouseKeyboardAgentPlayerSpec
from datagen.godot_generated_types import MouseKeyboardHumanPlayerSpec
from datagen.godot_generated_types import VRAgentPlayerSpec
from datagen.godot_generated_types import VRHumanPlayerSpec
from datagen.world_creation.world_generator import GenerateWorldParams


def create_mouse_keyboard_benchmark_config() -> AvalonSimSpec:
    with create_base_benchmark_config().mutable_clone() as config:
        assert isinstance(config.player, AgentPlayerSpec)
        config.player = MouseKeyboardAgentPlayerSpec.from_dict(config.player.to_dict())
        return config


def create_vr_benchmark_config() -> AvalonSimSpec:
    with create_base_benchmark_config().mutable_clone() as config:
        assert isinstance(config.player, AgentPlayerSpec)
        config.player = VRAgentPlayerSpec.from_dict(config.player.to_dict())
        return config


def get_null_mouse_keyboard_action() -> AttrsAction:
    return MouseKeyboardActionType(
        head_x=0.0,
        head_z=0.0,
        head_pitch=0.0,
        head_yaw=0.0,
        is_left_hand_grasping=0.0,
        is_right_hand_grasping=0.0,
        is_left_hand_throwing=0.0,
        is_right_hand_throwing=0.0,
        is_jumping=0.0,
        is_eating=0.0,
        is_crouching=0.0,
    )


def get_null_vr_action() -> AttrsAction:
    return VRActionType(
        head_x=0.0,
        head_y=0.0,
        head_z=0.0,
        head_pitch=0.0,
        head_yaw=0.0,
        head_roll=0.0,
        left_hand_x=0.0,
        left_hand_y=0.0,
        left_hand_z=0.0,
        left_hand_pitch=0.0,
        left_hand_yaw=0.0,
        left_hand_roll=0.0,
        is_left_hand_grasping=0.0,
        right_hand_x=0.0,
        right_hand_y=0.0,
        right_hand_z=0.0,
        right_hand_pitch=0.0,
        right_hand_yaw=0.0,
        right_hand_roll=0.0,
        is_right_hand_grasping=0.0,
        is_jumping=0.0,
    )


def create_env(
    config: AvalonSimSpec,
    action_type: Type[AttrsAction],
    observation_type: Type[AttrsObservation] = AvalonObservationType,
) -> GodotEnv:
    return GodotEnv(
        config=config,
        observation_type=observation_type,
        action_type=action_type,
        goal_evaluator=NullGoalEvaluator(),
        gpu_id=0,
    )


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class NullGoalEvaluator(GoalEvaluator[AvalonObservationType]):
    def calculate_goal_progress(self, observation: AvalonObservationType) -> GoalProgressResult:
        return GoalProgressResult(reward=0, is_done=False, log={})

    def reset(self, observation: AvalonObservationType, world_params: Optional[GenerateWorldParams] = None) -> None:
        pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class PlaybackGoalEvaluator(GoalEvaluator[AvalonObservationType]):
    def calculate_goal_progress(self, observation: AvalonObservationType) -> GoalProgressResult:
        return GoalProgressResult(reward=0, is_done=False, log={})

    def reset(self, observation: AvalonObservationType, world_params: Optional[GenerateWorldParams] = None) -> None:
        pass


def rgbd_to_video_tensor(rgbd_data: Iterable[npt.NDArray]) -> torch.Tensor:
    return torch.stack(
        [
            rearrange(
                torch.flipud(torch.tensor(rgbd[:, :, :3])),
                "h w c -> c h w",
            )
            / 255.0
            for rgbd in rgbd_data
        ]
    )


def observation_video_tensor(data: List[AvalonObservationType]) -> torch.Tensor:
    return rgbd_to_video_tensor(x.rgbd for x in data)


def observation_video(data: Union[List[AvalonObservationType], np.ndarray]) -> torch.Tensor:
    stack = [x.rgb for x in data] if isinstance(data, list) else data
    return torch.stack([rearrange(torch.tensor(x), "h w c -> c h w") / 255.0 for x in stack])


def display_video(data: List[AvalonObservationType], size: Optional[Tuple[int, int]] = None) -> None:
    if size is None:
        size = (512, 512)
    tensor = observation_video_tensor(data)
    visualize_tensor_as_video(tensor, normalize=False, size=size)


def better_display_video(data: List[npt.NDArray[np.uint8]], size: Optional[Tuple[int, int]] = None) -> None:
    if size is None:
        size = (512, 512)
    visualize_tensor_as_video(rgbd_to_video_tensor(data), normalize=False, size=size)


def get_action_type_from_config(config: AvalonSimSpec) -> Type[AttrsAction]:
    if isinstance(config.player, MouseKeyboardAgentPlayerSpec) or isinstance(
        config.player, MouseKeyboardHumanPlayerSpec
    ):
        return MouseKeyboardActionType
    elif isinstance(config.player, VRAgentPlayerSpec) or isinstance(config.player, VRHumanPlayerSpec):
        return VRActionType
    else:
        raise SwitchError(config.player)


def visualize_worlds_in_folder(world_paths: Iterable[Path], resolution=1024, num_frames=20):
    episode_seed = 0
    config = create_vr_benchmark_config()

    with config.mutable_clone() as config:
        config.recording_options.resolution_x = resolution
        config.recording_options.resolution_y = resolution
    action_type = VRActionType
    env = create_env(config, action_type)

    all_observations = []
    # if we want to take a few actions
    # null_action = get_null_vr_action()
    worlds_to_sort = []
    for world_path in world_paths:
        task, seed_str, difficulty_str = world_path.name.split("__")
        difficulty = float(difficulty_str.replace("_", "."))
        seed = int(seed_str)
        worlds_to_sort.append((task, difficulty, seed, world_path))

    for (task, difficulty, seed, world_path) in sorted(worlds_to_sort):
        print(f"Loading {world_path}")
        world_file = world_path / "main.tscn"
        observations = []
        observations.append(
            env.reset_nicely_with_specific_world(
                episode_seed=episode_seed,
                world_path=str(world_file),
            )
        )
        for i in range(num_frames):
            null_action = get_null_vr_action()
            obs, _ = env.act(null_action)
            observations.append(obs)

        all_observations.append(observations)

    return all_observations
