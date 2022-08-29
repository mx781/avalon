import gzip
import json
import os
import shutil
import struct
from collections import defaultdict
from io import BufferedReader
from math import prod
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import attr
import numpy as np
import numpy.typing as npt
from numpy import typing as npt

from common.errors import SwitchError
from datagen.env_helper import create_env
from datagen.env_helper import get_action_type_from_config
from datagen.godot_env import FAKE_TYPE_IMAGE
from datagen.godot_env import NP_DTYPE_MAP
from datagen.godot_env import AttrsAction
from datagen.godot_env import AvalonObservationType
from datagen.godot_env import GodotEnvActionLog
from datagen.godot_env import MouseKeyboardActionType
from datagen.godot_env import VRActionType
from datagen.godot_env import _from_bytes
from datagen.godot_env import _to_bytes
from datagen.godot_generated_types import ACTION_MESSAGE
from datagen.godot_generated_types import CLOSE_MESSAGE
from datagen.godot_generated_types import HUMAN_INPUT_MESSAGE
from datagen.godot_generated_types import QUERY_AVAILABLE_FEATURES_MESSAGE
from datagen.godot_generated_types import RESET_MESSAGE
from datagen.godot_generated_types import SEED_MESSAGE
from datagen.godot_generated_types import SELECT_FEATURES_MESSAGE
from datagen.godot_generated_types import AvalonSimSpec
from datagen.godot_generated_types import MouseKeyboardAgentPlayerSpec
from datagen.godot_generated_types import MouseKeyboardHumanPlayerSpec
from datagen.godot_generated_types import RecordingOptionsSpec
from datagen.godot_generated_types import VRAgentPlayerSpec
from datagen.godot_generated_types import VRHumanPlayerSpec


class PlaybackError(NamedTuple):
    frame: int
    key: str
    recorded_value: npt.NDArray[np.float32]
    playback_value: npt.NDArray[np.float32]
    max_error: float


class PlaybackResult(NamedTuple):
    is_error: bool
    errors: Dict[int, List[PlaybackError]]
    recorded_observations: List[AvalonObservationType]
    playback_observations: List[AvalonObservationType]
    human_inputs: List[AttrsAction]
    actions: List[AttrsAction]


SKIP_KEYS = ["rgb", "depth", "rgbd", "isometric_rgbd", "top_down_rgbd"]
DISCRETE_KEYS = [
    "left_hand_thing_colliding_with_hand",
    "left_hand_held_thing",
    "right_hand_thing_colliding_with_hand",
    "right_hand_held_thing",
    "nearest_food_id",
]


def _write_message(fp, message_type: int, message: bytes):
    fp.write(message_type.to_bytes(1, byteorder="little", signed=False))
    fp.write(message)


def get_replay_log_from_human_recording(
    new_action_record_path: Path,
    action_record_path: Path,
    metadata_path: Path,
    worlds_path: Path,
    selected_features,
    action_type: Type[AttrsAction],
) -> GodotEnvActionLog:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    episode_seed = metadata["episode_seed"]
    world_id = metadata["world_id"]
    world_path = get_world_path_from_world_id(root_path=worlds_path, world_id=world_id)

    with open(action_record_path, "rb") as f:
        content = f.read()

    Path(new_action_record_path).unlink(missing_ok=True)

    # need to manually add some header messages to our replay log
    with open(new_action_record_path, "wb") as f:
        _write_message(f, QUERY_AVAILABLE_FEATURES_MESSAGE, bytes())

        size_doubleword = (len(selected_features)).to_bytes(4, byteorder="little", signed=False)
        feature_names_bytes = ("\n".join(selected_features.keys()) + "\n").encode("UTF-8")
        _write_message(f, SELECT_FEATURES_MESSAGE, size_doubleword + feature_names_bytes)

        seed_message_bytes = episode_seed.to_bytes(8, byteorder="little", signed=True)
        _write_message(f, SEED_MESSAGE, seed_message_bytes)

        reset_message_bytes = (
            action_type.get_null_action().to_bytes()
            + episode_seed.to_bytes(8, byteorder="little", signed=True)
            + (world_path + "\n").encode("UTF-8")
            + _to_bytes(float, 1.0)
        )
        _write_message(f, RESET_MESSAGE, reset_message_bytes)

        f.write(content)

        _write_message(f, CLOSE_MESSAGE, bytes())

    return GodotEnvActionLog.parse(str(new_action_record_path), action_type)


def get_observations_from_human_recording(
    observations_path: Path, selected_features: OrderedDictType[str, Tuple[int, Tuple[int, ...]]]
) -> List[AvalonObservationType]:
    file_size = os.stat(observations_path).st_size
    observations = []
    with open(observations_path, "rb") as record_log:
        while record_log.tell() < file_size:
            feature_data = {}
            for feature_name, (data_type, dims) in selected_features.items():
                # TODO make a constant / is this still a problem?
                if feature_name in {"depth", "rgb", "rgbd"}:
                    continue
                size = prod(dims)
                if data_type != FAKE_TYPE_IMAGE:
                    size = size * 4
                feature_data[feature_name] = _read_shape(record_log, dims, size, dtype=NP_DTYPE_MAP[data_type])

            cleaned_feature_data = {
                feature.name: feature_data.get(feature.name, np.array([]))
                for feature in attr.fields(AvalonObservationType)
            }
            observations.append(AvalonObservationType(**cleaned_feature_data))
    return observations


def _read_shape(
    record_log: BufferedReader,
    shape: Tuple[int, ...],
    size: Optional[int] = None,
    dtype: npt.DTypeLike = np.uint8,
) -> npt.NDArray:
    byte_buffer = record_log.read(size if size is not None else prod(shape))

    return np.ndarray(shape=shape, dtype=dtype, buffer=byte_buffer)


def get_world_path_from_world_id(root_path: Path, world_id: str) -> str:
    return str(root_path / world_id / "main.tscn")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class MouseKeyboardHumanInputType(AttrsAction):
    is_move_forward_pressed: float
    is_move_backward_pressed: float
    is_move_left_pressed: float
    is_move_right_pressed: float
    is_jump_pressed: float
    is_eat_pressed: float
    is_grab_pressed: float
    is_throw_pressed: float
    is_crouch_pressed: float
    is_wheel_up_just_released: float
    is_wheel_down_just_released: float
    is_mouse_mode_toggled: float
    is_active_hand_toggled: float


def _to_bytes_with_double(value_type: Type, value: Any) -> bytes:
    if value_type is float:
        return struct.pack("<d", value)
    if value_type is int:
        return struct.pack("<i", value)
    raise SwitchError(f"{value_type} (value={value})")


def _from_bytes_with_double(value_type: Type, value_bytes: bytes) -> Tuple[Union[int, float], bytes]:
    byte_format = ""
    if value_type is float:
        byte_format = "<d"
    elif value_type is int:
        byte_format = "<i"
    else:
        raise SwitchError(f"{value_type} should be either float or int")

    size = struct.calcsize(byte_format)
    bytes_to_consume, bytes_remaining = value_bytes[:size], value_bytes[size:]
    return struct.unpack(byte_format, bytes_to_consume)[0], bytes_remaining


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class VRHumanInputType(AttrsAction):
    human_height: float
    arvr_origin_position_x: float
    arvr_origin_position_y: float
    arvr_origin_position_z: float
    arvr_origin_basis_x_x: float
    arvr_origin_basis_x_y: float
    arvr_origin_basis_x_z: float
    arvr_origin_basis_y_x: float
    arvr_origin_basis_y_y: float
    arvr_origin_basis_y_z: float
    arvr_origin_basis_z_x: float
    arvr_origin_basis_z_y: float
    arvr_origin_basis_z_z: float
    arvr_camera_position_x: float
    arvr_camera_position_y: float
    arvr_camera_position_z: float
    arvr_camera_basis_x_x: float
    arvr_camera_basis_x_y: float
    arvr_camera_basis_x_z: float
    arvr_camera_basis_y_x: float
    arvr_camera_basis_y_y: float
    arvr_camera_basis_y_z: float
    arvr_camera_basis_z_x: float
    arvr_camera_basis_z_y: float
    arvr_camera_basis_z_z: float
    arvr_left_hand_position_x: float
    arvr_left_hand_position_y: float
    arvr_left_hand_position_z: float
    arvr_left_hand_basis_x_x: float
    arvr_left_hand_basis_x_y: float
    arvr_left_hand_basis_x_z: float
    arvr_left_hand_basis_y_x: float
    arvr_left_hand_basis_y_y: float
    arvr_left_hand_basis_y_z: float
    arvr_left_hand_basis_z_x: float
    arvr_left_hand_basis_z_y: float
    arvr_left_hand_basis_z_z: float
    arvr_right_hand_position_x: float
    arvr_right_hand_position_y: float
    arvr_right_hand_position_z: float
    arvr_right_hand_basis_x_x: float
    arvr_right_hand_basis_x_y: float
    arvr_right_hand_basis_x_z: float
    arvr_right_hand_basis_y_x: float
    arvr_right_hand_basis_y_y: float
    arvr_right_hand_basis_y_z: float
    arvr_right_hand_basis_z_x: float
    arvr_right_hand_basis_z_y: float
    arvr_right_hand_basis_z_z: float
    look_stick_x: float
    look_stick_y: float
    strafe_stick_x: float
    strafe_stick_y: float
    is_left_hand_grab_pressed: float
    is_right_hand_grab_pressed: float
    is_jump_pressed: float

    @classmethod
    def from_bytes(cls: Type["VRHumanInputType"], action_bytes: bytes) -> "VRHumanInputType":
        fields: Dict[str, Any] = {}
        _size, remaining_bytes = _from_bytes_with_double(int, action_bytes)
        for field in attr.fields(cls):
            fields[field.name], remaining_bytes = _from_bytes_with_double(cast(Type, field.type), remaining_bytes)
        return cls(**fields)

    def to_bytes(self) -> bytes:
        action_bytes = b"".join(
            _to_bytes_with_double(
                cast(Type, field.type),
                getattr(self, field.name),
            )
            for field in attr.fields(self.__class__)
        )
        return _to_bytes_with_double(int, len(action_bytes)) + action_bytes


def parse_human_input(path: Path, human_input_type: Type[AttrsAction]) -> List[AttrsAction]:
    human_inputs = []
    with open(path, "rb") as f:
        while message_bytes := f.read(1):
            message = int.from_bytes(message_bytes, byteorder="little", signed=False)
            assert message == HUMAN_INPUT_MESSAGE
            size_bytes = f.read(4)
            size, _ = _from_bytes(int, size_bytes)
            human_input_bytes = size_bytes + f.read(cast(int, size))
            human_inputs.append(human_input_type.from_bytes(human_input_bytes))

    return human_inputs


def get_oculus_config_for_human_playback() -> AvalonSimSpec:
    # NOTE: make sure you get the type of player correct in config.json
    config = AvalonSimSpec.from_dict(json.load(open("datagen/godot/android/config.json", "r")))

    # update player to be agent player
    with config.mutable_clone() as config:
        # update recording options to make sense for the agent
        config.recording_options = RecordingOptionsSpec(
            user_id=None,
            apk_version=None,
            recorder_host=None,
            recorder_port=None,
            is_teleporter_enabled=False,
            resolution_x=64,
            resolution_y=64,
            is_recording_human_actions=False,
            is_remote_recording_enabled=False,
            is_adding_debugging_views=False,
            is_debugging_output_requested=False,
        )
        if isinstance(config.player, MouseKeyboardHumanPlayerSpec):
            config.player = MouseKeyboardAgentPlayerSpec.from_dict(config.player.to_dict())
        elif isinstance(config.player, VRHumanPlayerSpec):
            player_dict = config.player.to_dict()
            player_dict["arm_length"] = 10_000
            # player_dict["total_energy_coefficient"] = 1e-4
            # make vr arm length basically infinity
            config.player = VRAgentPlayerSpec.from_dict(player_dict)
        else:
            raise SwitchError(config.player)

    return config


def get_oculus_playback_config(is_using_human_input: bool):
    # NOTE: make sure you get the type of player correct in config.json
    config = AvalonSimSpec.from_dict(json.load(open("datagen/godot/android/config.json", "r")))

    # update player to be agent player
    with config.mutable_clone() as config:
        # update recording options to make sense for the agent
        config.recording_options = RecordingOptionsSpec(
            user_id=None,
            apk_version=None,
            recorder_host=None,
            recorder_port=None,
            is_teleporter_enabled=False,
            resolution_x=64,
            resolution_y=64,
            is_recording_human_actions=False,
            is_remote_recording_enabled=False,
            is_adding_debugging_views=False,
            is_debugging_output_requested=False,
        )
        if isinstance(config.player, MouseKeyboardHumanPlayerSpec):
            config.player = MouseKeyboardAgentPlayerSpec.from_dict(config.player.to_dict())
        elif isinstance(config.player, VRHumanPlayerSpec):
            player_dict = config.player.to_dict()
            player_dict["arm_length"] = 10_000
            player_dict["is_human_playback_enabled"] = is_using_human_input
            # make vr arm length basically infinity

            if is_using_human_input:
                config.player = VRHumanPlayerSpec.from_dict(player_dict)
            else:
                config.player = VRAgentPlayerSpec.from_dict(player_dict)
        else:
            raise SwitchError(config.player)
    return config


def validate_playback_recording(
    playback_path: Path,
    worlds_path: Path,
    threshold: float,
    is_using_human_input: bool,
) -> PlaybackResult:
    new_action_record_path = playback_path / "new_actions.out"
    action_record_path = playback_path / "actions.out"
    metadata_path = playback_path / "metadata.json"
    observations_path = playback_path / "observations.out"
    human_input_path = playback_path / "human_inputs.out"

    data_paths = [action_record_path, metadata_path, observations_path, human_input_path]

    for raw_path in data_paths:
        path = Path(f"{raw_path}.gz")
        if path.exists() and not raw_path.exists():
            with gzip.open(str(path), "rb") as f_in:
                with open(raw_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    config = get_oculus_playback_config(is_using_human_input)
    if is_using_human_input:
        action_path = human_input_path
        action_type = VRHumanInputType
    else:
        action_path = action_record_path
        action_type = get_action_type_from_config(config)

    env = create_env(config, action_type)
    selected_features = env.observation_context.selected_features
    # available_features = env.observation_context.available_features
    env.close()

    human_observations = get_observations_from_human_recording(
        observations_path=observations_path,
        selected_features=selected_features,
    )

    replay_log = get_replay_log_from_human_recording(
        new_action_record_path=new_action_record_path,
        action_record_path=action_path,
        metadata_path=metadata_path,
        worlds_path=worlds_path,
        selected_features=selected_features,
        action_type=action_type,
    )

    metadata = json.load(open(metadata_path))
    env = create_env(config, action_type)

    episode_id = replay_log.initial_episode_id
    world_id = metadata["world_id"]
    world_path = get_world_path_from_world_id(root_path=Path(worlds_path), world_id=world_id)
    env.seed_nicely(episode_id)

    actions = []
    observations = []
    for message in replay_log.messages:
        if message[0] in (ACTION_MESSAGE, HUMAN_INPUT_MESSAGE):
            action = message[1]
            obs, _ = env.act(cast(action_type, action))
            actions.append(action)
            observations.append(obs)
        elif message[0] == RESET_MESSAGE:
            observations.append(env.reset_nicely_with_specific_world(episode_seed=episode_id, world_path=world_path))
        else:
            raise SwitchError(f"Invalid replay message {message}")

    if is_using_human_input:
        human_input_type = action_type
    else:
        # TODO make this a function
        if action_type == MouseKeyboardActionType:
            human_input_type = MouseKeyboardHumanInputType
        elif action_type == VRActionType:
            human_input_type = VRHumanInputType
        else:
            raise SwitchError(action_type)

    human_inputs = [x for x in parse_human_input(human_input_path, human_input_type)]
    errors = defaultdict(list)
    for i, (agent_ob, player_ob) in enumerate(
        # TODO this feels a bit hacky -- need to revisit
        zip(observations[len(observations) - len(human_observations) :], human_observations)
    ):
        agent_observation_dict = attr.asdict(agent_ob)
        player_observation_dict = attr.asdict(player_ob)
        for key in player_observation_dict:
            if key in SKIP_KEYS:
                continue
            agent_value = agent_observation_dict[key]
            player_value = player_observation_dict[key]
            if key in DISCRETE_KEYS:
                max_error = (
                    1 if (agent_value != -1 and player_value == -1 or agent_value == -1 and player_value != -1) else 0
                )
            else:
                if np.isinf(agent_value.max()) or np.isinf(player_value.max()):
                    max_error = 1 if np.any(agent_value != player_value) else 0
                else:
                    delta: npt.NDArray[np.float32] = agent_value - player_value
                    max_error = np.abs(delta).max().item()
            if max_error > threshold:
                # mark video for error
                errors[i].append(
                    PlaybackError(
                        frame=i,
                        key=key,
                        recorded_value=player_value,
                        playback_value=agent_value,
                        max_error=max_error,
                    )
                )

    return PlaybackResult(
        is_error=len(errors) > 0,
        errors=errors,
        recorded_observations=human_observations,
        playback_observations=observations,
        human_inputs=human_inputs,
        actions=actions,
    )


def validate_all_playback_recordings(
    worlds_root_path: Path, playback_root_path: Path, threshold: float
) -> Dict[str, PlaybackResult]:
    results = {}
    for world_id_path in playback_root_path.iterdir():
        for playback_path in world_id_path.iterdir():
            key = f"{world_id_path.name}__{playback_path.name}"
            print(f"VALIDATING: {key}")
            results[key] = validate_playback_recording(playback_path, worlds_root_path, threshold)
            print()
            print()
    return results
