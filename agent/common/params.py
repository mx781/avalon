import uuid
from enum import Enum
from typing import Literal
from typing import Optional

import attr
import gym.spaces
import torch


class ClippedNormalMode(Enum):
    NO_CLIPPING = "NO_CLIPPING"
    SAMPLE_DIST = "SAMPLE_DIST"
    TRUNCATED_NORMAL = "TRUNCATED_NORMAL"


@attr.s(auto_attribs=True, frozen=True)
class EnvironmentParams:
    suite: Literal["godot", "gym", "dmc", "atari", "test"] = "godot"
    task: Optional[str] = None
    env_index: int = 0  # each environment will get a unique worker id in range [0, num_workers * num_worker_groups)
    env_count: int = 1  # this will get set automatically to num_workers
    action_repeat: int = 1
    time_limit: int = 100  # TODO: add a way to disable the time limit entirely
    reward_scale: float = 1  # scale reward magnitude by this amount
    pixel_obs_wrapper: bool = False  # use pixels instead of state vector for certain envs
    elapsed_time_obs: bool = False  # include an observation of the elapsed time
    mode: Literal["train", "val", "test"] = "train"


@attr.s(auto_attribs=True, frozen=True)
class Params:
    # wandb
    project: str
    name: Optional[str] = None  # wandb run name
    tag: str = "untagged"
    log_freq_hist: int = 50
    log_freq_scalar: int = 1
    log_freq_media: int = 250
    wandb_group: str = str(uuid.uuid4())[:10]

    # environment
    env_params: EnvironmentParams = EnvironmentParams()

    # worker
    multiprocessing: bool = True
    num_workers: int = 8  # number of environments per worker group
    time_limit_bootstrapping: bool = False
    obs_first: bool = True

    train_gpu: int = 0  # currently this does nothing (but will happen automatically)
    inference_gpus: list[int] = [0]  # only applied if there are multiple async worker managers
    godot_gpu: int = 0  # currently this does nothing (but should happen automatically)

    # training
    is_training: bool = True
    batch_size: int = 100
    resume_from_run: Optional[str] = None  # set this if you want to resume from an existing run
    resume_from_project: Optional[str] = None  # set this to resume from a different project than `project`
    resume_from_filename: str = "final.pt"
    checkpoint_every: int = 10_000
    total_env_steps: int = 1_000_000
    discount = 0.98
    normal_std_from_model: bool = True
    clipped_normal_mode: ClippedNormalMode = ClippedNormalMode.NO_CLIPPING

    # val/test
    is_testing: bool = False

    # spaces
    observation_space: Optional[gym.spaces.Dict] = None
    action_space: Optional[gym.spaces.Dict] = None

    @property
    def train_device(self) -> torch.device:
        return torch.device(f"cuda:{self.train_gpu}")
