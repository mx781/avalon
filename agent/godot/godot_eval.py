import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from threading import Thread
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import attrs
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.pyplot import bar
from numpy import typing as npt

from agent.common import wandb_lib
from agent.common.params import Params
from agent.common.storage import LambdaStorage
from agent.common.storage import StorageMode
from agent.common.types import Algorithm
from agent.common.types import StepData
from agent.common.worker import RolloutManager
from agent.godot.godot_gym import GodotEnvironmentParams
from common.log_utils import logger
from contrib.s3_utils import TEMP_BUCKET_NAME
from contrib.s3_utils import SimpleS3Client
from contrib.testing_utils import create_temp_file_path
from contrib.utils import TEMP_DIR

BIG_SEPARATOR = "-" * 80
RESULT_TAG = "DATALOADER:0 TEST RESULTS"


def log_rollout_stats_packed(
    packed_rollouts: Dict[str, npt.NDArray], infos: Dict[int, List[Dict[str, npt.NDArray]]], i
):
    successes = defaultdict(lambda: defaultdict(list))
    keys = ["success", "difficulty"]
    for worker, timestep in np.argwhere(packed_rollouts["dones"]):
        info = infos[worker][timestep]
        task = info["task"].lower()
        for field in keys:
            successes[task][f"final_{field}"].append(info[field])
    # Data is a dict (task) of dicts (keys) of lists
    for task, x in successes.items():
        for field, y in x.items():
            wandb_lib.log_histogram(f"train/{task}/{field}", y, i, hist_freq=10)
        wandb_lib.log_scalar(f"train/{task}/num_episodes", len(y), i)


def log_rollout_stats(rollouts: List[List[StepData]], i):
    """Log stats, when rollouts is a collection of episode fragments."""
    successes = defaultdict(lambda: defaultdict(list))
    keys = ["success", "difficulty"]
    for fragment in rollouts:
        for timestep in fragment:
            if timestep.done:
                task = timestep.info["task"].lower()
                for field in keys:
                    successes[task][f"final_{field}"].append(timestep.info[field])
    # Data is a dict (task) of dicts (keys) of lists
    for task, x in successes.items():
        for field, y in x.items():
            wandb_lib.log_histogram(f"train/{task}/{field}", y, i, hist_freq=10)
        wandb_lib.log_scalar(f"train/{task}/num_episodes", len(y), i)


Episode = List[StepData]
DifficultyBin = Tuple[float, float]  # [from, to)
EpisodesByDifficulty = DefaultDict[DifficultyBin, List[Episode]]
EpisodesByTaskByDifficulty = DefaultDict[str, EpisodesByDifficulty]


def get_difficulty_bin_name(difficulty_bin: DifficultyBin) -> str:
    return f"{difficulty_bin[0]:.2f}_to_{difficulty_bin[1]:.2f}"


def bin_episodes(rollouts: List[List[StepData]], difficulty_bin_size: float = 0.1) -> EpisodesByTaskByDifficulty:
    difficulty_bins = np.arange(difficulty_bin_size, 1 + difficulty_bin_size, difficulty_bin_size)  # interval ends
    episodes_by_task_by_difficulty_bin = defaultdict(lambda: defaultdict(list))
    for episode in rollouts:
        task = episode[-1].info["task"].lower()
        difficulty = episode[-1].info["difficulty"]
        difficulty_bin_idx = np.digitize(difficulty, difficulty_bins, right=True)
        difficulty_bin_end = difficulty_bins[difficulty_bin_idx]
        difficulty_bin_start = difficulty_bin_end - difficulty_bin_size
        difficulty_bin = difficulty_bin_start, difficulty_bin_end
        episodes_by_task_by_difficulty_bin[task][difficulty_bin].append(episode)
    return episodes_by_task_by_difficulty_bin


def get_episode_difficulty_bin(episode: Episode, difficulty_bin_size: float):
    difficulty_bins = np.arange(difficulty_bin_size, 1 + difficulty_bin_size, difficulty_bin_size)  # interval ends
    difficulty = episode[-1].info["difficulty"]
    difficulty_bin_idx = np.digitize(difficulty, difficulty_bins, right=True)
    difficulty_bin_end = difficulty_bins[difficulty_bin_idx]
    difficulty_bin_start = difficulty_bin_end - difficulty_bin_size
    difficulty_bin = difficulty_bin_start, difficulty_bin_end
    return difficulty_bin


def log_video_by_difficulty(
    episode: Episode, difficulty_bin: DifficultyBin, prefix: str = "test", infix: str = ""
) -> None:
    if infix != "":
        infix += "/"
    difficulty_bin_name = get_difficulty_bin_name(difficulty_bin)
    video = torch.stack([step.observation["rgbd"] for step in episode])
    wandb_lib.log_video(f"{prefix}/videos/{infix}{difficulty_bin_name}", video, step=None, normalize=True, freq=1)


def log_success_by_difficulty(
    successes_by_difficulty: DefaultDict[DifficultyBin, List[int]], prefix: str = "test", suffix: Optional[str] = None
) -> None:
    if suffix is not None:
        suffix = f"/{suffix}"
    data = []
    for difficulty_bin, successes in successes_by_difficulty.items():
        difficulty_bin_name = get_difficulty_bin_name(difficulty_bin)
        success_rate = np.mean(successes)
        data.append((difficulty_bin_name, success_rate))
    data = sorted(data, key=lambda item: item[0])
    difficulty_bins, success_rates = zip(*data)
    fig = plt.figure()
    plt.ylabel("success rate")
    plt.xlabel("difficulty bin")
    plt.ylim(0, 1)
    bar(difficulty_bins, success_rates)
    wandb.log({f"{prefix}/success_by_difficulty{suffix}": wandb.Image(fig)})


def load_worlds_from_s3(data_key: str, target_path: Path, bucket_name: str = TEMP_BUCKET_NAME) -> int:
    """Download pre-generated worlds from a S3 tarball to a local directory."""
    print("started loading fixed worlds from s3 file")
    shutil.rmtree(str(target_path), ignore_errors=True)
    target_path.mkdir(parents=True)
    s3_client = SimpleS3Client(bucket_name)
    logger.info(f"Downloading data from {data_key} to {target_path}")
    with create_temp_file_path() as temp_file_path:
        s3_client.download_to_file(data_key, temp_file_path)
        with tarfile.open(temp_file_path, "r:gz") as f:
            f.extractall(target_path)
    # TODO: figure out a better way to count episodes?
    episode_count = len(os.listdir(str(target_path)))
    print("finished loading fixed worlds from s3 file")
    return episode_count


def test(params: Params, model: Algorithm, log=True, exploration_mode="eval"):
    """Run evaluation for Godot."""
    # TODO: set gpu ids for godot and inference properly
    test_params = attrs.evolve(params, env_params=attrs.evolve(params.env_params, mode="test"))
    assert isinstance(test_params.env_params, GodotEnvironmentParams)
    assert test_params.env_params.env_index == 0

    if test_params.env_params.fixed_worlds_s3_key:
        # Load worlds from S3, if we have that enabled
        # e.g. "a5101fb5fca577a35a0749ba45ae28006823136f/test_worlds.tar.gz"
        # the worlds typically have to be put in a specific folder because they contain absolute paths...
        fixed_worlds_path = test_params.env_params.fixed_worlds_load_from_path
        if not fixed_worlds_path:
            fixed_worlds_path = Path(TEMP_DIR) / "eval_worlds"
        num_worlds = load_worlds_from_s3(test_params.env_params.fixed_worlds_s3_key, fixed_worlds_path)
        test_params = attrs.evolve(
            test_params, env_params=attrs.evolve(test_params.env_params, fixed_worlds_load_from_path=fixed_worlds_path)
        )
    elif test_params.env_params.fixed_worlds_load_from_path:
        # Got a path but no s3 key, assume the files are already locally available
        num_worlds = len(os.listdir(str(test_params.env_params.fixed_worlds_load_from_path)))
    else:
        num_worlds = params.env_params.test_episodes_per_task * params.env_params.num_tasks

    model.reset_state()
    model.eval()

    # Set up hooks for extracting episode info we care about
    difficulty_bin_size = 0.2
    seen_worlds: set[int] = set()
    success_by_task_and_difficulty_bin = defaultdict(lambda: defaultdict(list))
    world_scores = {}

    def collect_episode_stats(episode: Episode) -> None:
        world_index = episode[-1].info["world_index"]
        if world_index in seen_worlds:
            # We may run the same world twice, don't count them twice!
            return
        else:
            seen_worlds.add(world_index)
        task = episode[-1].info["task"].lower()
        success = episode[-1].info["success"]
        difficulty_bin = get_episode_difficulty_bin(episode, difficulty_bin_size)
        success_by_task_and_difficulty_bin[task][difficulty_bin].append(success)
        is_first_episode_in_group = len(success_by_task_and_difficulty_bin[task][difficulty_bin]) == 1
        if log and is_first_episode_in_group:
            # Note: wandb.log is not thread safe, this might cause issues. But it's fast :)
            thread = Thread(
                target=log_video_by_difficulty, args=(episode, difficulty_bin), kwargs={"infix": task}, daemon=True
            )
            thread.start()
            # log_video_by_difficulty(episode, difficulty_bin, infix=task)

        # Stuff for collecting scores
        world_scores[world_index] = episode[-1].info["score"]

    hooks = (collect_episode_stats,)
    storage_mode = StorageMode.EPISODE
    test_storage = LambdaStorage(params, hooks, storage_mode)

    # Maybe i should make a "free-running", "n_steps", and "n_episodes" worker.
    # And use the n_episodes version for eval.
    multiprocessing_context = torch.multiprocessing.get_context("spawn")
    # TODO: make a parameter for this
    num_workers = 16
    player = RolloutManager(
        params=test_params,
        num_workers=min(num_workers, num_worlds),
        is_multiprocessing=True,
        storage=test_storage,
        obs_space=params.observation_space,
        storage_mode=storage_mode,
        model=model,
        rollout_device=torch.device("cuda:0"),
        multiprocessing_context=multiprocessing_context,
    )
    test_storage.reset()

    logger.info(f"running {num_worlds} evaluation episodes")
    # assert num_episodes // num_workers > 0
    # Will potential run some worlds multiple times
    player.run_rollout(num_episodes=int(np.ceil(num_worlds / num_workers)), exploration_mode=exploration_mode)
    print("finished rollout, shutting down workers")
    player.shutdown()

    test_log = {}
    total_episodes_logged = 0
    all_successes = []
    for task, success_by_difficulty_bin in success_by_task_and_difficulty_bin.items():
        if log:
            log_success_by_difficulty(success_by_difficulty_bin, suffix=task)
        task_successes = sum(list(success_by_difficulty_bin.values()), [])
        all_successes.extend(task_successes)
        test_log[f"{task}_success_rate"] = np.mean(task_successes)
        for difficulty_bin, successes in success_by_difficulty_bin.items():
            total_episodes_logged += len(successes)
    test_log["overall_success_rate"] = np.mean(all_successes)
    assert (
        total_episodes_logged >= num_worlds
    ), f"Expected to log at least {num_worlds}, but only logged {total_episodes_logged}"

    if log:
        wandb.log({f"test/{k}": v for k, v in test_log.items()})
        logger.info(BIG_SEPARATOR)
        logger.info(RESULT_TAG)
        logger.info(test_log)
        logger.info(BIG_SEPARATOR)

    if test_params.env_params.fixed_worlds_load_from_path:
        # Special logging for the fixed evaluation worlds
        project = test_params.resume_from_project if test_params.resume_from_project else test_params.project
        # TODO: make this get the current wandb run_id or some other identifier if we're not loading a run
        run_id = test_params.resume_from_run
        filename = test_params.resume_from_filename
        fixed_world_key = (
            test_params.env_params.fixed_worlds_s3_key if test_params.env_params.fixed_worlds_s3_key else "test"
        )
        result_key = f"avalon_eval__{project}_{run_id}_{filename}__{fixed_world_key}__final"
        record_data = {
            "wandb_run": f"{project}/{run_id}/{filename}",
            "baseline": "PPO",
            "data_key": fixed_world_key,
            "all_results": world_scores,
        }
        print(record_data)

        print(f"Saving result to '{result_key}'")
        s3_client = SimpleS3Client()
        s3_client.save(result_key, json.dumps(record_data).encode())

    return test_log
