import signal
import time
import uuid
from abc import ABC
from abc import abstractmethod
from ctypes import c_int
from functools import partial
from pathlib import Path
from typing import Generic
from typing import Iterator
from typing import List
from typing import Optional
from typing import Protocol

import attr
import torch
import wandb
from torch.utils.data import DataLoader
from tree import map_structure

from agent.common import wandb_lib
from agent.common.dataloader import ReplayDataset
from agent.common.dataloader import worker_init_fn
from agent.common.envs import build_env
from agent.common.get_algorithm_cls import get_algorithm_cls
from agent.common.storage import InMemoryStorage
from agent.common.storage import ModelStorage
from agent.common.storage import StorageMode
from agent.common.storage import TrajectoryStorage
from agent.common.types import Algorithm
from agent.common.types import BatchSequenceData
from agent.common.types import ParamsType
from agent.common.util import pack_1d_list
from agent.common.util import postprocess_uint8_to_float
from agent.common.worker import AsyncRolloutManager
from agent.common.worker import RolloutManager
from agent.dreamer.params import OffPolicyParams
from agent.godot.godot_eval import log_rollout_stats
from agent.ppo.params import OnPolicyParams

# Just a temporary reminder to not use <1.11
assert int(torch.__version__.split(".")[1]) >= 11


class Cleanable(Protocol):
    def shutdown(self) -> None:
        pass


class Trainer(ABC, Generic[ParamsType]):
    def __init__(self, params: ParamsType):
        # TODO: fix this properly
        import openturns

        openturns.Tensor
        # Patch the issue where openturns resets the signal handler
        signal.signal(signal.SIGINT, signal.default_int_handler)

        # Use an explict context to fix this bug: https://github.com/pytorch/pytorch/issues/3492
        # Spawn is less fiddly, but fork launches faster
        self.multiprocessing_context = torch.multiprocessing.get_context("spawn")
        # Fixing the bug where torch gets really slow on small fast ops on CPU
        # https://github.com/pytorch/pytorch/issues/80777
        torch.set_num_threads(1)
        self.params = params
        self.to_cleanup: List[Cleanable] = []

        self.start = None
        self.i = 0
        self.env_step = 0

        self.get_spaces()
        self.train_storage = self.create_train_storage()
        self.algorithm = self.create_model()
        self.train_rollout_manager = self.create_rollout_manager()
        self.train_dataloader = self.create_dataloader()
        self.wandb_run = self.wandb_init()
        self.algorithm = self.algorithm.to(self.params.train_device)

    def get_spaces(self):
        dummy_env = build_env(self.params.env_params)
        self.params = attr.evolve(
            self.params, observation_space=dummy_env.observation_space, action_space=dummy_env.action_space
        )
        dummy_env.close()

    def create_train_storage(self) -> Optional[TrajectoryStorage]:
        return None

    def wandb_init(self):
        # This needs to happen after all other processes launch
        run_name = self.params.env_params.suite if not self.params.name else self.params.name
        run = wandb.init(
            name=run_name + "_train",
            project=self.params.project,
            config=attr.asdict(self.params),
            tags=[self.params.tag],
            group=self.params.wandb_group,
        )
        wandb_lib.SCALAR_FREQ = self.params.log_freq_scalar
        wandb_lib.HIST_FREQ = self.params.log_freq_hist
        wandb_lib.MEDIA_FREQ = self.params.log_freq_media
        return run

    @abstractmethod
    def create_rollout_manager(self):
        raise NotImplementedError

    def create_model(self) -> Algorithm:
        algorithm_cls = get_algorithm_cls(self.params)
        algorithm = algorithm_cls(self.params, self.params.observation_space, self.params.action_space)

        if self.params.resume_from_run:
            project = self.params.resume_from_project if self.params.resume_from_project else self.params.project
            checkpoint = wandb_lib.download_file(
                self.params.resume_from_run, project, self.params.resume_from_filename
            )
            algorithm.load_state_dict(torch.load(checkpoint, map_location=self.params.train_device))
            print("RESUMED MODEL FROM CHECKPOINT")
        return algorithm

    @abstractmethod
    def create_dataloader(self) -> Iterator:
        raise NotImplementedError

    def train(self):
        if not self.start:
            self.start = time.time()
        while True:
            self.train_step()

            wandb_lib.log_scalar("env_step", self.env_step, self.i)
            if self.env_step >= self.params.total_env_steps:
                break

        self.checkpoint(filename="final.pt")

    def train_step(self):
        rollouts: BatchSequenceData = next(self.train_dataloader)
        start = time.time()
        start_i = self.i
        rollouts = map_structure(lambda x: x.to(self.params.train_device, non_blocking=True), rollouts)
        rollouts = attr.evolve(rollouts, observation=postprocess_uint8_to_float(rollouts.observation))
        self.i = self.algorithm.train_step(rollouts, self.i)

        wandb_lib.log_scalar(
            "timings/train_fps", self.params.batch_size * (self.i - start_i) / (time.time() - start), self.i
        )

        if self.i % self.params.checkpoint_every == 0:
            self.checkpoint()

        wandb_lib.log_iteration_time(self.params.batch_size, self.i)
        wandb_lib.log_scalar("timings/cumulative_env_fps", self.env_step / (time.time() - self.start), self.i)
        wandb_lib.log_scalar(
            "timings/cumulative_train_fps", self.i * self.params.batch_size / (time.time() - self.start), self.i
        )

        # Visualize the observations
        if self.i % wandb_lib.MEDIA_FREQ == 0:
            # the post-transforms have already been applied.
            for k, v in self.params.observation_space.items():
                if len(v.shape) == 3:
                    obs_video = rollouts.observation[k][:8]
                    wandb_lib.log_video(f"videos/obs/{k}", obs_video + 0.5, self.i, freq=1)

        # Evaluation
        # val.update(algorithm, i, env_step)
        # self.i += 1

    def checkpoint(self, filename: Optional[str] = None):
        if not filename:
            filename = f"model_{self.i}.pt"
        model_filename = Path(wandb.run.dir) / filename  # type: ignore
        torch.save(self.algorithm.state_dict(), model_filename)
        wandb.save(str(model_filename), policy="now")  # type: ignore
        wandb_lib.log_scalar("last_checkpoint", self.i, self.i, freq=1)

    def test(self):
        if not self.params.is_train_only:
            raise NotImplementedError

    def shutdown(self):
        # The main thread won't join until we close all the processes we have open.
        for item in self.to_cleanup:
            item.shutdown()
        self.to_cleanup = []


class OffPolicyTrainer(Trainer[OffPolicyParams]):
    def __init__(self, params: OffPolicyParams):
        self.train_rollout_dir = str(Path(params.data_dir) / "train" / str(uuid.uuid4()))
        # Necessary to fix some bug: https://github.com/wandb/client/issues/1994
        wandb.require(experiment="service")
        wandb.setup()
        super().__init__(params)
        ModelStorage.clean()

    def create_rollout_manager(self):
        self.env_step_counters = []
        for i in range(self.params.worker_managers):
            # we don't need a lock here because there's only one writer and precise read/write ordering doesn't matter
            env_step_counter = self.multiprocessing_context.Value(c_int, lock=False)
            train_game_player = AsyncRolloutManager(
                params=self.params,
                obs_space=self.params.observation_space,
                action_space=self.params.action_space,
                rollout_manager_id=i,
                env_step_counter=env_step_counter,
                multiprocessing_context=self.multiprocessing_context,
                train_rollout_dir=self.train_rollout_dir,
            )
            self.env_step_counters.append(env_step_counter)
            self.to_cleanup.append(train_game_player)
            train_game_player.start()

    def create_dataloader(self) -> Iterator[BatchSequenceData]:
        train_dataset = ReplayDataset(self.params, self.train_rollout_dir, update_interval=4000)
        return iter(
            DataLoader(
                train_dataset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=self.params.num_dataloader_workers,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=2,
                worker_init_fn=worker_init_fn,
                collate_fn=partial(pack_1d_list, out_cls=BatchSequenceData),
            )
        )

    def train_step(self):
        super().train_step()
        self.env_step = sum([x.value for x in self.env_step_counters])
        if self.i % 1000 == 0:
            print("saving model")
            ModelStorage.push_model(self.algorithm)


class OnPolicyTrainer(Trainer[OnPolicyParams]):
    def __init__(self, params: OnPolicyParams):
        super().__init__(params)

    def create_rollout_manager(self):
        rollout_manager = RolloutManager(
            params=self.params,
            num_workers=self.params.num_workers,
            is_multiprocessing=self.params.multiprocessing,
            storage=self.train_storage,
            obs_space=self.params.observation_space,
            storage_mode=StorageMode.FRAGMENT,
            model=self.algorithm,
            rollout_device=self.params.inference_device,
            multiprocessing_context=self.multiprocessing_context,
        )
        self.to_cleanup.append(rollout_manager)
        return rollout_manager

    def create_train_storage(self) -> TrajectoryStorage:
        return InMemoryStorage(self.params)

    def rollout_step(self):
        # Run rollouts
        start = time.time()
        self.train_rollout_manager.run_rollout(
            num_steps=self.params.num_steps,
            exploration_mode="explore",
        )
        rollout_fps = (self.params.num_workers * self.params.num_steps) / (time.time() - start)
        wandb_lib.log_scalar("timings/rollout_fps", rollout_fps, self.i)
        self.env_step += self.params.num_workers * self.params.num_steps

        if self.params.env_params.suite == "godot":
            log_rollout_stats(self.train_storage.storage.values(), self.i)

    def create_dataloader(self) -> Iterator[BatchSequenceData]:
        def dataloader():
            while True:
                self.train_storage.reset()
                self.rollout_step()
                rollouts: BatchSequenceData = self.train_storage.to_packed()
                # go ahead and send to cuda now, will make the next step faster
                rollouts = map_structure(lambda x: x.to(self.params.train_device, non_blocking=True), rollouts)
                # TODO: do something about reaching in and grabbing the value like this - not great.
                # and this won't be correct in the case that we stopped to train at the same time an ep ended -
                # next_obs will actually be from the next ep. we need the terminal obs in that case.
                # Although in practice that might actually hurt performance a bit, in some cases.
                # Is it so bad to slice off one sample at the end here for training?
                # Could do something like - slice off the last step, and then insert it back in the buffer
                # to become part of the next rollout.
                final_observations = map_structure(
                    lambda x: x.to(self.params.train_device, non_blocking=True),
                    self.train_rollout_manager.next_obs,
                )
                # Add one more observation, to use for value backup.
                # TODO: this could be optimized, involves an unnecessary memory copy (but it's logically simpler this way)
                # TODO: shouldn't be mutating an immutable structure here
                for k, v in final_observations.items():
                    rollouts.observation[k] = torch.cat((rollouts.observation[k], torch.unsqueeze(v, 1)), dim=1)

                yield rollouts

        return iter(dataloader())
