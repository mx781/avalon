import os
import subprocess
from typing import Any
from typing import Tuple

import gym
import numpy as np

from agent.common import wrappers
from agent.common.params import EnvironmentParams
from agent.godot.godot_gym import AvalonGodotEnvWrapper
from agent.godot.godot_gym import GodotEnvironmentParams
from agent.godot.godot_gym import GodotObsTransformWrapper
from agent.godot.godot_gym import ScaleAndSquashAction


def build_env(env_params: EnvironmentParams) -> gym.Env:
    if env_params.suite == "dmc":
        # this has dict obs space with images at ["rgb"]
        assert env_params.task is not None
        env = DeepMindControl(env_params.task)
        env = wrappers.ActionRepeat(env, env_params.action_repeat)
        # rescales actions from standard ranges to the envs desired range.
        env = wrappers.TimeLimit(env, max_episode_steps=env_params.time_limit / env_params.action_repeat)
        env = wrappers.DictObsActionWrapper(env)
        env = wrappers.ImageTransformWrapper(env, key="rgb", greyscale=False, resolution=None)
    elif env_params.suite == "godot":
        assert isinstance(env_params, GodotEnvironmentParams)
        assert env_params.action_repeat == 1

        egl_driver_path = (
            subprocess.run("mount | grep x86_64 | grep EGL | cut -d ' ' -f 3", capture_output=True, shell=True)
            .stdout.decode("UTF-8")
            .strip()
        )
        assert egl_driver_path, "No EGL driver found! Maybe wrong AMI? Wrong docker command? Good luck."
        os.system(f"sudo ln -sf {egl_driver_path} /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0")

        env = AvalonGodotEnvWrapper(env_params)
        # We don't use the TimeLimit wrapper because the time limit is dynamic,
        # so we trust that the godot env gives the proper TimeLimit.truncated signal
        # (which it should) for the timelimit boostrapping to work properly if enabled.
        env = GodotObsTransformWrapper(env)
        if env_params.mode == "train":
            env = wrappers.CurriculumWrapper(
                env,
                task_difficulty_update=env_params.task_difficulty_update,
                meta_difficulty_update=env_params.meta_difficulty_update,
            )
        env = ScaleAndSquashAction(env, scale=1)
        env = wrappers.OneHotActionWrapper(env)
        # env = RewardSoftClipWrapper(env, scale=5)
    elif env_params.suite == "test":
        assert env_params.action_repeat == 1
        from agent.common.test_envs import get_env

        env = get_env(env_params.task, env_params)
        env = wrappers.DictObsActionWrapper(env)
        env = wrappers.OneHotActionWrapper(env)
    elif env_params.suite == "gym":
        assert env_params.action_repeat == 1
        # Annoyingly, gym envs apply their own time limit already.
        print("time limit arg ignored in gym envs")
        env = gym.make(env_params.task)
        # Hacky. Relies on the TimeWrapper being the outermost wrapper. Not sure the better way.
        max_steps = env._max_episode_steps
        print(f"env has a time limit of {max_steps} steps")
        # env = DiscreteActionToIntWrapper(env)
        if env_params.pixel_obs_wrapper:
            env = wrappers.PixelObsWrapper(env)
            env = wrappers.DictObsActionWrapper(env, obs_key="rgb")
        else:
            env = wrappers.DictObsActionWrapper(env, obs_key="state")
        if env_params.pixel_obs_wrapper:
            env = wrappers.ImageTransformWrapper(env, key="rgb", greyscale=True, resolution=64)
        env = wrappers.ClipActionWrapper(env)
        env = wrappers.OneHotActionWrapper(env)
        if env_params.elapsed_time_obs:
            env = wrappers.ElapsedTimeWrapper(env, max_steps)
    else:
        assert False
    env = wrappers.NormalizeActions(env)
    env = wrappers.ScaleRewards(env, env_params.reward_scale)
    # env = wrappers.NumpyToTorch(env)
    return env


class DeepMindControl(gym.Env):
    def __init__(
        self,
        name: str,
        size: Tuple[int, int] = (64, 64),
        camera: Any = None,
        include_state: bool = False,
        include_rgb: bool = True,
    ):
        from dm_control import suite

        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        self._env = suite.load(domain, task)
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.include_state = include_state
        self.include_rgb = include_rgb

    @property
    def observation_space(self) -> gym.spaces.Dict:
        spaces = {}
        if self.include_state:
            for key, value in self._env.observation_spec().items():
                print("warning: gym spaces do not give observation ranges. no rescaling will be applied.")
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        if self.include_rgb:
            spaces["rgb"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> gym.spaces.Box:
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):  # type: ignore
        time_step = self._env.step(action)
        obs = {}
        if self.include_state:
            obs |= dict(time_step.observation)
        if self.include_rgb:
            obs["rgb"] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):  # type: ignore
        time_step = self._env.reset()
        obs = {}
        if self.include_state:
            obs |= dict(time_step.observation)
        if self.include_rgb:
            obs["rgb"] = self.render()
        return obs

    def render(self, *args, **kwargs):  # type: ignore
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
