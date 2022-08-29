# mypy: ignore-errors
# disabling type checking in this test code
"""Simple custom state-based environments to quickly validate things."""
import gym
import numpy as np


def overwrite_params(params, new_params):
    for k, v in new_params.items():
        print(f"WARNING: want arg {k} but got {getattr(params, k)} with {v}")
        # setattr(params, k, v)


def get_env(task, params):
    if task == "dummy":
        return DummyEnv
    elif task == "case1":
        # One long episode, returning reward 1 each step.
        # TODO: change to having fully specified test conditions for each alg (because tests will be somewhat alg-specific)
        new_params = {"discount": 0.75}
        overwrite_params(params, new_params)
        # Use with discount = .75.
        # Expect value estimate to be 4. Value EV=1
        # No episode ends, reward always 1.
        # Actions should stay random - they are ignored.
        return Test1(1000000, discrete=True)
    elif task == "case1_continuous_action":
        new_params = {"discount": 0.75}
        overwrite_params(params, new_args)
        return Test1(1000000, discrete=False)
    elif task == "case2":
        # Reward is the value of the last obs (-1 or 1)
        # So the action should stay random
        # And the value should have an EV of about .5 (since we can only know one of the discounted rewards)
        new_args = {"discount": 0.75}
        overwrite_args(args, new_args)
        return Test2(1000000, reward_mode="obs")
    elif task == "case6":
        # Reward is 1 only if the action matches the last obs (-1 or 1)
        # So in a perfect run, the avg step reward should equal 1
        return Test2(1000000, reward_mode="obs_and_action")
        # return Test2(1000000, reward_mode="obs")
    elif task == "hybrid1":
        return TestHybridAction(1000000)
        # return Test2(1000000, reward_mode="obs")
    else:
        assert False


class DummyEnv:
    def __init__(self):
        self._random = np.random.RandomState(seed=0)
        self._step = None

    @property
    def observation_space(self):
        low = np.zeros([3, 84, 84], dtype=np.float32)
        high = 255 * np.ones([3, 84, 84], dtype=np.float32)
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        action_dim = 5
        low = -np.ones([action_dim], dtype=np.float32)
        high = np.ones([action_dim], dtype=np.float32)
        return gym.spaces.Box(low, high)

    def reset(self):
        self._step = 0
        obs = self.observation_space.sample()
        return obs

    def step(self, action):
        obs = np.zeros((3, 84, 84), dtype=np.float32)
        # obs = self.observation_space.sample()
        reward = self._random.uniform(0, 1)
        self._step += 1
        done = self._step >= 1000
        info = {}
        return obs, reward, done, info

    def close(self):
        pass


class Test1:
    """Agent gets reward of 1 every step for num_steps. Total reward is num_steps.

    Problem with this env is that you have to know the elapsed time.

    Ways to use:
    - set n = 1.
    - set n = inf. Set discount to .75. First step value est should converge to 4.

    """

    def __init__(self, num_steps, discrete=True):
        self._step = None
        self.num_steps = num_steps
        self.discrete = discrete

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(1,))

    @property
    def action_space(self):
        if self.discrete:
            return gym.spaces.Discrete(2)
        else:
            return gym.spaces.Box(-1, 1, shape=(1,))

    def reset(self):
        self._step = 0
        obs = np.zeros((1,))
        return obs

    def step(self, action):
        obs = np.zeros((1,))
        reward = 1
        self._step += 1
        done = self._step >= self.num_steps
        info = {}
        return obs, reward, done, info

    def close(self):
        pass


class Test2:
    """Random obs in the set (-1, 1). Binary action. Multiple reward modes."""

    def __init__(self, num_steps, reward_mode="obs_and_action"):
        self._step = None
        self.num_steps = num_steps
        self.last_obs = np.zeros((1,))
        self.reward_mode = reward_mode

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(1,))

    @property
    def action_space(self):
        return gym.spaces.Discrete(2)

    def reset(self):
        self._step = 0
        obs = np.zeros((1,))
        return obs

    def step(self, action):
        assert isinstance(action, int)
        if self.reward_mode == "obs_and_action":
            reward = self.last_obs.item() * (action * 2 - 1)
        elif self.reward_mode == "obs":
            reward = self.last_obs.item()
        else:
            assert False
        self._step += 1
        done = self._step >= self.num_steps
        info = {}
        # high is non-inclusive.
        obs = np.random.randint(0, 2, size=(1,)) * 2 - 1
        # obs = np.random.uniform(-1, 1, size=(1,))
        self.last_obs = obs
        return obs, reward, done, info

    def close(self):
        pass


class TestHybridAction:
    def __init__(self, num_steps):
        self._step = None
        self.num_steps = num_steps
        self.last_obs = np.zeros((1,))

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(1,))

    @property
    def action_space(self):
        return gym.spaces.Dict({"discrete": gym.spaces.Discrete(2), "continuous": gym.spaces.Box(0, 1, shape=(1,))})

    def reset(self):
        self._step = 0
        obs = np.zeros((1,))
        return obs

    def step(self, action):
        # TODO: continuous actions come in as an array, discrete come in as integers.
        # Standardize on always being an array, probably?
        continuous = action["continuous"].item()
        discrete = action["discrete"]

        # The reward is the distance between last_obs and continuous * sign(discrete)
        # Last obs ranges in (-1, 1)
        # Continuous is nonnegative, so it will control the scale. Discrete will control the sign.
        # Reward will be 0 if we are perfect.
        reward = -1 * (self.last_obs.item() - (continuous * (discrete * 2 - 1))) ** 2

        self._step += 1
        done = self._step >= self.num_steps
        info = {}
        obs = np.random.uniform(-1, 1, size=(1,))
        self.last_obs = obs
        return obs, reward, done, info

    def close(self):
        pass
