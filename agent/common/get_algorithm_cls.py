from typing import Type

from agent.common.params import Params
from agent.common.types import Algorithm


def get_algorithm_cls(params: Params) -> Type[Algorithm]:
    """This is a bit hacky because we don't really want to import code from an algorithm we're not using."""
    algorithm_cls: Type[Algorithm]
    if type(params).__name__ == "PPOParams":
        from agent.ppo.ppo import PPO

        algorithm_cls = PPO
    elif type(params).__name__ == "DreamerParams":
        from agent.dreamer.dreamer import Dreamer

        algorithm_cls = Dreamer
    else:
        raise NotImplementedError(type(params).__name__)
    return algorithm_cls
