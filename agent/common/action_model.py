from __future__ import annotations

from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import OneHotCategorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F

from agent.common.params import ClippedNormalMode
from agent.common.params import Params
from agent.common.types import ActionBatch
from agent.common.wrappers import OneHotMultiDiscrete
from agent.dreamer.tools import SampleDist
from agent.dreamer.tools import TanhBijector


class NormalWithMode(Normal):
    def mode(self) -> Tensor:
        mean = self.mean
        assert isinstance(mean, Tensor)
        return mean


class IndependentWithMode(Independent):
    def mode(self) -> Tensor:
        mode = self.base_dist.mode()
        assert isinstance(mode, Tensor)
        return mode


class NormalHead(nn.Module):
    """A module that builds a Diagonal Gaussian distribution from means.

    If model_provides_std=False, standard deviations are learned parameters in this module.
    Otherwise they are taken as inputs.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        model_provides_std: bool = False,
        min_std: float = 0.01,
        mode: ClippedNormalMode = ClippedNormalMode.NO_CLIPPING,
    ):
        super().__init__()
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
        self.num_outputs = action_space.shape[0]
        self.model_provides_std = model_provides_std
        self.min_std = min_std
        self.mode = mode
        if model_provides_std:
            # We take 2 values per output element, one for mean and one for (raw) std
            self.num_inputs = self.num_outputs * 2
            self.init_std = 1
            self.raw_init_std = np.log(np.exp(self.init_std) - 1)  # such that init_std = softplus(raw_init_std)
        else:
            # We'll use a constant learned std
            self.num_inputs = self.num_outputs
            # initial variance is e^0 = 1
            self.log_std = nn.Parameter(torch.zeros(self.num_outputs))

    def forward(self, x: Tensor) -> torch.distributions.Distribution:
        # x should have shape (..., action_dim)
        assert x.shape[-1] == self.num_inputs
        if self.model_provides_std:
            mean, raw_std = torch.chunk(x, 2, -1)
            # note that standard practice would be to use std = log_std.exp(), here we use a centered softplus instead
            std = F.softplus(raw_std + self.raw_init_std) + self.min_std
        else:
            mean = x
            std = self.log_std.exp() + self.min_std

        if self.mode == ClippedNormalMode.NO_CLIPPING:
            dist: torch.distributions.Distribution = NormalWithMode(mean, std)
            return IndependentWithMode(dist, reinterpreted_batch_ndims=1)
        elif self.mode == ClippedNormalMode.SAMPLE_DIST:
            # this mean_scale and tanh thing is a sort of soft clipping of the input.
            mean_scale = 5
            mean_scaled = mean_scale * torch.tanh(mean / mean_scale)
            dist = Normal(mean_scaled, std)
            transformed_dist = TransformedDistribution(dist, TanhBijector())
            independent_dist = Independent(transformed_dist, 1)
            sample_dist = SampleDist(independent_dist)
            return sample_dist
        elif self.mode == ClippedNormalMode.TRUNCATED_NORMAL:
            raise NotImplementedError
        assert False


class StraightThroughOneHotCategorical(OneHotCategorical):
    def rsample(self, sample_shape: torch.Size = torch.Size()):
        assert sample_shape == torch.Size()
        # Straight through biased gradient estimator.
        sample = self.sample(sample_shape).to(torch.float32)
        probs = self.probs
        assert sample.shape == probs.shape
        sample += probs - probs.detach()
        return sample.float()

    def mode(self) -> Tensor:
        return F.one_hot(self.probs.argmax(dim=-1), self.event_shape[-1]).float()  # type: ignore


class MultiCategoricalHead(nn.Module):
    """Represents multiple categorical dists. All must have the same number of categories."""

    def __init__(self, num_actions: int, num_categories: int):
        super().__init__()
        self.num_categories = num_categories
        self.num_actions = num_actions
        self.num_inputs = num_categories * num_actions
        self.num_outputs = num_categories * num_actions

    def forward(self, x: Tensor) -> torch.distributions.Distribution:
        assert x.shape[-1] == self.num_actions * self.num_categories
        x = rearrange(x, "... (a c) -> ... a c", a=self.num_actions, c=self.num_categories)
        x = x - x.mean(dim=-1, keepdim=True)
        x = torch.clamp(x, -4, 4)
        dist = StraightThroughOneHotCategorical(logits=x)
        independent_dist = IndependentWithMode(dist, 1)
        return independent_dist


class DictActionHead(torch.nn.Module):
    """This model handles generating a policy distribution from latents.

    Latents should be passed in with shape (..., self.num_inputs), and a DictActionDist will be returned.
    """

    def __init__(self, action_space: gym.spaces.Dict, params: Params):
        super().__init__()
        self.action_space = action_space

        # Build action heads
        self.num_inputs = 0
        self.num_outputs = 0
        action_heads = {}
        for k, space in self.action_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                # The input is (..., num_actions)
                # The output of sample() should be (..., num_actions)
                head: torch.nn.Module = NormalHead(
                    space, model_provides_std=params.normal_std_from_model, mode=params.clipped_normal_mode
                )
            elif isinstance(space, OneHotMultiDiscrete):
                # A MultiDiscrete is multiple independent Discrete spaces.
                # We coerce all discrete space types into this with a wrapper
                # This won't work if the discretes have different num_categories.
                assert len(set(space.nvec)) == 1
                head = MultiCategoricalHead(num_actions=len(space.nvec), num_categories=space.max_categories)
            else:
                assert False
            action_heads[k] = head
            self.num_inputs += action_heads[k].num_inputs  # type: ignore
            self.num_outputs += action_heads[k].num_outputs  # type: ignore
        self.action_heads = torch.nn.ModuleDict(action_heads)

    def forward(self, action_logits: Tensor) -> "DictActionDist":
        dists = {}
        i = 0
        for k, head in self.action_heads.items():
            logits = action_logits[..., i : i + head.num_inputs]
            dists[k] = head(logits)
            i += head.num_inputs
        return DictActionDist(dists)


class DictActionDist(torch.distributions.Distribution):
    """This is an instance of a torch Distribution that holds key-value pairs of other Distributions.

    It's used for e.g. sampling from an entire Dict action space in one operation,
    which will return a dictionary of samples.

    Operations like entropy() will reduce over all dists to return a single value (per batch element)."""

    def __init__(self, dists: Dict[str, torch.distributions.Distribution]):
        super().__init__(validate_args=False)
        self.dists = dists

    def sample(self) -> ActionBatch:  # type: ignore
        # Detach is just for good measure in case someone messes up the sample() impl :)
        return {k: v.sample().detach() for k, v in self.dists.items()}

    def rsample(self) -> ActionBatch:  # type: ignore
        return {k: v.rsample() for k, v in self.dists.items()}

    def log_prob(self, actions: Dict[str, Tensor]) -> Tensor:
        """Compute the log prob of the given action under the given dist (batchwise).

        Log prob is a scalar (per batch element.)"""
        # actions is a dict of tensors of shape (batch_size, num_outputs)
        log_probs = []
        for k, dist in self.dists.items():
            # batch_size = actions[k].shape[0]
            # assert actions[k].shape == (batch_size, self.action_heads[k].num_outputs)
            log_prob = dist.log_prob(actions[k])
            # The log_prob is over the entire action space, so it reduces to a single scalar per batch element
            # assert log_prob.shape == (batch_size,)
            log_probs.append(log_prob)

        # the output should have shape (batch_size, )
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def entropy(self) -> Tensor:
        # The entropy is over the entire action space, so it reduces to a single scalar per batch element
        entropies = [v.entropy() for v in self.dists.values()]
        return torch.stack(entropies, dim=1).sum(dim=1)

    def mean(self) -> Dict[str, Tensor]:
        return {k: v.mean() for k, v in self.dists.items()}

    def mode(self) -> Dict[str, Tensor]:
        return {k: v.mode() for k, v in self.dists.items()}

    # TODO: verify this is unnecessary and remove this.
    def distance_from_uninformative(self) -> Tensor:
        """The idea of this is to have a sort of distance from the "default" distribution.
        So for a Normal, it's the (KL) distance from N(0, 1).
        For a categorial, it's the (KL) distance from uniform.
        The overall result is the sum of all individual distances.
        Possibly this should be removed - I'm not sure that it's terribly well justified.
        """
        divs = []
        for k, original_dist in self.dists.items():
            is_indepdendent = False
            if isinstance(original_dist, Independent):
                dist = original_dist.base_dist
                is_indepdendent = True
            else:
                dist = original_dist

            if isinstance(dist, Normal):
                assert len(dist.event_shape) == 1
                uninformative: Distribution = Independent(Normal(loc=torch.zeros_like(dist.mean), scale=1), 1)
                div = torch.distributions.kl_divergence(dist, uninformative)
            elif isinstance(dist, OneHotCategorical):
                uninformative = OneHotCategorical(logits=torch.zeros_like(dist.logits))
                div = torch.distributions.kl_divergence(dist, uninformative)
            elif isinstance(dist, SampleDist):
                # I don't know how to compute a KL div for this, so just falling back to entropy.
                div = -1 * dist.entropy()
            else:
                assert False
            if is_indepdendent:
                div = div.sum(dim=-1)
            # assert div.shape == original_dist.batch_shape
            divs.append(div)
        return torch.stack(divs, dim=0).sum(dim=0)