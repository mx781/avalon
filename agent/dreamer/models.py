# mypy: ignore-errors
# TODO: type this file
from typing import Literal
from typing import Tuple

import gym
import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.distributions import Bernoulli
from torch.distributions import Independent
from torch.distributions import Normal
from torch.nn import functional as F
from tree import map_structure

from agent.common.action_model import DictActionHead
from agent.common.models import MLP
from agent.common.types import ActionBatch
from agent.common.types import LatentBatch
from agent.common.types import ObservationBatch
from agent.dreamer.tools import static_scan
from agent.ppo.model import mlp_init


def init_weights(m: torch.nn.Module, gain: float = 1.0) -> None:
    # It appears that tf uses glorot init with gain 1 by default for all weights, and zero for all biases
    # at least for these layers
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)


class RSSM(nn.Module):
    def __init__(self, embed_size=1024, stoch=30, deter=200, hidden=200, actdim=10, act=F.elu):
        super().__init__()
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self._actdim = actdim
        self.min_std = 0.1

        # These two layers are common/shared for the prior and posterior.
        self.img1 = nn.Linear(actdim + stoch, self._hidden_size)
        self._cell = nn.GRUCell(self._deter_size, self._deter_size)

        # Prior model
        self.img2 = nn.Linear(self._deter_size, self._hidden_size)
        self.img3 = nn.Linear(self._hidden_size, 2 * self._stoch_size)

        # Posterior model
        self._embed_size = embed_size
        self.obs1 = nn.Linear(self._deter_size + self._embed_size, self._hidden_size)
        self.obs2 = nn.Linear(self._hidden_size, 2 * self._stoch_size)

        self.apply(init_weights)

    def initial(self, batch_size: int, device) -> LatentBatch:
        """This is the initial latent state."""
        return dict(
            mean=torch.zeros([batch_size, self._stoch_size], device=device),
            std=torch.zeros([batch_size, self._stoch_size], device=device),
            stoch=torch.zeros([batch_size, self._stoch_size], device=device),
            deter=torch.zeros([batch_size, self._hidden_size], device=device),
        )

    def observe(self, embed: Tensor, action: ActionBatch, state=None) -> Tuple[LatentBatch, LatentBatch]:
        """Generates state estimations given a sequence of observations and actions.
        Only used in training?"""

        if state is None:
            state = self.initial(batch_size=embed.shape[0], device=embed.device)
        # these are moving the time axis to the front, for the static_scan
        embed = rearrange(embed, "b t ... -> t b ...")
        action = {k: rearrange(v, "b t ... -> t b ...") for k, v in action.items()}
        # here the nest structure is a tuple of tensors, and a tuple of dicts of tensors
        # (state, state) is just to give the proper structure for the output
        post, prior = static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs), (action, embed), (state, state)
        )
        # Moving the batch axis back in front?
        post = {k: v.permute(1, 0, 2) for k, v in post.items()}
        prior = {k: v.permute(1, 0, 2) for k, v in prior.items()}
        return post, prior

    def imagine(self, action: ActionBatch, state: LatentBatch) -> LatentBatch:
        # TODO: only used for the imagination viz. figure out how to share with train() imagine.
        # or probably just move this into the viz logic
        # Takes a list of actions and an initial state and rolls it out in imagination.
        assert isinstance(state, dict), state
        prior = static_scan(self.img_step, action, state)
        prior = {k: v.permute(1, 0, 2) for k, v in prior.items()}
        return prior

    def get_feat(self, state: LatentBatch) -> Tensor:
        """Combine stoch and deter state into a single vector."""
        feat = torch.cat([state["stoch"], state["deter"]], -1)
        return feat

    def get_dist(self, state: LatentBatch) -> torch.distributions.Distribution:
        # This is only used for sampling from the latents, as far as i can tell
        # A MultivariateNormalDiag dist
        # This janky stuff is to match what kinda looks like a bug in danijar's impl.
        # class FixedIndependent(Independent):
        #     def rsample(self):
        #         mean = self.base_dist.mean
        #         std = self.base_dist.stddev
        #         return mean + torch.randn(self.event_shape, device=mean.device) * std
        #
        dist = Independent(Normal(state["mean"], state["std"]), 1)
        return dist

    def obs_step(
        self, prev_state: LatentBatch, prev_action: ActionBatch, embed: Tensor
    ) -> Tuple[LatentBatch, LatentBatch]:
        """Observation is the process of taking s_{t-1}, a_{t-1}, and o_{t} and generating s_{t}.
        In other words, estimating the current (unobserved) state given the current observation.

        This is what the posterior model does."""
        prior = self.img_step(prev_state, prev_action)

        # Compute the posterior
        # embed is the observation embedding
        x = torch.cat([prior["deter"], embed], -1)
        x = self._activation(self.obs1(x))
        x = self.obs2(x)
        mean, std = torch.chunk(x, 2, -1)
        std = F.softplus(std) + self.min_std
        stoch_dist = self.get_dist({"mean": mean, "std": std})
        stoch = stoch_dist.rsample()
        post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    def img_step(self, prev_state: LatentBatch, prev_action: ActionBatch) -> LatentBatch:
        """Imagination is the process of taking s_{t-1} and a_{t-1} and generating s_{t}.
        In other words, imagining what the next state will be given the previous state and action.

        This is what the prior model does."""
        # Compute the RNN (this is the deterministic part)
        # prev_action = prev_action["wrapped"]
        # Flatten actions out into a vector per batch element. Batch must always only have 1 axis here.
        prev_action = torch.cat([x.reshape([x.shape[0], -1]) for x in prev_action.values()], dim=-1)
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self._activation(self.img1(x))
        deter = self._cell(x, prev_state["deter"])

        # Compute the prior
        y = self._activation(self.img2(deter))
        y = self.img3(y)
        mean, std = torch.chunk(y, 2, -1)
        std = F.softplus(std) + self.min_std
        stoch = self.get_dist({"mean": mean, "std": std}).rsample()
        prior = {"mean": mean, "std": std, "stoch": stoch, "deter": deter}
        return prior

    def kl_loss(
        self, post: LatentBatch, prior: LatentBatch, balance: float = 0.8, free: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        kld = torch.distributions.kl_divergence
        sg = lambda x: map_structure(torch.detach, x)
        value_lhs = value = kld(self.get_dist(post), self.get_dist(sg(prior)))
        value_rhs = kld(self.get_dist(sg(post)), self.get_dist(prior))
        loss_lhs = torch.clamp(value_lhs.mean(), min=free)
        loss_rhs = torch.clamp(value_rhs.mean(), min=free)
        loss = (1 - balance) * loss_lhs + balance * loss_rhs
        return loss, value


def is_image_space(x: gym.spaces.Space) -> bool:
    return isinstance(x, gym.spaces.Box) and len(x.shape) == 3


def is_vector_space(x: gym.spaces.Space) -> bool:
    return isinstance(x, gym.spaces.Box) and len(x.shape) == 1


class HybridEncoder(nn.Module):
    """Takes a dict obs space composed of image and vector Box spaces, and encodes it all to a single latent vector."""

    def __init__(self, obs_space: gym.spaces.Dict, embedding_dim: int):
        super().__init__()
        assert isinstance(obs_space, gym.spaces.Dict)
        self.obs_space = obs_space
        self.out_dim = embedding_dim

        self.image_keys = [k for k, v in obs_space.items() if is_image_space(v)]
        self.vector_keys = [k for k, v in obs_space.items() if is_vector_space(v)]
        assert len(obs_space) == len(self.image_keys) + len(self.vector_keys)

        if self.image_keys:
            # We expect images to be (c, h, w)
            img_size = obs_space[self.image_keys[0]].shape[1]
            self.img_channels = 0
            for key in self.image_keys:
                space = obs_space[key]
                assert space.shape[1] == img_size and space.shape[2] == img_size
                self.img_channels += space.shape[0]

            self.image_encoder = ConvEncoder(self.img_channels, img_size)
            # linear layer to match the conv output shape to the desired embedding shape
            self.image_encoder_fc = mlp_init(torch.nn.Linear(self.image_encoder.output_dim, self.out_dim))

        if self.vector_keys:
            self.vector_dim = 0
            for key in self.vector_keys:
                space = obs_space[key]
                self.vector_dim += space.shape[0]

            # TODO: consider orthonormal init here
            # TODO: move out this magic number to config
            # TODO: Dreamerv2 follows this with an activation
            self.vector_encoder_fc = MLP(self.vector_dim, hidden_dim=200, output_dim=self.out_dim, num_layers=3)

    def forward(self, obs: ObservationBatch) -> Tensor:
        encodings = []
        if self.image_keys:
            image_parts = [obs[key] for key in self.image_keys]
            # Ensure that we applied the post-transform properly everywhere
            assert all([x.dtype == torch.float32 for x in image_parts])
            image_obs = torch.cat(image_parts, dim=-3)
            img_encoding = self.image_encoder(image_obs)
            # TODO: dreamerv2 would follow this with an activation. (well actually it just wouldn't have this layer)
            img_encoding = self.image_encoder_fc(img_encoding)
            assert img_encoding.shape[-1] == self.out_dim
            encodings.append(img_encoding)

        if self.vector_keys:
            vector_parts = [obs[key] for key in self.vector_keys]
            vector_obs = torch.cat(vector_parts, dim=-1)
            assert vector_obs.shape[-1] == self.vector_dim
            encodings.append(self.vector_encoder_fc(vector_obs))

        encoding = torch.stack(encodings, dim=0).sum(dim=0)
        assert encoding.shape[-1] == self.out_dim
        return encoding


class HybridDecoder(nn.Module):
    """The inverse of the HybridEncoder"""

    def __init__(self, obs_space: gym.spaces.Dict, latent_dim: int, skip_keys: Tuple[str] = ()):
        super().__init__()
        assert isinstance(obs_space, gym.spaces.Dict)
        self.obs_space = obs_space

        self.image_keys = [k for k, v in obs_space.items() if is_image_space(v) if k not in skip_keys]
        self.vector_keys = [k for k, v in obs_space.items() if is_vector_space(v) if k not in skip_keys]
        skipped_keys = [k for k in obs_space.keys() if k in skip_keys]
        assert len(obs_space) == len(self.image_keys) + len(self.vector_keys) + len(skipped_keys)

        if self.image_keys:
            # We expect images to be (c, h, w)
            img_size = obs_space[self.image_keys[0]].shape[1]
            self.img_channels = 0
            for key in self.image_keys:
                space = obs_space[key]
                assert space.shape[1] == img_size and space.shape[2] == img_size
                self.img_channels += space.shape[0]
            # The decoder already has a linear to adapt the input size
            self.image_decoder = ConvDecoder(latent_dim, self.img_channels)

        if self.vector_keys:
            self.vector_dim = 0
            for key in self.vector_keys:
                space = obs_space[key]
                self.vector_dim += space.shape[0]
            # TODO: consider orthonormal init here
            self.vector_decoder = MLP(latent_dim, hidden_dim=200, output_dim=self.vector_dim, num_layers=3)

    def forward(self, latent: Tensor) -> ObservationBatch:
        # We're only going to handle latents of shape (b, t, latent_dim) here
        assert len(latent.shape) == 3
        batch_size, timesteps = latent.shape[:2]
        out = {}
        if self.image_keys:
            decoded_img = self.image_decoder(latent)
            # assert decoded_img.shape == (batch_size, timesteps, self.img_channels, 64, 64)
            start_channel = 0
            for key in self.image_keys:
                channels = self.obs_space[key].shape[0]
                out[key] = decoded_img[:, :, start_channel : start_channel + channels]
                start_channel += channels

        if self.vector_keys:
            decoded_vec = self.vector_decoder(latent)
            assert decoded_vec.shape == (batch_size, timesteps, self.vector_dim)
            start_dim = 0
            for key in self.vector_keys:
                dims = self.obs_space[key].shape[0]
                out[key] = decoded_vec[:, :, start_dim : start_dim + dims]
                start_dim += dims

        return out


class ConvEncoder(nn.Module):
    # TODO: clean up this class
    def __init__(self, input_channels: int = 3, input_res: int = 96, act=F.relu):
        super().__init__()
        self._act = act
        self.input_channels = input_channels
        assert input_res == 96
        self.input_res = input_res

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        self.output_dim = 1024
        self.apply(init_weights)

    def __call__(self, obs: Tensor) -> Tensor:
        # This can get called with shape (b, c, h, w) or (b, t, c, h, w)
        # TODO: clean up this double-shape code path?
        x = obs
        # Convert (b, t, c, h, w) to (b, c, h, w) if starting with the former
        x = torch.reshape(x, (-1,) + x.shape[-3:])
        assert x.shape[1:] == (self.input_channels, self.input_res, self.input_res)
        x = self._act(self.conv1(x))
        x = self._act(self.conv2(x))
        x = self._act(self.conv3(x))
        x = self._act(self.conv4(x))
        # Reshape back to the original b/t shape, flatten out the rest
        shape = obs.shape[:-3] + (1024,)
        x = torch.reshape(x, shape)
        return x


class ConvDecoder(nn.Module):
    # TODO: clean up this class
    def __init__(self, input_dim: int, out_channels: int, depth: int = 32, res: int = 96, act=F.relu):
        super().__init__()
        self._act = act
        # TODO: un-hardcode this stuff
        self._depth = depth
        assert depth == 32
        assert res == 96
        self.res = res
        self._shape = (out_channels, self.res, self.res)
        self.out_channels = out_channels

        embedding_size = 32 * self._depth

        # TODO: this is how dreamerv2 did it, but i'm not a huge fan of the asymmetric encoder/decoder.
        # And weird that we start with a [1024, 1, 1] shape vs [256, 2, 2] like the encoder outputs.
        self.fc1 = nn.Linear(input_dim, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, kernel_size=4, stride=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=6, stride=3, padding=1)
        self.conv5 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=2)

        self.apply(init_weights)

    def __call__(self, features: Tensor) -> Tensor:
        # c is stoch + deter
        b, t, c = features.shape
        x = self.fc1(features)
        x = rearrange(x, "b t c -> (b t) c 1 1", b=b, t=t, c=32 * self._depth)
        x = self._act(self.conv1(x))
        x = self._act(self.conv2(x))
        x = self._act(self.conv3(x))
        x = self._act(self.conv4(x))
        x = self.conv5(x)
        dist_mean = rearrange(x, "(b t) c h w-> b t c h w", b=b, t=t, h=self.res, w=self.res, c=self.out_channels)
        # shape is [50, 50, self.res, self.res, 3]
        return dist_mean
        # return Independent(Normal(mean, 1), len(self._shape))


class DenseDecoder(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, ...],
        layers: int,
        in_dim: int,
        units: int,
        dist: Literal["normal", "binary"] = "normal",
        act=F.elu,
    ):
        super().__init__()
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

        input_layer = nn.Linear(in_dim, self._units)
        # they don't seem to count the output layer in num_layers
        layers = [nn.Linear(self._units, self._units) for _ in range(self._layers - 1)]
        self.module_list = nn.ModuleList([input_layer] + layers)
        self.output_layer = nn.Linear(self._units, int(np.prod(self._shape)))

        self.apply(init_weights)

    def __call__(self, features: Tensor, raw_feats=False):
        # TODO: fix this to consistently return the same type all the time.
        x = features
        # TODO: review this logic more carefully. Replace with a clean MLP class.
        for layer in self.module_list:
            x = self._act(layer(x))
        x = self.output_layer(x)
        x = x.view(features.shape[:-1] + self._shape)
        if raw_feats:
            return x
        if self._dist == "normal":
            # The std 1 here is kinda weird. Why not let it predict its own std?
            return Independent(Normal(x, 1), len(self._shape))
        if self._dist == "binary":
            # x is just the batch shape, so this will properly construct a dist with proper batch shape and scalar event shape
            return Bernoulli(logits=x)
        raise NotImplementedError(self._dist)


class ActionDecoder(nn.Module):
    def __init__(self, params, action_space, layers, in_dim, units, act=F.elu):
        super().__init__()
        self._layers = layers
        self._units = units
        self._act = act

        self.action_head = DictActionHead(action_space, params)

        input_layer = nn.Linear(in_dim, self._units)
        layers = [nn.Linear(self._units, self._units) for _ in range(self._layers - 1)]
        self.module_list = nn.ModuleList([input_layer] + layers)
        self.output_layer = nn.Linear(self._units, self.action_head.num_inputs)

        self.apply(init_weights)

    def forward(self, x, return_raw=False):
        for layer in self.module_list:
            x = self._act(layer(x))
        x = self.output_layer(x)
        dist = self.action_head(x)

        # TODO: clean this up
        if return_raw:
            return dist, x, None, None
        else:
            return dist
