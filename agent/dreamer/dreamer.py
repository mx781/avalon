import itertools
from typing import Any
from typing import Literal
from typing import Optional
from typing import Tuple

import gym
import torch
from einops import rearrange
from einops import repeat
from torch import Tensor
from torch import nn
from torch.distributions import Independent
from torch.distributions import Normal
from tree import flatten
from tree import map_structure

from agent.common import wandb_lib
from agent.common.action_model import DictActionDist
from agent.common.action_model import StraightThroughOneHotCategorical
from agent.common.types import ActionBatch
from agent.common.types import Algorithm
from agent.common.types import AlgorithmInferenceExtraInfoBatch
from agent.common.types import BatchSequenceData
from agent.common.types import LatentBatch
from agent.common.types import ObservationBatch
from agent.common.types import StateActionBatch
from agent.common.util import explained_variance
from agent.dreamer.models import RSSM
from agent.dreamer.models import ActionDecoder
from agent.dreamer.models import DenseDecoder
from agent.dreamer.models import HybridDecoder
from agent.dreamer.models import HybridEncoder
from agent.dreamer.params import DreamerParams
from agent.dreamer.tools import lambda_return
from agent.dreamer.tools import pack_list_of_dicts


class Dreamer(Algorithm[DreamerParams]):
    def __init__(self, params: DreamerParams, obs_space: gym.spaces.Dict, action_space: gym.spaces.Dict):
        super().__init__(params, obs_space, action_space)

        self._encode = HybridEncoder(obs_space, self.params.embed_size)
        feature_size = self.params.deter_size + self.params.stoch_size
        self._decode = HybridDecoder(obs_space, feature_size)
        self._reward = DenseDecoder((), 2, feature_size, self.params.num_units)
        # TODO: parameterize the number of layers of all this stuff.
        # And figure out what the right default val should be.
        # self._value_current = DenseDecoder((), 3, feature_size, self._c.num_units)
        # self._value_lagged = DenseDecoder((), 3, feature_size, self._c.num_units)
        self._value_current = DenseDecoder((), 2, feature_size, self.params.num_units)
        self._value_lagged = DenseDecoder((), 2, feature_size, self.params.num_units)
        self._value_lagged.load_state_dict(self._value_current.state_dict())

        # self._actor = ActionDecoder(action_space, 4, feature_size, self._c.num_units)
        self._actor = ActionDecoder(params, action_space, 2, feature_size, self.params.num_units)
        self._actdim = self._actor.action_head.num_outputs
        self._dynamics = RSSM(
            self.params.embed_size,
            self.params.stoch_size,
            self.params.deter_size,
            self.params.deter_size,
            actdim=self._actdim,
        )
        if self.params.pcont:
            # Will use a Bernoulli output dist
            self._pcont = DenseDecoder((), 2, feature_size, self.params.num_units, "binary")

        model_modules = [self._encode, self._dynamics, self._decode, self._reward]
        if self.params.pcont:
            model_modules.append(self._pcont)
        self._model_params = list(itertools.chain(*[list(module.parameters()) for module in model_modules]))
        self._model_opt = torch.optim.Adam(self._model_params, self.params.model_lr, eps=1e-07)
        self._value_opt = torch.optim.Adam(self._value_current.parameters(), self.params.value_lr, eps=1e-07)
        self._actor_opt = torch.optim.Adam(self._actor.parameters(), self.params.actor_lr, eps=1e-07)

        # This is a tuple of (latent, action)
        self.last_rollout_state: Optional[StateActionBatch] = None

    def rollout_step(
        self,
        next_obs: ObservationBatch,
        dones: Tensor,
        indices_to_run: list[bool],
        exploration_mode: Literal["explore", "eval"],
    ) -> Tuple[ActionBatch, AlgorithmInferenceExtraInfoBatch]:
        # Only used in the GamePlayer
        # dones should be "did this env give a done signal after the last step".
        # In other words, obs should follow done. o_t, done_{t-1}

        # This computes the policy given an observation
        # The inputs should all be tensors/structures of tensors on GPU

        if self.last_rollout_state is None:
            device = next_obs[list(next_obs.keys())[0]].device
            batch_size = len(dones)
            prev_latent = self._dynamics.initial(batch_size, device)
            # Using random actions as an initial action probably isn't ideal, but zero isn't a valid 1-hot action..
            # so that seemed worse.
            prev_action = self.action_space.sample()
            prev_action = map_structure(lambda x: torch.tensor(x, device=device), prev_action)
            prev_action = map_structure(lambda x: repeat(x, "... -> b ...", b=batch_size), prev_action)
            self.last_rollout_state = (prev_latent, prev_action)

        # next_obs = {k: v[indices_to_run] for k, v in next_obs.items()}
        # Check the batch sizes match - that we're not carrying over an old rollout state.
        assert dones.shape[0] == flatten(self.last_rollout_state)[0].shape[0]

        # We want to set done to false for anything that claims to be done but isn't running this step.
        # This will result in no masking for those states.
        indices_to_run_torch = torch.tensor(indices_to_run, device=dones.device, dtype=torch.bool)
        dones = dones & indices_to_run_torch
        dones = dones.to(dtype=torch.float32)

        # Mask the state to 0 for any envs that have finished?
        mask = 1 - dones

        # we need this because the action can have a variable number of dims
        def multiply_vector_along_tensor_batch_dim(x: Tensor, vector: Tensor) -> Tensor:
            assert len(vector.shape) == 1
            extra_dims = (1,) * (x.dim() - 1)
            return x * vector.view(-1, *extra_dims)

        self.last_rollout_state = map_structure(
            lambda x: multiply_vector_along_tensor_batch_dim(x, mask) if x is not None else x, self.last_rollout_state
        )
        assert self.last_rollout_state is not None

        # sliced_state = map_structure(lambda x: x[indices_to_run], self.last_rollout_state)
        action, new_state = self.policy(next_obs, self.last_rollout_state, mode=exploration_mode)
        # action, new_state = self.policy(next_obs, sliced_state, mode="explore")

        # Not sure if map_structure works nicely with inplace operations, so we'll do it manually.
        for k1, v1 in enumerate(self.last_rollout_state):
            for k2, v2 in v1.items():
                v2[indices_to_run] = new_state[k1][k2][indices_to_run]

        action = map_structure(lambda x: x.cpu(), action)
        return action, AlgorithmInferenceExtraInfoBatch()

    def reset_state(self) -> None:
        self.last_rollout_state = None

    def policy(
        self, obs: ObservationBatch, prev_state: StateActionBatch, mode: Literal["train", "eval", "explore"] = "train"
    ) -> Tuple[ActionBatch, StateActionBatch]:
        """Encode the observation, pass it into the observation model along with the previous state/action
        to generate a new state estimate, and use that to generate a policy.

        state is a tuple of (latent, action)"""

        # Obs is/can be numpy array
        assert prev_state is not None
        prev_latent, prev_action = prev_state

        embed = self._encode(obs)
        latent, _ = self._dynamics.obs_step(prev_latent, prev_action, embed)
        feat = self._dynamics.get_feat(latent)
        if mode == "train":
            action = self._actor(feat).rsample()
        elif mode == "explore":
            action = self._actor(feat).rsample()
        elif mode == "eval":
            action = self._actor(feat).mode()
        else:
            assert False
        state = (latent, action)
        return action, state

    def _imagine_ahead(self, start: LatentBatch, dones: Tensor) -> dict[str, Any]:
        # In the sequence, at a given index, it's an (action -> state) pair. action comes first.
        # Thus the dummy "0" action at the front.

        start = {k: rearrange(v, "b t n -> (b t) n") for k, v in start.items()}
        start["feat"] = self._dynamics.get_feat(start)
        start["action"] = {k: torch.zeros_like(v) for k, v in self._actor(start["feat"]).rsample().items()}  # type: ignore
        seq = {k: [v] for k, v in start.items()}
        for _ in range(self.params.horizon):
            action = self._actor(seq["feat"][-1].detach()).rsample()
            state = self._dynamics.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self._dynamics.get_feat(state)
            for key, value in {**state, "action": action, "feat": feat}.items():
                seq[key].append(value)
        # These now have shape (imag_steps, batch_size * fragment_steps)
        seq_packed = {k: torch.stack(v, 0) for k, v in seq.items() if k != "action"}
        if self.params.pcont:
            disc = self.params.discount * self._pcont(seq_packed["feat"]).probs
            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            dones = rearrange(dones, "b t -> (b t)")
            true_first = 1.0 - dones.float()
            true_first *= self.params.discount
            disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = self.params.discount * torch.ones(seq_packed["feat"].shape[:-1], device=seq_packed["feat"].device)
        seq_packed["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        # TODO: I don't like how the same discount factor is used for value discounting and for this weight.
        # Seems like they have different purposes and should be treated separately.
        seq_packed["weight"] = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0)
        seq_packed["action"] = pack_list_of_dicts(seq["action"])  # type: ignore
        return seq_packed

    def train_step(self, batch_data: BatchSequenceData, step: int) -> int:
        self.train()
        # Shape of batch_data elements should be (batch_size, timesteps, ...)
        # Images should be (c, h, w), with range [-.5, .5]

        next_obs = batch_data.observation
        actions = batch_data.action
        rewards = batch_data.reward
        is_terminal = batch_data.is_terminal

        batch_size, timesteps = is_terminal.shape

        # Train the encoder and RSSM model. The actor and value models are not used anywhere here.
        embed = self._encode(next_obs)
        # This is where the loop over all timesteps happens
        post, prior = self._dynamics.observe(embed, actions)
        feat = self._dynamics.get_feat(post)
        assert len(feat.shape) == 3
        obs_pred = self._decode(feat)
        # Reinterpret all but the batch dim (no time dim here)
        # Note: log_likelihood of a Normal with constant std can be reinterpreted as a scaled MSE.
        # is a Normal with std 1 appropriate for vectors too? I guess why not, esp since MSE would seem appropriate.
        obs_dists = {k: Independent(Normal(mean, 1), len(mean.shape) - 2) for k, mean in obs_pred.items()}
        obs_likelihoods = {k: v.log_prob(next_obs[k]) for k, v in obs_dists.items()}
        assert all([v.shape == (batch_size, timesteps) for v in obs_likelihoods.values()])
        autoencoder_mask = 1 - is_terminal.int()
        assert autoencoder_mask.shape == (batch_size, timesteps)

        if step % 1000 == 0:
            # Note: these logs will havee some terms that are masked out in the loss.
            # No great way to log those, i guess we would reshape and slice them out if we were being proper.
            for k, pred in obs_pred.items():
                target = next_obs[k]
                wandb_lib.log_histogram(
                    f"train/observations/{k}_pred", pred, step, log_mean=False, mean_freq=1, hist_freq=1
                )
                wandb_lib.log_histogram(
                    f"train/observations/{k}_target", target, step, log_mean=False, mean_freq=1, hist_freq=1
                )
                # Log EVs of scalar observations
                if len(target.shape) == 3 and target.shape[-1] == 1:
                    ev = explained_variance(pred, target)
                    wandb_lib.log_scalar(f"train/observations/{k}_EV", ev, step, freq=1)

        reward_pred = self._reward(feat)
        # Note we mask out autoencoding of the terminal timestep.
        # This is because that observation actually comes from the start of the next episode (or is masked to grey,
        # as we actually do). We don't want to try to predict this, if we need to know if the episode ended,
        # we have a pcont signal for that.
        # For done but non-terminal (time limit), we should still be given the true next frame, so predicting that is fine.
        likelihoods: dict[str, Tensor] = {}
        likelihoods["obs"] = sum([(x * autoencoder_mask).mean() for x in obs_likelihoods.values()])
        likelihoods["reward"] = reward_pred.log_prob(rewards).mean()
        likelihoods["reward"] *= self.params.reward_loss_scale
        if self.params.pcont:
            # this is "probability of continue" - an estimator of the done signal
            pcont_pred = self._pcont(feat)
            # Label smoothing option
            # pcont_target = 1 - 0.9 * is_terminal.float()
            pcont_target = 1 - is_terminal.float()
            loss = -torch.binary_cross_entropy_with_logits(pcont_pred.logits, pcont_target)
            likelihoods["pcont"] = loss.mean()
            likelihoods["pcont"] *= self.params.pcont_loss_scale

            wandb_lib.log_histogram("train/model/pcont/pred", pcont_pred.probs, step)
            wandb_lib.log_scalar("train/model/pcont/target", pcont_target.mean(), step)
            wandb_lib.log_scalar("train/model/pcont/loss", -1 * likelihoods["pcont"], step)

        reward_ev = explained_variance(reward_pred.mean, rewards.float())
        wandb_lib.log_scalar("train/reward/ev", reward_ev, step)
        wandb_lib.log_histogram("train/reward/target", rewards.float(), step)
        wandb_lib.log_histogram("train/reward/pred", reward_pred.mean, step)

        prior_dist = self._dynamics.get_dist(prior)
        post_dist = self._dynamics.get_dist(post)

        # Dreamerv1 approach
        # FWIW, the dreamerv1 approach works differently than the dreamerv2 approach with balance=.5,
        # which doesn't make much sense. Same with kl_scale=2 to account for smaller grads.
        # Emperically, the v1 approach did better at generating high reward EVs in dmc_cartpole_balance
        # div = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
        # div_clipped = torch.maximum(div, torch.tensor(self._c.free_nats))
        # kl_loss = div_clipped
        # wandb_lib.log_scalar("train/model/div", div, step)

        # Dreamerv2 approach.
        kl_loss, kl_value = self._dynamics.kl_loss(
            post, prior, balance=self.params.kl_balance, free=self.params.free_nats
        )
        assert len(kl_loss.shape) == 0
        wandb_lib.log_scalar("train/model/div", kl_value.mean(), step)

        model_loss = self.params.kl_scale * kl_loss - sum(likelihoods.values())
        wandb_lib.log_scalar("train/model/kl_loss", self.params.kl_scale * kl_loss, step)

        self._model_opt.zero_grad(set_to_none=True)
        model_loss.backward()
        model_norm = nn.utils.clip_grad_norm_(self._model_params, self.params.clip_grad_norm)
        self._model_opt.step()

        wandb_lib.log_histogram("train/model/prior_ent", prior_dist.entropy(), step)
        wandb_lib.log_histogram("train/model/post_ent", post_dist.entropy(), step)
        for name, logprob in likelihoods.items():
            wandb_lib.log_scalar(f"train/model/{name}_loss", -logprob.mean(), step)
        wandb_lib.log_scalar("train/model/loss", model_loss.mean(), step)
        wandb_lib.log_scalar("train/model/grad_norm", model_norm.mean(), step)

        ## IMAGINE #############################

        if (self.params.actor_lr > 0 or self.params.value_lr > 0) and step > self.params.freeze_actor_steps:
            # Train the actor model
            seq = self._imagine_ahead({k: v.detach() for k, v in post.items()}, is_terminal)
            # These rewards have not seen any new actions.
            # So they are the rewards received *in* this timestep, ie before another action is taken.
            reward = self._reward(seq["feat"]).mean
            # NOTE: using the "target" value model here. This value computation is used as a target for training the value network too.
            # Wait, how does dynamics loss work if we're not backproping thru the value model?
            # Well, i guess gradients can still pass thru the lagged model even if it's not being trained?
            value = self._value_lagged(seq["feat"]).mean
            disc = seq["discount"]
            weight = seq["weight"].detach()
            # Skipping last time step because it is used for bootstrapping.
            # Value target i corresponds to the state in seq["feat"] i
            value_target = lambda_return(
                reward[:-1], value[:-1], disc[:-1], bootstrap=value[-1], lambda_=self.params.disclam, axis=0
            )
            assert len(value_target) == len(reward) - 1

        ## ACTOR #############################

        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2

        # Reward:      r0   [r1]  [r2]   r3
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.

        # Explaining the above:
        # A state is a (s, a) pair. So z1 is the state of the world right before we take action a1.
        # Let's think just about training the policy that predicted a1. a1 was predicted from z0.
        # The reward resulting from taking action a1 is r1. the value of state z1 (containing r1 and future rewards)
        # is v1.
        # The baseline is the value of the 'state' that existed before we decided on action a1. This doesn't exist
        # as a discrete state in this model, but instead we have state z0, which is the same logical state -
        # z0 is the state where we decided on the action before a1, and thus with a perfect model would know the resulting
        # state after stepping the env also.
        # So the baseline is v0.

        # Reinforce works by saying: if things worked out well in the rollout from this state, when action_x was taken,
        # update the policy to make it more likely to take that action.
        # "worked out well" specifically means "worked better than expected" - and "expected" is the value estimate
        # before seeing the action.
        # So we want the value at this state, the policy computed from this state,
        # and the action actually taken after observing that state.

        if self.params.actor_lr > 0 and step > self.params.freeze_actor_steps:
            policy = self._actor(seq["feat"][:-2].detach())
            if self.params.actor_grad == "dynamics":
                objective = value_target[1:]
            elif self.params.actor_grad == "reinforce":
                # Why do we recompute this here? We compute the same thing above. I guess different gradient flow somehow? but grads don't even go thru this.
                baseline = self._value_lagged(seq["feat"][:-2]).mean
                advantage = (value_target[1:] - baseline).detach()
                _action = {k: v[1:-1].detach() for k, v in seq["action"].items()}
                objective = policy.log_prob(_action) * advantage
            elif self.params.actor_grad == "hybrid":
                # Dynamics works well for continuous, REINFORCE works well for discrete.
                # We can compute REINFORCE just on the discrete actions. Can't split up the dynamics as easily tho
                # without recomputing the imagination rollout with carefully placed detach()es.
                # So here we just compute dynamics loss normally, and add the discrete-only REINFORCE loss.
                # TODO: this is hacky - the categorical should always be wrapped in an independent so this works..
                discrete_policy = DictActionDist(
                    {
                        k: v
                        for k, v in policy.dists.items()
                        if isinstance(v.base_dist, StraightThroughOneHotCategorical)
                    }
                )
                assert len(discrete_policy.dists) > 0

                baseline = self._value_lagged(seq["feat"][:-2]).mean
                advantage = (value_target[1:] - baseline).detach()
                discrete_action = {
                    k: v[1:-1].detach() for k, v in seq["action"].items() if k in discrete_policy.dists.keys()
                }
                discrete_reinforce_objective = discrete_policy.log_prob(discrete_action) * advantage
                dynamics_objective = value_target[1:]
                assert discrete_reinforce_objective.shape == dynamics_objective.shape
                # Could add weights here
                objective = discrete_reinforce_objective + dynamics_objective
            else:
                assert False

            actor_loss = -1 * objective * weight[:-2]

            # TODO: verify and switch back to regular entropy
            # I made this one to pull Normal actions towards center, in addition to controlling their entropy.
            # It's just entropy for discretes.
            actor_div = policy.distance_from_uninformative()
            # Entropy shape should be (batch * fragment_len, imag_len)
            actor_entropy = policy.entropy()
            # actor_div_loss = -1 * actor_entropy * self._c.policy_entropy_scale * weight[:-2]
            actor_div_loss = 1 * actor_div * self.params.policy_entropy_scale * weight[:-2]
            total_actor_loss = actor_loss + actor_div_loss

            self._actor_opt.zero_grad(set_to_none=True)
            total_actor_loss.mean().backward()
            actor_norm = nn.utils.clip_grad_norm_(self._actor.parameters(), self.params.clip_grad_norm)
            self._actor_opt.step()

            wandb_lib.log_histogram("train/actor/actor_loss", actor_loss, step)
            wandb_lib.log_scalar("train/actor/grad_norm", actor_norm.mean(), step)
            wandb_lib.log_scalar("train/actor/div_loss", actor_div_loss.mean(), step)
            wandb_lib.log_histogram("train/actor/entropy", actor_entropy, step)
            wandb_lib.log_histogram("train/actor/div_from_uninformative", actor_div, step)
            wandb_lib.log_histogram("train/actor/steps_imagined", seq["weight"].sum(dim=0), step)
            wandb_lib.log_histogram("train/reward/imagined", reward[:-1], step)

            # Visualize actions
            for k, v in self.action_space.items():
                if isinstance(v, gym.spaces.Discrete) and v.n == 2:
                    action_probs = policy.dists[k].probs
                    wandb_lib.log_histogram(f"train/imagine/actions/{k}", action_probs[..., 0], step)
                elif isinstance(v, gym.spaces.Box):
                    # TODO: really should split this by Normal dist type
                    means = policy.dists[k].mean
                    for dim in range(means.shape[-1]):
                        wandb_lib.log_histogram(f"train/imagine/actions/{k}_{dim}_mean", means[..., dim], step)
                    try:
                        # This won't work for a SampleDist
                        stds = policy.dists[k].stddev
                        for dim in range(means.shape[-1]):
                            wandb_lib.log_histogram(f"train/imagine/actions/{k}_{dim}_std", stds[..., dim], step)
                    except NotImplementedError:
                        pass

        ## VALUE #############################
        if self.params.value_lr > 0 and step > self.params.freeze_actor_steps:
            # Train the value model
            # I'm curious why they train the value model in imagination but not in the actual rollouts.
            # Oh, I guess it has to be on-policy and that's only the case in imagination.
            value_pred = self._value_current(seq["feat"].detach()[:-1], raw_feats=True)
            # isn't the log prob of a Normal with std 1 the same as MSE?
            # Not using an Independent is appropriate here since the event is a scalar and the batch is (t, b)
            value_pred = Normal(value_pred, 1)
            value_loss = -1 * (weight[:-1] * value_pred.log_prob(value_target.detach())).mean()

            self._value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_norm = nn.utils.clip_grad_norm_(self._value_current.parameters(), self.params.clip_grad_norm)
            self._value_opt.step()

            # Update the "target" network.
            # TODO: an EMA might be more appropriate.
            if step % 100 == 0:
                self._value_lagged.load_state_dict(self._value_current.state_dict())

            value_ev = explained_variance(value_pred.mean * weight[:-1], value_target.float() * weight[:-1])
            wandb_lib.log_scalar("train/value/ev", value_ev, step)
            # TODO: these histograms of things that are soft-masked don't really making sense.
            # Without the mask, we're showing hists containing junk. But if we apply the mask,
            # the hist ends up with wrong values. Ideally we'd make a "weighted hist".
            # In the meantime, just consider they'll be containing junk unless the imag rollouts rarely contain ep ends.
            wandb_lib.log_histogram("train/value/pred", value_pred.mean, step)
            wandb_lib.log_histogram("train/value/target", value_target, step)

            wandb_lib.log_scalar("train/value/grad_norm", value_norm.mean(), step)
            wandb_lib.log_scalar("train/value/loss", value_loss.mean(), step)
            wandb_lib.log_histogram("train/value/weight", weight, step)

        if "rgbd" in next_obs and step % self.params.log_freq_media == 0:
            batch_size = 6
            # Do an imagination rollout. Can we use the imagine_ahead logic instead of replicating here?
            truth = next_obs["rgbd"][:batch_size, :, :3] + 0.5
            recon = obs_pred["rgbd"][:batch_size, :, :3]
            # we observe the first 5 frames to estimate the state
            sliced_actions = {k: v[:batch_size, :5] for k, v in actions.items()}
            init, _ = self._dynamics.observe(embed[:batch_size, :5], sliced_actions)
            init = {k: v[:, -1] for k, v in init.items()}
            # Then do an imagination rollout from there
            actual_actions = {k: rearrange(v[:batch_size, 5:], "b t ... -> t b ...") for k, v in actions.items()}
            prior = self._dynamics.imagine(actual_actions, init)
            openl = self._decode(self._dynamics.get_feat(prior))["rgbd"][:, :, :3]
            # First 5 frames are recon, next are imagination
            model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
            error = (model - truth + 1) / 2
            openl = torch.cat([truth, model, error], 3)
            wandb_lib.log_video("video/openl", openl, step, freq=1, num_images_per_row=batch_size)

        return step + 1
