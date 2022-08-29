import attr

from agent.common.params import ClippedNormalMode
from agent.common.params import Params


@attr.s(auto_attribs=True, frozen=True)
class OffPolicyParams(Params):
    worker_managers: int = 1
    # ratio between these two effects sample efficiency
    rollout_steps: int = 500  # rollout steps per training iteration. this is per-worker.
    train_steps: int = 100  # train steps per training iteration
    min_fragment_len: int = 50  # min length of an episode fragment to train on
    max_fragment_len: int = 50  # max length of an episode fragment to train on

    data_dir: str = "/mnt/private/data/rollouts/"
    num_dataloader_workers: int = 4
    balanced_sampler: bool = True
    separate_ongoing: bool = True  # don't train on incomplete episodes
    prefill_steps: int = 5000
    replay_buffer_size_timesteps: int = 1_000_000
    obervation_compression: bool = False

    @property
    def replay_buffer_size_timesteps_per_manager(self) -> int:
        return int(self.replay_buffer_size_timesteps / self.worker_managers)


@attr.s(auto_attribs=True, frozen=True)
class DreamerParams(OffPolicyParams):

    # overrides
    project: str = "zack_dreamer_new"
    clip_grad_norm: float = 100.0
    reward_loss_scale: float = 10.0  # useful
    normal_std_from_model: bool = True
    clipped_normal_mode: ClippedNormalMode = ClippedNormalMode.SAMPLE_DIST
    discount: float = 0.98
    obs_first: bool = False

    # new dreamer stuff
    horizon: int = 15  # how long to roll out imagination
    stoch_size: int = 30
    deter_size: int = 200
    num_units: int = 400
    # TODO: dreamerv2 gets rid of this parameter and lets the effective embed size be dynamic
    # A lil tricky to implement tho bc we have to compute the output size of a conv.
    embed_size: int = 1024
    model_lr: float = 6e-4
    value_lr: float = 8e-5
    actor_lr: float = 8e-5
    free_nats: float = 1.0
    kl_scale: float = 1.0
    disclam: float = 0.95  # lambda for GAE
    # 1e-4 is the value used in dreamer for walker walk. Much larger in atari.
    policy_entropy_scale: float = 2e-3
    pcont: bool = True  # predict done signals
    pcont_loss_scale: float = 10.0  # weight of pcont loss
    actor_grad: str = "reinforce"  # which type of learning to use for the actor, "dynamics" to backprop through model
    # Note: .2 means most of the weight is on `post`, afaik. This is backwards from how it works in danijar's code
    # for unknown reasons.
    kl_balance: float = 0.2  # trains posterior faster than prior
    freeze_actor_steps: int = 500  # don't train the actor for the first n steps to let the world model stabilize a bit

    def __attrs_post_init__(self):
        # These should only be changed with great care.
        assert self.obs_first is False
        assert self.time_limit_bootstrapping is False
