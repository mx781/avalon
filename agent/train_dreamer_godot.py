from agent.common.trainer import OffPolicyTrainer
from agent.common.util import parse_args
from agent.dreamer.params import DreamerParams
from agent.godot.godot_eval import test
from agent.godot.godot_gym import GodotEnvironmentParams
from agent.godot.godot_gym import TrainingProtocolChoice

if __name__ == "__main__":
    fragment_length = 18
    params = DreamerParams(
        total_env_steps=50_000_000,
        pcont=False,  # this is an important parameter
        worker_managers=4,
        num_workers=4,
        discount=0.98,
        clip_grad_norm=50,
        env_params=GodotEnvironmentParams(
            task_difficulty_update=5e-4,
            meta_difficulty_update=5e-5,
            energy_cost_coefficient=1e-4,
            training_protocol=TrainingProtocolChoice.MULTI_TASK_ALL,
            fixed_world_max_difficulty=0.5,
            val_episodes_per_task=6,
            # Make the "eating" always be part of the final fragment, so the end is at least theoretically predictable.
            # Will make episodes where the reward is gotten in < 2 frames be skipped, though.
            num_frames_alive_after_food_is_gone=fragment_length - 2,
            gpu_id=3,
        ),
        batch_size=100,
        log_freq_hist=2000,
        log_freq_scalar=25,
        log_freq_media=5000,
        checkpoint_every=2500,
        free_nats=3,
        kl_scale=15,
        policy_entropy_scale=2e-3,
        pcont_loss_scale=20,
        reward_loss_scale=20,
        kl_balance=0.1,
        disclam=0.9,
        stoch_size=60,
        freeze_actor_steps=500,
        min_fragment_len=fragment_length,
        max_fragment_len=fragment_length,
        train_gpu=0,
        inference_gpus=[1, 2],
        godot_gpu=3,
    )
    params = parse_args(params)
    trainer = OffPolicyTrainer(params)
    try:
        if params.is_training:
            trainer.train()
        if params.is_testing:
            test(trainer.params, trainer.algorithm, log=True, exploration_mode="eval")
    finally:
        trainer.shutdown()
