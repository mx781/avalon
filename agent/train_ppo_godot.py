import torch

from agent.common.parse_args import parse_args
from agent.common.trainer import OnPolicyTrainer
from agent.godot.godot_eval import test
from agent.godot.godot_gym import GodotEnvironmentParams
from agent.godot.godot_gym import TrainingProtocolChoice
from agent.ppo.params import PPOParams

if __name__ == "__main__":
    num_steps = 200
    num_workers = 16
    params = PPOParams(
        total_env_steps=50_000_000,
        num_steps=num_steps,
        num_workers=num_workers,
        batch_size=num_steps * num_workers,  # 3200 ~= 20GB GPU memory
        ppo_epochs=2,
        discount=0.99,
        lam=0.83,
        value_loss_coef=1,
        entropy_coef=1.5e-4,
        clip_range=0.03,
        lr=2.5e-4,
        clip_grad_norm=0.5,
        env_params=GodotEnvironmentParams(
            task_difficulty_update=3e-4,
            meta_difficulty_update=3e-5,
            is_meta_curriculum_used=False,
            energy_cost_coefficient=1e-8,
            training_protocol=TrainingProtocolChoice.MULTI_TASK_BASIC,
            test_episodes_per_task=101,
            fixed_world_max_difficulty=0.5,
        ),
        log_freq_hist=500,
        log_freq_scalar=50,
        log_freq_media=500,
        checkpoint_every=2500,
    )
    # log_utils.enable_debug_logging()
    assert params.num_batches == 1
    params = parse_args(params)

    trainer = OnPolicyTrainer(params)
    try:
        if params.is_training:
            trainer.train()
            trainer.shutdown()
            trainer.train_storage.reset()
            torch.cuda.empty_cache()  # just for seeing what's going on
        if params.is_testing:
            test(trainer.params, trainer.algorithm, log=True, exploration_mode="eval")

    finally:
        trainer.shutdown()
