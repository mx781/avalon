
Gym mujoco state-based:
`PYTHONPATH=$PYTHONPATH:../../science:../../computronium python -m agent.main_ppo --project zack_test --env gym_Reacher-v2  --model mlp`
- shared mem impl:
    - (8 workers, 256 steps, 4 batches, 10 PPO epochs)
        - with training disabled: 4000 env steps/sec
        - with training enabled: ~3k env steps/sec
    - new storage, new action space:
        - with training: ~3k
    - (32 workers, 256 steps, 4 batches, 10 PPO epochs)
        - with training disabled: 8000 env steps/sec
            - this is ~1 iteration/sec
            - not terrible but i would expect the scaling to be linear in num workers. and it's dramatically not.
    - (32 workers, 256 steps, 4 batches, 10 PPO epochs)
        - with training disabled: 10000 env steps/sec

For pixel-based Gym mujoco envs:
`sudo apt install xvfb python-opengl ffmpeg`
`PYTHONPATH=$PYTHONPATH:../../science:../../computronium xvfb-run -a -s "-screen 0 1400x900x24" python -m quarantine.zack.zack_ppo_dict.main_ppo --project zack_test --env gym_Reacher-v2 --pixel_obs_wrapper True --mp_method forkserver --model cnn`

For dummy pixel env:
`PYTHONPATH=$PYTHONPATH:../../science:../../computronium python -m quarantine.zack.zack_ppo_dict.main_ppo --project zack_test --env test_dummy  --model cnn --num_workers 8`
- shared mem impl:
    - (8 workers, 256 steps, 4 batches, 10 PPO epochs)
        - no training: 2000 env steps/sec
        - no policy actual model run: 2200
        - no sending obs to cuda: 4000
        - no policy eval: 10000
            - same thing but with pipe instead of shared mem: 8000
        - also no wandb, no gae: 17000
    - reworked (8 workers, 256 steps, 4 batches, 10 PPO epochs)
        - with cuda: 7500
        - with model:
        - with all but training: 4000
    - after new storage, new action space:
        - all but training: 3600
        - with training: 2000
    - (32 workers, 256 steps, 4 batches, 10 PPO epochs)
        - with cuda: 6500
        - with model: 4500


For gym envs:
- `pip install -U gym`
- `pip install -U numpy`
- but will have to downgrade back to 0.21.0 to get godot working, probably.


For Godot:
- PYTHONPATH=$PYTHONPATH:../../science:../../computronium:../../bones python -m agent.train_ppo_godot
- tracing:
  - PYTHONPATH=$PYTHONPATH:../../science:../../computronium viztracer --ignore_c_function --ignore_frozen --min_duration 50us --max_stack_depth 25 -m agent.train_ppo_godot --num_workers 8 --num_steps 100 --total_env_steps 1600

- to evaluate on fixed worlds:
- source /opt/secrets/environment_vars/bashenv_secrets.sh
- PYTHONPATH=$PYTHONPATH:../../science:../../computronium:../../bones python -m agent.train_ppo_godot --is_training False --is_testing True --env_params.fixed_worlds_s3_key avalon_worlds__0824_full.tar.gz --resume_from_run 1wli6rk8 --resume_from_project zack_zack_ppo --env_params.fixed_worlds_load_from_path /mnt/private/avalon/worlds/viewable_worlds/

For Bones:
- PYTHONPATH=$PYTHONPATH:../../science:../../computronium:../../bones python opt_ppo.py
  train_once
- python science/bin/science launch_experiment --project standalone/avalon --cluster_spec provider:physical,priority:4000,gpus:0 --name zack_ppo_opt1 --command "opt_ppo.py run"

Data sequence for a single timestep: obs, computed value + action/policy from that obs -> step -> reward, done


The implementation here is designed to match sb3 exactly, at least the codepaths that have been tested:
- state-based observation mujoco gym environments
- discrete and continuous actions


Not tested:
- atari
- frame skipping
- end_on_life_loss (best practice is to not use this anyways)

