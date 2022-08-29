import sys

from loguru import logger

from computronium.api import LOG_FORMAT_WITH_COMMAND_LABELS
from computronium.api import run_commands
from computronium.data_types import LabeledCommand
from computronium.machine_spec import MachineSpec

logger.remove()
logger.add(sys.stdout, format=LOG_FORMAT_WITH_COMMAND_LABELS, level="INFO")


gym_envs = [
    "Reacher-v2",
    "HalfCheetah-v3",
    "Hopper-v3",  # v2 also valid
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2",
    "Swimmer-v3",
    "Walker2d-v3",  # v2 also valid
    "CartPole-v1",
]

dmc_tasks = [
    "acrobot-swingup",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "cup-catch",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "hopper-hop",
    "hopper-stand",
    "pendulum-swingup",
    "quadruped-run",
    "quadruped-walk",
    "reacher-easy",
    "reacher-hard",
    "walker-walk",
    "walker-stand",
    "walker-run",
]
dmc_tasks = [f"{t.replace('-', '_')}" for t in dmc_tasks]

atari_tasks = [
    "adventure",
    "air_raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank_heist",
    "battle_zone",
    "beam_rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper_command",
    "crazy_climber",
    "defender",
    "demon_attack",
    "double_dunk",
    "elevator_action",
    "enduro",
    "fishing_derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice_hockey",
    "jamesbond",
    "journey_escape",
    "kaboom",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "montezuma_revenge",
    "ms_pacman",
    "name_this_game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private_eye",
    "qbert",
    "riverraid",
    "road_runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space_invaders",
    "star_gunner",
    "tennis",
    "time_pilot",
    "tutankham",
    "up_n_down",
    "venture",
    "video_pinball",
    "wizard_of_wor",
    "yars_revenge",
    "zaxxon",
]
atari_tasks = [f"atari_{task}" for task in atari_tasks]


secrets = "/opt/secrets/environment_vars/bashenv_secrets.sh"


def launch(commands: list[LabeledCommand], machine_spec: MachineSpec, max_workers: int):
    # machine_spec = parse_machine_spec(machine_spec, image)
    worker_count = min(len(commands), max_workers)
    run_commands(
        iter(commands), machine_spec, worker_count=worker_count, secret_file_path=secrets, is_retaining_workers=False
    )
