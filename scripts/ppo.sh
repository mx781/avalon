#!/bin/bash
set -e
set -u

CODE=/opt/projects/avalon
DATA=/mnt/private

sudo mkdir -p -m 777 "${DATA}"

export PATH=/opt/venv/bin:$PATH
export PYTHONPATH=$CODE

exec python -m agent.train_ppo_godot --is_training True --is_testing True
