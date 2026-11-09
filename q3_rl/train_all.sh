#!/usr/bin/env bash
# Q3: 3 seeds x 3 training worlds, 3M steps each. The three worlds of one seed run in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
for seed in 0 1 2; do
  for world in physics learned mujoco; do
    $PY q3_rl/ppo.py --world $world --timesteps 3000000 --seed $seed > q3_rl/runs_${world}_s${seed}.log 2>&1 &
  done
  wait
done
