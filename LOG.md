# Log

## 2026-11-09, Step 0 (Dell)

- Machine: Dell, Ubuntu 24.04, Linux 6.17, Python 3.12.3, 24 CPU threads, NVIDIA RTX 4000 Ada.
- Inventory: `STATUS_2026-11-09.md`. Exam 1 cloned read-only from GitHub at `90150da`.
- Python environment: a virtual environment with `requirements.txt` (MuJoCo 3.16.0, PyTorch 2.7.0 CPU, Gymnasium 1.1.1, OpenCV 4.11.0).

## 2026-11-09, Steps 1 to 3 (Dell)

| step | command | outcome |
| :--- | :--- | :--- |
| MuJoCo dog | `exam2/mjdog.py` | Unitree `go2.xml` walked by the project's `mjx_s0` walker; 10 s of simulated walking in 0.09 s on one core |
| simulated dataset | `q2_data/collect_sim.py` (train 200, val 30, test 30, shift_isaac 30 episodes of 30 s) | 290 episodes, 0 falls, 6.3 s for the training split on 20 processes; 42 MB compressed |
| ensemble, first run | `q1_vision_ml/train_ensemble.py` | training finished; evaluation crashed (`operands could not be broadcast`, 21 against 24 inputs): the rollout dropped one command from the history. Fixed in `exam2/models.py` `Ensemble.rollout`, then retrained from scratch; the numbers matched the `--eval-only` run exactly |
| ensemble | same, 161 s | 0.5 s ahead, vx RMSE 0.033 ± 0.001 m/s against 0.060 ± 0.002 for the fitted lag-delay model (n = 30 episodes); fitted lag-delay delay 0 s, time constants 0.24 to 0.28 s |
| pyramid and optical flow | `q1_vision_ml/flow_pyramid.py` | on Exam 1's 4 real frames; 4 pyramid levels track 0.992 of corners within 1 px at a 64 px shift, single level 0.007 |
| sensor statistics | `q2_data/sensor_stats.py` | first table printed body-height sd as 0.0000; now printed with 2 significant digits (2.2e-06 m) |
| L-path replay | `q2_data/l_path_replay.py` | both learned walkers barely turn in place (2.8 and 7.8 deg of 90); the Isaac walker does not move at 0.2 m/s |
| small commands | `q2_data/small_commands.py` | MuJoCo dog with the `mjx_s0` walker matches the real forward (0.63 against 0.65) and lateral 0.3 m/s (0.56 against 0.57) shares. A 50 % rise-time column was dropped: its definition with the rate limiter differs from the real measurement |
| navigation, simulation against real | `q2_data/nav_sim_vs_real.py` | real 0.737 ± 0.101 (n = 19) against 0.780 to 0.835 in simulation (n = 200) |
| PPO from scratch | `q3_rl/ppo.py`, speed test of 82k steps per world | 5,780 (physics), 5,856 (learned), 3,978 (MuJoCo) steps per second with 16 processes. First try failed in the worker processes (`No module named 'go2x'`); the worker now adds the project path |
| PPO, 9 runs | `q3_rl/train_all.sh` | 3 seeds x 3 worlds x 3M steps, one seed's 3 worlds in parallel |
| tests | `python -m pytest -q tests` | 6 passed. On the Dell, `PYTHONPATH` points at ROS 2 Jazzy, whose pytest plugins fail to import (`No module named 'lark'`); run with `env -u PYTHONPATH` |
| references | arXiv API and Crossref | titles, authors and years checked for references 1 to 11, 14 and 15; 12 and 13 have no DOI |

## 2026-11-09, Step 3 continued and Step 6 (Dell)

| step | command | outcome |
| :--- | :--- | :--- |
| PPO, 9 runs | `q3_rl/train_all.sh` | all 9 finished 3,002,368 steps (664 to 922 s each). The wrapper's last line (`tail -1` over several logs) exited with an error after training; no run was affected |
| cross-evaluation | `q3_rl/cross_eval.py` | every PPO policy beats go-to-goal in every world; paired by scenario, the training worlds differ by less than two standard errors in the MuJoCo dog |
| fresh clone | `git clone --recurse-submodules`, then tests, `sensor_stats`, `flow_pyramid`, `nav_sim_vs_real`, `small_commands`, `l_path_replay`, `make_figures`, `collect_sim --split val`, `train_ensemble` | 6 tests passed; every result file and figure came out byte-identical, except `ensemble_eval.json`'s run time (161.3 against 277.8 s) |
| secrets scan | `git grep -nIiE "serial|token|password|api[_-]?key|secret|gho_|ghp_|GO2_IP"` and an IPv4 pattern, outside `project/` | no secrets, robot addresses or serial numbers; the project's `.env` does not exist on the Dell |
| sizes | `find . -size +10M` | largest file `q2_data/dataset/sim/train.npz`, 28.9 MB; repository without `.git` and `project/` about 60 MB |

## 2026-11-10, raw logs and the four remaining analyses (cloud session, linked to the MacBook)

| step | command | outcome |
| :--- | :--- | :--- |
| look for the raw logs on GitHub | `git ls-remote` of the project; every branch's tree | not on any branch (`data/` is git-ignored) |
| copy from the MacBook | the session's file bridge, read-only | 25 navigation run folders and a link check (2026-11-06), 16 motion-test folders (2026-10-17). The MacBook has no `Exam2` folder in the course folder |
| privacy scan | IPv4 pattern and key-name search (`serial`, `sn`, `ip`, `addr`, `host`, `ssid`, `key`) over every `.json` and `.jsonl` | no addresses or serial numbers; only `"host": "Aidan-MacBook"` in headers |
| velocity frame | `velocity` integrated against `position`, 24 runs | body frame: integrated end error 0.03 to 0.18 m against up to 2.6 m for a world-frame reading |
| camera videos | `ffmpeg -c:v libx264 -preset veryfast -crf 26`, frame counts compared before and after | 96 MB to 8 MB, every frame kept |
| unitree_mujoco | `git clone`, checkout `1eb6642` | for the MuJoCo-dog replay |
| analyses | `python q2_data/real_runs.py` (12.5 s) | odometry reads 0.827 ± 0.008 of the camera path; drift 0.146 m per m. Replay end error: physics model 0.16 m, MuJoCo dog 0.39 m, ensemble 0.48 m. Ensemble on real motion: best on the motion tests, worst on turning in the navigation runs |
| first figure | floor-plan panels | axis lines drawn across the panels; spines turned off |
| twin figure | `python q2_data/twin_frames.py` | camera frame and twin frame at the last policy step of 4 runs |
| tests | `python -m pytest -q tests` | 9 passed (3 new: the raw logs give the run table's counts, the odometry mapping, the motion-test loader) |
