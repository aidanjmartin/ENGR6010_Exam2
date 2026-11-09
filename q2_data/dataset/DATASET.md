# Go2 dataset for the Second Test

Three parts: real recordings of the Go2 X, simulated recordings of the MuJoCo Go2, and the per-run tables of the real robot runs. The simulated part follows the record layout of [`ShijieZhao36/unitree_go2_create_dataset`](https://github.com/ShijieZhao36/unitree_go2_create_dataset) (`cf01a90`). Its `robot_state` message holds `foot_force[4]`, `motor_q[12]`, `motor_dq[12]`, `motor_ddq[12]` and `motor_tau_est[12]`, recorded while a velocity command (`vx`, `vy`, `vyaw`) drives the robot. The simulated records keep those fields and add the command, the body velocity and the pose.

## 1. Real Go2 X, standing still (`real_exam1/`)

Recorded in Exam 1 (`aidanjmartin/ENGR6010_Exam1` `90150da`, `q3_apps/output/webrtc/` and `q2_sdk/output/`) over WebRTC on the wired network, robot standing, no commands. Copied unchanged; the robot's address was already replaced by `<GO2_IP>` in Exam 1 and appears nowhere here.

| file | rows | fields |
| :--- | ---: | :--- |
| `sportmodestate.csv` | 400 (19.9 s, about 20 Hz) | `t_s` (s since start), `mode`, `gait_type`, `body_height_m`, `vx`, `vy`, `vz` (m/s, the robot's own state estimate) |
| `lowstate.csv` | 20 (about 1 Hz) | `t_s`, `roll`, `pitch`, `yaw` (rad, IMU), `battery_soc_pct`, `power_v` (V), `foot_force_0` to `foot_force_3` (raw sensor units) |
| `camera_frames.jpg` | 4 frames | front camera over WebRTC (1280 x 720 stream), tiled 2 x 2 at 640 x 360 each, in time order left to right, top to bottom |
| `real_front_camera.jpg` | 1 frame | front camera over DDS (Ethernet, Unitree SDK `GetImageSample`), 960 x 540 |

## 2. Simulated Go2 (`sim/`)

Made by `q2_data/collect_sim.py`:
- **Robot:** Unitree's `go2.xml` (`unitree_mujoco` `1eb6642`) in MuJoCo 3.16.0, every collision geom.
- **Controller:** walked by the project's MuJoCo-trained walking policy (`project/results/phase2/mujoco_walk/mjx_s0/policy.pt`); `shift_isaac` uses the Isaac Lab walker (`project/results/phase2/walk_train/policy.pt`).
- **Timing:** 5 ms physics steps, walking policy at 50 Hz, records at 20 Hz.
- **Commands:** piecewise-constant holds of 0.5 to 3 s, plus stretches that change every 1 to 4 steps as a navigation policy does. Ranges: vx -0.5 to 0.8 m/s, vy -0.4 to 0.4 m/s, wz -1.0 to 1.0 rad/s. Each axis is zero with probability 0.3, and 15 % of holds are full stops.
- **Per episode:** foot friction drawn from 0.3 to 1.0, and a payload of 0 to 3 kg on the base.

| split | episodes | seeds | samples | walker | falls |
| :--- | ---: | :--- | ---: | :--- | ---: |
| `train` | 200 x 30 s | 0 to 199 | 120,000 | MuJoCo-trained (`mjx_s0`) | 0 |
| `val` | 30 x 30 s | 1000 to 1029 | 18,000 | MuJoCo-trained | 0 |
| `test` | 30 x 30 s | 2000 to 2029 | 18,000 | MuJoCo-trained | 0 |
| `shift_isaac` | 30 x 30 s | 3000 to 3029 | 18,000 | Isaac Lab walker (not in training) | 0 |

Each `<split>.npz` (compressed numpy) holds arrays of shape (episodes, 600, ...) at 20 Hz; samples past `length` are zero.

| key | shape | units, meaning |
| :--- | :--- | :--- |
| `cmd` | (E, T, 3) | velocity command held during the next 0.05 s: vx, vy (m/s, body frame), wz (rad/s) |
| `vel` | (E, T, 3) | body velocity at the sample: vx, vy (m/s, body frame), wz (rad/s) |
| `pose` | (E, T, 3) | x, y (m, world), yaw (rad) |
| `z` | (E, T, 1) | base height (m) |
| `rpy` | (E, T, 3) | roll, pitch, yaw (rad) |
| `gyro` | (E, T, 3) | IMU angular velocity (rad/s, body frame) |
| `foot_force` | (E, T, 4) | normal contact force (N), FL, FR, RL, RR |
| `motor_q` | (E, T, 12) | joint angle (rad) |
| `motor_dq` | (E, T, 12) | joint velocity (rad/s) |
| `motor_ddq` | (E, T, 12) | joint acceleration (rad/s^2) |
| `motor_tau_est` | (E, T, 12) | applied joint torque (N m) |
| `seed`, `friction`, `payload_kg`, `fell`, `length` | (E,) | episode seed, foot friction, added base mass (kg), fell or not, valid samples |
| `dt` | () | 0.05 s |

Joint order (policy order): FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf. `<split>.json` records the episode count, seeds, falls, walker, run time and file size.

## 3. Real robot runs (tables in `results/robot/`)

| file | content |
| :--- | :--- |
| `l_path_trials_webrtc.md` | Phase 1 (2026-10-17): 5 L-path trials (4 ran), forward 0.5 m, turn 90 deg, forward 0.5 m, robot odometry and IMU |
| `nav_camera_runs_webrtc.md` | Phase 2 (2026-11-06): 25 runs of the navigation policy with the overhead camera; 19 with policy steps; outcome, steps, start and end distance to the target, path length, camera or odometry pose per step |

The Phase 1 single-axis tests used in `q2_data/small_commands.py` come from the project's `LOG.md` on `main` (`3cc7e15`).

## Not in this repository

| data | where | why |
| :--- | :--- | :--- |
| raw logs of the 2026-10-17 motion tests and the 2026-11-06 camera runs (`state.jsonl`, `lowstate.jsonl`, `commands.jsonl`, `events.jsonl`, `policy.npz`, `tag_fixes.jsonl`, `detections.jsonl`, `perception.jsonl`) | the MacBook's project clone, `data/` (git-ignored) | never pushed; this repository was built on the Dell, which does not have them |
| Isaac Lab walking transitions, 97,451 windows (`transitions_walk_eval.npz`, 13 MB) | the Dell, `~/dell-runner/data/ENGR6010_Go2X_SafeRL/040-walk-eval/` | job output, not in git |
| 60 rendered atrium scenes with ground truth | the Dell, job 045 output | job output, not in git |
