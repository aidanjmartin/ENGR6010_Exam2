# Go2 dataset for the Second Test

Five parts: a real standing recording of the Go2 X, simulated recordings of the MuJoCo Go2, the raw logs of the real navigation runs and of the Phase 1 motion tests, and the per-run tables of the real robot runs. The simulated part follows the record layout of [`ShijieZhao36/unitree_go2_create_dataset`](https://github.com/ShijieZhao36/unitree_go2_create_dataset) (`cf01a90`). Its `robot_state` message holds `foot_force[4]`, `motor_q[12]`, `motor_dq[12]`, `motor_ddq[12]` and `motor_tau_est[12]`, recorded while a velocity command (`vx`, `vy`, `vyaw`) drives the robot. The simulated records keep those fields and add the command, the body velocity and the pose.

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

## 3. Real navigation runs, 2026-11-06 (`real_runs_20261106/`)

The navigation policy (Question 3, attempt 4) on the Go2 X over the robot's Wi-Fi hotspot (WebRTC, `unitree_webrtc_connect` `e0abc57`), with the robot pose from the overhead camera. Written by the project's `scripts/robot/run_nav_policy.py`; copied from the MacBook's project clone (`data/`) on 2026-11-10. One folder per run, named by its start time in UTC (`20261107T015345Z` is 7:53:45 PM on 2026-11-06 in Nashville). 25 navigation runs (19 with policy steps) and one read-only link check. The logs hold no robot address or serial number (checked with an IPv4 pattern and a key-name search); `header.json` names the laptop (`Aidan-MacBook`).

All times are the MacBook's monotonic clock (`t_mono_ns`, ns) and wall clock (`t_wall_ns`, ns since 1970).

| file | content |
| :--- | :--- |
| `header.json` | script, backend, safety limits (`max_vx` 0.8 m/s, `max_vy` 0.3 m/s, `max_wz` 1.0 rad/s, acceleration limits, 20 Hz), the policy file and its training job, the target |
| `run.json` | scenario (start pose, chairs `[x, y, yaw, color]` with color 0 red, 1 blue, 2 green, 3 yellow by tag; table `[x, y, yaw]`; target `["chair", index]` or `["table", 0]`), pose source, start mapping, outcome |
| `state.jsonl` | sport-mode state, about 20 Hz: `position` (odometry, m), `velocity` (vx, vy in the body frame, vz; m/s), `yaw_speed` (rad/s), `imu_state` (quaternion, gyroscope, accelerometer, roll-pitch-yaw), `body_height`, `foot_force`, `foot_position_body`, `foot_speed_body`, `mode`, `gait_type` |
| `lowstate.jsonl` | low-level state, about 1 Hz: IMU roll-pitch-yaw and the 12 motor angles (`motor_state[i].q`, rad) with temperatures |
| `commands.jsonl` | every command at 20 Hz: `requested` (vx, vy, wz before the safety layer) and `sent` (after its clamps and rate limits) |
| `events.jsonl` | connection, stand-up, `policy_start` (odometry mapping), one `policy_step` per policy step (pose used, `pose_source` `camera` or `odometry`, action, command, distance to target), `policy_end` (outcome), stops |
| `policy.npz` | per policy step: `t` (s since policy start), `pose` (x, y, yaw in the square frame, m and rad), `odom` (raw odometry x, y, yaw), `speed`, `action`, `cmd_policy`, `cmd_sent`, `distance`, `frame` (the 4 x 64 x 64 drawn image the policy read), `people` |
| `detections.jsonl` | OWLv2 boxes (pixels), scores and labels about once a second, and the objects mapped to the floor |
| `perception.jsonl` | the perceived layout (scenario) after each detection |
| `camera.mp4` | the overhead camera (1280 x 720) with the tag and detection overlay, one frame per policy step at 10 frames per second. Re-encoded for size (H.264, CRF 26) on 2026-11-10; every frame kept |
| `twin.mp4` | digital twin rendered in MuJoCo from the logs (`project/scripts/robot/render_twin.py`), one frame per sport-state message, 20 frames per second, 1600 x 480: fixed camera, chase camera, policy input |

The square frame: origin at the center of the 3.048 m square, the corner tags 0 to 3 at its corners. Runs without policy steps have no camera or twin video; `20261107T014956Z` never connected and has no `run.json`. Per-run outcomes: `results/robot/nav_camera_runs_webrtc.md`.

## 4. Real motion tests, 2026-10-17 (`real_motion_20261017/`)

The Phase 1 motion scripts (project `main`, `scripts/motion/`) on the Go2 X over the Ethernet tether (WebRTC). Copied from the MacBook's course folder on 2026-11-10. 16 folders: a state read, two stand tests, forward 0.5 m at 0.2 m/s, rotate 90 degrees at 0.3 and 0.5 rad/s, lateral 0.3 m at 0.15, 0.2 and 0.3 m/s, the Phase 1 RL policy toward a goal, and L-path trials 1 to 5 (trial 1's first attempt and trial 5 never connected). Same `header.json`, `state.jsonl`, `commands.jsonl` and `events.jsonl` formats as above; no camera. Safety limits: `max_vx` 0.2 m/s, `max_vy` 0.2 m/s, `max_wz` 0.5 rad/s for the single-axis tests.

## 5. Real robot run tables (`results/robot/`)

| file | content |
| :--- | :--- |
| `l_path_trials_webrtc.md` | Phase 1 (2026-10-17): 5 L-path trials (4 ran), forward 0.5 m, turn 90 deg, forward 0.5 m, robot odometry and IMU |
| `nav_camera_runs_webrtc.md` | Phase 2 (2026-11-06): 25 runs of the navigation policy with the overhead camera; 19 with policy steps; outcome, steps, start and end distance to the target, path length, camera or odometry pose per step |

The Phase 1 single-axis tests used in `q2_data/small_commands.py` come from the project's `LOG.md` on `main` (`3cc7e15`).

## Not in this repository

| data | where | why |
| :--- | :--- | :--- |
| 2026-11-06 calibration images: ChArUco board views (`intrinsics_20261107T003255Z/`, 102 MB) and floor frames (`floor_frames/`, 131 MB) | the MacBook's project clone, `data/` | large; the calibration results are in the project's `config/` |
| Isaac Lab walking transitions, 97,451 windows (`transitions_walk_eval.npz`, 13 MB) | the Dell, `~/dell-runner/data/ENGR6010_Go2X_SafeRL/040-walk-eval/` | job output, not in git |
| 60 rendered atrium scenes with ground truth | the Dell, job 045 output | job output, not in git |
