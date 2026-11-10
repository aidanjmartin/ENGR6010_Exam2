# The real Go2 X runs of 2026-11-06 against the overhead camera

19 runs with policy steps (q2_data/dataset/real_runs_20261106/). Ground truth: the overhead camera's robot pose at every policy step where it saw the robot's tag. Mean ± standard error over runs unless stated.

## 1. Camera against odometry

Odometry anchored at each run's first camera pose (507 camera steps):

| quantity | value |
| :--- | ---: |
| position error at the run's end (m) | 0.147 ± 0.021 |
| heading error at the run's end (deg) | 6.5 ± 1.6 |
| end error per m walked, runs over 0.5 m (n = 13) | 0.142 ± 0.007 |
| odometry path / camera path, same runs | 0.827 ± 0.008 |
| least-squares slope of error against distance, all steps (m per m) | 0.146 |

| distance walked (m) | steps | median error (m) | 90th percentile (m) | median heading error (deg) |
| :--- | ---: | ---: | ---: | ---: |
| 0 to 0.5 | 340 | 0.021 | 0.066 | 3.5 |
| 0.5 to 1.0 | 87 | 0.090 | 0.143 | 3.3 |
| 1.0 to 1.5 | 50 | 0.209 | 0.242 | 4.2 |
| 1.5 to 2.5 | 30 | 0.243 | 0.279 | 4.8 |

Per run:

| run (UTC) | camera steps | camera path (m) | odometry path (m) | end error (m) | max error (m) | end heading error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261107T012917Z | 13 | 0.21 | 0.18 | 0.042 | 0.042 | 5.6 |
| 20261107T013603Z | 22 | 0.45 | 0.40 | 0.066 | 0.066 | 3.1 |
| 20261107T013658Z | 21 | 1.22 | 1.00 | 0.216 | 0.216 | 4.4 |
| 20261107T013815Z | 13 | 0.20 | 0.19 | 0.014 | 0.014 | 2.0 |
| 20261107T013905Z | 17 | 0.88 | 0.73 | 0.120 | 0.120 | 3.5 |
| 20261107T014031Z | 31 | 1.42 | 1.18 | 0.225 | 0.225 | 4.0 |
| 20261107T014135Z | 37 | 1.64 | 1.36 | 0.267 | 0.291 | 8.3 |
| 20261107T014228Z | 27 | 0.50 | 0.46 | 0.137 | 0.137 | 31.4 |
| 20261107T014349Z | 17 | 0.28 | 0.22 | 0.060 | 0.063 | 6.3 |
| 20261107T014646Z | 33 | 0.77 | 0.60 | 0.068 | 0.068 | 2.1 |
| 20261107T015033Z | 9 | 0.11 | 0.11 | 0.004 | 0.012 | 5.5 |
| 20261107T015237Z | 22 | 0.89 | 0.69 | 0.122 | 0.236 | 4.5 |
| 20261107T015345Z | 50 | 2.06 | 1.70 | 0.241 | 0.241 | 7.0 |
| 20261107T015441Z | 31 | 1.58 | 1.35 | 0.263 | 0.274 | 16.1 |
| 20261107T015621Z | 39 | 1.64 | 1.33 | 0.255 | 0.255 | 1.6 |
| 20261107T015741Z | 22 | 1.58 | 1.31 | 0.251 | 0.251 | 3.7 |
| 20261107T015848Z | 35 | 1.19 | 1.02 | 0.133 | 0.133 | 3.3 |
| 20261107T020049Z | 49 | 1.61 | 1.30 | 0.238 | 0.244 | 4.8 |
| 20261107T020150Z | 19 | 0.59 | 0.52 | 0.073 | 0.073 | 5.9 |

## 2. Recorded commands replayed through the dynamics models

Each run's sent commands (20 Hz) replayed open loop from its first camera pose, at rest; the model path is compared with the camera path at the camera steps. The odometry row is the dog's own estimate, for reference.

| model | mean position error (m) | end position error (m) | end heading error (deg) | path length / camera path (runs over 0.5 m) |
| :--- | ---: | ---: | ---: | ---: |
| odometry (the dog's own estimate) | 0.059 ± 0.009 | 0.147 ± 0.021 | 6.5 ± 1.6 | 0.83 ± 0.01 |
| physics, Phase 1 (unfitted) | 0.070 ± 0.010 | 0.160 ± 0.025 | 10.6 ± 1.6 | 0.92 ± 0.02 |
| physics, fitted to the MuJoCo dog | 0.073 ± 0.009 | 0.165 ± 0.024 | 9.1 ± 1.5 | 0.94 ± 0.02 |
| physics, fitted to the real dog (in-sample) | 0.053 ± 0.007 | 0.107 ± 0.015 | 6.9 ± 1.4 | 0.95 ± 0.02 |
| learned ensemble (mean) | 0.148 ± 0.031 | 0.480 ± 0.113 | 45.1 ± 6.0 | 0.82 ± 0.01 |
| MuJoCo dog, MuJoCo walker | 0.118 ± 0.025 | 0.386 ± 0.093 | 37.2 ± 5.2 | 0.89 ± 0.01 |

## 3. Velocity prediction on the real dog

Body velocity on the 20 Hz command grid, predicted with the commands over the horizon known. RMSE at the horizon per run, then mean ± standard error over runs. vx and vy in m/s, wz in rad/s. The lag-delay model fitted to the real dog is fitted on the other 18 navigation runs for the navigation runs, and on all 19 navigation runs for the motion tests.

### navigation runs, 2026-11-06

19 runs, 1515 steps.

**0.1 s ahead** (1325 windows)

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.096 ± 0.013 | 0.070 ± 0.005 | 0.130 ± 0.006 |
| lag-delay, Phase 1 (unfitted) | 0.082 ± 0.011 | 0.063 ± 0.004 | 0.117 ± 0.005 |
| lag-delay, fitted to the MuJoCo dog | 0.079 ± 0.011 | 0.061 ± 0.004 | 0.101 ± 0.005 |
| lag-delay, fitted to the real dog | 0.075 ± 0.010 | 0.062 ± 0.004 | 0.085 ± 0.006 |
| ensemble (5 networks) | 0.090 ± 0.013 | 0.068 ± 0.005 | 0.233 ± 0.008 |

**0.5 s ahead** (1173 windows)

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.184 ± 0.027 | 0.127 ± 0.009 | 0.425 ± 0.020 |
| lag-delay, Phase 1 (unfitted) | 0.101 ± 0.014 | 0.093 ± 0.003 | 0.220 ± 0.011 |
| lag-delay, fitted to the MuJoCo dog | 0.089 ± 0.012 | 0.086 ± 0.003 | 0.179 ± 0.009 |
| lag-delay, fitted to the real dog | 0.084 ± 0.011 | 0.085 ± 0.004 | 0.103 ± 0.007 |
| ensemble (5 networks) | 0.091 ± 0.012 | 0.067 ± 0.004 | 0.277 ± 0.012 |

One-step 2-sigma coverage (1439 pairs): vx 0.440, vy 0.510, wz 0.277 (0.954 if Gaussian and calibrated).

### Phase 1 motion tests, 2026-10-17 (02_forward, 03_rotate, 03_rotate, 04_lateral, 04_lateral, 04_lateral, 06_rl_goal, 07_l_path_t1, 07_l_path_t2, 07_l_path_t3, 07_l_path_t4)

11 runs, 2238 steps.

**0.1 s ahead** (2128 windows)

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.021 ± 0.004 | 0.023 ± 0.003 | 0.046 ± 0.005 |
| lag-delay, Phase 1 (unfitted) | 0.026 ± 0.005 | 0.025 ± 0.004 | 0.056 ± 0.007 |
| lag-delay, fitted to the MuJoCo dog | 0.026 ± 0.005 | 0.024 ± 0.004 | 0.053 ± 0.006 |
| lag-delay, fitted to the real dog | 0.032 ± 0.008 | 0.023 ± 0.003 | 0.081 ± 0.015 |
| ensemble (5 networks) | 0.021 ± 0.004 | 0.021 ± 0.003 | 0.062 ± 0.010 |

**0.5 s ahead** (2040 windows)

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.041 ± 0.008 | 0.033 ± 0.008 | 0.084 ± 0.015 |
| lag-delay, Phase 1 (unfitted) | 0.050 ± 0.013 | 0.041 ± 0.010 | 0.105 ± 0.020 |
| lag-delay, fitted to the MuJoCo dog | 0.049 ± 0.013 | 0.038 ± 0.009 | 0.099 ± 0.018 |
| lag-delay, fitted to the real dog | 0.052 ± 0.015 | 0.036 ± 0.008 | 0.104 ± 0.022 |
| ensemble (5 networks) | 0.030 ± 0.006 | 0.021 ± 0.002 | 0.082 ± 0.014 |

One-step 2-sigma coverage (2194 pairs): vx 0.753, vy 0.823, wz 0.668 (0.954 if Gaussian and calibrated).

Lag-delay fitted to all 19 navigation runs: tau [0.14, 0.38, 0.08] s, delay [0.0, 0.0, 0.0] s (vx, vy, wz). Fitted to the MuJoCo dog (Q1): tau [0.26, 0.24, 0.28] s, delay [0.0, 0.0, 0.0] s.

