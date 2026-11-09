# Velocity prediction on held-out simulated episodes

RMSE of the predicted body velocity at the horizon, per episode, then mean ± standard error over episodes (n = 30 episodes per split). vx and vy in m/s, wz in rad/s. Commands over the horizon are known to every model.

**test: same walker as training (mjx_s0), 0.1 s ahead**

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.070 ± 0.002 | 0.056 ± 0.002 | 0.162 ± 0.007 |
| lag-delay, Phase 1 (unfitted) | 0.051 ± 0.002 | 0.047 ± 0.001 | 0.177 ± 0.006 |
| lag-delay, fitted | 0.039 ± 0.001 | 0.037 ± 0.001 | 0.123 ± 0.004 |
| ensemble (5 networks) | 0.023 ± 0.001 | 0.022 ± 0.001 | 0.043 ± 0.002 |

**test: same walker as training (mjx_s0), 0.5 s ahead**

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.174 ± 0.005 | 0.109 ± 0.003 | 0.247 ± 0.008 |
| lag-delay, Phase 1 (unfitted) | 0.075 ± 0.002 | 0.064 ± 0.002 | 0.238 ± 0.006 |
| lag-delay, fitted | 0.060 ± 0.002 | 0.053 ± 0.001 | 0.194 ± 0.006 |
| ensemble (5 networks) | 0.033 ± 0.001 | 0.030 ± 0.001 | 0.048 ± 0.002 |

One-step 2-sigma coverage (test): vx 0.966, vy 0.969, wz 0.965 (0.954 if Gaussian and calibrated).

**shift: Isaac Lab walker, never seen in training, 0.1 s ahead**

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.045 ± 0.001 | 0.041 ± 0.001 | 0.165 ± 0.006 |
| lag-delay, Phase 1 (unfitted) | 0.056 ± 0.002 | 0.050 ± 0.001 | 0.186 ± 0.005 |
| lag-delay, fitted | 0.049 ± 0.001 | 0.042 ± 0.001 | 0.131 ± 0.003 |
| ensemble (5 networks) | 0.034 ± 0.001 | 0.028 ± 0.001 | 0.052 ± 0.002 |

**shift: Isaac Lab walker, never seen in training, 0.5 s ahead**

| model | vx | vy | wz |
| :--- | ---: | ---: | ---: |
| constant velocity | 0.120 ± 0.004 | 0.077 ± 0.003 | 0.240 ± 0.008 |
| lag-delay, Phase 1 (unfitted) | 0.115 ± 0.003 | 0.090 ± 0.003 | 0.245 ± 0.006 |
| lag-delay, fitted | 0.107 ± 0.003 | 0.085 ± 0.003 | 0.202 ± 0.005 |
| ensemble (5 networks) | 0.074 ± 0.003 | 0.054 ± 0.003 | 0.059 ± 0.002 |

One-step 2-sigma coverage (shift_isaac): vx 0.883, vy 0.940, wz 0.909 (0.954 if Gaussian and calibrated).

Fitted lag-delay: tau [0.26, 0.24, 0.28] s, delay [0.0, 0.0, 0.0] s (vx, vy, wz).
