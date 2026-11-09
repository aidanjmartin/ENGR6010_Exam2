# Small single-axis commands: share of the commanded travel delivered

Real: one run each on the Go2 X (2026-10-17, odometry and IMU). Simulated: mean over n runs (sd in the JSON). 1.00 means the full distance or angle.

| world | forward 0.2 m/s, 0.5 m | rotate 0.3 rad/s, 90 deg | rotate 0.5 rad/s, 90 deg | lateral 0.15 m/s, 0.3 m | lateral 0.2 m/s, 0.3 m | lateral 0.3 m/s, 0.3 m |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| real Go2 X (Unitree controller) | 0.65 | 0.14 | 0.47 | 0.03 | 0.13 | 0.57 |
| physics model, Phase 1 ranges | 0.99 | 1.00 | 1.00 | 0.99 | 0.98 | 0.97 |
| physics model, fitted to the MuJoCo dog | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |
| learned ensemble | 0.26 | 0.02 | 0.04 | 0.02 | 0.05 | 0.36 |
| MuJoCo dog, MuJoCo walker (mjx_s0) | 0.63 | 0.03 | 0.09 | 0.06 | 0.32 | 0.56 |
| MuJoCo dog, Isaac Lab walker | 0.09 | 0.02 | 0.05 | 0.00 | 0.00 | 0.01 |
