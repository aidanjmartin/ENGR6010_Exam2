# L-path test: real Go2 X against simulation

Mean ± sd over trials (n per row). Ideal: legs 0.5 m, turn 90 deg, end +0.5 m forward and +0.5 m left. Real values are the robot's odometry and IMU, as in the simulated rows.

| world | n | leg 1 (m) | turn (deg) | leg 2 (m) | end forward (m) | end left (m) | heading change (deg) | segment timeouts |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real Go2 X (Unitree controller) | 4 | 0.369 ± 0.063 | 82.3 ± 2.5 | 0.353 ± 0.194 | 0.414 ± 0.089 | 0.414 ± 0.193 | 83.9 ± 1.7 | 3 |
| physics model, Phase 1 ranges | 10 | 0.525 ± 0.012 | 92.7 ± 1.9 | 0.525 ± 0.012 | 0.500 ± 0.029 | 0.524 ± 0.012 | 93.2 ± 2.3 | 0 |
| physics model, fitted to the MuJoCo dog | 1 | 0.508 ± 0.000 | 91.1 ± 0.0 | 0.508 ± 0.000 | 0.495 ± 0.000 | 0.507 ± 0.000 | 91.7 ± 0.0 | 0 |
| learned ensemble | 10 | 0.454 ± 0.030 | 3.7 ± 7.9 | 0.450 ± 0.035 | 0.911 ± 0.071 | -0.058 ± 0.096 | 0.9 ± 9.1 | 13 |
| MuJoCo dog, MuJoCo walker (mjx_s0) | 10 | 0.446 ± 0.000 | 2.8 ± 0.0 | 0.453 ± 0.001 | 0.897 ± 0.001 | -0.070 ± 0.001 | -2.0 ± 0.0 | 10 |
| MuJoCo dog, Isaac Lab walker | 10 | 0.046 ± 0.000 | 7.8 ± 0.0 | 0.010 ± 0.000 | 0.044 ± 0.000 | -0.017 ± 0.000 | 6.9 ± 0.0 | 30 |
