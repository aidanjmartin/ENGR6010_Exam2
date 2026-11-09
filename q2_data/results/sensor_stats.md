# The standing dog: real sensor statistics against the MuJoCo dog

Real: Exam 1 WebRTC recording, 400 sport-state messages over 19.9 s (median period 50.0 ms, 95th percentile 51.7 ms, longest 61.3 ms) and 20 low-state messages. Simulated: 215 settled standing stretches, 4300 samples at 20 Hz.

| quantity | real mean | real sd | simulated mean | simulated sd |
| :--- | ---: | ---: | ---: | ---: |
| body height (m) | +0.3207 | 2.2e-06 | +0.3177 | 0.0019 |
| velocity vx (m/s) | +0.0004 | 0.0073 | -0.0003 | 0.0028 |
| velocity vy (m/s) | -0.0010 | 0.0074 | +0.0000 | 0.0011 |
| roll (rad) | -0.0310 | 0.0002 | +0.0048 | 0.0044 |
| pitch (rad) | -0.0136 | 0.00032 | -0.0150 | 0.008 |
| velocity vz (m/s) | -0.0293 | 0.002 | not recorded | |

Real body height drift: +0.000 mm/s. Real foot force sensors (raw units, not newtons): foot_0 90.3 ± sd 1.0, foot_1 80.6 ± sd 0.9, foot_2 89.7 ± sd 0.5, foot_3 92.9 ± sd 0.8 (n = 20). Simulated total foot normal force 164.9 N (sd 8.4).
