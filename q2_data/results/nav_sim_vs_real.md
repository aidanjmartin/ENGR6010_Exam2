# Navigation policy (attempt 4): simulation against the real Go2 X

Success rate ± binomial standard error (successes / runs). Start distance: robot front to target footprint.

| world | all | start < 1.9 m | start >= 1.9 m |
| :--- | ---: | ---: | ---: |
| real Go2 X (Unitree controller), 2026-11-06 | 0.737 ± 0.101 (14/19) | 0.889 ± 0.105 (8/9) | 0.600 ± 0.155 (6/10) |
| Isaac Lab (job 060) | 0.780 ± 0.029 (156/200) | 0.811 ± 0.030 (137/169) | 0.613 ± 0.087 (19/31) |
| MuJoCo, MuJoCo walker (job 140) | 0.810 ± 0.028 (162/200) | 0.834 ± 0.029 (141/169) | 0.677 ± 0.084 (21/31) |
| MuJoCo, Isaac Lab walker (job 140) | 0.835 ± 0.026 (167/200) | 0.858 ± 0.027 (145/169) | 0.710 ± 0.082 (22/31) |

Simulated start distances: 1.02 to 2.51 m (median 1.48); real: 0.52 to 2.40 m.
