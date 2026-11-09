# Q3 cross-evaluation

100 seeded scenarios (seeds 10000 to 10099). PPO rows: mean over 3 training seeds ± standard error over seeds. Baseline row: one deterministic controller, binomial standard error over the 100 scenarios.

**success rate**

| controller | in physics | in learned | in mujoco |
| :--- | ---: | ---: | ---: |
| PPO trained in physics | 0.757 ± 0.015 | 0.730 ± 0.015 | 0.747 ± 0.009 |
| PPO trained in learned | 0.783 ± 0.012 | 0.793 ± 0.009 | 0.783 ± 0.003 |
| PPO trained in mujoco | 0.777 ± 0.009 | 0.760 ± 0.010 | 0.760 ± 0.010 |
| go-to-goal baseline (Phase 1) | 0.690 ± 0.046 | 0.660 ± 0.047 | 0.640 ± 0.048 |

**collision rate**

| controller | in physics | in learned | in mujoco |
| :--- | ---: | ---: | ---: |
| PPO trained in physics | 0.243 ± 0.015 | 0.270 ± 0.015 | 0.253 ± 0.009 |
| PPO trained in learned | 0.217 ± 0.012 | 0.207 ± 0.009 | 0.217 ± 0.003 |
| PPO trained in mujoco | 0.223 ± 0.009 | 0.237 ± 0.007 | 0.237 ± 0.007 |
| go-to-goal baseline (Phase 1) | 0.310 ± 0.046 | 0.340 ± 0.047 | 0.360 ± 0.048 |

**fell rate**

| controller | in physics | in learned | in mujoco |
| :--- | ---: | ---: | ---: |
| PPO trained in physics | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| PPO trained in learned | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| PPO trained in mujoco | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| go-to-goal baseline (Phase 1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

**timeout rate**

| controller | in physics | in learned | in mujoco |
| :--- | ---: | ---: | ---: |
| PPO trained in physics | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| PPO trained in learned | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| PPO trained in mujoco | 0.000 ± 0.000 | 0.003 ± 0.003 | 0.003 ± 0.003 |
| go-to-goal baseline (Phase 1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
