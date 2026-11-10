# Decisions

Each line: the decision, then the reason.

1. Build Exam 2 on the Dell, in `~/ENGR6010_Exam2`, as a new repository. Aidan asked for the parts the Dell can do; the MacBook's `Exam2` folder is not reachable from here.
2. Keep the project clone read-only and read project code through the pinned `project/` submodule (`46db755`). This follows the task plan.
3. Use simulated data from the MuJoCo Go2 (Unitree's `go2.xml` with the project's trained walking policy) for the supervised dynamics model. The real Phase 1 and 2026-11-06 raw logs are on the MacBook only.
4. Use Exam 1's 20 s stationary WebRTC recording as the real stationary data for sensor statistics. It is the only raw real-dog state on GitHub.
5. Run the optical-flow work on Exam 1's four real front-camera frames, with known synthetic shifts as ground truth. These are the only real camera frames available.
6. Compare simulation with the real dog through the per-trial and per-run tables in git (4 L-path trials, 19 camera runs with policy steps). The raw paths needed for a path-error comparison are on the MacBook.
7. Use the MuJoCo dog as a third evaluation world for the Exam 2 PPO policies. It is the closest model to the real dog available offline.
8. No `caffeinate`, because the Dell runs Linux and stays awake.
9. Push to `aidanjmartin/ENGR6010_Exam2` as a private repository. Aidan asked for the result on GitHub; the name is free on GitHub.
10. Write PPO out in full in `q3_rl/ppo.py` instead of calling Stable-Baselines3, so every RL component the question asks for is in this repository's code. Its settings follow the Phase 1 Stable-Baselines3 run, which gives a reference result on the same scenarios.
11. Train the Exam 2 PPO on the Phase 1 navigation task (LiDAR, random rooms). It runs on the CPU in minutes with any dynamics model; the camera-image task needs Isaac Lab and the GPU.
12. Train the Exam 2 PPO in a third world, the MuJoCo dog, besides the physics model and the ensemble. It shows how a policy trained in each simple model compares with one trained in the closest available model of the real dog.

2026-11-10, with Aidan's go-ahead to finish the exam ("do whatever you need to complete"):

13. Copy the raw logs from the MacBook into this repository instead of linking to them: the 2026-11-06 navigation runs (`~/ENGR6010_Go2X_SafeRL/data/`) and the 2026-10-17 motion tests (`~/Desktop/Uni/FALL26/ENGR6010/project/data/`). They are on no GitHub branch, and the analyses and graders need them. Only real-robot folders (`_webrtc`) are copied, not the MuJoCo-backend test runs.
14. Leave out the 233 MB of 2026-11-06 calibration images. The calibration results they produced are in the project's `config/camera_real.yaml`, and no analysis here needs the images.
15. Re-encode the overhead camera videos (H.264, CRF 26) with every frame kept, which brings them from 96 MB to 8 MB. They are overlays for viewing; no analysis reads them except the contact-sheet figure.
16. Use the overhead camera as ground truth and anchor the odometry at each run's first camera pose. The run's own odometry mapping (`run.json`) is anchored at the scenario's planned start, not where the robot stood.
17. Replay the commands as sent (after the safety layer), at rest, from the first camera pose, open loop. These are exactly what the robot received; closing the loop would hide the model error.
18. Fit a lag-delay model to the real dog and compare it too: leave one run out for the velocity test, and on the other day's runs for the motion tests. The replay uses the fit on all 19 runs and is labeled in-sample.
