# Handoff (2026-11-09)

Built on the Dell, not the MacBook, at Aidan's request ("do the parts you can from the Dell"). The repository is complete for all three questions, using the data the Dell can reach: Exam 1's real recordings, the per-run tables of the real robot runs, the project's Phase 2 results, and new simulated data.

## End state of every item

See `STATUS_2026-11-09.md`, section "End state". In short:
- **Done:** Q1 dynamics ensemble, Q1 pyramid and optical flow, Q2 dataset and `DATASET.md`, Q2 sensor statistics, Q2 simulation against the real dog (single-axis tests, L-path, navigation by start distance), Q3 PPO in three worlds with cross-evaluation, README.
- **Not run, need the MacBook's raw logs** (`data/` in the MacBook's project clone):
  - camera against odometry drift on the 2026-11-06 runs;
  - replay of those runs' command logs through the models, with path error;
  - digital twins of the real runs;
  - a test of the ensemble on real motion.

## Decisions taken without Aidan

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

## Needs Aidan

1. **The MacBook's `Exam2` folder.** An earlier unattended session may have work there that is not in this repository. Compare before deciding which to keep; this repository does not include it.
2. **Raw robot logs.** To add the four analyses above, run them on the MacBook against `data/`, or push the 2026-11-06 run folders to this repository after removing the robot's serial number and address.
3. **Public or not, and Canvas.** The repository is private. To make it public:
   - with the Dell's `gh` 2.45: `gh repo edit aidanjmartin/ENGR6010_Exam2 --visibility public`;
   - with a newer `gh`, which asks for an extra flag: `gh repo edit aidanjmartin/ENGR6010_Exam2 --visibility public --accept-visibility-change-consequences`.

   The link to post on Canvas: https://github.com/aidanjmartin/ENGR6010_Exam2
4. **Submodule access.** `project/` points at the public `aidanjmartin/ENGR6010_Go2X_SafeRL`. If that repository is made private, graders cannot fetch the submodule.

## Push

PUSH_PLACEHOLDER
