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
