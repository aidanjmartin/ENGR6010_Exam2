# Handoff (2026-11-10)

The repository is complete for all three questions and ready to submit. It was built on the Dell on 2026-11-09. On 2026-11-10 the raw robot logs were copied from the MacBook, and the four analyses that needed them were added.

## End state of every item

- **Done on 2026-11-09:** Q1 dynamics ensemble, Q1 pyramid and optical flow, Q2 dataset and `DATASET.md`, Q2 sensor statistics, Q2 simulation against the real dog (single-axis tests, L-path, navigation by start distance), Q3 PPO in three worlds with cross-evaluation, README.
- **Done on 2026-11-10,** from the raw logs of the 2026-11-06 navigation runs and the 2026-10-17 motion tests, now in `q2_data/dataset/`:
  - the camera against odometry drift (README 4.5);
  - the recorded commands replayed through the models, with path error (README 4.6);
  - the digital twins of the 19 real runs, with figures (README 4.7);
  - the ensemble tested on real motion (README 3.2, "On the real dog").
- **Checked on 2026-11-10:** the MacBook has no `Exam2` folder under `~/Desktop/Uni/FALL26/ENGR6010/`, so no earlier work is missing from this repository.

## Decisions taken without Aidan

See `DECISIONS.md` (decisions 1 to 12 on 2026-11-09, 13 to 18 on 2026-11-10).

## Needs Aidan

1. **Canvas.** The repository is public. The link to post: https://github.com/aidanjmartin/ENGR6010_Exam2
2. **Submodule access.** `project/` points at the public `aidanjmartin/ENGR6010_Go2X_SafeRL`. If that repository is made private, graders cannot fetch the submodule.

## Push

`main` was first pushed on 2026-11-09 (created private with `gh repo create --private`). It was made public on 2026-11-10 at Aidan's request (`gh repo edit --visibility public`). The 2026-11-10 additions were pushed the same day (`LOG.md`, 2026-11-10).
