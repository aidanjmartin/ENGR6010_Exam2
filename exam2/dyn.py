"""Dynamics models behind the Phase 1 interface (project/go2x/sim/dynamics.py DynamicsModel): reset(rng, pose)
and step(cmd, dt), with .pose (x, y, yaw) and .vel (vx, vy, wz). These plug into the Phase 1 navigation environment
(go2x.sim.nav_env.Go2NavEnv) and into the L-path replay.

- PhysicsModel: the Phase 1 lag-delay model, unchanged (go2x.sim.dynamics.LagDelayDynamics), with per-episode
  parameters drawn from config/sim.yaml, or fixed ones.
- LearnedModel: the Q1 ensemble. Each episode uses one member drawn at random (trajectory sampling, as in PETS),
  and samples its predicted Gaussian noise, so the model is stochastic like the dog.
- MujocoModel: the MuJoCo dog (exam2/mjdog.py).
"""
import numpy as np

from exam2 import DT
from exam2.models import H, Ensemble, features
from go2x.sim.dynamics import DynamicsModel, LagDelayDynamics, integrate_pose


class PhysicsModel(LagDelayDynamics):
    def __init__(self, tau_range=(0.1, 0.4), delay_range=(0.0, 0.15), fixed=None):
        super().__init__(tau_range, delay_range, substeps=10, dt=DT)
        self.fixed = fixed                                   # (tau, delay) per axis, or None to draw

    def reset(self, rng, pose):
        if self.fixed is None:
            return super().reset(rng, pose)
        return super().reset(rng, pose, tau=self.fixed[0], delay=self.fixed[1])


class LearnedModel(DynamicsModel):
    def __init__(self, path, sample_noise=True, member=None):
        super().__init__()
        self.ens = Ensemble(path)
        self.sample_noise = sample_noise
        self.fixed_member = member

    def reset(self, rng, pose):
        super().reset(rng, pose)
        self.rng = rng
        self.m = self.fixed_member if self.fixed_member is not None else int(rng.integers(self.ens.n_members))
        self.vh = np.zeros((1, H, 3))
        self.ch = np.zeros((1, H - 1, 3))

    def step(self, cmd, dt):
        full = np.concatenate([self.ch, np.asarray(cmd, float).reshape(1, 1, 3)], 1)
        mu, var = self.ens.member(self.m, features(self.vh, full))
        dv = mu[0] + (self.rng.normal(0, 1, 3) * np.sqrt(var[0]) if self.sample_noise else 0.0)
        v_new = self.vel + dv
        n = 10
        for i in range(n):                                   # velocity moves linearly over the step
            v = self.vel + (v_new - self.vel) * (i + 0.5) / n
            self.pose = integrate_pose(self.pose, v, dt / n)
        self.vel = v_new
        self.vh = np.concatenate([self.vh[:, 1:], v_new.reshape(1, 1, 3)], 1)
        self.ch = full[:, 1:]


class MujocoModel(DynamicsModel):
    def __init__(self, **kw):
        super().__init__()
        from exam2.mjdog import MujocoDog
        self.dog = MujocoDog(**kw)
        self.fell = False

    def reset(self, rng, pose):
        st = self.dog.reset(tuple(pose))
        self.pose, self.vel, self.fell = st["pose"].copy(), st["vel"].copy(), False

    def step(self, cmd, dt):
        st = self.dog.step(cmd)
        self.pose, self.vel = st["pose"].copy(), st["vel"].copy()
        self.fell = self.fell or self.dog.fallen()
