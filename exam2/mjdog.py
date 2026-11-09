"""The MuJoCo dog: Unitree's full go2.xml driven by a trained walking policy, commanded with body velocities.

This is the same physics and walker as the project's MuJoCo robot backend (go2x/deploy/mujoco_robot.py) and
walking evaluation (project/scripts/mjx/eval_walk.py): 45-value observation, PD torque at every 5 ms physics step,
policy at 50 Hz. A velocity command is held for one 0.05 s step. step() returns what a robot log holds.
"""
import math

import mujoco
import numpy as np
import torch

from exam2 import DT, MODEL_DIR, WALKER  # noqa: F401  (sets the project path)
from go2x.config import load_config
from go2x.mjx.go2_model import eval_model, indices
from go2x.sim2sim import joint_targets, observation, quat_rotate_inverse, tilt

FEET = ("FL", "FR", "RL", "RR")


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


class MujocoDog:
    def __init__(self, model_dir=MODEL_DIR, policy=str(WALKER), friction=None, payload=0.0):
        torch.set_num_threads(1)
        self.MJ, self.WALK = load_config("mujoco"), load_config("walk")
        self.m = eval_model(model_dir, self.MJ["timestep"])
        if friction is not None:
            for f in FEET:
                self.m.geom_friction[self.m.geom(f).id, 0] = friction
        self.m.body_mass[self.m.body("base_link").id] += payload
        self.d = mujoco.MjData(self.m)
        ix = indices(self.m)
        self.qadr, self.vadr, self.act, self.limit = ix["qadr"], ix["vadr"], ix["act"], ix["limit"]
        self.kp, self.kd = self.WALK["actuator"]["kp"], self.WALK["actuator"]["kd"]
        self.scale = self.WALK["actuator"]["action_scale"]
        self.policy = torch.jit.load(policy, map_location="cpu").eval()
        self.gyro = self.m.sensor("imu_gyro").adr[0]
        self.quat = self.m.sensor("imu_quat").adr[0]
        self.n_phys = int(round(DT / self.MJ["timestep"]))          # 10 physics steps per 0.05 s
        self.sub = self.MJ["pd_substeps"]                           # policy every 4 physics steps
        self.foot_geoms = {self.m.geom(f).id: i for i, f in enumerate(FEET)}

    def reset(self, pose=(0.0, 0.0, 0.0)):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        x, y, yaw = pose
        d.qpos[0:3] = [x, y, self.MJ["start_height"]]
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        hold = joint_targets(np.zeros(12), self.scale)
        d.qpos[self.qadr] = hold
        mujoco.mj_forward(m, d)
        for _ in range(int(round(self.MJ["settle_s"] / self.MJ["timestep"]))):
            d.ctrl[self.act] = np.clip(self.kp * (hold - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr],
                                       -self.limit, self.limit)
            mujoco.mj_step(m, d)
        self.last = np.zeros(12, np.float32)
        self.target = hold
        self.k = 0
        return self.state()

    def step(self, cmd):
        """Hold body-velocity command (vx, vy, wz) for 0.05 s. Returns the state after it."""
        m, d = self.m, self.d
        cmd = np.asarray(cmd, float)
        for _ in range(self.n_phys):
            if self.k % self.sub == 0:
                obs = observation(d.sensordata[self.gyro:self.gyro + 3], d.sensordata[self.quat:self.quat + 4],
                                  cmd, d.qpos[self.qadr], d.qvel[self.vadr], self.last)
                with torch.no_grad():
                    self.last = self.policy(torch.from_numpy(obs)[None])[0].numpy().astype(np.float32)
                self.target = joint_targets(self.last, self.scale)
            d.ctrl[self.act] = np.clip(self.kp * (self.target - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr],
                                       -self.limit, self.limit)
            mujoco.mj_step(m, d)
            self.k += 1
        return self.state()

    def fallen(self):
        return tilt(self.d.qpos[3:7]) > self.MJ["fall"]["tilt_rad"] or self.d.qpos[2] < self.MJ["fall"]["base_height_m"]

    def foot_force(self):
        """Normal contact force on each foot (N), FL FR RL RR."""
        f = np.zeros(4)
        c6 = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            for g in (con.geom1, con.geom2):
                if g in self.foot_geoms:
                    mujoco.mj_contactForce(self.m, self.d, i, c6)
                    f[self.foot_geoms[g]] += abs(c6[0])
        return f

    def state(self):
        d = self.d
        q = d.qpos[3:7]
        v_b = quat_rotate_inverse(q, d.qvel[0:3])
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return {
            "vel": np.array([v_b[0], v_b[1], d.qvel[5]]),          # body vx, vy (m/s), yaw rate (rad/s)
            "pose": np.array([d.qpos[0], d.qpos[1], yaw_of(q)]),   # world x, y (m), yaw (rad)
            "z": float(d.qpos[2]), "rpy": np.array([roll, pitch, yaw_of(q)]),
            "gyro": d.sensordata[self.gyro:self.gyro + 3].copy(),
            "motor_q": d.qpos[self.qadr].copy(), "motor_dq": d.qvel[self.vadr].copy(),
            "motor_ddq": d.qacc[self.vadr].copy(), "motor_tau_est": d.ctrl[self.act].copy(),
            "foot_force": self.foot_force(),
        }
