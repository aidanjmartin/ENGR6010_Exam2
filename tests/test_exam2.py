"""CPU tests for the Second Test code. MuJoCo tests need Unitree's models (GO2X_UNITREE_MUJOCO)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "q3_rl"))
from exam2 import MODEL_DIR  # noqa: E402
from exam2.models import H, Ensemble, LagDelay  # noqa: E402

needs_model = pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "unitree_robots/go2/go2.xml")),
                                 reason="unitree_mujoco not found (set GO2X_UNITREE_MUJOCO)")
ENS = ROOT / "q1_vision_ml/results/ensemble.npz"


def test_lag_delay_matches_phase1_model():
    """The vectorized lag-delay rollout equals the Phase 1 LagDelayDynamics on the same commands."""
    from go2x.sim.dynamics import LagDelayDynamics
    rng = np.random.default_rng(0)
    tau, delay = [0.2, 0.3, 0.15], [0.05, 0.0, 0.1]
    cmds = rng.uniform(-0.5, 0.5, (30, 3))
    ref = LagDelayDynamics(substeps=10, dt=0.05)
    ref.reset(rng, np.zeros(3), tau=tau, delay=delay)
    want = []
    for c in cmds:
        ref.step(c, 0.05)
        want.append(ref.vel.copy())
    got = LagDelay(tau, delay).rollout(np.zeros((1, 3)), np.zeros((1, 8, 3)), cmds[None])[0]
    assert np.abs(got - np.array(want)).max() < 1e-9


def test_ensemble_numpy_matches_torch_and_rollout_shape():
    torch = pytest.importorskip("torch")
    z = np.load(ENS)
    ens = Ensemble(ENS)
    x = np.random.default_rng(1).normal(size=(7, 6 * H))
    h = torch.tensor((x - z["x_mean"]) / z["x_std"])
    for i in range(4):
        h = h @ torch.tensor(z[f"m0_W{i}"]) + torch.tensor(z[f"m0_b{i}"])
        if i < 3:
            h = torch.nn.functional.silu(h)
    mu = h[:, :3].numpy() * z["y_std"] + z["y_mean"]
    assert np.abs(ens.member(0, x)[0] - mu).max() < 1e-9
    out = ens.rollout(np.zeros((2, H, 3)), np.zeros((2, H - 1, 3)), np.zeros((2, 10, 3)))
    assert out.shape == (2, 10, 3) and np.isfinite(out).all()


def test_learned_model_in_phase1_env():
    from exam2.dyn import LearnedModel
    from go2x.sim.nav_env import Go2NavEnv
    env = Go2NavEnv(dynamics=LearnedModel(ENS))
    o, _ = env.reset(seed=3)
    x0 = env.dynamics.pose.copy()
    for _ in range(40):
        o, r, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0]))
        if term or trunc:
            break
    assert o.shape == env.observation_space.shape and np.isfinite(o).all()
    assert np.hypot(*(env.dynamics.pose[:2] - x0[:2])) > 0.2       # it moved forward


def test_ppo_gae_simple_case():
    from ppo import gae
    rew = np.array([[1.0], [1.0], [1.0]])
    val = np.zeros((3, 1))
    done = np.array([[0.0], [0.0], [1.0]])
    adv, ret = gae(rew, val, done, np.array([5.0]), gamma=0.5, lam=1.0)
    assert np.allclose(ret[:, 0], [1 + 0.5 + 0.25, 1 + 0.5, 1])        # the episode ends; last_val is not used


def test_dataset_fields():
    z = np.load(ROOT / "q2_data/dataset/sim/val.npz")
    for k, d in (("motor_q", 12), ("motor_dq", 12), ("motor_ddq", 12), ("motor_tau_est", 12), ("foot_force", 4), ("cmd", 3)):
        assert z[k].shape[-1] == d
    meta = json.loads((ROOT / "q2_data/dataset/sim/val.json").read_text())
    assert meta["episodes"] == len(z["seed"]) == 30


@needs_model
def test_mujoco_dog_walks_forward():
    from exam2.mjdog import MujocoDog
    dog = MujocoDog()
    dog.reset((0.0, 0.0, 0.0))
    for _ in range(60):
        st = dog.step([0.5, 0.0, 0.0])
    assert not dog.fallen() and st["pose"][0] > 0.8 and st["foot_force"].sum() > 50
