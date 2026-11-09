"""Q3: proximal policy optimization (PPO; Schulman et al., 2017) written out in full, for the Phase 1 navigation task
(project/go2x/sim/nav_env.py Go2NavEnv) with a choice of dynamics model ("world"):

    physics  the Phase 1 lag-delay model, parameters drawn per episode from config/sim.yaml (unfitted ranges)
    learned  the Q1 learned ensemble (one member per episode, sampled noise)
    mujoco   the MuJoCo dog: Unitree's go2.xml walked by the trained walking policy

Every reinforcement learning component is in this file:
    environment and episode end   make_env -> Go2NavEnv (success, collision, 40 s timeout)
    state (observation)           Go2NavEnv._obs: goal distance and bearing, body velocity, previous action, 32 LiDAR ranges
    action                        Gaussian policy over [-1, 1]^3, scaled to vx, vy, wz limits by the environment
    reward                        Go2NavEnv.step: progress, success, collision and time terms (config/sim.yaml)
    policy network                ActorCritic.actor (2 x 256 tanh) with a learned log standard deviation
    value network                 ActorCritic.critic (2 x 256 tanh)
    rollout and advantages        collect() and gae(): generalized advantage estimation (Schulman et al., 2016),
                                  bootstrapping from the value of the final state on timeouts
    PPO update                    update(): clipped surrogate, value loss, entropy, gradient clipping, minibatches
    training loop                 train()
    evaluation                    q3_rl/cross_eval.py (deterministic mean action, 100 seeded scenarios)
Settings follow the Phase 1 Stable-Baselines3 run (project main, scripts/rl/train_ppo.py).

    python q3_rl/ppo.py --world physics --timesteps 3000000 --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import exam2  # noqa: E402,F401  (puts the project/ submodule on the path)

HP = {"n_envs": 16, "n_steps": 256, "epochs": 10, "minibatch": 256, "lr": 3e-4, "gamma": 0.99, "lam": 0.95,
      "clip": 0.2, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5, "hidden": 256, "log_std_init": 0.0}


def make_dynamics(world):
    from exam2.dyn import LearnedModel, MujocoModel, PhysicsModel
    if world == "physics":
        return PhysicsModel()
    if world == "learned":
        return LearnedModel(ROOT / "q1_vision_ml/results/ensemble.npz")
    if world == "mujoco":
        return MujocoModel()
    raise ValueError(world)


def make_env(world):
    def thunk():
        torch.set_num_threads(1)
        sys.path.insert(0, str(ROOT))
        import exam2  # noqa: F401
        from go2x.sim.nav_env import Go2NavEnv
        return Go2NavEnv(dynamics=make_dynamics(world))
    return thunk


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_act, hidden):
        super().__init__()
        def mlp(out, gain):
            net = nn.Sequential(nn.Linear(n_obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))
            for i, layer in enumerate(l for l in net if isinstance(l, nn.Linear)):   # orthogonal initialization
                nn.init.orthogonal_(layer.weight, gain if i == 2 else np.sqrt(2))
                nn.init.zeros_(layer.bias)
            return net
        self.actor = mlp(n_act, 0.01)
        self.critic = mlp(1, 1.0)
        self.log_std = nn.Parameter(torch.full((n_act,), HP["log_std_init"]))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def gae(rew, val, done, last_val, gamma, lam):
    """Generalized advantage estimation over a (T, N) rollout. done marks the last step of an episode."""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    g = np.zeros(rew.shape[1])
    for t in reversed(range(T)):
        nxt = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nxt * (1 - done[t]) - val[t]
        g = delta + gamma * lam * (1 - done[t]) * g
        adv[t] = g
    return adv, adv + val


def update(ac, opt, b):
    """PPO clipped-surrogate update over minibatches; returns mean losses and the approximate KL divergence."""
    n = len(b["obs"])
    stats = []
    for _ in range(HP["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, HP["minibatch"]):
            j = perm[i:i + HP["minibatch"]]
            d = ac.dist(b["obs"][j])
            logp = d.log_prob(b["act"][j]).sum(-1)
            ratio = torch.exp(logp - b["logp"][j])
            adv = b["adv"][j]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - HP["clip"], 1 + HP["clip"])).mean()
            v = ac.value(b["obs"][j])
            vf = 0.5 * ((v - b["ret"][j]) ** 2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pg + HP["vf_coef"] * vf - HP["ent_coef"] * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), HP["max_grad_norm"])
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean()
            stats.append([pg.item(), vf.item(), ent.item(), kl.item()])
    return dict(zip(["policy_loss", "value_loss", "entropy", "approx_kl"], np.mean(stats, 0).tolist()))


def train(world, timesteps, seed, out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = gym.vector.AsyncVectorEnv([make_env(world) for _ in range(HP["n_envs"])],
                                     autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = envs.reset(seed=seed)
    ac = ActorCritic(envs.single_observation_space.shape[0], envs.single_action_space.shape[0], HP["hidden"])
    opt = torch.optim.Adam(ac.parameters(), lr=HP["lr"], eps=1e-5)
    N, T = HP["n_envs"], HP["n_steps"]
    log, ep_ret, ep_len = [], np.zeros(N), np.zeros(N)
    finished = []
    t0, steps = time.time(), 0
    while steps < timesteps:
        B = {k: np.zeros((T, N) + s, np.float32) for k, s in (("obs", obs.shape[1:]), ("act", (3,)))}
        rew, val, done, logp = (np.zeros((T, N)) for _ in range(4))
        for t in range(T):                                                     # collect a rollout
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = ac.dist(o)
                a = d.sample()
                logp[t] = d.log_prob(a).sum(-1).numpy()
                val[t] = ac.value(o).numpy()
            nobs, r, term, trunc, info = envs.step(a.numpy())
            B["obs"][t], B["act"][t] = obs, a.numpy()
            r = r.copy()
            if trunc.any():                                                    # bootstrap on timeouts
                fo = np.stack([info["final_obs"][i] if trunc[i] else nobs[i] for i in range(N)])
                with torch.no_grad():
                    fv = ac.value(torch.as_tensor(fo, dtype=torch.float32)).numpy()
                r = r + HP["gamma"] * fv * trunc
            rew[t], done[t] = r, np.logical_or(term, trunc)
            ep_ret += r
            ep_len += 1
            for i in np.flatnonzero(done[t]):
                fi = info["final_info"]
                finished.append({"return": float(ep_ret[i]), "length": int(ep_len[i]),
                                 "success": bool(fi["is_success"][i]), "collision": bool(fi["collision"][i])})
                ep_ret[i] = ep_len[i] = 0
            obs = nobs
            steps += N
        with torch.no_grad():
            last_val = ac.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        adv, ret = gae(rew, val, done, last_val, HP["gamma"], HP["lam"])
        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), dtype=torch.float32)  # noqa: E731
        batch = {"obs": flat(B["obs"]), "act": flat(B["act"]), "logp": flat(logp), "adv": flat(adv), "ret": flat(ret)}
        st = update(ac, opt, batch)
        recent = finished[-200:]
        row = {"steps": steps, "wall_s": round(time.time() - t0, 1), "episodes": len(finished),
               "success": float(np.mean([e["success"] for e in recent])) if recent else 0.0,
               "collision": float(np.mean([e["collision"] for e in recent])) if recent else 0.0,
               "return": float(np.mean([e["return"] for e in recent])) if recent else 0.0,
               "action_std": float(ac.log_std.exp().mean()), **st}
        log.append(row)
        if len(log) % 20 == 0:
            print(f"[ppo {world} s{seed}] {steps:>9,} steps {row['wall_s']:7.0f} s  success {row['success']:.2f} "
                  f"collision {row['collision']:.2f}  return {row['return']:6.2f}  std {row['action_std']:.2f}", flush=True)
    envs.close()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ac.state_dict(), out / "policy.pt")
    (out / "progress.json").write_text(json.dumps(log))
    (out / "run.json").write_text(json.dumps({"world": world, "seed": seed, "timesteps": steps, "hp": HP,
                                              "wall_s": round(time.time() - t0, 1),
                                              "steps_per_s": round(steps / (time.time() - t0))}, indent=1))
    print(f"[ppo {world} s{seed}] done: {steps:,} steps in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=["physics", "learned", "mujoco"])
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / f"q3_rl/runs/{args.world}_s{args.seed}"
    train(args.world, args.timesteps, args.seed, out)


if __name__ == "__main__":
    main()
