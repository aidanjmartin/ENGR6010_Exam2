"""Dynamics models of the dog's body velocity under velocity commands, all at the 0.05 s step:

- LagDelay: per-axis first-order lag with dead time (the Phase 1 physics model, project/go2x/sim/dynamics.py),
  vectorized over many rollouts.
- Ensemble: 5 probabilistic networks (Chua et al., PETS, NeurIPS 2018), each predicting a Gaussian over the next
  velocity change from the last H velocities and commands. numpy inference, so it can run inside a Gymnasium env.
"""
import numpy as np

DT = 0.05
H = 4                                   # history length (steps) in the ensemble input


class LagDelay:
    """v <- v + (1 - exp(-h/tau)) (u(t - delay) - v) on substeps h = DT/10 (same discretization as Phase 1)."""

    def __init__(self, tau, delay, substeps=10):
        self.tau, self.delay = np.asarray(tau, float), np.asarray(delay, float)
        self.h = DT / substeps
        self.n = substeps
        self.alpha = 1 - np.exp(-self.h / self.tau)
        self.lag = np.rint(self.delay / self.h).astype(int)

    def rollout(self, v0, cmd_past, cmd_future):
        """v0 (B,3); cmd_past (B,P,3) commands of the P steps before t (oldest first); cmd_future (B,K,3).
        Returns velocities after each of the K steps (B,K,3)."""
        B, K = cmd_future.shape[0], cmd_future.shape[1]
        sub = np.repeat(np.concatenate([cmd_past, cmd_future], 1), self.n, axis=1)   # substep commands
        start = cmd_past.shape[1] * self.n
        v = v0.copy()
        out = np.zeros((B, K, 3))
        ax = np.arange(3)
        for j in range(K * self.n):
            i = start + j
            u = sub[:, i - self.lag, ax]
            v = v + self.alpha * (u - v)
            if (j + 1) % self.n == 0:
                out[:, (j + 1) // self.n - 1] = v
        return out


def features(vel_hist, cmd_hist):
    """vel_hist, cmd_hist (B,H,3), oldest first; cmd_hist[:, -1] is the command held during the next step."""
    return np.concatenate([vel_hist.reshape(len(vel_hist), -1), cmd_hist.reshape(len(cmd_hist), -1)], 1)


class Ensemble:
    """numpy forward pass of the trained ensemble (weights from q1_vision_ml/train_ensemble.py)."""

    def __init__(self, path):
        z = np.load(path)
        self.n_members = int(z["n_members"])
        self.n_layers = int(z["n_layers"])
        self.x_mean, self.x_std = z["x_mean"], z["x_std"]
        self.y_mean, self.y_std = z["y_mean"], z["y_std"]
        self.W = [[z[f"m{m}_W{i}"] for i in range(self.n_layers)] for m in range(self.n_members)]
        self.b = [[z[f"m{m}_b{i}"] for i in range(self.n_layers)] for m in range(self.n_members)]
        self.max_logvar, self.min_logvar = z["max_logvar"], z["min_logvar"]

    def member(self, m, x):
        h = (x - self.x_mean) / self.x_std
        for i in range(self.n_layers):
            h = h @ self.W[m][i] + self.b[m][i]
            if i < self.n_layers - 1:
                h = h / (1 + np.exp(-h))                            # SiLU
        mu, lv = h[..., :3], h[..., 3:]
        lv = self.max_logvar - np.logaddexp(0, self.max_logvar - lv)
        lv = self.min_logvar + np.logaddexp(0, lv - self.min_logvar)
        return mu * self.y_std + self.y_mean, np.exp(lv) * self.y_std ** 2

    def predict(self, x):
        """Mean over members of the next velocity change, and total variance (mean aleatoric + epistemic)."""
        mus, vars_ = zip(*(self.member(m, x) for m in range(self.n_members)))
        mus, vars_ = np.stack(mus), np.stack(vars_)
        return mus.mean(0), vars_.mean(0) + mus.var(0)

    def rollout(self, vel_hist, cmd_hist, cmd_future, member=None):
        """Autoregressive rollout of K steps with known commands. vel_hist, cmd_hist (B,H,3) end at time t
        (cmd_hist excludes the command at t, which is cmd_future[:, 0]). member=None uses the ensemble mean."""
        vh, ch = vel_hist.copy(), cmd_hist.copy()
        K = cmd_future.shape[1]
        out = np.zeros((len(vh), K, 3))
        for k in range(K):
            full = np.concatenate([ch, cmd_future[:, k:k + 1]], 1)        # H commands, the last one held next
            x = features(vh, full)
            ch = full[:, 1:]
            dv = self.predict(x)[0] if member is None else self.member(member, x)[0]
            v = vh[:, -1] + dv
            out[:, k] = v
            vh = np.concatenate([vh[:, 1:], v[:, None]], 1)
        return out
