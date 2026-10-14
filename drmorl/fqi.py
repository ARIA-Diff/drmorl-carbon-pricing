"""Model-informed fitted Q-iteration with robust targets, Section 4.1 / Algorithm 1.

    Q_t(s,a) = c_t^w(s,a,xi=0) + phi_t(s,a)' theta_t                      Eq. (param)
    Z_lj     = c_t^w(s_l,a_l,xi_j) + gamma min_a' Q_{t+1}(s'_lj, a')      Eq. (targets)
    y_l      = worst-case mean-CVaR of {Z_lj}_j over the Wasserstein ball Eq. (robusttarget)
    theta_t  = ridge regression of y_l - c_t^w on phi_t                   Eq. (ridge)
    pi_t(s)  = argmin over the 25-point action grid of Q_t(s,a)           Eq. (greedy)
"""
import numpy as np

from .risk import robust_target
from .utils import Timer, make_rng

_IU = np.triu_indices(6)
N_FEATURES = 1 + 6 + 21 + 3          # = 31 features
_CHUNK = 20000


class QPolicy:
    """Stage-wise linear value function and its greedy pricing rule."""

    def __init__(self, econ, spec, w, nu):
        self.econ, self.spec = econ, spec
        self.w, self.nu = np.asarray(w, dtype=float), float(nu)
        self.theta = np.zeros((econ.T, N_FEATURES))
        self.r2 = np.full(econ.T, np.nan)
        self.train_time = 0.0

    # --------------------------------------------------------------- features
    def features(self, t, s, a):
        e = self.econ
        x = np.stack(np.broadcast_arrays(
            s[..., 0] / 100.0,
            10.0 * np.log(s[..., 1] / e.sigma_bar[t]),
            10.0 * np.log(s[..., 2] / e.Y_bar[t]),
            10.0 * np.log(s[..., 3] / e.Pb_bar[t]),
            (s[..., 4] - e.B * t / e.T) / (5.0 * e.E0),
            a / 100.0), axis=-1)
        quad = x[..., _IU[0]] * x[..., _IU[1]]
        pa, pp = x[..., 5], x[..., 0]
        extra = np.stack([pa ** 3, pa ** 4, pa ** 3 * pp], axis=-1)
        one = np.ones(x.shape[:-1] + (1,))
        return np.concatenate([one, x, quad, extra], axis=-1)

    # ---------------------------------------------------------- value function
    def q(self, t, s, a):
        out = self.features(t, s, a) @ self.theta[t]
        if self.spec.offset:
            out = out + self.econ.stage_cost_nominal(t, s, a, self.w)
        return out

    def value_min(self, t, s):
        """min_{a' in A(s)} Q_t(s, a'); equals the terminal cost for t = T."""
        if t >= self.econ.T:
            return self.econ.terminal_cost(s, self.w, self.nu)
        shape = s.shape[:-1]
        flat = s.reshape(-1, 5)
        out = np.empty(flat.shape[0])
        for i in range(0, flat.shape[0], _CHUNK):
            sc = flat[i:i + _CHUNK]
            grid = self.econ.action_grid(sc[:, 0])
            out[i:i + _CHUNK] = self.q(t, sc[:, None, :], grid).min(1)
        return out.reshape(shape)

    def greedy(self, t, s):
        flat = s.reshape(-1, 5)
        a = np.empty(flat.shape[0])
        for i in range(0, flat.shape[0], _CHUNK):
            sc = flat[i:i + _CHUNK]
            grid = self.econ.action_grid(sc[:, 0])
            idx = self.q(t, sc[:, None, :], grid).argmin(1)
            a[i:i + _CHUNK] = grid[np.arange(sc.shape[0]), idx]
        return a.reshape(s.shape[:-1])

    __call__ = greedy


def explore_states(econ, sm, spec, n_s, rng):
    """Training states for t = 0..T-1 from the economy under a randomised price policy."""
    s = econ.initial_state(n_s)
    states = []
    for t in range(econ.T):
        states.append(s)
        lo, hi = econ.action_bounds(s[:, 0])
        a = rng.uniform(lo, hi)
        if spec.support == "gaussian":
            xi = sm.gaussian_draw((n_s,), rng)
        else:
            xi = sm.resample((n_s,), rng)
        s, _ = econ.step(t, s, a, xi)
    return states


def train_policy(econ, spec, sm, w, nu, cfg, seed, n_s=None, rep=0):
    """Backward fitted Q-iteration (inner loop of Algorithm 1) for given (w, nu)."""
    algo = cfg["algo"]
    n_s = int(n_s or algo["n_s"])
    T, gamma = econ.T, econ.gamma
    rng_x = make_rng(seed, 101, rep)       # exploratory states
    rng_a = make_rng(seed, 102, rep)       # training actions
    rng_sup = make_rng(seed, 103, rep)     # candidate support / SP scenarios
    rng_one = make_rng(seed, 104, rep)     # single transitions (model-free RL)

    pol = QPolicy(econ, spec, w, nu)
    with Timer() as tm:
        states = explore_states(econ, sm, spec, n_s, rng_x)

        # candidate support and transport costs (Algorithm 1, line 1)
        d, n_exact, support = None, None, None      # n_exact=None: all atoms are equally weighted
        if spec.support == "mean":
            support = sm.mean[None, :]
        elif spec.support == "augmented":
            if spec.eps > 0.0:
                support = sm.augmented_support(spec.M, rng_sup)
                d = sm.transport_cost(support)
            else:
                support = sm.record                  # eps = 0: empirical distribution
            n_exact = sm.N

        ridge = algo["ridge"]
        for t in range(T - 1, -1, -1):
            s = states[t]
            lo, hi = econ.action_bounds(s[:, 0])
            a = rng_a.uniform(lo, hi)

            if spec.support == "single":
                xi = sm.resample((n_s, 1), rng_one)
            elif spec.support == "gaussian":
                xi = sm.gaussian_support(spec.M, rng_sup)[None, :, :]
            else:
                xi = support[None, :, :]

            s_next, out = econ.step(t, s[:, None, :], a[:, None], xi)
            Z = out["c"] @ pol.w + gamma * pol.value_min(t + 1, s_next)     # (n_s, M)

            y = robust_target(Z, d, spec.eps, spec.alpha, spec.kappa, n_exact or Z.shape[1],
                              algo["K_eta"], algo["K_lam"],
                              algo["eta_levels"], algo["lam_log10"])

            Phi = pol.features(t, s, a)
            off = econ.stage_cost_nominal(t, s, a, pol.w) if spec.offset else 0.0
            A = Phi.T @ Phi + ridge * n_s * np.eye(N_FEATURES)
            pol.theta[t] = np.linalg.solve(A, Phi.T @ (y - off))

            fit = off + Phi @ pol.theta[t]
            sst = np.sum((y - y.mean()) ** 2)
            pol.r2[t] = 1.0 - np.sum((y - fit) ** 2) / sst if sst > 0 else np.nan
    pol.train_time = tm.elapsed
    return pol
