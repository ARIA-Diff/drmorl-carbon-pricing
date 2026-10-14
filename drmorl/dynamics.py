"""Environmental-economic dynamics, Section 3.1-3.2 of the manuscript.

State  s_t = (p_{t-1}, sigma_t, Y_t, P^b_t, C_t)                    Eq. (state)
Action a_t = p_t in [max(0,p_{t-1}-D), min(p_bar,p_{t-1}+D)]         Eq. (action)
Shock  xi_t = (g_t, d_t, b_t, iota_t)                                Eq. (shock)
All functions broadcast over leading dimensions.
"""
import numpy as np


class CarbonEconomy:
    def __init__(self, cfg):
        e = cfg["economy"]
        self.T = int(e["T"])
        self.Y0, self.sigma0 = e["Y0"], e["sigma0"]
        self.g_sigma, self.Pb0, self.tau_b = e["g_sigma"], e["Pb0"], e["tau_b"]
        self.theta2, self.phi = e["theta2"], e["phi"]
        self.p_init, self.p_max, self.dp_max = e["p_init"], e["p_max"], e["dp_max"]
        self.B, self.delta = e["budget"], e["delta"]
        self.gamma, self.H = e["gamma"], e["H"]
        self.E0 = self.sigma0 * self.Y0
        self.n_actions = int(cfg["algo"]["n_actions"])
        self.w = np.asarray(cfg["weights"], dtype=float)

        t = np.arange(self.T + 1)
        # trend growth rate and business-as-usual (BAU) trend paths
        self.gbar = e["g_floor"] + e["g_amp"] * np.exp(-t / e["g_tau"])
        self.sigma_bar = self.sigma0 * (1.0 - self.g_sigma) ** t
        self.Y_bar = self.Y0 * np.concatenate([[1.0], np.cumprod(1.0 + self.gbar[:-1])])
        self.Pb_bar = self.Pb0 * (1.0 - self.tau_b) ** t
        self._u = np.linspace(-1.0, 1.0, self.n_actions)

    # ------------------------------------------------------------------ state
    def initial_state(self, n):
        s = np.empty((n, 5))
        s[:, 0] = self.p_init
        s[:, 1] = self.sigma0
        s[:, 2] = self.Y0
        s[:, 3] = self.Pb0
        s[:, 4] = 0.0
        return s

    # ---------------------------------------------------------------- actions
    def action_bounds(self, p_prev):
        lo = np.maximum(0.0, p_prev - self.dp_max)
        hi = np.minimum(self.p_max, p_prev + self.dp_max)
        return lo, hi

    def action_grid(self, p_prev):
        """25 admissible prices: p_prev + D*u, u equally spaced in [-1,1], clipped."""
        g = np.asarray(p_prev)[..., None] + self.dp_max * self._u
        return np.clip(g, 0.0, self.p_max)

    # --------------------------------------------------------- abatement cost
    def abatement(self, p, Pb):
        """mu = min{1, (p/Pb)^(1/(theta2-1))}, Eq. (abate)."""
        return np.minimum(1.0, (np.maximum(p, 0.0) / Pb) ** (1.0 / (self.theta2 - 1.0)))

    def cost_share(self, mu, Pb, sigma):
        """Lambda = theta1 * mu^theta2, theta1 = Pb*sigma/(1000*theta2), Eq. (cost)."""
        theta1 = Pb * sigma / (1000.0 * self.theta2)
        return theta1 * mu ** self.theta2

    # -------------------------------------------------------------- transition
    def step(self, t, s, a, xi):
        """One transition s_{t+1} = f_t(s_t, a_t, xi_t) together with the stage costs.

        Returns (s_next, out) with out = dict(E, Lam, mu, loss, c) where
        c = (c1, c2, c3) from Eq. (costs) stacked on the last axis.
        """
        p_prev, sigma, Y, Pb, C = (s[..., k] for k in range(5))
        g, d, b, iota = (xi[..., k] for k in range(4))
        mu = self.abatement(a, Pb)
        Lam = self.cost_share(mu, Pb, sigma)
        E = sigma * Y * (1.0 - mu) * np.exp(d)                         # Eq. (emis)
        Y1 = Y * (1.0 + self.gbar[t] + g) * (1.0 - self.phi * Lam)     # Eq. (output)
        sigma1 = sigma * (1.0 - self.g_sigma) * np.exp(iota)           # Eq. (intensity)
        Pb1 = Pb * (1.0 - self.tau_b - b)                              # Eq. (backstop)
        C1 = C + E                                                     # Eq. (cum)
        loss = Lam * Y
        c1 = 10.0 * E / self.E0
        c2 = loss / (0.01 * self.Y0)
        c3 = ((a - p_prev) / 10.0) ** 2
        parts = np.broadcast_arrays(a, sigma1, Y1, Pb1, C1)
        s_next = np.stack(parts, axis=-1)
        c = np.stack(np.broadcast_arrays(c1, c2, c3), axis=-1)
        return s_next, dict(E=E, Lam=Lam, mu=mu, loss=loss, c=c)

    def stage_cost_nominal(self, t, s, a, w):
        """Known part of the Q-function: w' c_t(s, a, xi=0)."""
        zero = np.zeros(np.broadcast(s[..., 0], a).shape + (4,))
        _, out = self.step(t, s, a, zero)
        return out["c"] @ w

    def terminal_cost(self, s, w, nu):
        """c_T^{w,nu}(s_T) = H (w1 c1 + w2 c2)|_{p=p_{T-1}} + nu 10 (C_T-B)_+ / E0, Eq. (lagr)."""
        p_prev, sigma, Y, Pb, C = (s[..., k] for k in range(5))
        mu = self.abatement(p_prev, Pb)
        Lam = self.cost_share(mu, Pb, sigma)
        E = sigma * Y * (1.0 - mu)
        c1 = 10.0 * E / self.E0
        c2 = Lam * Y / (0.01 * self.Y0)
        return self.H * (w[0] * c1 + w[1] * c2) + nu * 10.0 * np.maximum(C - self.B, 0.0) / self.E0
