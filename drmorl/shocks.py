"""Shock process, uncertainty structures S1-S5 and the candidate support (Sections 4.1, 5.2).

Shock components are ordered (g, d, b, iota): output growth, energy demand,
technological progress and emission intensity.
"""
import numpy as np

CLIP_KEYS = ("g", "d", "b", "iota")


def _std_t(df, shape, rng):
    """Student-t draws rescaled to unit variance (df > 2)."""
    return rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2.0))


def scenario_spec(cfg, name):
    """Complete specification of an uncertainty structure (nominal + overrides)."""
    sh = cfg["shocks"]
    spec = dict(rec_prob=sh["rec_prob"], d_df=sh["d_df"], iota_df=None,
                scale=1.0, mean_shift=[0.0, 0.0, 0.0, 0.0], regime=None)
    over = {k: v for k, v in cfg["scenarios"][name].items() if k != "label"}
    spec.update(over)
    return spec


def clip_shocks(xi, cfg):
    cl = cfg["shocks"]["clip"]
    lo = np.array([cl[k][0] for k in CLIP_KEYS])
    hi = np.array([cl[k][1] for k in CLIP_KEYS])
    return np.clip(xi, lo, hi)


def draw_shocks(cfg, spec, n, T, rng):
    """Draw an (n, T, 4) array of annual shock vectors from an uncertainty structure.

    * growth: Gaussian innovation plus a recession regime that lowers growth by
      ``rec_shift``; the innovation is centred at the nominal recession
      probability, so that the nominal process has zero mean;
    * demand: scaled Student-t (``d_df`` degrees of freedom, unit-variance scaling);
    * technology: Gaussian;
    * intensity: correlated with demand (``rho``), Gaussian unless ``iota_df`` is set;
    * ``scale`` multiplies all innovations, ``mean_shift`` shifts their means and
      ``regime`` adds a persistent Markov crisis regime.
    """
    sh = cfg["shocks"]
    shape = (n, T)
    g = rng.normal(0.0, sh["g_sd"], shape)
    rec = rng.random(shape) < spec["rec_prob"]
    g = g - sh["rec_shift"] * (rec.astype(float) - sh["rec_prob"])

    zd = _std_t(spec["d_df"], shape, rng) if spec["d_df"] else rng.standard_normal(shape)
    ze = _std_t(spec["iota_df"], shape, rng) if spec["iota_df"] else rng.standard_normal(shape)
    d = sh["d_sd"] * zd
    b = rng.normal(0.0, sh["b_sd"], shape)
    iota = sh["iota_sd"] * (sh["rho"] * zd + np.sqrt(1.0 - sh["rho"] ** 2) * ze)

    xi = np.stack([g, d, b, iota], axis=-1) * spec["scale"]
    xi = xi + np.asarray(spec["mean_shift"], dtype=float)

    reg = spec.get("regime")
    if reg:
        u = rng.random(shape)
        crisis = np.zeros(shape, dtype=bool)
        state = np.zeros(n, dtype=bool)
        for t in range(T):
            state = np.where(state, u[:, t] < reg["stay"], u[:, t] < reg["enter"])
            crisis[:, t] = state
        xi = xi + crisis[..., None] * np.asarray(reg["shift"], dtype=float)
    return clip_shocks(xi, cfg)


def draw_record(cfg, N, rng):
    """Observed record of N annual shocks from the nominal process (S1)."""
    return draw_shocks(cfg, scenario_spec(cfg, "S1"), 1, N, rng)[0]


class ShockModel:
    """Everything the learners know about the shocks, built from the record only."""

    def __init__(self, cfg, record):
        self.cfg = cfg
        self.record = np.asarray(record, dtype=float)
        self.N = self.record.shape[0]
        self.mean = self.record.mean(0)
        self.cov = np.cov(self.record, rowvar=False)
        self.sd = self.record.std(0, ddof=1)   # diagonal of Sigma^{1/2}, Eq. (wass)

    # transport cost  d(xi, xi') = || Sigma^{-1/2} (xi - xi') ||_2
    def transport_cost(self, support):
        diff = (self.record[:, None, :] - support[None, :, :]) / self.sd
        return np.sqrt((diff ** 2).sum(-1))                        # (N, M)

    def augmented_support(self, M, rng):
        """N observed shocks plus M-N perturbed resampled observations."""
        extra = M - self.N
        if extra <= 0:
            return self.record.copy()
        idx = rng.integers(0, self.N, extra)
        noise = self.cfg["shocks"]["perturb_sd"] * self.sd * rng.standard_normal((extra, 4))
        pts = clip_shocks(self.record[idx] + noise, self.cfg)
        return np.concatenate([self.record, pts], axis=0)

    def gaussian_support(self, M, rng):
        """M scenarios from the multivariate Gaussian fitted to the record (SP)."""
        return clip_shocks(rng.multivariate_normal(self.mean, self.cov, size=M), self.cfg)

    def resample(self, shape, rng):
        """i.i.d. draws from the empirical distribution; shape + (4,)."""
        return self.record[rng.integers(0, self.N, size=shape)]

    def gaussian_draw(self, shape, rng):
        n = int(np.prod(shape))
        x = rng.multivariate_normal(self.mean, self.cov, size=n).reshape(tuple(shape) + (4,))
        return clip_shocks(x, self.cfg)
