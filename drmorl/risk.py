"""Worst-case mean-CVaR over a Wasserstein ball, Lemma 1 / Eq. (robusttarget).

    y = min_{eta, lambda>=0} { lambda*eps + (1/N) sum_i max_j [ h_eta(Z_j) - lambda d_ij ] }
    h_eta(z) = (1-kappa) z + kappa ( eta + (z-eta)_+ / (1-alpha) )

The minimisation is carried out on a grid of empirical quantiles for eta and a
logarithmic grid for lambda scaled by the spread of the targets (upper
approximation).  For eps = 0 the empirical mean-CVaR is computed exactly by sorting.
"""
import numpy as np

from .utils import mean_cvar_empirical


def robust_target(Z, d, eps, alpha, kappa, n_exact, K_eta, K_lam,
                  eta_levels=(0.5, 1.0), lam_log10=(-2.0, 1.0), max_block=2.0e7):
    """Robust targets for each row of Z.

    Z        : (n, M) cost-to-go values under the M candidate shocks
    d        : (N, M) transport costs between observed (rows) and candidate (columns) shocks
    n_exact  : number of leading atoms forming the empirical distribution when eps = 0
    """
    Z = np.asarray(Z, dtype=float)
    n, M = Z.shape
    if eps <= 0.0:
        return mean_cvar_empirical(Z[:, :n_exact], alpha, kappa)

    N = d.shape[0]
    k_eta = K_eta if kappa > 0.0 else 1          # eta is irrelevant for kappa = 0
    levels = np.linspace(eta_levels[0], eta_levels[1], K_eta)
    lam_mult = np.logspace(lam_log10[0], lam_log10[1], K_lam)
    block = max(1, int(max_block // (k_eta * K_lam * N * M)))
    out = np.empty(n)
    for a in range(0, n, block):
        Zc = Z[a:a + block]                                            # (c, M)
        c = Zc.shape[0]
        spread = Zc.max(1) - Zc.min(1) + 1e-12
        lam = spread[:, None] * lam_mult[None, :]                      # (c, K_lam)
        if kappa > 0.0:
            eta = np.quantile(Zc, levels, axis=1).T                    # (c, K_eta)
        else:
            eta = np.zeros((c, 1))
        H = ((1.0 - kappa) * Zc[:, None, :]
             + kappa * (eta[:, :, None] + np.maximum(Zc[:, None, :] - eta[:, :, None], 0.0) / (1.0 - alpha)))
        # (c, k_eta, K_lam, N, M)
        V = H[:, :, None, None, :] - lam[:, None, :, None, None] * d[None, None, None, :, :]
        inner = V.max(-1).mean(-1)                                     # (c, k_eta, K_lam)
        val = lam[:, None, :] * eps + inner
        out[a:a + c] = val.reshape(c, -1).min(1)
    return out
